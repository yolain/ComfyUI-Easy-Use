import os
import torch
import safetensors.torch
from typing import Tuple, TypedDict, Callable, NamedTuple

import folder_paths
import comfy.model_management
from comfy.diffusers_convert import convert_unet_state_dict
from comfy.model_patcher import ModelPatcher
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
from comfy.model_base import BaseModel
from comfy.conds import CONDRegular

class UnetParams(TypedDict):
    input: torch.Tensor
    timestep: torch.Tensor
    c: dict
    cond_or_uncond: torch.Tensor

class ICLight:

    @staticmethod
    def apply_c_concat(cond, uncond, c_concat: torch.Tensor):
        def write_c_concat(cond):
            new_cond = []
            for t in cond:
                n = [t[0], t[1].copy()]
                if "model_conds" not in n[1]:
                    n[1]["model_conds"] = {}
                n[1]["model_conds"]["c_concat"] = CONDRegular(c_concat)
                new_cond.append(n)
            return new_cond

        return (write_c_concat(cond), write_c_concat(uncond))

    @staticmethod
    def create_custom_conv(
        original_conv: torch.nn.Module,
        dtype: torch.dtype,
        device=torch.device,
    ) -> torch.nn.Module:
        with torch.no_grad():
            new_conv_in = torch.nn.Conv2d(
                8,
                original_conv.out_channels,
                original_conv.kernel_size,
                original_conv.stride,
                original_conv.padding,
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(original_conv.weight)
            new_conv_in.bias = original_conv.bias
            return new_conv_in.to(dtype=dtype, device=device)


    def apply(self, model_path, model: ModelPatcher, c_concat: dict,) -> Tuple[ModelPatcher]:

        device = comfy.model_management.get_torch_device()
        dtype = comfy.model_management.unet_dtype()
        work_model = model.clone()
        c_concat_samples: torch.Tensor = c_concat["samples"]

        def wrapped_unet(unet_apply: Callable, params: UnetParams):
            # Apply concat.
            sample = params["input"]
            params["c"]["c_concat"] = torch.cat(
                (
                        [c_concat_samples.to(sample.device)]
                        * (sample.shape[0] // c_concat_samples.shape[0])
                )
                + params["c"].get("c_concat", []),
                dim=0,
            )
            return unet_apply(x=sample, t=params["timestep"], **params["c"])

        work_model.add_object_patch(
            "diffusion_model.input_blocks.0.0",
            self.create_custom_conv(
                original_conv=work_model.get_model_object("diffusion_model.input_blocks.0.0"),
                dtype=dtype,
                device=device,
            ),
        )
        work_model.set_model_unet_function_wrapper(wrapped_unet)
        sd_offset = convert_unet_state_dict(safetensors.torch.load_file(model_path))

        work_model.add_patches(
            patches={
                ("diffusion_model." + key): (
                    sd_offset[key].to(dtype=dtype, device=device),
                )
                for key in sd_offset.keys()
            }
        )

        return work_model