import torch
import comfy.model_management
from enum import Enum
from comfy.utils import load_torch_file
from comfy.conds import CONDRegular
from comfy_extras.nodes_compositing import JoinImageWithAlpha
from .model import ModelPatcher, TransparentVAEDecoder, calculate_weight_adjust_channel
from .attension_sharing import AttentionSharingPatcher
from ..config import LAYER_DIFFUSION, LAYER_DIFFUSION_DIR, LAYER_DIFFUSION_VAE
from ..libs.utils import to_lora_patch_dict, get_local_filepath, get_sd_version

load_layer_model_state_dict = load_torch_file
class LayerMethod(Enum):
    FG_ONLY_ATTN = "Attention Injection"
    FG_ONLY_CONV = "Conv Injection"
    FG_TO_BLEND = "Foreground"
    FG_BLEND_TO_BG = "Foreground to Background"
    BG_TO_BLEND = "Background"
    BG_BLEND_TO_FG = "Background to Foreground"

class LayerDiffuse:

    def __init__(self) -> None:
        self.vae_transparent_decoder = None
        self.frames = 1
        try:
            ModelPatcher.calculate_weight = calculate_weight_adjust_channel(ModelPatcher.calculate_weight)
        except:
            pass

    def get_layer_diffusion_method(self, method, has_blend_latent):
        method = LayerMethod(method)
        if method == LayerMethod.BG_TO_BLEND and has_blend_latent:
            method = LayerMethod.BG_BLEND_TO_FG
        elif method == LayerMethod.FG_TO_BLEND and has_blend_latent:
            method = LayerMethod.FG_BLEND_TO_BG
        return method

    def apply_layer_c_concat(self, cond, uncond, c_concat):
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

    def apply_layer_diffusion(self, model: ModelPatcher, method, weight, samples, blend_samples, positive, negative, control_img=None):
        sd_version = get_sd_version(model)
        model_url = LAYER_DIFFUSION[method.value][sd_version]["model_url"]
        if method in [LayerMethod.FG_ONLY_CONV, LayerMethod.FG_ONLY_ATTN] and sd_version == 'sd15':
            self.frames = 1
        elif method == [LayerMethod.BG_TO_BLEND, LayerMethod.FG_TO_BLEND] and sd_version == 'sd15':
            self.frames = 2
        elif method == [LayerMethod.BG_BLEND_TO_FG, LayerMethod.FG_BLEND_TO_BG] and sd_version == 'sd15':
            self.frames = 3
        if model_url is None:
            raise Exception(f"{method.value} is not supported for {sd_version} model")

        model_path = get_local_filepath(model_url, LAYER_DIFFUSION_DIR)
        layer_lora_state_dict = load_layer_model_state_dict(model_path)
        work_model = model.clone()
        if sd_version == 'sd15':
            patcher = AttentionSharingPatcher(
                work_model, self.frames, use_control=control_img is not None
            )
            patcher.load_state_dict(layer_lora_state_dict, strict=True)
            if control_img is not None:
                patcher.set_control(control_img)
        else:
            layer_lora_patch_dict = to_lora_patch_dict(layer_lora_state_dict)
            work_model.add_patches(layer_lora_patch_dict, weight)

        # cond_contact
        if method in [LayerMethod.FG_ONLY_ATTN, LayerMethod.FG_ONLY_CONV]:
            samp_model = work_model
        else:
            if method in [LayerMethod.BG_TO_BLEND, LayerMethod.FG_TO_BLEND]:
                c_concat = model.model.latent_format.process_in(samples["samples"])
            else:
                c_concat = model.model.latent_format.process_in(torch.cat([samples["samples"], blend_samples["samples"]], dim=1))
            samp_model, positive, negative = (work_model,) + self.apply_layer_c_concat(positive, negative, c_concat)

        return samp_model, positive, negative

    def join_image_with_alpha(self, image, alpha):
        out = image.movedim(-1, 1)
        if out.shape[1] == 3:  # RGB
            out = torch.cat([out, torch.ones_like(out[:, :1, :, :])], dim=1)
        for i in range(out.shape[0]):
            out[i, 3, :, :] = alpha
        return out.movedim(1, -1)

    def layer_diffusion_decode(self, layer_diffusion_method, latent, blend_samples, samp_images, model):
        alpha = None
        if layer_diffusion_method is not None:
            method = self.get_layer_diffusion_method(layer_diffusion_method, blend_samples is not None)
            print(method.value)
            if method in [LayerMethod.FG_ONLY_CONV, LayerMethod.FG_ONLY_ATTN, LayerMethod.BG_BLEND_TO_FG]:
                if self.vae_transparent_decoder is None:
                    sd_version = get_sd_version(model)
                    print(sd_version)
                    if sd_version not in ['sdxl', 'sd15']:
                        raise Exception(f"Only SDXL and SD1.5 model supported for Layer Diffusion")
                    model_url = LAYER_DIFFUSION_VAE['decode'][sd_version]["model_url"]
                    if model_url is None:
                        raise Exception(f"{method.value} is not supported for {sd_version} model")
                    decoder_file = get_local_filepath(model_url, LAYER_DIFFUSION_DIR)
                    self.vae_transparent_decoder = TransparentVAEDecoder(
                        load_torch_file(decoder_file),
                        device=comfy.model_management.get_torch_device(),
                        dtype=(torch.float16 if comfy.model_management.should_use_fp16() else torch.float32),
                    )

                pixel = samp_images.movedim(-1, 1)  # [B, H, W, C] => [B, C, H, W]
                decoded = []
                sub_batch_size = 16
                for start_idx in range(0, latent.shape[0], sub_batch_size):
                    decoded.append(
                        self.vae_transparent_decoder.decode_pixel(
                            pixel[start_idx: start_idx + sub_batch_size],
                            latent[start_idx: start_idx + sub_batch_size],
                        )
                    )
                pixel_with_alpha = torch.cat(decoded, dim=0)
                # [B, C, H, W] => [B, H, W, C]
                pixel_with_alpha = pixel_with_alpha.movedim(1, -1)
                image = pixel_with_alpha[..., 1:]
                alpha = pixel_with_alpha[..., 0]

                alpha = 1.0 - alpha
                new_images, = JoinImageWithAlpha().join_image_with_alpha(image, alpha)
            else:
                new_images = samp_images
        else:
            new_images = samp_images
        return (new_images, samp_images, alpha)