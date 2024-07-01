

import os
import torch
import comfy.supported_models_base
import comfy.latent_formats
import comfy.model_patcher
import comfy.model_base
import comfy.utils
import comfy.conds

from comfy import model_management
from tqdm import tqdm
from transformers import AutoTokenizer, modeling_utils
from transformers import T5Config, T5EncoderModel, BertConfig, BertModel

class EXM_HYDiT(comfy.supported_models_base.BASE):
    unet_config = {}
    unet_extra_config = {}
    latent_format = comfy.latent_formats.SDXL

    def __init__(self, model_conf):
        self.unet_config = model_conf.get("unet_config", {})
        self.sampling_settings = model_conf.get("sampling_settings", {})
        self.latent_format = self.latent_format()
        # UNET is handled by extension
        self.unet_config["disable_unet_model_creation"] = True

    def model_type(self, state_dict, prefix=""):
        return comfy.model_base.ModelType.V_PREDICTION


class EXM_HYDiT_Model(comfy.model_base.BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)

        for name in ["context_t5", "context_mask", "context_t5_mask"]:
            out[name] = comfy.conds.CONDRegular(kwargs[name])

        src_size_cond = kwargs.get("src_size_cond", None)
        if src_size_cond is not None:
            out["src_size_cond"] = comfy.conds.CONDRegular(torch.tensor(src_size_cond))

        return out


def load_hydit(model_path, model_conf):
    state_dict = comfy.utils.load_torch_file(model_path)
    state_dict = state_dict.get("model", state_dict)

    parameters = comfy.utils.calculate_parameters(state_dict)
    unet_dtype = model_management.unet_dtype(model_params=parameters)
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()

    # ignore fp8/etc and use directly for now
    manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device)
    if manual_cast_dtype:
        print(f"HunYuanDiT: falling back to {manual_cast_dtype}")
        unet_dtype = manual_cast_dtype

    model_conf = EXM_HYDiT(model_conf)
    model = EXM_HYDiT_Model(
        model_conf,
        model_type=comfy.model_base.ModelType.V_PREDICTION,
        device=model_management.get_torch_device()
    )

    from .models.models import HunYuanDiT
    model.diffusion_model = HunYuanDiT(
        **model_conf.unet_config,
        log_fn=tqdm.write,
    )

    model.diffusion_model.load_state_dict(state_dict)
    model.diffusion_model.dtype = unet_dtype
    model.diffusion_model.eval()
    model.diffusion_model.to(unet_dtype)

    model_patcher = comfy.model_patcher.ModelPatcher(
        model,
        load_device=load_device,
        offload_device=offload_device,
        current_device="cpu",
    )
    return model_patcher


# CLIP Model
class hyCLIPModel(torch.nn.Module):
	def __init__(self, textmodel_json_config=None, device="cpu", max_length=77, freeze=True, dtype=None):
		super().__init__()
		self.device = device
		self.dtype = dtype
		self.max_length = max_length
		if textmodel_json_config is None:
			textmodel_json_config = os.path.join(
				os.path.dirname(os.path.realpath(__file__)),
				f"config_clip.json"
			)
		config = BertConfig.from_json_file(textmodel_json_config)
		with modeling_utils.no_init_weights():
			self.transformer = BertModel(config)
		self.to(dtype)
		if freeze:
			self.freeze()

	def freeze(self):
		self.transformer = self.transformer.eval()
		for param in self.parameters():
			param.requires_grad = False

	def load_sd(self, sd):
		return self.transformer.load_state_dict(sd, strict=False)

	def to(self, *args, **kwargs):
		return self.transformer.to(*args, **kwargs)

class EXM_HyDiT_Tenc_Temp:
	def __init__(self, no_init=False, device="cpu", dtype=None, model_class="mT5", *kwargs):
		if no_init:
			return

		size = 8 if model_class == "mT5" else 2
		if dtype == torch.float32:
			size *= 2
		size *= (1024**3)

		if device == "auto":
			self.load_device = model_management.text_encoder_device()
			self.offload_device = model_management.text_encoder_offload_device()
			self.init_device = "cpu"
		elif device == "cpu":
			size = 0 # doesn't matter
			self.load_device = "cpu"
			self.offload_device = "cpu"
			self.init_device="cpu"
		elif device.startswith("cuda"):
			print("Direct CUDA device override!\nVRAM will not be freed by default.")
			size = 0 # not used
			self.load_device = device
			self.offload_device = device
			self.init_device = device
		else:
			self.load_device = model_management.get_torch_device()
			self.offload_device = "cpu"
			self.init_device="cpu"

		self.dtype = dtype
		self.device = self.load_device
		if model_class == "mT5":
			self.cond_stage_model = mT5Model(
				device         = self.load_device,
				dtype          = self.dtype,
			)
			tokenizer_args = {"subfolder": "t2i/mt5"} # web
			tokenizer_path = os.path.join( # local
				os.path.dirname(os.path.realpath(__file__)),
				"mt5_tokenizer",
			)
		else:
			self.cond_stage_model = hyCLIPModel(
				device         = self.load_device,
				dtype          = self.dtype,
			)
			tokenizer_args = {"subfolder": "t2i/tokenizer",} # web
			tokenizer_path = os.path.join( # local
				os.path.dirname(os.path.realpath(__file__)),
				"tokenizer",
			)
		# self.tokenizer = AutoTokenizer.from_pretrained(
			# "Tencent-Hunyuan/HunyuanDiT",
			# **tokenizer_args
		# )
		self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
		self.patcher = comfy.model_patcher.ModelPatcher(
			self.cond_stage_model,
			load_device    = self.load_device,
			offload_device = self.offload_device,
			current_device = self.load_device,
			size           = size,
		)

	def clone(self):
		n = EXM_HyDiT_Tenc_Temp(no_init=True)
		n.patcher = self.patcher.clone()
		n.cond_stage_model = self.cond_stage_model
		n.tokenizer = self.tokenizer
		return n

	def load_sd(self, sd):
		return self.cond_stage_model.load_sd(sd)

	def get_sd(self):
		return self.cond_stage_model.state_dict()

	def load_model(self):
		if self.load_device != "cpu":
			model_management.load_model_gpu(self.patcher)
		return self.patcher

	def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
		return self.patcher.add_patches(patches, strength_patch, strength_model)

	def get_key_patches(self):
		return self.patcher.get_key_patches()

# MT5 model
class mT5Model(torch.nn.Module):
	def __init__(self, textmodel_json_config=None, device="cpu", max_length=256, freeze=True, dtype=None):
		super().__init__()
		self.device = device
		self.dtype = dtype
		self.max_length = max_length
		if textmodel_json_config is None:
			textmodel_json_config = os.path.join(
				os.path.dirname(os.path.realpath(__file__)),
				f"config_mt5.json"
			)
		config = T5Config.from_json_file(textmodel_json_config)
		with modeling_utils.no_init_weights():
			self.transformer = T5EncoderModel(config)
		self.to(dtype)
		if freeze:
			self.freeze()

	def freeze(self):
		self.transformer = self.transformer.eval()
		for param in self.parameters():
			param.requires_grad = False

	def load_sd(self, sd):
		return self.transformer.load_state_dict(sd, strict=False)

	def to(self, *args, **kwargs):
		return self.transformer.to(*args, **kwargs)

