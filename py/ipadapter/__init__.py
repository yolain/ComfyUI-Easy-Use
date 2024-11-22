#credit to shakker-labs and instantX for this module
#from https://github.com/Shakker-Labs/ComfyUI-IPAdapter-Flux
import torch
from PIL import Image
import numpy as np
from .attention_processor import IPAFluxAttnProcessor2_0
from .utils import is_model_pathched, FluxUpdateModules

class MLPProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x


class InstantXFluxIpadapterApply:
    def __init__(self, num_tokens=128):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_tokens = num_tokens
        self.ip_ckpt = None
        self.clip_vision = None
        self.image_encoder = None
        self.clip_image_processor = None
        # state_dict
        self.state_dict = None
        self.joint_attention_dim = 4096
        self.hidden_size = 3072

    def set_ip_adapter(self, flux_model, weight, timestep_percent_range=(0.0, 1.0)):
        s = flux_model.model_sampling
        percent_to_timestep_function = lambda a: s.percent_to_sigma(a)
        timestep_range = (percent_to_timestep_function(timestep_percent_range[0]),
                          percent_to_timestep_function(timestep_percent_range[1]))
        ip_attn_procs = {}  # 19+38=57
        dsb_count = len(flux_model.diffusion_model.double_blocks)
        for i in range(dsb_count):
            name = f"double_blocks.{i}"
            ip_attn_procs[name] = IPAFluxAttnProcessor2_0(
                hidden_size=self.hidden_size,
                cross_attention_dim=self.joint_attention_dim,
                num_tokens=self.num_tokens,
                scale=weight,
                timestep_range=timestep_range
            ).to(self.device, dtype=torch.bfloat16)
        ssb_count = len(flux_model.diffusion_model.single_blocks)
        for i in range(ssb_count):
            name = f"single_blocks.{i}"
            ip_attn_procs[name] = IPAFluxAttnProcessor2_0(
                hidden_size=self.hidden_size,
                cross_attention_dim=self.joint_attention_dim,
                num_tokens=self.num_tokens,
                scale=weight,
                timestep_range=timestep_range
            ).to(self.device, dtype=torch.bfloat16)
        return ip_attn_procs

    def load_ip_adapter(self, flux_model, weight, timestep_percent_range=(0.0, 1.0)):
        self.image_proj_model.load_state_dict(self.state_dict["image_proj"], strict=True)
        ip_attn_procs = self.set_ip_adapter(flux_model, weight, timestep_percent_range)
        ip_layers = torch.nn.ModuleList(ip_attn_procs.values())
        ip_layers.load_state_dict(self.state_dict["ip_adapter"], strict=True)
        return ip_attn_procs

    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        # outputs = self.clip_vision.encode_image(pil_image)
        # clip_image_embeds = outputs['image_embeds']
        # clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.bfloat16)
        # image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(
                clip_image.to(self.device, dtype=self.image_encoder.dtype)).pooler_output
            clip_image_embeds = clip_image_embeds.to(dtype=torch.bfloat16)
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.bfloat16)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        return image_prompt_embeds

    def apply_ipadapter_flux(self, model, ipadapter, image, weight, start_at, end_at):
        if "clipvision" in ipadapter:
            # self.clip_vision = ipadapter["clipvision"]['model']
            self.image_encoder = ipadapter["clipvision"]['model']['image_encoder'].to(self.device, dtype=torch.bfloat16)
            self.clip_image_processor = ipadapter["clipvision"]['model']['clip_image_processor']
        if "ipadapter" in ipadapter:
            self.ip_ckpt = ipadapter["ipadapter"]['file']
            self.state_dict = ipadapter["ipadapter"]['model']

        pil_image = image.numpy()[0] * 255.0
        pil_image = Image.fromarray(pil_image.astype(np.uint8))
        # initialize ipadapter
        self.image_proj_model = MLPProjModel(
            cross_attention_dim=self.joint_attention_dim,  # 4096
            id_embeddings_dim=1152,
            num_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.bfloat16)
        ip_attn_procs = self.load_ip_adapter(model.model, weight, (start_at, end_at))
        # process control image
        image_prompt_embeds = self.get_image_embeds(pil_image=pil_image, clip_image_embeds=None)
        # set model
        is_patched = is_model_pathched(model.model)
        bi = model.clone()
        tyanochky = bi.model
        FluxUpdateModules(tyanochky, ip_attn_procs, image_prompt_embeds, is_patched)

        return (bi, image)

