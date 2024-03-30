
import torch

from collections import OrderedDict

from comfy import model_base
from comfy import utils
from comfy import diffusers_convert

from comfy import sd2_clip

from comfy import supported_models_base
from comfy import latent_formats

from ..lvdm.modules.encoders.resampler import Resampler

DYNAMICRAFTER_CONFIG = {
    'in_channels': 8,
    'out_channels': 4,
    'model_channels': 320,
    'attention_resolutions': [4, 2, 1],
    'num_res_blocks': 2,
    'channel_mult': [1, 2, 4, 4],
    'num_head_channels': 64,
    'transformer_depth': 1,
    'context_dim': 1024,
    'use_linear': True,
    'use_checkpoint': False,
    'temporal_conv': True,
    'temporal_attention': True,
    'temporal_selfatt_only': True,
    'use_relative_position': False,
    'use_causal_attention': False,
    'temporal_length': 16,
    'addition_attention': True,
    'image_cross_attention': True,
    'image_cross_attention_scale_learnable': True,
    'default_fs': 3,
    'fs_condition': True
}

IMAGE_PROJ_CONFIG = {
    "dim": 1024,
    "depth": 4,
    "dim_head": 64,
    "heads": 12,
    "num_queries": 16,
    "embedding_dim": 1280,
    "output_dim": 1024,
    "ff_mult": 4,
    "video_length": 16
}

def process_list_or_str(target_key_or_keys, k):
    if isinstance(target_key_or_keys, list):
        return any([list_k in k for list_k in target_key_or_keys])
    else:
        return target_key_or_keys in k

def simple_state_dict_loader(state_dict: dict, target_key: str, target_dict: dict = None):
    out_dict = {}
    
    if target_dict is None:
        for k, v in state_dict.items():
            if process_list_or_str(target_key, k):
                out_dict[k] = v
    else:
        for k, v in target_dict.items():
            out_dict[k] = state_dict[k]
            
    return out_dict

def load_image_proj_dict(state_dict: dict):
    return simple_state_dict_loader(state_dict, 'image_proj')

def load_dynamicrafter_dict(state_dict: dict):
    return simple_state_dict_loader(state_dict, 'model.diffusion_model')

def load_vae_dict(state_dict: dict):
    return simple_state_dict_loader(state_dict, 'first_stage_model')

def get_base_model(state_dict: dict, version_checker=False):

    is_256_model = False

    for k in state_dict.keys():
         if "framestride_embed" in k:
            is_256_model = True
            break

def get_image_proj_model(state_dict: dict):

    state_dict = {k.replace('image_proj_model.', ''): v for k, v in state_dict.items()}
    #target_dict = Resampler().state_dict()

    ImageProjModel = Resampler(**IMAGE_PROJ_CONFIG)
    ImageProjModel.load_state_dict(state_dict)

    print("Image Projection Model loaded successfully")
    #del target_dict
    return ImageProjModel

class DynamiCrafterBase(supported_models_base.BASE):
    unet_config = {}
    unet_extra_config = {}

    latent_format = latent_formats.SD15

    def process_clip_state_dict(self, state_dict):
        replace_prefix = {}
        replace_prefix["conditioner.embedders.0.model."] = "clip_h." #SD2 in sgm format
        replace_prefix["cond_stage_model.model."] = "clip_h."
        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=True)
        state_dict = utils.clip_text_transformers_convert(state_dict, "clip_h.", "clip_h.transformer.")
        return state_dict

    def process_clip_state_dict_for_saving(self, state_dict):
        replace_prefix = {}
        replace_prefix["clip_h"] = "cond_stage_model.model"
        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix)
        state_dict = diffusers_convert.convert_text_enc_state_dict_v20(state_dict)
        return state_dict

    def clip_target(self):
        return supported_models_base.ClipTarget(sd2_clip.SD2Tokenizer, sd2_clip.SD2ClipModel)

    def process_dict_version(self, state_dict: dict):
        processed_dict = OrderedDict()
        is_eps = False
        
        for k in list(state_dict.keys()):
            if "framestride_embed" in k:
                new_key = k.replace("framestride_embed", "fps_embedding")
                processed_dict[new_key] = state_dict[k]
                is_eps = True
                continue
            
            processed_dict[k] = state_dict[k]

        return processed_dict, is_eps



