import os
import folder_paths
from pathlib import Path

BASE_RESOLUTIONS = [
    ("自定义", "自定义"),
    (512, 512),
    (512, 768),
    (576, 1024),
    (768, 512),
    (768, 768),
    (768, 1024),
    (768, 1280),
    (768, 1344),
    (768, 1536),
    (816, 1920),
    (832, 1152),
    (896, 1152),
    (896, 1088),
    (1024, 1024),
    (1024, 576),
    (1024, 768),
    (1080, 1920),
    (1440, 2560),
    (1088, 896),
    (1152, 832),
    (1152, 896),
    (1280, 768),
    (1344, 768),
    (1536, 640),
    (1536, 768),
    (1920, 816),
    (1920, 1080),
    (2560, 1440),
]
MAX_SEED_NUM = 1125899906842624


RESOURCES_DIR = os.path.join(Path(__file__).parent.parent, "resources")

# inpaint
INPAINT_DIR = os.path.join(folder_paths.models_dir, "inpaint")
FOOOCUS_STYLES_DIR = os.path.join(Path(__file__).parent.parent, "styles")
FOOOCUS_STYLES_SAMPLES = 'https://raw.githubusercontent.com/lllyasviel/Fooocus/main/sdxl_styles/samples/'
FOOOCUS_INPAINT_HEAD = {
    "fooocus_inpaint_head": {
        "model_url": "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth"
    }
}
FOOOCUS_INPAINT_PATCH = {
    "inpaint_v26 (1.32GB)": {
        "model_url": "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch"
    },
    "inpaint_v25 (2.58GB)": {
        "model_url": "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v25.fooocus.patch"
    },
    "inpaint (1.32GB)": {
        "model_url": "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint.fooocus.patch"
    },
}
BRUSHNET_MODELS = {
    "random_mask": {
        "sd1": {
            "model_url": "https://huggingface.co/Kijai/BrushNet-fp16/resolve/main/brushnet_random_mask_fp16.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/yolain/brushnet/resolve/main/brushnet_random_mask_sdxl.safetensors"
        }
    },
    "segmentation_mask": {
        "sd1": {
            "model_url": "https://huggingface.co/Kijai/BrushNet-fp16/resolve/main/brushnet_segmentation_mask_fp16.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/yolain/brushnet/resolve/main/brushnet_segmentation_mask_sdxl.safetensors"
        }
    }
}
POWERPAINT_MODELS = {
    "base_fp16": {
        "model_url": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/text_encoder/model.fp16.safetensors"
    },
    "v2.1": {
        "model_url": "https://huggingface.co/JunhaoZhuang/PowerPaint-v2-1/resolve/main/PowerPaint_Brushnet/diffusion_pytorch_model.safetensors",
        "clip_url": "https://huggingface.co/JunhaoZhuang/PowerPaint-v2-1/resolve/main/PowerPaint_Brushnet/pytorch_model.bin",
    }
}

# layerDiffuse
LAYER_DIFFUSION_DIR = os.path.join(folder_paths.models_dir, "layer_model")
LAYER_DIFFUSION_VAE = {
    "encode": {
        "sdxl": {
            "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/vae_transparent_encoder.safetensors"
        }
    },
    "decode": {
        "sd1": {
            "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_sd15_vae_transparent_decoder.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/vae_transparent_decoder.safetensors"
        }
    }
}
LAYER_DIFFUSION = {
    "Attention Injection": {
        "sd1": {
          "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_sd15_transparent_attn.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_transparent_attn.safetensors"
        },
    },
    "Conv Injection": {
        "sdxl": {
            "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_transparent_conv.safetensors"
        },
        "sd1": {
            "model_url": None
        }
    },
    "Everything": {
        "sd1": {
          "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_sd15_joint.safetensors"
        },
        "sdxl": {
            "model_url": None
        }
    },
    "Foreground": {
        "sd1": {
          "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_sd15_fg2bg.safetensors"
        },
        "sdxl": {
           "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_fg2ble.safetensors"
        }
    },
    "Foreground to Background": {
        "sd1": {
          "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_sd15_fg2bg.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_fgble2bg.safetensors"
        }
    },
    "Background": {
        "sd1": {
          "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_sd15_bg2fg.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_bg2ble.safetensors"
        }
    },
    "Background to Foreground": {
        "sd1": {
            "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_sd15_bg2fg.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_bgble2fg.safetensors"
        }
    },
}

# IC Light
IC_LIGHT_MODELS = {
    "Foreground": {
        "sd1": {
            "model_url": "https://huggingface.co/huchenlei/IC-Light-ldm/resolve/main/iclight_sd15_fc_unet_ldm.safetensors"
        },
        "sdxl": {
            "model_url": None
        }
    },
    "Foreground&Background": {
        "sd1": {
            "model_url": "https://huggingface.co/huchenlei/IC-Light-ldm/resolve/main/iclight_sd15_fbc_unet_ldm.safetensors"
        },
        "sdxl": {
            "model_url": None
        }
    }
}


# REMBG
REMBG_DIR = os.path.join(folder_paths.models_dir, "rembg")
REMBG_MODELS = {
    "RMBG-1.4": {
        "model_url": "https://huggingface.co/briaai/RMBG-1.4/resolve/main/model.pth"
    }
}

#ipadapter
IPADAPTER_DIR = os.path.join(folder_paths.models_dir, "ipadapter")
IPADAPTER_MODELS = {
    "LIGHT - SD1.5 only (low strength)": {
        "sd15": {
            "model_url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_light_v11.bin"
        },
        "sdxl": {
            "model_url": ""
        }
    },
    "STANDARD (medium strength)": {
        "sd15": {
            "model_url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors"
        }
    },
    "VIT-G (medium strength)": {
        "sd15": {
            "model_url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_vit-G.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.safetensors"
        }
    },
    "PLUS (high strength)": {
        "sd15": {
            "model_url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors"
        }
    },
    "PLUS FACE (portraits)": {
        "sd15": {
            "model_url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors"
        }
    },
    "FULL FACE - SD1.5 only (portraits stronger)": {
        "sd15": {
            "model_url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-full-face_sd15.safetensors"
        },
        "sdxl": {
            "model_url": ""
        }
    },
    "FACEID": {
        "sd15": {
            "model_url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin",
            "lora_url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15_lora.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl.bin",
            "lora_url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl_lora.safetensors"
        }
    },
    "FACEID PLUS - SD1.5 only": {
        "sd15": {
            "model_url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plus_sd15.bin",
            "lora_url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plus_sd15_lora.safetensors"
        },
        "sdxl": {
            "model_url": "",
            "lora_url": ""
        }
    },
    "FACEID PLUS V2": {
        "sd15": {
            "model_url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin",
            "lora_url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15_lora.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl.bin",
            "lora_url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl_lora.safetensors"
        }
    },
    "FACEID PORTRAIT (style transfer)": {
        "sd15": {
            "model_url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait-v11_sd15.bin",
        },
        "sdxl": {
            "model_url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait_sdxl.bin",
        }
    },
    "COMPOSITION": {
        "sd15": {
            "model_url": "https://huggingface.co/ostris/ip-composition-adapter/resolve/main/ip_plus_composition_sd15.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/ostris/ip-composition-adapter/resolve/main/ip_plus_composition_sdxl.safetensors"
        }
    }
}

# dynamiCrafter
DYNAMICRAFTER_DIR = os.path.join(folder_paths.models_dir, "dynamicrafter_models")
DYNAMICRAFTER_MODELS = {
    "dynamicrafter_unet_512 (2.98GB)": {
        "model_url": "https://huggingface.co/ExponentialML/DynamiCrafterUNet/resolve/main/dynamicrafter_unet_512.safetensors",
        "vae_url": "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors",
        "clip_url": "https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/text_encoder/model.safetensors",
        "clip_vision_url": "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.safetensors",
    },
    "dynamicrafter_unet_512_interp (2.98GB)": {
        "model_url": "https://huggingface.co/ExponentialML/DynamiCrafterUNet/resolve/main/dynamicrafter_unet_512_interp.safetensors"
    },
    "dynamicrafter_unet_1024 (2.98GB)": {
        "model_url": "https://huggingface.co/ExponentialML/DynamiCrafterUNet/resolve/main/dynamicrafter_unet_1024.safetensors"
    },
    "dynamicrafter_unet_256 (2.98GB)": {
        "model_url": "https://huggingface.co/ExponentialML/DynamiCrafterUNet/resolve/main/dynamicrafter_unet_256.safetensors"
    },
}

#humanParsing
HUMANPARSING_MODELS = {
    "parsing_lip": {
        "model_url": "https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/humanparsing/parsing_lip.onnx",
    },
}

#mediapipe
MEDIAPIPE_DIR = os.path.join(folder_paths.models_dir, "mediapipe")
MEDIAPIPE_MODELS = {
    "selfie_multiclass_256x256": {
        "model_url": "https://huggingface.co/yolain/selfie_multiclass_256x256/resolve/main/selfie_multiclass_256x256.tflite"
    }
}