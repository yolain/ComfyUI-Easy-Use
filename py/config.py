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

# fooocus
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

# layerDiffuse
LAYER_DIFFUSION_DIR = os.path.join(folder_paths.models_dir, "layer_model")
LAYER_DIFFUSION_VAE = {
    "encode": {
        "sdxl": {
            "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/vae_transparent_encoder.safetensors"
        }
    },
    "decode": {
        "sd15": {
            "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_sd15_vae_transparent_decoder.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/vae_transparent_decoder.safetensors"
        }
    }
}
LAYER_DIFFUSION = {
    "Attention Injection": {
        "sd15": {
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
        "sd15": {
            "model_url": None
        }
    },
    "Everything": {
        "sd15": {
          "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_sd15_joint.safetensors"
        },
        "sdxl": {
            "model_url": None
        }
    },
    "Foreground": {
        "sd15": {
          "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_sd15_fg2bg.safetensors"
        },
        "sdxl": {
           "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_fg2ble.safetensors"
        }
    },
    "Foreground to Background": {
        "sd15": {
          "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_sd15_fg2bg.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_fgble2bg.safetensors"
        }
    },
    "Background": {
        "sd15": {
          "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_sd15_bg2fg.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_bg2ble.safetensors"
        }
    },
    "Background to Foreground": {
        "sd15": {
            "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_sd15_bg2fg.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_bgble2fg.safetensors"
        }
    },
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
            "model_url": "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.safetensors"
        }
    },
    "VIT-G (medium strength)": {
        "sd15": {
            "model_url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_vit-G.safetensors"
        },
        "sdxl": {
            "model_url": "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors"
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
    }
}