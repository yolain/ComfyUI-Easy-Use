"""List of all HYDiT model types / settings"""
sampling_settings = {
	"beta_schedule" : "linear",
	"linear_start"  : 0.00085,
	"linear_end"    : 0.03,
	"timesteps"     : 1000,
}

from argparse import Namespace
hydit_args = Namespace(**{ # normally from argparse
	"infer_mode": "torch",
	"norm": "layer",
	"learn_sigma": True,
	"text_states_dim": 1024,
	"text_states_dim_t5": 2048,
	"text_len": 77,
	"text_len_t5": 256,
})

hydit_conf = {
	"G/2": { # Seems to be the main one
		"unet_config": {
			"depth"       :   40,
			"num_heads"   :   16,
			"patch_size"  :    2,
			"hidden_size" : 1408,
			"mlp_ratio" : 4.3637,
			"input_size": (1024//8, 1024//8),
			"args": hydit_args,
		},
		"sampling_settings" : sampling_settings,
	},
}

dtypes = ["default", "auto (comfy)", "FP32", "FP16", "BF16"]
devices = ["auto", "cpu", "gpu"]


# these are the same as regular DiT, I think
from ..config import dit_conf
for name in ["XL/2", "L/2", "B/2"]:
	hydit_conf[name] = {
		"unet_config": dit_conf[name]["unet_config"].copy(),
		"sampling_settings": sampling_settings,
	}
	hydit_conf[name]["unet_config"]["args"] = hydit_args