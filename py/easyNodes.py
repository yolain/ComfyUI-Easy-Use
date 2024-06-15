import sys, os, re, json, time
import torch
import folder_paths
import numpy as np
import comfy.utils, comfy.sample, comfy.samplers, comfy.controlnet, comfy.model_base, comfy.model_management
try:
    import comfy.sampler_helpers, comfy.supported_models
except:
    pass
from comfy.sd import CLIP, VAE
from comfy.model_patcher import ModelPatcher
from comfy_extras.chainner_models import model_loading
from comfy_extras.nodes_mask import LatentCompositeMasked, GrowMask
from comfy_extras.nodes_compositing import JoinImageWithAlpha
from comfy.clip_vision import load as load_clip_vision
from urllib.request import urlopen
from PIL import Image

from server import PromptServer
from nodes import MAX_RESOLUTION, LatentFromBatch, RepeatLatentBatch, NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS, ConditioningSetMask, ConditioningConcat, CLIPTextEncode, VAEEncodeForInpaint, InpaintModelConditioning
from .config import MAX_SEED_NUM, BASE_RESOLUTIONS, RESOURCES_DIR, INPAINT_DIR, FOOOCUS_STYLES_DIR, FOOOCUS_INPAINT_HEAD, FOOOCUS_INPAINT_PATCH, BRUSHNET_MODELS, POWERPAINT_MODELS, IPADAPTER_DIR, IPADAPTER_MODELS, DYNAMICRAFTER_DIR, DYNAMICRAFTER_MODELS, IC_LIGHT_MODELS
from .layer_diffuse import LayerDiffuse, LayerMethod
from .xyplot import XYplot_ModelMergeBlocks, XYplot_CFG, XYplot_Lora, XYplot_Checkpoint, XYplot_Denoise, XYplot_Steps, XYplot_PromptSR, XYplot_Positive_Cond, XYplot_Negative_Cond, XYplot_Positive_Cond_List, XYplot_Negative_Cond_List, XYplot_SeedsBatch, XYplot_Control_Net, XYplot_Sampler_Scheduler

from .libs.log import log_node_info, log_node_error, log_node_warn
from .libs.adv_encode import advanced_encode
from .libs.wildcards import process_with_loras, get_wildcard_list, process
from .libs.utils import find_wildcards_seed, is_linked_styles_selector, easySave, get_local_filepath, AlwaysEqualProxy, get_sd_version
from .libs.loader import easyLoader
from .libs.sampler import easySampler, alignYourStepsScheduler
from .libs.xyplot import easyXYPlot
from .libs.controlnet import easyControlnet
from .libs.conditioning import prompt_to_cond, set_cond
from .libs.easing import EasingBase
from .libs.translate import has_chinese, zh_to_en
from .libs import cache as backend_cache

sampler = easySampler()
easyCache = easyLoader()
# ---------------------------------------------------------------提示词 开始----------------------------------------------------------------------#

# 正面提示词
class positivePrompt:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "positive": ("STRING", {"default": "", "multiline": True, "placeholder": "Positive"}),}
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "main"

    CATEGORY = "EasyUse/Prompt"

    @staticmethod
    def main(positive):
        if has_chinese(positive):
            return zh_to_en([positive])
        return positive,

# 通配符提示词
class wildcardsPrompt:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        wildcard_list = get_wildcard_list()
        return {"required": {
            "text": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False, "placeholder": "(Support Lora Block Weight and wildcard)"}),
            "Select to add LoRA": (["Select the LoRA to add to the text"] + folder_paths.get_filename_list("loras"),),
            "Select to add Wildcard": (["Select the Wildcard to add to the text"] + wildcard_list,),
            "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED_NUM}),
            "multiline_mode": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "populated_text")
    OUTPUT_IS_LIST = (True, True)
    OUTPUT_NODE = True
    FUNCTION = "main"

    CATEGORY = "EasyUse/Prompt"

    def translate(self, text):
        if has_chinese(text):
            return zh_to_en([text])[0]
        else:
            return text

    def main(self, *args, **kwargs):
        prompt = kwargs["prompt"] if "prompt" in kwargs else None
        seed = kwargs["seed"]

        # Clean loaded_objects
        if prompt:
            easyCache.update_loaded_objects(prompt)

        text = kwargs['text']
        if "multiline_mode" in kwargs and kwargs["multiline_mode"]:
            populated_text = []
            _text = []
            text = text.split("\n")
            for t in text:
                t = self.translate(t)
                _text.append(t)
                populated_text.append(process(t, seed))
            text = _text
        else:
            text = self.translate(text)
            populated_text = [process(text, seed)]
            text = [text]
        return {"ui": {"value": [seed]}, "result": (text, populated_text)}

# 负面提示词
class negativePrompt:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "negative": ("STRING", {"default": "", "multiline": True, "placeholder": "Negative"}),}
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("negative",)
    FUNCTION = "main"

    CATEGORY = "EasyUse/Prompt"

    @staticmethod
    def main(negative):
        if has_chinese(negative):
            return zh_to_en([negative])
        else:
            return negative,

# 风格提示词选择器
class stylesPromptSelector:

    @classmethod
    def INPUT_TYPES(s):
        styles = ["fooocus_styles"]
        styles_dir = FOOOCUS_STYLES_DIR
        for file_name in os.listdir(styles_dir):
            file = os.path.join(styles_dir, file_name)
            if os.path.isfile(file) and file_name.endswith(".json") and "styles" in file_name.split(".")[0]:
                styles.append(file_name.split(".")[0])
        return {
            "required": {
               "styles": (styles, {"default": "fooocus_styles"}),
            },
            "optional": {
                "positive": ("STRING", {"forceInput": True}),
                "negative": ("STRING", {"forceInput": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("positive", "negative",)

    CATEGORY = 'EasyUse/Prompt'
    FUNCTION = 'run'
    OUTPUT_NODE = True

    def run(self, styles, positive='', negative='', prompt=None, extra_pnginfo=None, my_unique_id=None):
        values = []
        all_styles = {}
        positive_prompt, negative_prompt = '', negative
        if styles == "fooocus_styles":
            file = os.path.join(RESOURCES_DIR,  styles + '.json')
        else:
            file = os.path.join(FOOOCUS_STYLES_DIR, styles + '.json')
        f = open(file, 'r', encoding='utf-8')
        data = json.load(f)
        f.close()
        for d in data:
            all_styles[d['name']] = d
        if my_unique_id in prompt:
            if prompt[my_unique_id]["inputs"]['select_styles']:
                values = prompt[my_unique_id]["inputs"]['select_styles'].split(',')

        has_prompt = False
        if len(values) == 0:
            return (positive, negative)

        for index, val in enumerate(values):
            if 'prompt' in all_styles[val]:
                if "{prompt}" in all_styles[val]['prompt'] and has_prompt == False:
                    positive_prompt = all_styles[val]['prompt'].format(prompt=positive)
                    has_prompt = True
                else:
                    positive_prompt += ', ' + all_styles[val]['prompt'].replace(', {prompt}', '').replace('{prompt}', '')
            if 'negative_prompt' in all_styles[val]:
                negative_prompt += ', ' + all_styles[val]['negative_prompt'] if negative_prompt else all_styles[val]['negative_prompt']

        if has_prompt == False and positive:
            positive_prompt = positive + ', '

        return (positive_prompt, negative_prompt)

#prompt
class prompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Prompt"}),
            "main": ([
                'none',
                'beautiful woman, detailed face',
                'handsome man, detailed face',
                'pretty girl',
                'handsome boy',
                'dog',
                'cat',
                'Buddha',
                'toy'
            ], {"default": "none"}),
            "lighting": ([
                'none',
                'sunshine from window',
                'neon light, city',
                'sunset over sea',
                'golden time',
                'sci-fi RGB glowing, cyberpunk',
                'natural lighting',
                'warm atmosphere, at home, bedroom',
                'magic lit',
                'evil, gothic, Yharnam',
                'light and shadow',
                'shadow from window',
                'soft studio lighting',
                'home atmosphere, cozy bedroom illumination',
                'neon, Wong Kar-wai, warm',
                'cinemative lighting',
                'neo punk lighting, cyberpunk',
            ],{"default":'none'})
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "doit"

    CATEGORY = "EasyUse/Prompt"

    def doit(self, prompt, main, lighting):
        if has_chinese(prompt):
            prompt = zh_to_en([prompt])[0]
        if lighting != 'none' and main != 'none':
            prompt = main + ',' + lighting + ',' + prompt
        elif lighting != 'none' and main == 'none':
            prompt = prompt + ',' + lighting
        elif main != 'none':
            prompt = main + ',' + prompt

        return prompt,
#promptList
class promptList:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prompt_1": ("STRING", {"multiline": True, "default": ""}),
            "prompt_2": ("STRING", {"multiline": True, "default": ""}),
            "prompt_3": ("STRING", {"multiline": True, "default": ""}),
            "prompt_4": ("STRING", {"multiline": True, "default": ""}),
            "prompt_5": ("STRING", {"multiline": True, "default": ""}),
        },
            "optional": {
                "optional_prompt_list": ("LIST",)
            }
        }

    RETURN_TYPES = ("LIST", "STRING")
    RETURN_NAMES = ("prompt_list", "prompt_strings")
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "run"
    CATEGORY = "EasyUse/Prompt"

    def run(self, **kwargs):
        prompts = []

        if "optional_prompt_list" in kwargs:
            for l in kwargs["optional_prompt_list"]:
                prompts.append(l)

        # Iterate over the received inputs in sorted order.
        for k in sorted(kwargs.keys()):
            v = kwargs[k]

            # Only process string input ports.
            if isinstance(v, str) and v != '':
                if has_chinese(v):
                    v = zh_to_en([v])[0]
                prompts.append(v)

        return (prompts, prompts)

#promptLine
class promptLine:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "prompt": ("STRING", {"multiline": True, "default": "text"}),
                    "start_index": ("INT", {"default": 0, "min": 0, "max": 9999}),
                     "max_rows": ("INT", {"default": 1000, "min": 1, "max": 9999}),
                    },
            "hidden":{
                "workflow_prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ("STRING", AlwaysEqualProxy('*'))
    RETURN_NAMES = ("STRING", "COMBO")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "generate_strings"
    CATEGORY = "EasyUse/Prompt"

    def generate_strings(self, prompt, start_index, max_rows, workflow_prompt=None, my_unique_id=None):
        lines = prompt.split('\n')
        lines = [zh_to_en([v])[0] if has_chinese(v) else v for v in lines]

        start_index = max(0, min(start_index, len(lines) - 1))

        end_index = min(start_index + max_rows, len(lines))

        rows = lines[start_index:end_index]

        return (rows, rows)

class promptConcat:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
        },
            "optional": {
                "prompt1": ("STRING", {"multiline": False, "default": "", "forceInput": True}),
                "prompt2": ("STRING", {"multiline": False, "default": "", "forceInput": True}),
                "separator": ("STRING", {"multiline": False, "default": ""}),
            },
        }
    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("prompt", )
    FUNCTION = "concat_text"
    CATEGORY = "EasyUse/Prompt"

    def concat_text(self, prompt1="", prompt2="", separator=""):

        return (prompt1 + separator + prompt2,)

class promptReplace:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
            },
            "optional": {
                "find1": ("STRING", {"multiline": False, "default": ""}),
                "replace1": ("STRING", {"multiline": False, "default": ""}),
                "find2": ("STRING", {"multiline": False, "default": ""}),
                "replace2": ("STRING", {"multiline": False, "default": ""}),
                "find3": ("STRING", {"multiline": False, "default": ""}),
                "replace3": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "replace_text"
    CATEGORY = "EasyUse/Prompt"

    def replace_text(self, prompt, find1="", replace1="", find2="", replace2="", find3="", replace3=""):

        prompt = prompt.replace(find1, replace1)
        prompt = prompt.replace(find2, replace2)
        prompt = prompt.replace(find3, replace3)

        return (prompt,)


# 肖像大师
# Created by AI Wiz Art (Stefano Flore)
# Version: 2.2
# https://stefanoflore.it
# https://ai-wiz.art
class portraitMaster:

    @classmethod
    def INPUT_TYPES(s):
        max_float_value = 1.95
        prompt_path = os.path.join(RESOURCES_DIR, 'portrait_prompt.json')
        if not os.path.exists(prompt_path):
            response = urlopen('https://raw.githubusercontent.com/yolain/ComfyUI-Easy-Use/main/resources/portrait_prompt.json')
            temp_prompt = json.loads(response.read())
            prompt_serialized = json.dumps(temp_prompt, indent=4)
            with open(prompt_path, "w") as f:
                f.write(prompt_serialized)
            del response, temp_prompt
        # Load local
        with open(prompt_path, 'r') as f:
            list = json.load(f)
        keys = [
            ['shot', 'COMBO', {"key": "shot_list"}], ['shot_weight', 'FLOAT'],
            ['gender', 'COMBO', {"default": "Woman", "key": "gender_list"}], ['age', 'INT', {"default": 30, "min": 18, "max": 90, "step": 1, "display": "slider"}],
            ['nationality_1', 'COMBO', {"default": "Chinese", "key": "nationality_list"}], ['nationality_2', 'COMBO', {"key": "nationality_list"}], ['nationality_mix', 'FLOAT'],
            ['body_type', 'COMBO', {"key": "body_type_list"}], ['body_type_weight', 'FLOAT'], ['model_pose', 'COMBO', {"key": "model_pose_list"}], ['eyes_color', 'COMBO', {"key": "eyes_color_list"}],
            ['facial_expression', 'COMBO', {"key": "face_expression_list"}], ['facial_expression_weight', 'FLOAT'], ['face_shape', 'COMBO', {"key": "face_shape_list"}], ['face_shape_weight', 'FLOAT'], ['facial_asymmetry', 'FLOAT'],
            ['hair_style', 'COMBO', {"key": "hair_style_list"}], ['hair_color', 'COMBO', {"key": "hair_color_list"}], ['disheveled', 'FLOAT'], ['beard', 'COMBO', {"key": "beard_list"}],
            ['skin_details', 'FLOAT'], ['skin_pores', 'FLOAT'], ['dimples', 'FLOAT'], ['freckles', 'FLOAT'],
            ['moles', 'FLOAT'], ['skin_imperfections', 'FLOAT'], ['skin_acne', 'FLOAT'], ['tanned_skin', 'FLOAT'],
            ['eyes_details', 'FLOAT'], ['iris_details', 'FLOAT'], ['circular_iris', 'FLOAT'], ['circular_pupil', 'FLOAT'],
            ['light_type', 'COMBO', {"key": "light_type_list"}], ['light_direction', 'COMBO', {"key": "light_direction_list"}], ['light_weight', 'FLOAT']
        ]
        widgets = {}
        for i, obj in enumerate(keys):
            if obj[1] == 'COMBO':
                key = obj[2]['key'] if obj[2] and 'key' in obj[2] else obj[0]
                _list = list[key].copy()
                _list.insert(0, '-')
                widgets[obj[0]] = (_list, {**obj[2]})
            elif obj[1] == 'FLOAT':
                widgets[obj[0]] = ("FLOAT", {"default": 0, "step": 0.05, "min": 0, "max": max_float_value, "display": "slider",})
            elif obj[1] == 'INT':
                widgets[obj[0]] = (obj[1], obj[2])
        del list
        return {
            "required": {
                **widgets,
                "photorealism_improvement": (["enable", "disable"],),
                "prompt_start": ("STRING", {"multiline": True, "default": "raw photo, (realistic:1.5)"}),
                "prompt_additional": ("STRING", {"multiline": True, "default": ""}),
                "prompt_end": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("positive", "negative",)

    FUNCTION = "pm"

    CATEGORY = "EasyUse/Prompt"

    def pm(self, shot="-", shot_weight=1, gender="-", body_type="-", body_type_weight=0, eyes_color="-",
           facial_expression="-", facial_expression_weight=0, face_shape="-", face_shape_weight=0,
           nationality_1="-", nationality_2="-", nationality_mix=0.5, age=30, hair_style="-", hair_color="-",
           disheveled=0, dimples=0, freckles=0, skin_pores=0, skin_details=0, moles=0, skin_imperfections=0,
           wrinkles=0, tanned_skin=0, eyes_details=1, iris_details=1, circular_iris=1, circular_pupil=1,
           facial_asymmetry=0, prompt_additional="", prompt_start="", prompt_end="", light_type="-",
           light_direction="-", light_weight=0, negative_prompt="", photorealism_improvement="disable", beard="-",
           model_pose="-", skin_acne=0):

        prompt = []

        if gender == "-":
            gender = ""
        else:
            if age <= 25 and gender == 'Woman':
                gender = 'girl'
            if age <= 25 and gender == 'Man':
                gender = 'boy'
            gender = " " + gender + " "

        if nationality_1 != '-' and nationality_2 != '-':
            nationality = f"[{nationality_1}:{nationality_2}:{round(nationality_mix, 2)}]"
        elif nationality_1 != '-':
            nationality = nationality_1 + " "
        elif nationality_2 != '-':
            nationality = nationality_2 + " "
        else:
            nationality = ""

        if prompt_start != "":
            prompt.append(f"{prompt_start}")

        if shot != "-" and shot_weight > 0:
            prompt.append(f"({shot}:{round(shot_weight, 2)})")

        prompt.append(f"({nationality}{gender}{round(age)}-years-old:1.5)")

        if body_type != "-" and body_type_weight > 0:
            prompt.append(f"({body_type}, {body_type} body:{round(body_type_weight, 2)})")

        if model_pose != "-":
            prompt.append(f"({model_pose}:1.5)")

        if eyes_color != "-":
            prompt.append(f"({eyes_color} eyes:1.25)")

        if facial_expression != "-" and facial_expression_weight > 0:
            prompt.append(
                f"({facial_expression}, {facial_expression} expression:{round(facial_expression_weight, 2)})")

        if face_shape != "-" and face_shape_weight > 0:
            prompt.append(f"({face_shape} shape face:{round(face_shape_weight, 2)})")

        if hair_style != "-":
            prompt.append(f"({hair_style} hairstyle:1.25)")

        if hair_color != "-":
            prompt.append(f"({hair_color} hair:1.25)")

        if beard != "-":
            prompt.append(f"({beard}:1.15)")

        if disheveled != "-" and disheveled > 0:
            prompt.append(f"(disheveled:{round(disheveled, 2)})")

        if prompt_additional != "":
            prompt.append(f"{prompt_additional}")

        if skin_details > 0:
            prompt.append(f"(skin details, skin texture:{round(skin_details, 2)})")

        if skin_pores > 0:
            prompt.append(f"(skin pores:{round(skin_pores, 2)})")

        if skin_imperfections > 0:
            prompt.append(f"(skin imperfections:{round(skin_imperfections, 2)})")

        if skin_acne > 0:
            prompt.append(f"(acne, skin with acne:{round(skin_acne, 2)})")

        if wrinkles > 0:
            prompt.append(f"(skin imperfections:{round(wrinkles, 2)})")

        if tanned_skin > 0:
            prompt.append(f"(tanned skin:{round(tanned_skin, 2)})")

        if dimples > 0:
            prompt.append(f"(dimples:{round(dimples, 2)})")

        if freckles > 0:
            prompt.append(f"(freckles:{round(freckles, 2)})")

        if moles > 0:
            prompt.append(f"(skin pores:{round(moles, 2)})")

        if eyes_details > 0:
            prompt.append(f"(eyes details:{round(eyes_details, 2)})")

        if iris_details > 0:
            prompt.append(f"(iris details:{round(iris_details, 2)})")

        if circular_iris > 0:
            prompt.append(f"(circular iris:{round(circular_iris, 2)})")

        if circular_pupil > 0:
            prompt.append(f"(circular pupil:{round(circular_pupil, 2)})")

        if facial_asymmetry > 0:
            prompt.append(f"(facial asymmetry, face asymmetry:{round(facial_asymmetry, 2)})")

        if light_type != '-' and light_weight > 0:
            if light_direction != '-':
                prompt.append(f"({light_type} {light_direction}:{round(light_weight, 2)})")
            else:
                prompt.append(f"({light_type}:{round(light_weight, 2)})")

        if prompt_end != "":
            prompt.append(f"{prompt_end}")

        prompt = ", ".join(prompt)
        prompt = prompt.lower()

        if photorealism_improvement == "enable":
            prompt = prompt + ", (professional photo, balanced photo, balanced exposure:1.2), (film grain:1.15)"

        if photorealism_improvement == "enable":
            negative_prompt = negative_prompt + ", (shinny skin, reflections on the skin, skin reflections:1.25)"

        log_node_info("Portrait Master as generate the prompt:", prompt)

        return (prompt, negative_prompt,)

# ---------------------------------------------------------------提示词 结束----------------------------------------------------------------------#

# ---------------------------------------------------------------潜空间 开始----------------------------------------------------------------------#
# 潜空间sigma相乘
class latentNoisy:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            "steps": ("INT", {"default": 10000, "min": 0, "max": 10000}),
            "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
            "end_at_step": ("INT", {"default": 10000, "min": 1, "max": 10000}),
            "source": (["CPU", "GPU"],),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        },
        "optional": {
            "pipe": ("PIPE_LINE",),
            "optional_model": ("MODEL",),
            "optional_latent": ("LATENT",)
        }}

    RETURN_TYPES = ("PIPE_LINE", "LATENT", "FLOAT",)
    RETURN_NAMES = ("pipe", "latent", "sigma",)
    FUNCTION = "run"

    CATEGORY = "EasyUse/Latent"

    def run(self, sampler_name, scheduler, steps, start_at_step, end_at_step, source, seed, pipe=None, optional_model=None, optional_latent=None):
        model = optional_model if optional_model is not None else pipe["model"]
        batch_size = pipe["loader_settings"]["batch_size"]
        empty_latent_height = pipe["loader_settings"]["empty_latent_height"]
        empty_latent_width = pipe["loader_settings"]["empty_latent_width"]

        if optional_latent is not None:
            samples = optional_latent
        else:
            torch.manual_seed(seed)
            if source == "CPU":
                device = "cpu"
            else:
                device = comfy.model_management.get_torch_device()
            noise = torch.randn((batch_size, 4, empty_latent_height // 8, empty_latent_width // 8), dtype=torch.float32,
                                device=device).cpu()

            samples = {"samples": noise}

        device = comfy.model_management.get_torch_device()
        end_at_step = min(steps, end_at_step)
        start_at_step = min(start_at_step, end_at_step)
        comfy.model_management.load_model_gpu(model)
        model_patcher = comfy.model_patcher.ModelPatcher(model.model, load_device=device, offload_device=comfy.model_management.unet_offload_device())
        sampler = comfy.samplers.KSampler(model_patcher, steps=steps, device=device, sampler=sampler_name,
                                          scheduler=scheduler, denoise=1.0, model_options=model.model_options)
        sigmas = sampler.sigmas
        sigma = sigmas[start_at_step] - sigmas[end_at_step]
        sigma /= model.model.latent_format.scale_factor
        sigma = sigma.cpu().numpy()

        samples_out = samples.copy()

        s1 = samples["samples"]
        samples_out["samples"] = s1 * sigma

        if pipe is None:
            pipe = {}
        new_pipe = {
            **pipe,
            "samples": samples_out
        }
        del pipe

        return (new_pipe, samples_out, sigma)

# Latent遮罩复合
class latentCompositeMaskedWithCond:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "text_combine": ("LIST",),
                "source_latent": ("LATENT",),
                "source_mask": ("MASK",),
                "destination_mask": ("MASK",),
                "text_combine_mode": (["add", "replace", "cover"], {"default": "add"}),
                "replace_text": ("STRING", {"default": ""})
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
        }

    OUTPUT_IS_LIST = (False, False, True)
    RETURN_TYPES = ("PIPE_LINE", "LATENT", "CONDITIONING")
    RETURN_NAMES = ("pipe", "latent", "conditioning",)
    FUNCTION = "run"
    OUTPUT_NODE = True

    CATEGORY = "EasyUse/Latent"

    def run(self, pipe, text_combine, source_latent, source_mask, destination_mask, text_combine_mode, replace_text, prompt=None, extra_pnginfo=None, my_unique_id=None):
        positive = None
        clip = pipe["clip"]
        destination_latent = pipe["samples"]

        conds = []

        for text in text_combine:
            if text_combine_mode == 'cover':
                positive = text
            elif text_combine_mode == 'replace' and replace_text != '':
                positive = pipe["loader_settings"]["positive"].replace(replace_text, text)
            else:
                positive = pipe["loader_settings"]["positive"] + ',' + text
            positive_token_normalization = pipe["loader_settings"]["positive_token_normalization"]
            positive_weight_interpretation = pipe["loader_settings"]["positive_weight_interpretation"]
            a1111_prompt_style = pipe["loader_settings"]["a1111_prompt_style"]
            positive_cond = pipe["positive"]

            log_node_warn("正在处理提示词编码...")
            steps = pipe["loader_settings"]["steps"] if "steps" in pipe["loader_settings"] else 1
            positive_embeddings_final = advanced_encode(clip, positive,
                                         positive_token_normalization,
                                         positive_weight_interpretation, w_max=1.0,
                                         apply_to_pooled='enable', a1111_prompt_style=a1111_prompt_style, steps=steps)

            # source cond
            (cond_1,) = ConditioningSetMask().append(positive_cond, source_mask, "default", 1)
            (cond_2,) = ConditioningSetMask().append(positive_embeddings_final, destination_mask, "default", 1)
            positive_cond = cond_1 + cond_2

            conds.append(positive_cond)
        # latent composite masked
        (samples,) = LatentCompositeMasked().composite(destination_latent, source_latent, 0, 0, False)

        new_pipe = {
            **pipe,
            "samples": samples,
            "loader_settings": {
                **pipe["loader_settings"],
                "positive": positive,
            }
        }

        del pipe

        return (new_pipe, samples, conds)

# 噪声注入到潜空间
class injectNoiseToLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 200.0, "step": 0.0001}),
            "normalize": ("BOOLEAN", {"default": False}),
            "average": ("BOOLEAN", {"default": False}),
        },
            "optional": {
                "pipe_to_noise": ("PIPE_LINE",),
                "image_to_latent": ("IMAGE",),
                "latent": ("LATENT",),
                "noise": ("LATENT",),
                "mask": ("MASK",),
                "mix_randn_amount": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.001}),
                # "seed": ("INT", {"default": 123, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "inject"
    CATEGORY = "EasyUse/Latent"

    def inject(self,strength, normalize, average, pipe_to_noise=None, noise=None, image_to_latent=None, latent=None, mix_randn_amount=0, mask=None):

        vae = pipe_to_noise["vae"] if pipe_to_noise is not None else pipe_to_noise["vae"]
        batch_size = pipe_to_noise["loader_settings"]["batch_size"] if pipe_to_noise is not None and "batch_size" in pipe_to_noise["loader_settings"] else 1
        if noise is None and pipe_to_noise is not None:
            noise = pipe_to_noise["samples"]
        elif noise is None:
            raise Exception("InjectNoiseToLatent: No noise provided")

        if image_to_latent is not None and vae is not None:
            samples = {"samples": vae.encode(image_to_latent[:, :, :, :3])}
            latents = RepeatLatentBatch().repeat(samples, batch_size)[0]
        elif latent is not None:
            latents = latent
        else:
            raise Exception("InjectNoiseToLatent: No input latent provided")

        samples = latents.copy()
        if latents["samples"].shape != noise["samples"].shape:
            raise ValueError("InjectNoiseToLatent: Latent and noise must have the same shape")
        if average:
            noised = (samples["samples"].clone() + noise["samples"].clone()) / 2
        else:
            noised = samples["samples"].clone() + noise["samples"].clone() * strength
        if normalize:
            noised = noised / noised.std()
        if mask is not None:
            mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
                                                   size=(noised.shape[2], noised.shape[3]), mode="bilinear")
            mask = mask.expand((-1, noised.shape[1], -1, -1))
            if mask.shape[0] < noised.shape[0]:
                mask = mask.repeat((noised.shape[0] - 1) // mask.shape[0] + 1, 1, 1, 1)[:noised.shape[0]]
            noised = mask * noised + (1 - mask) * latents["samples"]
        if mix_randn_amount > 0:
            # if seed is not None:
            #     torch.manual_seed(seed)
            rand_noise = torch.randn_like(noised)
            noised = ((1 - mix_randn_amount) * noised + mix_randn_amount *
                      rand_noise) / ((mix_randn_amount ** 2 + (1 - mix_randn_amount) ** 2) ** 0.5)
        samples["samples"] = noised
        return (samples,)

# ---------------------------------------------------------------潜空间 结束----------------------------------------------------------------------#

# ---------------------------------------------------------------随机种 开始----------------------------------------------------------------------#
# 随机种
class easySeed:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED_NUM}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "doit"

    CATEGORY = "EasyUse/Seed"

    OUTPUT_NODE = True

    def doit(self, seed=0, prompt=None, extra_pnginfo=None, my_unique_id=None):
        return seed,

# 全局随机种
class globalSeed:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": 0, "max": MAX_SEED_NUM}),
                "mode": ("BOOLEAN", {"default": True, "label_on": "control_before_generate", "label_off": "control_after_generate"}),
                "action": (["fixed", "increment", "decrement", "randomize",
                            "increment for each node", "decrement for each node", "randomize for each node"], ),
                "last_seed": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "doit"

    CATEGORY = "EasyUse/Seed"

    OUTPUT_NODE = True

    def doit(self, **kwargs):
        return {}

# ---------------------------------------------------------------随机种 结束----------------------------------------------------------------------#

# ---------------------------------------------------------------加载器 开始----------------------------------------------------------------------#
class setCkptName:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            }
        }

    RETURN_TYPES = (AlwaysEqualProxy('*'),)
    RETURN_NAMES = ("ckpt_name",)
    FUNCTION = "set_name"
    CATEGORY = "EasyUse/Util"

    def set_name(self, ckpt_name):
        return (ckpt_name,)

class setControlName:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                "controlnet_name": (folder_paths.get_filename_list("controlnet"),),
            }
        }

    RETURN_TYPES = (AlwaysEqualProxy('*'),)
    RETURN_NAMES = ("controlnet_name",)
    FUNCTION = "set_name"
    CATEGORY = "EasyUse/Util"

    def set_name(self, controlnet_name):
        return (controlnet_name,)

# 简易加载器完整
class fullLoader:

    @classmethod
    def INPUT_TYPES(cls):
        resolution_strings = [f"{width} x {height}" for width, height in BASE_RESOLUTIONS]
        a1111_prompt_style_default = False

        return {"required": {
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            "config_name": (["Default", ] + folder_paths.get_filename_list("configs"), {"default": "Default"}),
            "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
            "clip_skip": ("INT", {"default": -1, "min": -24, "max": 0, "step": 1}),

            "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
            "lora_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

            "resolution": (resolution_strings,),
            "empty_latent_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "empty_latent_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),

            "positive": ("STRING", {"default": "", "placeholder": "Positive", "multiline": True}),
            "positive_token_normalization": (["none", "mean", "length", "length+mean"],),
            "positive_weight_interpretation": (["comfy",  "A1111", "comfy++", "compel", "fixed attention"],),

            "negative": ("STRING", {"default": "", "placeholder": "Negative", "multiline": True}),
            "negative_token_normalization": (["none", "mean", "length", "length+mean"],),
            "negative_weight_interpretation": (["comfy",  "A1111", "comfy++", "compel", "fixed attention"],),

            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
        },
            "optional": {"model_override": ("MODEL",), "clip_override": ("CLIP",), "vae_override": ("VAE",), "optional_lora_stack": ("LORA_STACK",), "optional_controlnet_stack": ("CONTROL_NET_STACK",), "a1111_prompt_style": ("BOOLEAN", {"default": a1111_prompt_style_default}),},
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE", "CLIP", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("pipe", "model", "vae", "clip", "positive", "negative", "latent")

    FUNCTION = "adv_pipeloader"
    CATEGORY = "EasyUse/Loaders"

    def adv_pipeloader(self, ckpt_name, config_name, vae_name, clip_skip,
                       lora_name, lora_model_strength, lora_clip_strength,
                       resolution, empty_latent_width, empty_latent_height,
                       positive, positive_token_normalization, positive_weight_interpretation,
                       negative, negative_token_normalization, negative_weight_interpretation,
                       batch_size, model_override=None, clip_override=None, vae_override=None, optional_lora_stack=None, optional_controlnet_stack=None, a1111_prompt_style=False, prompt=None,
                       my_unique_id=None
                       ):

        # Clean models from loaded_objects
        easyCache.update_loaded_objects(prompt)

        # Load models
        log_node_warn("正在加载模型...")
        model, clip, vae, clip_vision, lora_stack = easyCache.load_main(ckpt_name, config_name, vae_name, lora_name, lora_model_strength, lora_clip_strength, optional_lora_stack, model_override, clip_override, vae_override, prompt)

        # Create Empty Latent
        sd3 = True if get_sd_version(model) == 'sd3' else False
        samples = sampler.emptyLatent(resolution, empty_latent_width, empty_latent_height, batch_size, sd3=sd3)

        # Prompt to Conditioning
        positive_embeddings_final, positive_wildcard_prompt, model, clip = prompt_to_cond('positive', model, clip, clip_skip, lora_stack, positive, positive_token_normalization, positive_weight_interpretation, a1111_prompt_style, my_unique_id, prompt, easyCache)
        negative_embeddings_final, negative_wildcard_prompt, model, clip = prompt_to_cond('negative', model, clip, clip_skip, lora_stack, negative, negative_token_normalization, negative_weight_interpretation, a1111_prompt_style, my_unique_id, prompt, easyCache)

        # Conditioning add controlnet
        if optional_controlnet_stack is not None and len(optional_controlnet_stack) > 0:
            for controlnet in optional_controlnet_stack:
                positive_embeddings_final, negative_embeddings_final = easyControlnet().apply(controlnet[0], controlnet[5], positive_embeddings_final, negative_embeddings_final, controlnet[1], start_percent=controlnet[2], end_percent=controlnet[3], control_net=None, scale_soft_weights=controlnet[4], mask=None, easyCache=easyCache, use_cache=True)

        log_node_warn("加载完毕...")
        pipe = {
            "model": model,
            "positive": positive_embeddings_final,
            "negative": negative_embeddings_final,
            "vae": vae,
            "clip": clip,

            "samples": samples,
            "images": None,

            "loader_settings": {
                "ckpt_name": ckpt_name,
                "vae_name": vae_name,
                "lora_name": lora_name,
                "lora_model_strength": lora_model_strength,
                "lora_clip_strength": lora_clip_strength,
                "lora_stack": lora_stack,

                "clip_skip": clip_skip,
                "a1111_prompt_style": a1111_prompt_style,
                "positive": positive,
                "positive_token_normalization": positive_token_normalization,
                "positive_weight_interpretation": positive_weight_interpretation,
                "negative": negative,
                "negative_token_normalization": negative_token_normalization,
                "negative_weight_interpretation": negative_weight_interpretation,
                "resolution": resolution,
                "empty_latent_width": empty_latent_width,
                "empty_latent_height": empty_latent_height,
                "batch_size": batch_size,
            }
        }

        return {"ui": {"positive": positive_wildcard_prompt, "negative": negative_wildcard_prompt}, "result": (pipe, model, vae, clip, positive_embeddings_final, negative_embeddings_final, samples)}

# A1111简易加载器
class a1111Loader(fullLoader):
    @classmethod
    def INPUT_TYPES(cls):
        resolution_strings = [f"{width} x {height}" for width, height in BASE_RESOLUTIONS]
        a1111_prompt_style_default = False
        checkpoints = folder_paths.get_filename_list("checkpoints")
        loras = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "ckpt_name": (checkpoints,),
                "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                "clip_skip": ("INT", {"default": -1, "min": -24, "max": 0, "step": 1}),

                "lora_name": (loras,),
                "lora_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "resolution": (resolution_strings, {"default": "512 x 512"}),
                "empty_latent_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "empty_latent_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),

                "positive": ("STRING", {"default":"", "placeholder": "Positive", "multiline": True}),
                "negative": ("STRING", {"default":"", "placeholder": "Negative", "multiline": True}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "optional_lora_stack": ("LORA_STACK",),
                "optional_controlnet_stack": ("CONTROL_NET_STACK",),
                "a1111_prompt_style": ("BOOLEAN", {"default": a1111_prompt_style_default}),
            },
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE")
    RETURN_NAMES = ("pipe", "model", "vae")

    FUNCTION = "a1111loader"
    CATEGORY = "EasyUse/Loaders"

    def a1111loader(self, ckpt_name, vae_name, clip_skip,
                       lora_name, lora_model_strength, lora_clip_strength,
                       resolution, empty_latent_width, empty_latent_height,
                       positive, negative, batch_size, optional_lora_stack=None, optional_controlnet_stack=None, a1111_prompt_style=False, prompt=None,
                       my_unique_id=None):

        return super().adv_pipeloader(ckpt_name, 'Default', vae_name, clip_skip,
             lora_name, lora_model_strength, lora_clip_strength,
             resolution, empty_latent_width, empty_latent_height,
             positive, 'mean', 'A1111',
             negative,'mean','A1111',
             batch_size, None, None,  None, optional_lora_stack=optional_lora_stack, optional_controlnet_stack=optional_controlnet_stack,a1111_prompt_style=a1111_prompt_style, prompt=prompt,
             my_unique_id=my_unique_id
        )

# Comfy简易加载器
class comfyLoader(fullLoader):
    @classmethod
    def INPUT_TYPES(cls):
        resolution_strings = [f"{width} x {height}" for width, height in BASE_RESOLUTIONS]
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                "clip_skip": ("INT", {"default": -1, "min": -24, "max": 0, "step": 1}),

                "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
                "lora_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "resolution": (resolution_strings, {"default": "512 x 512"}),
                "empty_latent_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "empty_latent_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),

                "positive": ("STRING", {"default": "", "placeholder": "Positive", "multiline": True}),
                "negative": ("STRING", {"default": "", "placeholder": "Negative", "multiline": True}),

                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {"optional_lora_stack": ("LORA_STACK",), "optional_controlnet_stack": ("CONTROL_NET_STACK",),},
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE")
    RETURN_NAMES = ("pipe", "model", "vae")

    FUNCTION = "comfyloader"
    CATEGORY = "EasyUse/Loaders"

    def comfyloader(self, ckpt_name, vae_name, clip_skip,
                       lora_name, lora_model_strength, lora_clip_strength,
                       resolution, empty_latent_width, empty_latent_height,
                       positive, negative, batch_size, optional_lora_stack=None, optional_controlnet_stack=None, prompt=None,
                      my_unique_id=None):
        return super().adv_pipeloader(ckpt_name, 'Default', vae_name, clip_skip,
             lora_name, lora_model_strength, lora_clip_strength,
             resolution, empty_latent_width, empty_latent_height,
             positive, 'none', 'comfy',
             negative, 'none', 'comfy',
             batch_size, None, None, None, optional_lora_stack=optional_lora_stack, optional_controlnet_stack=optional_controlnet_stack, a1111_prompt_style=False, prompt=prompt,
             my_unique_id=my_unique_id
         )

# stable Cascade
class cascadeLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        resolution_strings = [f"{width} x {height}" for width, height in BASE_RESOLUTIONS]

        return {"required": {
            "stage_c": (folder_paths.get_filename_list("unet") + folder_paths.get_filename_list("checkpoints"),),
            "stage_b": (folder_paths.get_filename_list("unet") + folder_paths.get_filename_list("checkpoints"),),
            "stage_a": (["Baked VAE"]+folder_paths.get_filename_list("vae"),),
            "clip_name": (["None"] + folder_paths.get_filename_list("clip"),),

            "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
            "lora_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

            "resolution": (resolution_strings, {"default": "1024 x 1024"}),
            "empty_latent_width": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
            "empty_latent_height": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
            "compression": ("INT", {"default": 42, "min": 32, "max": 64, "step": 1}),

            "positive": ("STRING", {"default":"", "placeholder": "Positive", "multiline": True}),
            "negative": ("STRING", {"default":"", "placeholder": "Negative", "multiline": True}),

            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
        },
            "optional": {"optional_lora_stack": ("LORA_STACK",), },
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "LATENT", "VAE")
    RETURN_NAMES = ("pipe", "model_c", "latent_c", "vae")

    FUNCTION = "adv_pipeloader"
    CATEGORY = "EasyUse/Loaders"

    def is_ckpt(self, name):
        is_ckpt = False
        path = folder_paths.get_full_path("checkpoints", name)
        if path is not None:
            is_ckpt = True
        return is_ckpt

    def adv_pipeloader(self, stage_c, stage_b, stage_a, clip_name, lora_name, lora_model_strength, lora_clip_strength,
                       resolution, empty_latent_width, empty_latent_height, compression,
                       positive, negative, batch_size, optional_lora_stack=None,prompt=None,
                       my_unique_id=None):

        vae: VAE | None = None
        model_c: ModelPatcher | None = None
        model_b: ModelPatcher | None = None
        clip: CLIP | None = None
        can_load_lora = True
        pipe_lora_stack = []

        # Clean models from loaded_objects
        easyCache.update_loaded_objects(prompt)

        # Create Empty Latent
        samples = sampler.emptyLatent(resolution, empty_latent_width, empty_latent_height, batch_size, compression)

        if self.is_ckpt(stage_c):
            model_c, clip, vae_c, clip_vision = easyCache.load_checkpoint(stage_c)
        else:
            model_c = easyCache.load_unet(stage_c)
            vae_c = None
        if self.is_ckpt(stage_b):
            model_b, clip, vae_b, clip_vision = easyCache.load_checkpoint(stage_b)
        else:
            model_b = easyCache.load_unet(stage_b)
            vae_b = None

        if optional_lora_stack is not None and can_load_lora:
            for lora in optional_lora_stack:
                lora = {"lora_name": lora[0], "model": model_c, "clip": clip, "model_strength": lora[1], "clip_strength": lora[2]}
                model_c, clip = easyCache.load_lora(lora)
                lora['model'] = model_c
                lora['clip'] = clip
                pipe_lora_stack.append(lora)

        if lora_name != "None" and can_load_lora:
            lora = {"lora_name": lora_name, "model": model_c, "clip": clip, "model_strength": lora_model_strength,
                    "clip_strength": lora_clip_strength}
            model_c, clip = easyCache.load_lora(lora)
            pipe_lora_stack.append(lora)

        model = (model_c, model_b)
        # Load clip
        if clip_name != 'None':
            clip = easyCache.load_clip(clip_name, "stable_cascade")
        # Load vae
        if stage_a not in ["Baked VAE", "Baked-VAE"]:
            vae_b = easyCache.load_vae(stage_a)

        vae = (vae_c, vae_b)
        # 判断是否连接 styles selector
        is_positive_linked_styles_selector = is_linked_styles_selector(prompt, my_unique_id, 'positive')
        is_negative_linked_styles_selector = is_linked_styles_selector(prompt, my_unique_id, 'negative')

        log_node_warn("正在处理提示词...")
        positive_seed = find_wildcards_seed(my_unique_id, positive, prompt)
        # Translate cn to en
        if has_chinese(positive):
            positive = zh_to_en([positive])[0]
        model_c, clip, positive, positive_decode, show_positive_prompt, pipe_lora_stack = process_with_loras(positive,
                                                                                                           model_c, clip,
                                                                                                           "positive",
                                                                                                           positive_seed,
                                                                                                           can_load_lora,
                                                                                                           pipe_lora_stack,
                                                                                                           easyCache)
        positive_wildcard_prompt = positive_decode if show_positive_prompt or is_positive_linked_styles_selector else ""
        negative_seed = find_wildcards_seed(my_unique_id, negative, prompt)
        # Translate cn to en
        if has_chinese(negative):
            negative = zh_to_en([negative])[0]
        model_c, clip, negative, negative_decode, show_negative_prompt, pipe_lora_stack = process_with_loras(negative,
                                                                                                           model_c, clip,
                                                                                                           "negative",
                                                                                                           negative_seed,
                                                                                                           can_load_lora,
                                                                                                           pipe_lora_stack,
                                                                                                           easyCache)
        negative_wildcard_prompt = negative_decode if show_negative_prompt or is_negative_linked_styles_selector else ""

        tokens = clip.tokenize(positive)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        positive_embeddings_final = [[cond, {"pooled_output": pooled}]]

        tokens = clip.tokenize(negative)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        negative_embeddings_final = [[cond, {"pooled_output": pooled}]]

        image = easySampler.pil2tensor(Image.new('RGB', (1, 1), (0, 0, 0)))

        log_node_warn("处理结束...")
        pipe = {
            "model": model,
            "positive": positive_embeddings_final,
            "negative": negative_embeddings_final,
            "vae": vae,
            "clip": clip,

            "samples": samples,
            "images": image,
            "seed": 0,

            "loader_settings": {
                "vae_name": stage_a,
                "lora_name": lora_name,
                "lora_model_strength": lora_model_strength,
                "lora_clip_strength": lora_clip_strength,
                "lora_stack": pipe_lora_stack,

                "positive": positive,
                "positive_token_normalization": 'none',
                "positive_weight_interpretation": 'comfy',
                "negative": negative,
                "negative_token_normalization": 'none',
                "negative_weight_interpretation": 'comfy',
                "resolution": resolution,
                "empty_latent_width": empty_latent_width,
                "empty_latent_height": empty_latent_height,
                "batch_size": batch_size,
                "compression": compression
            }
        }

        return {"ui": {"positive": positive_wildcard_prompt, "negative": negative_wildcard_prompt},
                "result": (pipe, model_c, model_b, vae)}

# Zero123简易加载器 (3D)
try:
    from comfy_extras.nodes_stable3d import camera_embeddings
except FileNotFoundError:
    log_node_error("EasyUse[zero123Loader]", "请更新ComfyUI到最新版本")

class zero123Loader:

    @classmethod
    def INPUT_TYPES(cls):
        def get_file_list(filenames):
            return [file for file in filenames if file != "put_models_here.txt" and "zero123" in file.lower()]

        return {"required": {
            "ckpt_name": (get_file_list(folder_paths.get_filename_list("checkpoints")),),
            "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),

            "init_image": ("IMAGE",),
            "empty_latent_width": ("INT", {"default": 256, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
            "empty_latent_height": ("INT", {"default": 256, "min": 16, "max": MAX_RESOLUTION, "step": 8}),

            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),

            "elevation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
            "azimuth": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
        },
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE")
    RETURN_NAMES = ("pipe", "model", "vae")

    FUNCTION = "adv_pipeloader"
    CATEGORY = "EasyUse/Loaders"

    def adv_pipeloader(self, ckpt_name, vae_name, init_image, empty_latent_width, empty_latent_height, batch_size, elevation, azimuth, prompt=None, my_unique_id=None):
        model: ModelPatcher | None = None
        vae: VAE | None = None
        clip: CLIP | None = None
        clip_vision = None

        # Clean models from loaded_objects
        easyCache.update_loaded_objects(prompt)

        model, clip, vae, clip_vision = easyCache.load_checkpoint(ckpt_name, "Default", True)

        output = clip_vision.encode_image(init_image)
        pooled = output.image_embeds.unsqueeze(0)
        pixels = comfy.utils.common_upscale(init_image.movedim(-1, 1), empty_latent_width, empty_latent_height, "bilinear", "center").movedim(1, -1)
        encode_pixels = pixels[:, :, :, :3]
        t = vae.encode(encode_pixels)
        cam_embeds = camera_embeddings(elevation, azimuth)
        cond = torch.cat([pooled, cam_embeds.repeat((pooled.shape[0], 1, 1))], dim=-1)

        positive = [[cond, {"concat_latent_image": t}]]
        negative = [[torch.zeros_like(pooled), {"concat_latent_image": torch.zeros_like(t)}]]
        latent = torch.zeros([batch_size, 4, empty_latent_height // 8, empty_latent_width // 8])
        samples = {"samples": latent}

        image = easySampler.pil2tensor(Image.new('RGB', (1, 1), (0, 0, 0)))

        pipe = {"model": model,
                "positive": positive,
                "negative": negative,
                "vae": vae,
                "clip": clip,

                "samples": samples,
                "images": image,
                "seed": 0,

                "loader_settings": {"ckpt_name": ckpt_name,
                                    "vae_name": vae_name,

                                    "positive": positive,
                                    "negative": negative,
                                    "empty_latent_width": empty_latent_width,
                                    "empty_latent_height": empty_latent_height,
                                    "batch_size": batch_size,
                                    "seed": 0,
                                    }
                }

        return (pipe, model, vae)

# SV3D加载器
class sv3DLoader(EasingBase):

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        def get_file_list(filenames):
            return [file for file in filenames if file != "put_models_here.txt" and "sv3d" in file]

        return {"required": {
            "ckpt_name": (get_file_list(folder_paths.get_filename_list("checkpoints")),),
            "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),

            "init_image": ("IMAGE",),
            "empty_latent_width": ("INT", {"default": 576, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
            "empty_latent_height": ("INT", {"default": 576, "min": 16, "max": MAX_RESOLUTION, "step": 8}),

            "batch_size": ("INT", {"default": 21, "min": 1, "max": 4096}),
            "interp_easing": (["linear", "ease_in", "ease_out", "ease_in_out"], {"default": "linear"}),
            "easing_mode": (["azimuth", "elevation", "custom"], {"default": "azimuth"}),
        },
            "optional": {"scheduler": ("STRING", {"default": "",  "multiline": True})},
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "STRING")
    RETURN_NAMES = ("pipe", "model", "interp_log")

    FUNCTION = "adv_pipeloader"
    CATEGORY = "EasyUse/Loaders"

    def adv_pipeloader(self, ckpt_name, vae_name, init_image, empty_latent_width, empty_latent_height, batch_size, interp_easing, easing_mode, scheduler='',prompt=None, my_unique_id=None):
        model: ModelPatcher | None = None
        vae: VAE | None = None
        clip: CLIP | None = None

        # Clean models from loaded_objects
        easyCache.update_loaded_objects(prompt)

        model, clip, vae, clip_vision = easyCache.load_checkpoint(ckpt_name, "Default", True)

        output = clip_vision.encode_image(init_image)
        pooled = output.image_embeds.unsqueeze(0)
        pixels = comfy.utils.common_upscale(init_image.movedim(-1, 1), empty_latent_width, empty_latent_height, "bilinear", "center").movedim(1,
                                                                                                                    -1)
        encode_pixels = pixels[:, :, :, :3]
        t = vae.encode(encode_pixels)

        azimuth_points = []
        elevation_points = []
        if easing_mode == 'azimuth':
            azimuth_points = [(0, 0), (batch_size-1, 360)]
            elevation_points = [(0, 0)] * batch_size
        elif easing_mode == 'elevation':
            azimuth_points = [(0, 0)] * batch_size
            elevation_points = [(0, -90), (batch_size-1, 90)]
        else:
            schedulers = scheduler.rstrip('\n')
            for line in schedulers.split('\n'):
                frame_str, point_str = line.split(':')
                point_str = point_str.strip()[1:-1]
                point = point_str.split(',')
                azimuth_point = point[0]
                elevation_point = point[1] if point[1] else 0.0
                frame = int(frame_str.strip())
                azimuth = float(azimuth_point)
                azimuth_points.append((frame, azimuth))
                elevation_val = float(elevation_point)
                elevation_points.append((frame, elevation_val))
            azimuth_points.sort(key=lambda x: x[0])
            elevation_points.sort(key=lambda x: x[0])

        #interpolation
        next_point = 1
        next_elevation_point = 1
        elevations = []
        azimuths = []
        # For azimuth interpolation
        for i in range(batch_size):
            # Find the interpolated azimuth for the current frame
            while next_point < len(azimuth_points) and i >= azimuth_points[next_point][0]:
                next_point += 1
            if next_point == len(azimuth_points):
                next_point -= 1
            prev_point = max(next_point - 1, 0)

            if azimuth_points[next_point][0] != azimuth_points[prev_point][0]:
                timing = (i - azimuth_points[prev_point][0]) / (
                            azimuth_points[next_point][0] - azimuth_points[prev_point][0])
                interpolated_azimuth = self.ease(azimuth_points[prev_point][1], azimuth_points[next_point][1], self.easing(timing, interp_easing))
            else:
                interpolated_azimuth = azimuth_points[prev_point][1]

            # Interpolate the elevation
            next_elevation_point = 1
            while next_elevation_point < len(elevation_points) and i >= elevation_points[next_elevation_point][0]:
                next_elevation_point += 1
            if next_elevation_point == len(elevation_points):
                next_elevation_point -= 1
            prev_elevation_point = max(next_elevation_point - 1, 0)

            if elevation_points[next_elevation_point][0] != elevation_points[prev_elevation_point][0]:
                timing = (i - elevation_points[prev_elevation_point][0]) / (
                            elevation_points[next_elevation_point][0] - elevation_points[prev_elevation_point][0])
                interpolated_elevation = self.ease(elevation_points[prev_point][1], elevation_points[next_point][1], self.easing(timing, interp_easing))
            else:
                interpolated_elevation = elevation_points[prev_elevation_point][1]

            azimuths.append(interpolated_azimuth)
            elevations.append(interpolated_elevation)

        log_node_info("easy sv3dLoader", "azimuths:" + str(azimuths))
        log_node_info("easy sv3dLoader", "elevations:" + str(elevations))

        log = 'azimuths:' + str(azimuths) + '\n\n' + "elevations:" + str(elevations)
        # Structure the final output
        positive = [[pooled, {"concat_latent_image": t, "elevation": elevations, "azimuth": azimuths}]]
        negative = [[torch.zeros_like(pooled),
                           {"concat_latent_image": torch.zeros_like(t), "elevation": elevations, "azimuth": azimuths}]]

        latent = torch.zeros([batch_size, 4, empty_latent_height // 8, empty_latent_width // 8])
        samples = {"samples": latent}

        image = easySampler.pil2tensor(Image.new('RGB', (1, 1), (0, 0, 0)))


        pipe = {"model": model,
                "positive": positive,
                "negative": negative,
                "vae": vae,
                "clip": clip,

                "samples": samples,
                "images": image,
                "seed": 0,

                "loader_settings": {"ckpt_name": ckpt_name,
                                    "vae_name": vae_name,

                                    "positive": positive,
                                    "negative": negative,
                                    "empty_latent_width": empty_latent_width,
                                    "empty_latent_height": empty_latent_height,
                                    "batch_size": batch_size,
                                    "seed": 0,
                                    }
                }

        return (pipe, model, log)

#svd加载器
class svdLoader:

    @classmethod
    def INPUT_TYPES(cls):
        resolution_strings = [f"{width} x {height}" for width, height in BASE_RESOLUTIONS]
        def get_file_list(filenames):
            return [file for file in filenames if file != "put_models_here.txt" and "svd" in file.lower()]

        return {"required": {
                "ckpt_name": (get_file_list(folder_paths.get_filename_list("checkpoints")),),
                "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                "clip_name": (["None"] + folder_paths.get_filename_list("clip"),),

                "init_image": ("IMAGE",),
                "resolution": (resolution_strings, {"default": "1024 x 576"}),
                "empty_latent_width": ("INT", {"default": 256, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "empty_latent_height": ("INT", {"default": 256, "min": 16, "max": MAX_RESOLUTION, "step": 8}),

                "video_frames": ("INT", {"default": 14, "min": 1, "max": 4096}),
                "motion_bucket_id": ("INT", {"default": 127, "min": 1, "max": 1023}),
                "fps": ("INT", {"default": 6, "min": 1, "max": 1024}),
                "augmentation_level": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01})
            },
            "optional": {
                "optional_positive": ("STRING", {"default": "", "multiline": True}),
                "optional_negative": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE")
    RETURN_NAMES = ("pipe", "model", "vae")

    FUNCTION = "adv_pipeloader"
    CATEGORY = "EasyUse/Loaders"

    def adv_pipeloader(self, ckpt_name, vae_name, clip_name, init_image, resolution, empty_latent_width, empty_latent_height, video_frames, motion_bucket_id, fps, augmentation_level, optional_positive=None, optional_negative=None, prompt=None, my_unique_id=None):
        model: ModelPatcher | None = None
        vae: VAE | None = None
        clip: CLIP | None = None
        clip_vision = None

        # resolution
        if resolution != "自定义 x 自定义":
            try:
                width, height = map(int, resolution.split(' x '))
                empty_latent_width = width
                empty_latent_height = height
            except ValueError:
                raise ValueError("Invalid base_resolution format.")

        # Clean models from loaded_objects
        easyCache.update_loaded_objects(prompt)

        model, clip, vae, clip_vision = easyCache.load_checkpoint(ckpt_name, "Default", True)

        output = clip_vision.encode_image(init_image)
        pooled = output.image_embeds.unsqueeze(0)
        pixels = comfy.utils.common_upscale(init_image.movedim(-1, 1), empty_latent_width, empty_latent_height, "bilinear", "center").movedim(1, -1)
        encode_pixels = pixels[:, :, :, :3]
        if augmentation_level > 0:
            encode_pixels += torch.randn_like(pixels) * augmentation_level
        t = vae.encode(encode_pixels)
        positive = [[pooled,
                     {"motion_bucket_id": motion_bucket_id, "fps": fps, "augmentation_level": augmentation_level,
                      "concat_latent_image": t}]]
        negative = [[torch.zeros_like(pooled),
                     {"motion_bucket_id": motion_bucket_id, "fps": fps, "augmentation_level": augmentation_level,
                      "concat_latent_image": torch.zeros_like(t)}]]
        if optional_positive is not None and optional_positive != '':
            if clip_name == 'None':
                raise Exception("You need choose a open_clip model when positive is not empty")
            clip = easyCache.load_clip(clip_name)
            if has_chinese(optional_positive):
                optional_positive = zh_to_en([optional_positive])[0]
            positive_embeddings_final, = CLIPTextEncode().encode(clip, optional_positive)
            positive, = ConditioningConcat().concat(positive, positive_embeddings_final)
        if optional_negative is not None and optional_negative != '':
            if clip_name == 'None':
                raise Exception("You need choose a open_clip model when negative is not empty")
            if has_chinese(optional_negative):
                optional_positive = zh_to_en([optional_negative])[0]
            negative_embeddings_final, = CLIPTextEncode().encode(clip, optional_negative)
            negative, = ConditioningConcat().concat(negative, negative_embeddings_final)

        latent = torch.zeros([video_frames, 4, empty_latent_height // 8, empty_latent_width // 8])
        samples = {"samples": latent}

        image = easySampler.pil2tensor(Image.new('RGB', (1, 1), (0, 0, 0)))

        pipe = {"model": model,
                "positive": positive,
                "negative": negative,
                "vae": vae,
                "clip": clip,

                "samples": samples,
                "images": image,
                "seed": 0,

                "loader_settings": {"ckpt_name": ckpt_name,
                                    "vae_name": vae_name,

                                    "positive": positive,
                                    "negative": negative,
                                    "resolution": resolution,
                                    "empty_latent_width": empty_latent_width,
                                    "empty_latent_height": empty_latent_height,
                                    "batch_size": 1,
                                    "seed": 0,
                                     }
                }

        return (pipe, model, vae)

#dynamiCrafter加载器
from .dynamiCrafter import DynamiCrafter
class dynamiCrafterLoader(DynamiCrafter):

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        resolution_strings = [f"{width} x {height}" for width, height in BASE_RESOLUTIONS]

        return {"required": {
                "model_name": (list(DYNAMICRAFTER_MODELS.keys()),),
                "clip_skip": ("INT", {"default": -2, "min": -24, "max": 0, "step": 1}),

                "init_image": ("IMAGE",),
                "resolution": (resolution_strings, {"default": "512 x 512"}),
                "empty_latent_width": ("INT", {"default": 256, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "empty_latent_height": ("INT", {"default": 256, "min": 16, "max": MAX_RESOLUTION, "step": 8}),

                "positive": ("STRING", {"default": "", "multiline": True}),
                "negative": ("STRING", {"default": "", "multiline": True}),

                "use_interpolate": ("BOOLEAN", {"default": False}),
                "fps": ("INT", {"default": 15, "min": 1, "max": 30, "step": 1},),
                "frames": ("INT", {"default": 16}),
                "scale_latents": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "optional_vae": ("VAE",),
            },
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE")
    RETURN_NAMES = ("pipe", "model", "vae")

    FUNCTION = "adv_pipeloader"
    CATEGORY = "EasyUse/Loaders"

    def get_clip_file(self, node_name):
        clip_list = folder_paths.get_filename_list("clip")
        pattern = 'sd2-1-open-clip|model\.(safetensors|bin)$'
        clip_files = [e for e in clip_list if re.search(pattern, e, re.IGNORECASE)]

        clip_name = clip_files[0] if len(clip_files)>0 else None
        clip_file = folder_paths.get_full_path("clip", clip_name) if clip_name else None
        if clip_name is not None:
            log_node_info(node_name, f"Using {clip_name}")

        return clip_file, clip_name

    def get_clipvision_file(self, node_name):
        clipvision_list = folder_paths.get_filename_list("clip_vision")
        pattern = '(ViT.H.14.*s32B.b79K|ipadapter.*sd15|sd1.?5.*model|open_clip_pytorch_model\.(bin|safetensors))'
        clipvision_files = [e for e in clipvision_list if re.search(pattern, e, re.IGNORECASE)]

        clipvision_name = clipvision_files[0] if len(clipvision_files)>0 else None
        clipvision_file = folder_paths.get_full_path("clip_vision", clipvision_name) if clipvision_name else None
        if clipvision_name is not None:
            log_node_info(node_name, f"Using {clipvision_name}")

        return clipvision_file, clipvision_name

    def get_vae_file(self, node_name):
        vae_list = folder_paths.get_filename_list("vae")
        pattern = 'vae-ft-mse-840000-ema-pruned\.(pt|bin|safetensors)$'
        vae_files = [e for e in vae_list if re.search(pattern, e, re.IGNORECASE)]

        vae_name = vae_files[0] if len(vae_files)>0 else None
        vae_file = folder_paths.get_full_path("vae", vae_name) if vae_name else None
        if vae_name is not None:
            log_node_info(node_name, f"Using {vae_name}")

        return vae_file, vae_name

    def adv_pipeloader(self, model_name, clip_skip, init_image, resolution, empty_latent_width, empty_latent_height, positive, negative, use_interpolate, fps, frames, scale_latents, optional_vae=None, prompt=None, my_unique_id=None):
        positive_embeddings_final, negative_embeddings_final = None, None
        # resolution
        if resolution != "自定义 x 自定义":
            try:
                width, height = map(int, resolution.split(' x '))
                empty_latent_width = width
                empty_latent_height = height
            except ValueError:
                raise ValueError("Invalid base_resolution format.")

        # Clean models from loaded_objects
        easyCache.update_loaded_objects(prompt)

        models_0 = list(DYNAMICRAFTER_MODELS.keys())[0]

        if optional_vae:
            vae = optional_vae
            vae_name = None
        else:
            vae_file, vae_name = self.get_vae_file("easy dynamiCrafterLoader")
            if vae_file is None:
                vae_name = "vae-ft-mse-840000-ema-pruned.safetensors"
                get_local_filepath(DYNAMICRAFTER_MODELS[models_0]['vae_url'], os.path.join(folder_paths.models_dir, "vae"),
                                   vae_name)
            vae = easyCache.load_vae(vae_name)

        clip_file, clip_name = self.get_clip_file("easy dynamiCrafterLoader")
        if clip_file is None:
            clip_name = 'sd2-1-open-clip.safetensors'
            get_local_filepath(DYNAMICRAFTER_MODELS[models_0]['clip_url'], os.path.join(folder_paths.models_dir, "clip"),
                           clip_name)

        clip = easyCache.load_clip(clip_name)
        # load clip vision
        clip_vision_file, clip_vision_name = self.get_clipvision_file("easy dynamiCrafterLoader")
        if clip_vision_file is None:
            clip_vision_name = 'CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors'
            clip_vision_file = get_local_filepath(DYNAMICRAFTER_MODELS[models_0]['clip_vision_url'], os.path.join(folder_paths.models_dir, "clip_vision"),
                                   clip_vision_name)
        clip_vision = load_clip_vision(clip_vision_file)
        # load unet model
        model_path = get_local_filepath(DYNAMICRAFTER_MODELS[model_name]['model_url'], DYNAMICRAFTER_DIR)
        model_patcher, image_proj_model = self.load_dynamicrafter(model_path)

        # rescale cfg

        # apply
        model, empty_latent, image_latent = self.process_image_conditioning(model_patcher, clip_vision, vae, image_proj_model, init_image, use_interpolate, fps, frames, scale_latents)

        clipped = clip.clone()
        if clip_skip != 0:
            clipped.clip_layer(clip_skip)

        if positive is not None and positive != '':
            if has_chinese(positive):
                positive = zh_to_en([positive])[0]
            positive_embeddings_final, = CLIPTextEncode().encode(clipped, positive)
        if negative is not None and negative != '':
            if has_chinese(negative):
                negative = zh_to_en([negative])[0]
            negative_embeddings_final, = CLIPTextEncode().encode(clipped, negative)

        image = easySampler.pil2tensor(Image.new('RGB', (1, 1), (0, 0, 0)))

        pipe = {"model": model,
                "positive": positive_embeddings_final,
                "negative": negative_embeddings_final,
                "vae": vae,
                "clip": clip,
                "clip_vision": clip_vision,

                "samples": empty_latent,
                "images": image,
                "seed": 0,

                "loader_settings": {"ckpt_name": model_name,
                                    "vae_name": vae_name,

                                    "positive": positive,
                                    "negative": negative,
                                    "resolution": resolution,
                                    "empty_latent_width": empty_latent_width,
                                    "empty_latent_height": empty_latent_height,
                                    "batch_size": 1,
                                    "seed": 0,
                                     }
                }

        return (pipe, model, vae)

# lora
class loraStack:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        max_lora_num = 10
        inputs = {
            "required": {
                "toggle": ("BOOLEAN", {"label_on": "enabled", "label_off": "disabled"}),
                "mode": (["simple", "advanced"],),
                "num_loras": ("INT", {"default": 1, "min": 1, "max": max_lora_num}),
            },
            "optional": {
                "optional_lora_stack": ("LORA_STACK",),
            },
        }

        for i in range(1, max_lora_num+1):
            inputs["optional"][f"lora_{i}_name"] = (
            ["None"] + folder_paths.get_filename_list("loras"), {"default": "None"})
            inputs["optional"][f"lora_{i}_strength"] = (
            "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["optional"][f"lora_{i}_model_strength"] = (
            "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["optional"][f"lora_{i}_clip_strength"] = (
            "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})

        return inputs

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "stack"

    CATEGORY = "EasyUse/Loaders"

    def stack(self, toggle, mode, num_loras, optional_lora_stack=None, **kwargs):
        if (toggle in [False, None, "False"]) or not kwargs:
            return (None,)

        loras = []

        # Import Stack values
        if optional_lora_stack is not None:
            loras.extend([l for l in optional_lora_stack if l[0] != "None"])

        # Import Lora values
        for i in range(1, num_loras + 1):
            lora_name = kwargs.get(f"lora_{i}_name")

            if not lora_name or lora_name == "None":
                continue

            if mode == "simple":
                lora_strength = float(kwargs.get(f"lora_{i}_strength"))
                loras.append((lora_name, lora_strength, lora_strength))
            elif mode == "advanced":
                model_strength = float(kwargs.get(f"lora_{i}_model_strength"))
                clip_strength = float(kwargs.get(f"lora_{i}_clip_strength"))
                loras.append((lora_name, model_strength, clip_strength))
        return (loras,)

class controlnetStack:

    def get_file_list(filenames):
        return [file for file in filenames if file != "put_models_here.txt" and "lllite" not in file]

    @classmethod
    def INPUT_TYPES(s):
        max_cn_num = 3
        inputs = {
            "required": {
                "toggle": ("BOOLEAN", {"label_on": "enabled", "label_off": "disabled"}),
                "mode": (["simple", "advanced"],),
                "num_controlnet": ("INT", {"default": 1, "min": 1, "max": max_cn_num}),
            },
            "optional": {
                "optional_controlnet_stack": ("CONTROL_NET_STACK",),
            }
        }

        for i in range(1, max_cn_num+1):
            inputs["optional"][f"controlnet_{i}"] = (["None"] + s.get_file_list(folder_paths.get_filename_list("controlnet")), {"default": "None"})
            inputs["optional"][f"controlnet_{i}_strength"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},)
            inputs["optional"][f"start_percent_{i}"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},)
            inputs["optional"][f"end_percent_{i}"] = ("FLOAT",{"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},)
            inputs["optional"][f"scale_soft_weight_{i}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},)
            inputs["optional"][f"image_{i}"] = ("IMAGE",)
        return inputs

    RETURN_TYPES = ("CONTROL_NET_STACK",)
    RETURN_NAMES = ("controlnet_stack",)
    FUNCTION = "stack"
    CATEGORY = "EasyUse/Loaders"

    def stack(self, toggle, mode, num_controlnet, optional_controlnet_stack=None, **kwargs):
        if (toggle in [False, None, "False"]) or not kwargs:
            return (None,)

        controlnets = []

        # Import Stack values
        if optional_controlnet_stack is not None:
            controlnets.extend([l for l in optional_controlnet_stack if l[0] != "None"])

        # Import Controlnet values
        for i in range(1, num_controlnet+1):
            controlnet_name = kwargs.get(f"controlnet_{i}")

            if not controlnet_name or controlnet_name == "None":
                continue

            controlnet_strength = float(kwargs.get(f"controlnet_{i}_strength"))
            start_percent = float(kwargs.get(f"start_percent_{i}")) if mode == "advanced" else 0
            end_percent = float(kwargs.get(f"end_percent_{i}")) if mode == "advanced" else 1.0
            scale_soft_weights = float(kwargs.get(f"scale_soft_weight_{i}"))
            image = kwargs.get(f"image_{i}")

            controlnets.append((controlnet_name, controlnet_strength, start_percent, end_percent, scale_soft_weights, image, True))

        return (controlnets,)
# controlnet
class controlnetSimple:
    @classmethod
    def INPUT_TYPES(s):
        def get_file_list(filenames):
            return [file for file in filenames if file != "put_models_here.txt" and "lllite" not in file]

        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "image": ("IMAGE",),
                "control_net_name": (get_file_list(folder_paths.get_filename_list("controlnet")),),
            },
            "optional": {
                "control_net": ("CONTROL_NET",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "scale_soft_weights": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},),
            }
        }

    RETURN_TYPES = ("PIPE_LINE", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("pipe", "positive", "negative")
    OUTPUT_NODE = True

    FUNCTION = "controlnetApply"
    CATEGORY = "EasyUse/Loaders"

    def controlnetApply(self, pipe, image, control_net_name, control_net=None, strength=1, scale_soft_weights=1):

        positive, negative = easyControlnet().apply(control_net_name, image, pipe["positive"], pipe["negative"], strength, 0, 1, control_net, scale_soft_weights, None, easyCache)

        new_pipe = {
            "model": pipe['model'],
            "positive": positive,
            "negative": negative,
            "vae": pipe['vae'],
            "clip": pipe['clip'],

            "samples": pipe["samples"],
            "images": pipe["images"],
            "seed": 0,

            "loader_settings": pipe["loader_settings"]
        }

        del pipe
        return (new_pipe, positive, negative)

# controlnetADV
class controlnetAdvanced:

    @classmethod
    def INPUT_TYPES(s):
        def get_file_list(filenames):
            return [file for file in filenames if file != "put_models_here.txt" and "lllite" not in file]

        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "image": ("IMAGE",),
                "control_net_name": (get_file_list(folder_paths.get_filename_list("controlnet")),),
            },
            "optional": {
                "control_net": ("CONTROL_NET",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "scale_soft_weights": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},),
            }
        }

    RETURN_TYPES = ("PIPE_LINE", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("pipe", "positive", "negative")
    OUTPUT_NODE = True

    FUNCTION = "controlnetApply"
    CATEGORY = "EasyUse/Loaders"


    def controlnetApply(self, pipe, image, control_net_name, control_net=None, strength=1, start_percent=0, end_percent=1, scale_soft_weights=1):
        positive, negative = easyControlnet().apply(control_net_name, image, pipe["positive"], pipe["negative"],
                                                    strength, start_percent, end_percent, control_net, scale_soft_weights, None, easyCache)

        new_pipe = {
            "model": pipe['model'],
            "positive": positive,
            "negative": negative,
            "vae": pipe['vae'],
            "clip": pipe['clip'],

            "samples": pipe["samples"],
            "images": pipe["images"],
            "seed": 0,

            "loader_settings": pipe["loader_settings"]
        }

        del pipe

        return (new_pipe, positive, negative)

# LLLiteLoader
from .libs.lllite import load_control_net_lllite_patch
class LLLiteLoader:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        def get_file_list(filenames):
            return [file for file in filenames if file != "put_models_here.txt" and "lllite" in file]

        return {
            "required": {
                "model": ("MODEL",),
                "model_name": (get_file_list(folder_paths.get_filename_list("controlnet")),),
                "cond_image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 200, "step": 1}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "end_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lllite"
    CATEGORY = "EasyUse/Loaders"

    def load_lllite(self, model, model_name, cond_image, strength, steps, start_percent, end_percent):
        # cond_image is b,h,w,3, 0-1

        model_path = os.path.join(folder_paths.get_full_path("controlnet", model_name))

        model_lllite = model.clone()
        patch = load_control_net_lllite_patch(model_path, cond_image, strength, steps, start_percent, end_percent)
        if patch is not None:
            model_lllite.set_model_attn1_patch(patch)
            model_lllite.set_model_attn2_patch(patch)

        return (model_lllite,)

# ---------------------------------------------------------------加载器 结束----------------------------------------------------------------------#

#---------------------------------------------------------------Inpaint 开始----------------------------------------------------------------------#

# FooocusInpaint
from .libs.fooocus import InpaintHead, InpaintWorker
inpaint_head_model = None

class applyFooocusInpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "head": (list(FOOOCUS_INPAINT_HEAD.keys()),),
                "patch": (list(FOOOCUS_INPAINT_PATCH.keys()),),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    CATEGORY = "EasyUse/Inpaint"
    FUNCTION = "apply"

    def apply(self, model, latent, head, patch):

        global inpaint_head_model

        head_file = get_local_filepath(FOOOCUS_INPAINT_HEAD[head]["model_url"], INPAINT_DIR)
        if inpaint_head_model is None:
            inpaint_head_model = InpaintHead()
            sd = torch.load(head_file, map_location='cpu')
            inpaint_head_model.load_state_dict(sd)

        patch_file = get_local_filepath(FOOOCUS_INPAINT_PATCH[patch]["model_url"], INPAINT_DIR)
        inpaint_lora = comfy.utils.load_torch_file(patch_file, safe_load=True)

        patch = (inpaint_head_model, inpaint_lora)
        worker = InpaintWorker(node_name="easy kSamplerInpainting")
        cloned = model.clone()

        m, = worker.patch(cloned, latent, patch)
        return (m,)

# brushnet
from .brushnet import BrushNet
class applyBrushNet:

    def get_files_with_extension(folder='inpaint', extensions='.safetensors'):
        return [file for file in folder_paths.get_filename_list(folder) if file.endswith(extensions)]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "brushnet": (s.get_files_with_extension(),),
                "dtype": (['float16', 'bfloat16', 'float32', 'float64'], ),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "start_at": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at": ("INT", {"default": 10000, "min": 0, "max": 10000}),
            },
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    CATEGORY = "EasyUse/Inpaint"
    FUNCTION = "apply"

    def apply(self, pipe, image, mask, brushnet, dtype, scale, start_at, end_at):

        model = pipe['model']
        vae = pipe['vae']
        positive = pipe['positive']
        negative = pipe['negative']
        cls = BrushNet()
        if brushnet in backend_cache.cache:
            log_node_info("easy brushnetApply", f"Using {brushnet} Cached")
            _, brushnet_model = backend_cache.cache[brushnet][1]
        else:
            brushnet_file = os.path.join(folder_paths.get_full_path("inpaint", brushnet))
            brushnet_model, = cls.load_brushnet_model(brushnet_file, dtype)
            backend_cache.update_cache(brushnet, 'brushnet', (False, brushnet_model))
        m, positive, negative, latent = cls.brushnet_model_update(model=model, vae=vae, image=image, mask=mask,
                                                           brushnet=brushnet_model, positive=positive,
                                                           negative=negative, scale=scale, start_at=start_at,
                                                           end_at=end_at)
        new_pipe = {
            **pipe,
            "model": m,
            "positive": positive,
            "negative": negative,
            "samples": latent,
        }
        del pipe
        return (new_pipe,)

# #powerpaint
class applyPowerPaint:
    def get_files_with_extension(folder='inpaint', extensions='.safetensors'):
        return [file for file in folder_paths.get_filename_list(folder) if file.endswith(extensions)]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "powerpaint_model": (s.get_files_with_extension(),),
                "powerpaint_clip": (s.get_files_with_extension(extensions='.bin'),),
                "dtype": (['float16', 'bfloat16', 'float32', 'float64'],),
                "fitting": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 1.0}),
                "function": (['text guided', 'shape guided', 'object removal', 'context aware', 'image outpainting'],),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "start_at": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "save_memory": (['none', 'auto', 'max'],),
            },
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    CATEGORY = "EasyUse/Inpaint"
    FUNCTION = "apply"

    def apply(self, pipe, image, mask, powerpaint_model, powerpaint_clip, dtype, fitting, function, scale, start_at, end_at, save_memory='none'):
        model = pipe['model']
        vae = pipe['vae']
        positive = pipe['positive']
        negative = pipe['negative']

        cls = BrushNet()
        # load powerpaint clip
        if powerpaint_clip in backend_cache.cache:
            log_node_info("easy powerpaintApply", f"Using {powerpaint_clip} Cached")
            _, ppclip = backend_cache.cache[powerpaint_clip][1]
        else:
            model_url = POWERPAINT_MODELS['base_fp16']['model_url']
            base_clip = get_local_filepath(model_url, os.path.join(folder_paths.models_dir, 'clip'))
            ppclip, = cls.load_powerpaint_clip(base_clip, os.path.join(folder_paths.get_full_path("inpaint", powerpaint_clip)))
            backend_cache.update_cache(powerpaint_clip, 'ppclip', (False, ppclip))
        # load powerpaint model
        if powerpaint_model in backend_cache.cache:
            log_node_info("easy powerpaintApply", f"Using {powerpaint_model} Cached")
            _, powerpaint = backend_cache.cache[powerpaint_model][1]
        else:
            powerpaint_file = os.path.join(folder_paths.get_full_path("inpaint", powerpaint_model))
            powerpaint, = cls.load_brushnet_model(powerpaint_file, dtype)
            backend_cache.update_cache(powerpaint_model, 'powerpaint', (False, powerpaint))
        m, positive, negative, latent = cls.powerpaint_model_update(model=model, vae=vae, image=image, mask=mask, powerpaint=powerpaint,
                                                           clip=ppclip, positive=positive,
                                                           negative=negative, fitting=fitting, function=function,
                                                           scale=scale, start_at=start_at, end_at=end_at, save_memory=save_memory)
        new_pipe = {
            **pipe,
            "model": m,
            "positive": positive,
            "negative": negative,
            "samples": latent,
        }
        del pipe
        return (new_pipe,)

class applyInpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "inpaint_mode": (('normal', 'fooocus_inpaint', 'brushnet_random', 'brushnet_segmentation', 'powerpaint'),),
                "encode": (('none', 'vae_encode_inpaint', 'inpaint_model_conditioning', 'different_diffusion'), {"default": "none"}),
                "grow_mask_by": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),
                "dtype": (['float16', 'bfloat16', 'float32', 'float64'],),
                "fitting": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 1.0}),
                "function": (['text guided', 'shape guided', 'object removal', 'context aware', 'image outpainting'],),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "start_at": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at": ("INT", {"default": 10000, "min": 0, "max": 10000}),
            },
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    CATEGORY = "EasyUse/Inpaint"
    FUNCTION = "apply"

    def inpaint_model_conditioning(self, pipe, image, vae, mask, grow_mask_by):
        if grow_mask_by >0:
            mask, = GrowMask().expand_mask(mask, grow_mask_by, False)
        positive, negative, latent = InpaintModelConditioning().encode(pipe['positive'], pipe['negative'], image,
                                                                       vae, mask)
        pipe['positive'] = positive
        pipe['negative'] = negative
        pipe['samples'] = latent

        return pipe

    def get_brushnet_model(self, type, model):
        model_type = 'sdxl' if isinstance(model.model.model_config, comfy.supported_models.SDXL) else 'sd1'
        if type == 'brushnet_random':
            brush_model = BRUSHNET_MODELS['random_mask'][model_type]['model_url']
            if model_type == 'sdxl':
                pattern = 'brushnet.random.mask.sdxl.*\.(safetensors|bin)$'
            else:
                pattern = 'brushnet.random.mask.*\.(safetensors|bin)$'
        elif type == 'brushnet_segmentation':
            brush_model = BRUSHNET_MODELS['segmentation_mask'][model_type]['model_url']
            if model_type == 'sdxl':
                pattern = 'brushnet.segmentation.mask.sdxl.*\.(safetensors|bin)$'
            else:
                pattern = 'brushnet.segmentation.mask.*\.(safetensors|bin)$'


        brushfile = [e for e in folder_paths.get_filename_list('inpaint') if re.search(pattern, e, re.IGNORECASE)]
        brushname = brushfile[0] if brushfile else None
        if not brushname:
            from urllib.parse import urlparse
            get_local_filepath(brush_model, INPAINT_DIR)
            parsed_url = urlparse(brush_model)
            brushname = os.path.basename(parsed_url.path)
        return brushname

    def get_powerpaint_model(self, model):
        model_type = 'sdxl' if isinstance(model.model.model_config, comfy.supported_models.SDXL) else 'sd1'
        if model_type == 'sdxl':
            raise Exception("Powerpaint not supported for SDXL models")

        powerpaint_model = POWERPAINT_MODELS['v2.1']['model_url']
        powerpaint_clip = POWERPAINT_MODELS['v2.1']['clip_url']

        from urllib.parse import urlparse
        get_local_filepath(powerpaint_model, os.path.join(INPAINT_DIR, 'powerpaint'))
        model_parsed_url = urlparse(powerpaint_model)
        clip_parsed_url = urlparse(powerpaint_clip)
        model_name = os.path.join("powerpaint",os.path.basename(model_parsed_url.path))
        clip_name = os.path.join("powerpaint",os.path.basename(clip_parsed_url.path))
        return model_name, clip_name

    def apply(self, pipe, image, mask, inpaint_mode, encode, grow_mask_by, dtype, fitting, function, scale, start_at, end_at):
        new_pipe = {
            **pipe,
        }
        del pipe
        if inpaint_mode in ['brushnet_random', 'brushnet_segmentation']:
            brushnet = self.get_brushnet_model(inpaint_mode, new_pipe['model'])
            new_pipe, = applyBrushNet().apply(new_pipe, image, mask, brushnet, dtype, scale, start_at, end_at)
        elif inpaint_mode == 'powerpaint':
            powerpaint_model, powerpaint_clip = self.get_powerpaint_model(new_pipe['model'])
            new_pipe, = applyPowerPaint().apply(new_pipe, image, mask, powerpaint_model, powerpaint_clip, dtype, fitting, function, scale, start_at, end_at)

        vae = new_pipe['vae']
        if encode == 'none':
            if inpaint_mode == 'fooocus_inpaint':
                model, = applyFooocusInpaint().apply(new_pipe['model'], new_pipe['samples'],
                                                     list(FOOOCUS_INPAINT_HEAD.keys())[0],
                                                     list(FOOOCUS_INPAINT_PATCH.keys())[0])
                new_pipe['model'] = model
        elif encode == 'vae_encode_inpaint':
            latent, = VAEEncodeForInpaint().encode(vae, image, mask, grow_mask_by)
            new_pipe['samples'] = latent
            if inpaint_mode == 'fooocus_inpaint':
                model, = applyFooocusInpaint().apply(new_pipe['model'], new_pipe['samples'],
                                                     list(FOOOCUS_INPAINT_HEAD.keys())[0],
                                                     list(FOOOCUS_INPAINT_PATCH.keys())[0])
                new_pipe['model'] = model
        elif encode == 'inpaint_model_conditioning':
            if inpaint_mode == 'fooocus_inpaint':
                latent, = VAEEncodeForInpaint().encode(vae, image, mask, grow_mask_by)
                new_pipe['samples'] = latent
                model, = applyFooocusInpaint().apply(new_pipe['model'], new_pipe['samples'],
                                                     list(FOOOCUS_INPAINT_HEAD.keys())[0],
                                                     list(FOOOCUS_INPAINT_PATCH.keys())[0])
                new_pipe['model'] = model
                new_pipe = self.inpaint_model_conditioning(new_pipe, image, vae, mask, 0)
            else:
                new_pipe = self.inpaint_model_conditioning(new_pipe, image, vae, mask, grow_mask_by)
        elif encode == 'different_diffusion':
            if inpaint_mode == 'fooocus_inpaint':
                latent, = VAEEncodeForInpaint().encode(vae, image, mask, grow_mask_by)
                new_pipe['samples'] = latent
                model, = applyFooocusInpaint().apply(new_pipe['model'], new_pipe['samples'],
                                                     list(FOOOCUS_INPAINT_HEAD.keys())[0],
                                                     list(FOOOCUS_INPAINT_PATCH.keys())[0])
                new_pipe['model'] = model
                new_pipe = self.inpaint_model_conditioning(new_pipe, image, vae, mask, 0)
            else:
                new_pipe = self.inpaint_model_conditioning(new_pipe, image, vae, mask, grow_mask_by)
            cls = ALL_NODE_CLASS_MAPPINGS['DifferentialDiffusion']
            if cls is not None:
                model, = cls().apply(new_pipe['model'])
                new_pipe['model'] = model
            else:
                raise Exception("Differential Diffusion not found,please update comfyui")

        return (new_pipe,)
# ---------------------------------------------------------------Inpaint 结束----------------------------------------------------------------------#

#---------------------------------------------------------------适配器 开始----------------------------------------------------------------------#

# 风格对齐
from .libs.styleAlign import styleAlignBatch, SHARE_NORM_OPTIONS, SHARE_ATTN_OPTIONS
class styleAlignedBatchAlign:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "share_norm": (SHARE_NORM_OPTIONS,),
                "share_attn": (SHARE_ATTN_OPTIONS,),
                "scale": ("FLOAT", {"default": 1, "min": 0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "align"
    CATEGORY = "EasyUse/Adapter"

    def align(self, model, share_norm, share_attn, scale):
        return (styleAlignBatch(model, share_norm, share_attn, scale),)

# 光照对齐
from .ic_light.__init__ import ICLight, VAEEncodeArgMax
class icLightApply:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (list(IC_LIGHT_MODELS.keys()),),
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "lighting": (['None', 'Left Light', 'Right Light', 'Top Light', 'Bottom Light', 'Circle Light'],{"default": "None"}),
                "source": (['Use Background Image', 'Use Flipped Background Image', 'Left Light', 'Right Light', 'Top Light', 'Bottom Light', 'Ambient'],{"default": "Use Background Image"}),
                "remove_bg": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MODEL", "IMAGE")
    RETURN_NAMES = ("model", "lighting_image")
    FUNCTION = "apply"
    OUTPUT_NODE = True
    CATEGORY = "EasyUse/Adapter"

    def batch(self, image1, image2):
        if image1.shape[1:] != image2.shape[1:]:
            image2 = comfy.utils.common_upscale(image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "bilinear",
                                                "center").movedim(1, -1)
        s = torch.cat((image1, image2), dim=0)
        return s

    def removebg(self, image):
        if "easy imageRemBg" not in ALL_NODE_CLASS_MAPPINGS:
            raise Exception("Please re-install ComfyUI-Easy-Use")
        cls = ALL_NODE_CLASS_MAPPINGS['easy imageRemBg']
        results = cls().remove('RMBG-1.4', image, 'Hide', 'ComfyUI')
        if "result" in results:
            image, _ = results['result']
            return image

    def apply(self, mode, model, image, vae, lighting, source, remove_bg):
        model_type = get_sd_version(model)
        if model_type == 'sdxl':
            raise Exception("IC Light model is not supported for SDXL now")

        batch_size, height, width, channel = image.shape
        if channel == 3:
            # remove bg
            if mode == 'Foreground' or batch_size == 1:
                if remove_bg:
                    image = self.removebg(image)
                else:
                    mask = torch.full((1, height, width), 1.0, dtype=torch.float32, device="cpu")
                    image, = JoinImageWithAlpha().join_image_with_alpha(image, mask)

        iclight = ICLight()
        if mode == 'Foreground':
          lighting_image = iclight.generate_lighting_image(image, lighting)
        else:
          lighting_image = iclight.generate_source_image(image, source)
          if source not in ['Use Background Image', 'Use Flipped Background Image']:
              _, height, width, _ = lighting_image.shape
              mask = torch.full((1, height, width), 1.0, dtype=torch.float32, device="cpu")
              lighting_image, = JoinImageWithAlpha().join_image_with_alpha(lighting_image, mask)
              if batch_size < 2:
                image = self.batch(image, lighting_image)
              else:
                original_image = [img.unsqueeze(0) for img in image]
                original_image = self.removebg(original_image[0])
                image = self.batch(original_image, lighting_image)

        latent, = VAEEncodeArgMax().encode(vae, image)
        key = 'iclight_' + mode + '_' + model_type
        model_path = get_local_filepath(IC_LIGHT_MODELS[mode]['sd1']["model_url"],
                                        os.path.join(folder_paths.models_dir, "unet"))
        ic_model = None
        if key in backend_cache.cache:
            log_node_info("easy icLightApply", f"Using icLightModel {mode+'_'+model_type} Cached")
            _, ic_model = backend_cache.cache[key][1]
            m, _ = iclight.apply(model_path, model, latent, ic_model)
        else:
            m, ic_model = iclight.apply(model_path, model, latent, ic_model)
            backend_cache.update_cache(key, 'iclight', (False, ic_model))
        return (m, lighting_image)


def insightface_loader(provider):
    try:
        from insightface.app import FaceAnalysis
    except ImportError as e:
        raise Exception(e)

    path = os.path.join(folder_paths.models_dir, "insightface")
    model = FaceAnalysis(name="buffalo_l", root=path, providers=[provider + 'ExecutionProvider', ])
    model.prepare(ctx_id=0, det_size=(640, 640))
    return model

# Apply Ipadapter
class ipadapter:

    def __init__(self):
        self.normal_presets = [
            'LIGHT - SD1.5 only (low strength)',
            'STANDARD (medium strength)',
            'VIT-G (medium strength)',
            'PLUS (high strength)',
            'PLUS FACE (portraits)',
            'FULL FACE - SD1.5 only (portraits stronger)',
            'COMPOSITION'
        ]
        self.faceid_presets = [
            'FACEID',
            'FACEID PLUS - SD1.5 only',
            'FACEID PLUS V2',
            'FACEID PORTRAIT (style transfer)'
        ]
        self.weight_types = ["linear", "ease in", "ease out", 'ease in-out', 'reverse in-out', 'weak input', 'weak output', 'weak middle', 'strong middle', 'style transfer', 'composition']
        self.presets = self.normal_presets + self.faceid_presets


    def error(self):
        raise Exception(f"[ERROR] To use ipadapterApply, you need to install 'ComfyUI_IPAdapter_plus'")

    def get_clipvision_file(self, preset, node_name):
        preset = preset.lower()
        clipvision_list = folder_paths.get_filename_list("clip_vision")

        if preset.startswith("vit-g"):
            pattern = '(ViT.bigG.14.*39B.b160k|ipadapter.*sdxl|sdxl.*model\.(bin|safetensors))'
        else:
            pattern = '(ViT.H.14.*s32B.b79K|ipadapter.*sd15|sd1.?5.*model\.(bin|safetensors))'
        clipvision_files = [e for e in clipvision_list if re.search(pattern, e, re.IGNORECASE)]

        clipvision_name = clipvision_files[0] if len(clipvision_files)>0 else None
        clipvision_file = folder_paths.get_full_path("clip_vision", clipvision_name) if clipvision_name else None
        # if clipvision_name is not None:
        #     log_node_info(node_name, f"Using {clipvision_name}")

        return clipvision_file, clipvision_name

    def get_ipadapter_file(self, preset, is_sdxl, node_name):
        preset = preset.lower()
        ipadapter_list = folder_paths.get_filename_list("ipadapter")
        is_insightface = False
        lora_pattern = None

        if preset.startswith("light"):
            if is_sdxl:
                raise Exception("light model is not supported for SDXL")
            pattern = 'sd15.light.v11\.(safetensors|bin)$'
            # if light model v11 is not found, try with the old version
            if not [e for e in ipadapter_list if re.search(pattern, e, re.IGNORECASE)]:
                pattern = 'sd15.light\.(safetensors|bin)$'
        elif preset.startswith("standard"):
            if is_sdxl:
                pattern = 'ip.adapter.sdxl.vit.h\.(safetensors|bin)$'
            else:
                pattern = 'ip.adapter.sd15\.(safetensors|bin)$'
        elif preset.startswith("vit-g"):
            if is_sdxl:
                pattern = 'ip.adapter.sdxl\.(safetensors|bin)$'
            else:
                pattern = 'sd15.vit.g\.(safetensors|bin)$'
        elif preset.startswith("plus ("):
            if is_sdxl:
                pattern = 'plus.sdxl.vit.h\.(safetensors|bin)$'
            else:
                pattern = 'ip.adapter.plus.sd15\.(safetensors|bin)$'
        elif preset.startswith("plus face"):
            if is_sdxl:
                pattern = 'plus.face.sdxl.vit.h\.(safetensors|bin)$'
            else:
                pattern = 'plus.face.sd15\.(safetensors|bin)$'
        elif preset.startswith("full"):
            if is_sdxl:
                raise Exception("full face model is not supported for SDXL")
            pattern = 'full.face.sd15\.(safetensors|bin)$'
        elif preset.startswith("composition"):
            if is_sdxl:
                pattern = 'plus.composition.sdxl\.(safetensors|bin)$'
            else:
                pattern = 'plus.composition.sd15\.(safetensors|bin)$'
        elif preset.startswith("faceid portrait"):
            if is_sdxl:
                pattern = 'portrait.sdxl\.(safetensors|bin)$'
            else:
                pattern = 'portrait.v11.sd15\.(safetensors|bin)$'
                # if v11 is not found, try with the old version
                if not [e for e in ipadapter_list if re.search(pattern, e, re.IGNORECASE)]:
                    pattern = 'portrait.sd15\.(safetensors|bin)$'
            is_insightface = True
        elif preset == "faceid":
            if is_sdxl:
                pattern = 'faceid.sdxl\.(safetensors|bin)$'
                lora_pattern = 'faceid.sdxl.lora\.safetensors$'
            else:
                pattern = 'faceid.sd15\.(safetensors|bin)$'
                lora_pattern = 'faceid.sd15.lora\.safetensors$'
            is_insightface = True
        elif preset.startswith("faceid plus -"):
            if is_sdxl:
                raise Exception("faceid plus model is not supported for SDXL")
            pattern = 'faceid.plus.sd15\.(safetensors|bin)$'
            lora_pattern = 'faceid.plus.sd15.lora\.safetensors$'
            is_insightface = True
        elif preset.startswith("faceid plus v2"):
            if is_sdxl:
                pattern = 'faceid.plusv2.sdxl\.(safetensors|bin)$'
                lora_pattern = 'faceid.plusv2.sdxl.lora\.safetensors$'
            else:
                pattern = 'faceid.plusv2.sd15\.(safetensors|bin)$'
                lora_pattern = 'faceid.plusv2.sd15.lora\.safetensors$'
            is_insightface = True
        else:
            raise Exception(f"invalid type '{preset}'")

        ipadapter_files = [e for e in ipadapter_list if re.search(pattern, e, re.IGNORECASE)]
        ipadapter_name = ipadapter_files[0] if len(ipadapter_files)>0 else None
        ipadapter_file = folder_paths.get_full_path("ipadapter", ipadapter_name) if ipadapter_name else None
        # if ipadapter_name is not None:
        #     log_node_info(node_name, f"Using {ipadapter_name}")

        return ipadapter_file, ipadapter_name, is_insightface, lora_pattern

    def get_lora_file(self, preset, pattern, model_type, model, model_strength, clip_strength, clip=None):
        lora_list = folder_paths.get_filename_list("loras")
        lora_files = [e for e in lora_list if re.search(pattern, e, re.IGNORECASE)]
        lora_name = lora_files[0] if lora_files else None
        if lora_name:
            return easyCache.load_lora({"model": model, "clip": clip, "lora_name": lora_name, "model_strength":model_strength, "clip_strength":clip_strength},)
        else:
            if "lora_url" in IPADAPTER_MODELS[preset][model_type]:
                lora_name = get_local_filepath(IPADAPTER_MODELS[preset][model_type]["lora_url"], os.path.join(folder_paths.models_dir, "loras"))
                return easyCache.load_lora({"model": model, "clip": clip, "lora_name": lora_name, "model_strength":model_strength, "clip_strength":clip_strength},)
            return (model, clip)

    def ipadapter_model_loader(self, file):
        model = comfy.utils.load_torch_file(file, safe_load=True)

        if file.lower().endswith(".safetensors"):
            st_model = {"image_proj": {}, "ip_adapter": {}}
            for key in model.keys():
                if key.startswith("image_proj."):
                    st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
                elif key.startswith("ip_adapter."):
                    st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
            model = st_model
            del st_model

        if not "ip_adapter" in model.keys() or not model["ip_adapter"]:
            raise Exception("invalid IPAdapter model {}".format(file))

        if 'plusv2' in file.lower():
            model["faceidplusv2"] = True

        return model

    def load_model(self, model, preset, lora_model_strength, provider="CPU", clip_vision=None, optional_ipadapter=None, cache_mode='none', node_name='easy ipadapterApply'):
        pipeline = {"clipvision": {'file': None, 'model': None}, "ipadapter": {'file': None, 'model': None},
                    "insightface": {'provider': None, 'model': None}}
        if optional_ipadapter is not None:
            pipeline = optional_ipadapter

        # 1. Load the clipvision model
        if not clip_vision:
            clipvision_file, clipvision_name = self.get_clipvision_file(preset, node_name)
            if clipvision_file is None:
                raise Exception("ClipVision model not found.")
            if clipvision_file == pipeline['clipvision']['file']:
                clip_vision = pipeline['clipvision']['model']
            elif cache_mode in ["all", "clip_vision only"] and clipvision_name in backend_cache.cache:
                log_node_info("easy ipadapterApply", f"Using ClipVisonModel {clipvision_name} Cached")
                _, clip_vision = backend_cache.cache[clipvision_name][1]
            else:
                clip_vision = load_clip_vision(clipvision_file)
                log_node_info("easy ipadapterApply", f"Using ClipVisonModel {clipvision_name}")
                if cache_mode in ["all", "clip_vision only"]:
                    backend_cache.update_cache(clipvision_name, 'clip_vision', (False, clip_vision))
            pipeline['clipvision']['file'] = clipvision_file
        pipeline['clipvision']['model'] = clip_vision

        # 2. Load the ipadapter model
        is_sdxl = isinstance(model.model, comfy.model_base.SDXL)
        ipadapter_file, ipadapter_name, is_insightface, lora_pattern = self.get_ipadapter_file(preset, is_sdxl, node_name)
        model_type = 'sdxl' if is_sdxl else 'sd15'
        if ipadapter_file is None:
            model_url = IPADAPTER_MODELS[preset][model_type]["model_url"]
            ipadapter_file = get_local_filepath(model_url, IPADAPTER_DIR)
            ipadapter_name = os.path.basename(model_url)
        if ipadapter_file == pipeline['ipadapter']['file']:
            ipadapter = pipeline['ipadapter']['model']
        elif cache_mode in ["all", "ipadapter only"] and ipadapter_name in backend_cache.cache:
            log_node_info("easy ipadapterApply", f"Using IpAdapterModel {ipadapter_name} Cached")
            _, ipadapter = backend_cache.cache[ipadapter_name][1]
        else:
            ipadapter = self.ipadapter_model_loader(ipadapter_file)
            pipeline['ipadapter']['file'] = ipadapter_file
            log_node_info("easy ipadapterApply", f"Using IpAdapterModel {ipadapter_name}")
            if cache_mode in ["all", "ipadapter only"]:
                backend_cache.update_cache(ipadapter_name, 'ipadapter', (False, ipadapter))

        pipeline['ipadapter']['model'] = ipadapter

        # 3. Load the lora model if needed
        if lora_pattern is not None:
            if lora_model_strength > 0:
              model, _ = self.get_lora_file(preset, lora_pattern, model_type, model, lora_model_strength, 1)

        # 4. Load the insightface model if needed
        if is_insightface:
            icache_key = 'insightface-' + provider
            if provider == pipeline['insightface']['provider']:
                insightface = pipeline['insightface']['model']
            elif cache_mode in ["all", "insightface only"] and icache_key in backend_cache.cache:
                log_node_info("easy ipadapterApply", f"Using InsightFaceModel {icache_key} Cached")
                _, insightface = backend_cache.cache[icache_key][1]
            else:
                insightface = insightface_loader(provider)
                if cache_mode in ["all", "insightface only"]:
                    backend_cache.update_cache(icache_key, 'insightface',(False, insightface))
            pipeline['insightface']['provider'] = provider
            pipeline['insightface']['model'] = insightface

        return (model, pipeline,)

class ipadapterApply(ipadapter):
    def __init__(self):
        super().__init__()
        pass

    @classmethod
    def INPUT_TYPES(cls):
        presets = cls().presets
        return {
            "required": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "preset": (presets,),
                "lora_strength": ("FLOAT", {"default": 0.6, "min": 0, "max": 1, "step": 0.01}),
                "provider": (["CPU", "CUDA", "ROCM", "DirectML", "OpenVINO", "CoreML"],),
                "weight": ("FLOAT", {"default": 1.0, "min": -1, "max": 3, "step": 0.05}),
                "weight_faceidv2": ("FLOAT", { "default": 1.0, "min": -1, "max": 5.0, "step": 0.05 }),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "cache_mode": (["insightface only", "clip_vision only", "ipadapter only", "all", "none"], {"default": "all"},),
                "use_tiled": ("BOOLEAN", {"default": False},),
            },

            "optional": {
                "attn_mask": ("MASK",),
                "optional_ipadapter": ("IPADAPTER",),
            }
        }

    RETURN_TYPES = ("MODEL", "IMAGE", "MASK", "IPADAPTER",)
    RETURN_NAMES = ("model", "images", "masks", "ipadapter", )
    CATEGORY = "EasyUse/Adapter"
    FUNCTION = "apply"

    def apply(self, model, image, preset, lora_strength, provider, weight, weight_faceidv2, start_at, end_at, cache_mode, use_tiled, attn_mask=None, optional_ipadapter=None):
        images, masks = image, [None]
        model, ipadapter = self.load_model(model, preset, lora_strength, provider, clip_vision=None, optional_ipadapter=optional_ipadapter, cache_mode=cache_mode)
        if use_tiled and preset not in self.faceid_presets:
            if "IPAdapterTiled" not in ALL_NODE_CLASS_MAPPINGS:
                self.error()
            cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterTiled"]
            model, images, masks = cls().apply_tiled(model, ipadapter, image, weight, "linear", start_at, end_at, sharpening=0.0, combine_embeds="concat", image_negative=None, attn_mask=attn_mask, clip_vision=None, embeds_scaling='V only')
        else:
            if preset in ['FACEID PLUS V2', 'FACEID PORTRAIT (style transfer)']:
                if "IPAdapterAdvanced" not in ALL_NODE_CLASS_MAPPINGS:
                    self.error()
                cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterAdvanced"]
                model, images = cls().apply_ipadapter(model, ipadapter, start_at=start_at, end_at=end_at, weight=weight, weight_type="linear", combine_embeds="concat", weight_faceidv2=weight_faceidv2, image=image, image_negative=None, clip_vision=None, attn_mask=attn_mask, insightface=None, embeds_scaling='V only')
            else:
                if "IPAdapter" not in ALL_NODE_CLASS_MAPPINGS:
                    self.error()
                cls = ALL_NODE_CLASS_MAPPINGS["IPAdapter"]
                model, images = cls().apply_ipadapter(model, ipadapter, image, weight, start_at, end_at, weight_type='standard', attn_mask=attn_mask)
        if images is None:
            images = image
        return (model, images, masks, ipadapter,)

class ipadapterApplyAdvanced(ipadapter):
    def __init__(self):
        super().__init__()
        pass

    @classmethod
    def INPUT_TYPES(cls):
        ipa_cls = cls()
        presets = ipa_cls.presets
        weight_types = ipa_cls.weight_types
        return {
            "required": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "preset": (presets,),
                "lora_strength": ("FLOAT", {"default": 0.6, "min": 0, "max": 1, "step": 0.01}),
                "provider": (["CPU", "CUDA", "ROCM", "DirectML", "OpenVINO", "CoreML"],),
                "weight": ("FLOAT", {"default": 1.0, "min": -1, "max": 3, "step": 0.05}),
                "weight_faceidv2": ("FLOAT", {"default": 1.0, "min": -1, "max": 5.0, "step": 0.05 }),
                "weight_type": (weight_types,),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'],),
                "cache_mode": (["insightface only", "clip_vision only","ipadapter only", "all", "none"], {"default": "all"},),
                "use_tiled": ("BOOLEAN", {"default": False},),
                "use_batch": ("BOOLEAN", {"default": False},),
                "sharpening": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },

            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
                "optional_ipadapter": ("IPADAPTER",),
            }
        }

    RETURN_TYPES = ("MODEL", "IMAGE", "MASK", "IPADAPTER",)
    RETURN_NAMES = ("model", "images", "masks", "ipadapter", )
    CATEGORY = "EasyUse/Adapter"
    FUNCTION = "apply"

    def apply(self, model, image, preset, lora_strength, provider, weight, weight_faceidv2, weight_type, combine_embeds, start_at, end_at, embeds_scaling, cache_mode, use_tiled, use_batch, sharpening, weight_style=1.0, weight_composition=1.0, image_style=None, image_composition=None, expand_style=False, image_negative=None, clip_vision=None, attn_mask=None, optional_ipadapter=None):
        images, masks = image, [None]
        model, ipadapter = self.load_model(model, preset, lora_strength, provider, clip_vision=clip_vision, optional_ipadapter=optional_ipadapter, cache_mode=cache_mode)
        if use_tiled:
            if use_batch:
                if "IPAdapterTiledBatch" not in ALL_NODE_CLASS_MAPPINGS:
                    self.error()
                cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterTiledBatch"]
            else:
                if "IPAdapterTiled" not in ALL_NODE_CLASS_MAPPINGS:
                    self.error()
                cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterTiled"]
            model, images, masks = cls().apply_tiled(model, ipadapter, image=image, weight=weight, weight_type=weight_type, start_at=start_at, end_at=end_at, sharpening=sharpening, combine_embeds=combine_embeds, image_negative=image_negative, attn_mask=attn_mask, clip_vision=clip_vision, embeds_scaling=embeds_scaling)
        else:
            if use_batch:
                if "IPAdapterBatch" not in ALL_NODE_CLASS_MAPPINGS:
                    self.error()
                cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterBatch"]
            else:
                if "IPAdapterAdvanced" not in ALL_NODE_CLASS_MAPPINGS:
                    self.error()
                cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterAdvanced"]
            model, images = cls().apply_ipadapter(model, ipadapter, weight=weight, weight_type=weight_type, start_at=start_at, end_at=end_at, combine_embeds=combine_embeds, weight_faceidv2=weight_faceidv2, image=image, image_negative=image_negative, weight_style=1.0, weight_composition=1.0, image_style=image_style, image_composition=image_composition, expand_style=expand_style, clip_vision=clip_vision, attn_mask=attn_mask, insightface=None, embeds_scaling=embeds_scaling)
        if images is None:
            images = image
        return (model, images, masks, ipadapter)

class ipadapterStyleComposition(ipadapter):
    def __init__(self):
        super().__init__()
        pass

    @classmethod
    def INPUT_TYPES(cls):
        ipa_cls = cls()
        normal_presets = ipa_cls.normal_presets
        weight_types = ipa_cls.weight_types
        return {
            "required": {
                "model": ("MODEL",),
                "image_style": ("IMAGE",),
                "preset": (normal_presets,),
                "weight_style": ("FLOAT", {"default": 1.0, "min": -1, "max": 5, "step": 0.05}),
                "weight_composition": ("FLOAT", {"default": 1.0, "min": -1, "max": 5, "step": 0.05}),
                "expand_style": ("BOOLEAN", {"default": False}),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"], {"default": "average"}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'],),
                "cache_mode": (["insightface only", "clip_vision only", "ipadapter only", "all", "none"],
                               {"default": "all"},),
            },
            "optional": {
                "image_composition": ("IMAGE",),
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
                "optional_ipadapter": ("IPADAPTER",),
            }
        }

    CATEGORY = "EasyUse/Adapter"

    RETURN_TYPES = ("MODEL", "IPADAPTER",)
    RETURN_NAMES = ("model", "ipadapter",)
    CATEGORY = "EasyUse/Adapter"
    FUNCTION = "apply"

    def apply(self, model, preset, weight_style, weight_composition, expand_style, combine_embeds, start_at, end_at, embeds_scaling, cache_mode, image_style=None , image_composition=None, image_negative=None, clip_vision=None, attn_mask=None, optional_ipadapter=None):
        model, ipadapter = self.load_model(model, preset, 0, 'CPU', clip_vision=None, optional_ipadapter=optional_ipadapter, cache_mode=cache_mode)

        if "IPAdapterAdvanced" not in ALL_NODE_CLASS_MAPPINGS:
            self.error()
        cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterAdvanced"]

        model, image = cls().apply_ipadapter(model, ipadapter, start_at=start_at, end_at=end_at, weight_style=weight_style, weight_composition=weight_composition, weight_type='linear', combine_embeds=combine_embeds, weight_faceidv2=weight_composition, image_style=image_style, image_composition=image_composition, image_negative=image_negative, expand_style=expand_style, clip_vision=clip_vision, attn_mask=attn_mask, insightface=None, embeds_scaling=embeds_scaling)
        return (model, ipadapter)

class ipadapterApplyEncoder(ipadapter):
    def __init__(self):
        super().__init__()
        pass

    @classmethod
    def INPUT_TYPES(cls):
        ipa_cls = cls()
        normal_presets = ipa_cls.normal_presets
        max_embeds_num = 4
        inputs = {
            "required": {
                "model": ("MODEL",),
                "clip_vision": ("CLIP_VISION",),
                "image1": ("IMAGE",),
                "preset": (normal_presets,),
                "num_embeds":  ("INT", {"default": 2, "min": 1, "max": max_embeds_num}),
            },
            "optional": {}
        }

        for i in range(1, max_embeds_num + 1):
            if i > 1:
                inputs["optional"][f"image{i}"] = ("IMAGE",)
        for i in range(1, max_embeds_num + 1):
            inputs["optional"][f"mask{i}"] = ("MASK",)
            inputs["optional"][f"weight{i}"] = ("FLOAT", {"default": 1.0, "min": -1, "max": 3, "step": 0.05})
        inputs["optional"]["combine_method"] = (["concat", "add", "subtract", "average", "norm average", "max", "min"],)
        inputs["optional"]["optional_ipadapter"] = ("IPADAPTER",)
        inputs["optional"]["pos_embeds"] = ("EMBEDS",)
        inputs["optional"]["neg_embeds"] = ("EMBEDS",)
        return inputs

    RETURN_TYPES = ("MODEL", "CLIP_VISION","IPADAPTER", "EMBEDS", "EMBEDS", )
    RETURN_NAMES = ("model", "clip_vision","ipadapter", "pos_embed", "neg_embed",)
    CATEGORY = "EasyUse/Adapter"
    FUNCTION = "apply"

    def batch(self, embeds, method):
        if method == 'concat' and len(embeds) == 1:
            return (embeds[0],)

        embeds = [embed for embed in embeds if embed is not None]
        embeds = torch.cat(embeds, dim=0)

        match method:
            case "add":
                embeds = torch.sum(embeds, dim=0).unsqueeze(0)
            case "subtract":
                embeds = embeds[0] - torch.mean(embeds[1:], dim=0)
                embeds = embeds.unsqueeze(0)
            case "average":
                embeds = torch.mean(embeds, dim=0).unsqueeze(0)
            case "norm average":
                embeds = torch.mean(embeds / torch.norm(embeds, dim=0, keepdim=True), dim=0).unsqueeze(0)
            case "max":
                embeds = torch.max(embeds, dim=0).values.unsqueeze(0)
            case "min":
                embeds = torch.min(embeds, dim=0).values.unsqueeze(0)

        return embeds

    def apply(self, **kwargs):
        model = kwargs['model']
        clip_vision = kwargs['clip_vision']
        preset = kwargs['preset']
        if 'optional_ipadapter' in kwargs:
            ipadapter = kwargs['optional_ipadapter']
        else:
            model, ipadapter = self.load_model(model, preset, 0, 'CPU', clip_vision=clip_vision, optional_ipadapter=None, cache_mode='none')

        if "IPAdapterEncoder" not in ALL_NODE_CLASS_MAPPINGS:
            self.error()
        encoder_cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterEncoder"]
        pos_embeds = kwargs["pos_embeds"] if "pos_embeds" in kwargs else []
        neg_embeds = kwargs["neg_embeds"] if "neg_embeds" in kwargs else []
        for i in range(1, kwargs['num_embeds'] + 1):
            if f"image{i}" not in kwargs:
                raise Exception(f"image{i} is required")
            kwargs[f"mask{i}"] = kwargs[f"mask{i}"] if f"mask{i}" in kwargs else None
            kwargs[f"weight{i}"] = kwargs[f"weight{i}"] if f"weight{i}" in kwargs else 1.0

            pos, neg = encoder_cls().encode(ipadapter, kwargs[f"image{i}"], kwargs[f"weight{i}"], kwargs[f"mask{i}"], clip_vision=clip_vision)
            pos_embeds.append(pos)
            neg_embeds.append(neg)

        pos_embeds = self.batch(pos_embeds, kwargs['combine_method'])
        neg_embeds = self.batch(neg_embeds, kwargs['combine_method'])

        return (model,clip_vision, ipadapter, pos_embeds, neg_embeds)

class ipadapterApplyEmbeds(ipadapter):
    def __init__(self):
        super().__init__()
        pass

    @classmethod
    def INPUT_TYPES(cls):
        ipa_cls = cls()
        weight_types = ipa_cls.weight_types
        return {
            "required": {
                "model": ("MODEL",),
                "clip_vision": ("CLIP_VISION",),
                "ipadapter": ("IPADAPTER",),
                "pos_embed": ("EMBEDS",),
                "weight": ("FLOAT", {"default": 1.0, "min": -1, "max": 3, "step": 0.05}),
                "weight_type": (weight_types,),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'],),
            },

            "optional": {
                "neg_embed": ("EMBEDS",),
                "attn_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MODEL", "IPADAPTER",)
    RETURN_NAMES = ("model", "ipadapter", )
    CATEGORY = "EasyUse/Adapter"
    FUNCTION = "apply"

    def apply(self, model, ipadapter, clip_vision, pos_embed, weight, weight_type, start_at, end_at, embeds_scaling, attn_mask=None, neg_embed=None,):
        if "IPAdapterEmbeds" not in ALL_NODE_CLASS_MAPPINGS:
            self.error()

        cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterEmbeds"]
        model, image = cls().apply_ipadapter(model, ipadapter, pos_embed, weight, weight_type, start_at, end_at, neg_embed=neg_embed, attn_mask=attn_mask, clip_vision=clip_vision, embeds_scaling=embeds_scaling)

        return (model, ipadapter)

class ipadapterApplyRegional(ipadapter):
    def __init__(self):
        super().__init__()
        pass

    @classmethod
    def INPUT_TYPES(cls):
        ipa_cls = cls()
        weight_types = ipa_cls.weight_types
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "image": ("IMAGE",),
                "positive": ("STRING", {"default": "", "placeholder": "positive", "multiline": True}),
                "negative": ("STRING", {"default": "", "placeholder": "negative",  "multiline": True}),
                "image_weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 3.0, "step": 0.05}),
                "prompt_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "weight_type": (weight_types,),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },

            "optional": {
                "mask": ("MASK",),
                "optional_ipadapter_params": ("IPADAPTER_PARAMS",),
            },
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "IPADAPTER_PARAMS", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("pipe", "ipadapter_params", "positive", "negative")
    CATEGORY = "EasyUse/Adapter"
    FUNCTION = "apply"

    def apply(self, pipe, image, positive, negative, image_weight, prompt_weight, weight_type, start_at, end_at, mask=None, optional_ipadapter_params=None, prompt=None, my_unique_id=None):
        model = pipe['model']
        clip = pipe['clip']
        clip_skip = pipe['loader_settings']['clip_skip']
        a1111_prompt_style = pipe['loader_settings']['a1111_prompt_style']
        pipe_lora_stack = pipe['loader_settings']['lora_stack']
        positive_token_normalization = pipe['loader_settings']['positive_token_normalization']
        positive_weight_interpretation = pipe['loader_settings']['positive_weight_interpretation']
        negative_token_normalization = pipe['loader_settings']['negative_token_normalization']
        negative_weight_interpretation = pipe['loader_settings']['negative_weight_interpretation']
        if positive == '':
            positive = pipe['loader_settings']['positive']
        if negative == '':
            negative = pipe['loader_settings']['negative']

        if not clip:
            raise Exception("No CLIP found")

        positive_embeddings_final, positive_wildcard_prompt, model, clip = prompt_to_cond('positive', model, clip, clip_skip, pipe_lora_stack, positive, positive_token_normalization, positive_weight_interpretation, a1111_prompt_style, my_unique_id, prompt, easyCache)
        negative_embeddings_final, negative_wildcard_prompt, model, clip = prompt_to_cond('negative', model, clip, clip_skip, pipe_lora_stack, negative, negative_token_normalization, negative_weight_interpretation, a1111_prompt_style, my_unique_id, prompt, easyCache)

        #ipadapter regional
        if "IPAdapterRegionalConditioning" not in ALL_NODE_CLASS_MAPPINGS:
            self.error()

        cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterRegionalConditioning"]
        ipadapter_params, new_positive_embeds, new_negative_embeds = cls().conditioning(image, image_weight, prompt_weight, weight_type, start_at, end_at, mask=mask, positive=positive_embeddings_final, negative=negative_embeddings_final)

        if optional_ipadapter_params is not None:
            positive_embeds = pipe['positive'] + new_positive_embeds
            negative_embeds = pipe['negative'] + new_negative_embeds
            _ipadapter_params = {
                "image": optional_ipadapter_params["image"] + ipadapter_params["image"],
                "attn_mask": optional_ipadapter_params["attn_mask"] + ipadapter_params["attn_mask"],
                "weight": optional_ipadapter_params["weight"] + ipadapter_params["weight"],
                "weight_type": optional_ipadapter_params["weight_type"] + ipadapter_params["weight_type"],
                "start_at": optional_ipadapter_params["start_at"] + ipadapter_params["start_at"],
                "end_at": optional_ipadapter_params["end_at"] + ipadapter_params["end_at"],
            }
            ipadapter_params = _ipadapter_params
            del _ipadapter_params
        else:
            positive_embeds = new_positive_embeds
            negative_embeds = new_negative_embeds

        new_pipe = {
            **pipe,
            "positive": positive_embeds,
            "negative": negative_embeds,
        }

        del pipe

        return (new_pipe, ipadapter_params, positive_embeds, negative_embeds)

class ipadapterApplyFromParams(ipadapter):
    def __init__(self):
        super().__init__()
        pass

    @classmethod
    def INPUT_TYPES(cls):
        ipa_cls = cls()
        normal_presets = ipa_cls.normal_presets
        return {
            "required": {
                "model": ("MODEL",),
                "preset": (normal_presets,),
                "ipadapter_params": ("IPADAPTER_PARAMS",),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average", "max", "min"],),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'],),
                "cache_mode": (["insightface only", "clip_vision only", "ipadapter only", "all", "none"],
                               {"default": "insightface only"},{"default":"none"}),
            },

            "optional": {
                "optional_ipadapter": ("IPADAPTER",),
                "image_negative": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MODEL", "IPADAPTER",)
    RETURN_NAMES = ("model", "ipadapter", )
    CATEGORY = "EasyUse/Adapter"
    FUNCTION = "apply"

    def apply(self, model, preset, ipadapter_params, combine_embeds, embeds_scaling, cache_mode, optional_ipadapter=None, image_negative=None,):
        model, ipadapter = self.load_model(model, preset, 0, 'CPU', clip_vision=None, optional_ipadapter=optional_ipadapter, cache_mode=cache_mode)
        if "IPAdapterFromParams" not in ALL_NODE_CLASS_MAPPINGS:
            self.error()
        cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterFromParams"]
        model, image = cls().apply_ipadapter(model, ipadapter, clip_vision=None, combine_embeds=combine_embeds, embeds_scaling=embeds_scaling, image_negative=image_negative, ipadapter_params=ipadapter_params)

        return (model, ipadapter)

#Apply InstantID
class instantID:

    def error(self):
        raise Exception(f"[ERROR] To use instantIDApply, you need to install 'ComfyUI_InstantID'")

    def run(self, pipe, image, instantid_file, insightface, control_net_name, cn_strength, cn_soft_weights, weight, start_at, end_at, noise, image_kps=None, mask=None, control_net=None, positive=None, negative=None, prompt=None, extra_pnginfo=None, my_unique_id=None):
        instantid_model, insightface_model, face_embeds = None, None, None
        model = pipe['model']
        # Load InstantID
        cache_key = 'instantID'
        if cache_key in backend_cache.cache:
            log_node_info("easy instantIDApply","Using InstantIDModel Cached")
            _, instantid_model = backend_cache.cache[cache_key][1]
        if "InstantIDModelLoader" in ALL_NODE_CLASS_MAPPINGS:
            load_instant_cls = ALL_NODE_CLASS_MAPPINGS["InstantIDModelLoader"]
            instantid_model, = load_instant_cls().load_model(instantid_file)
            backend_cache.update_cache(cache_key, 'instantid', (False, instantid_model))
        else:
            self.error()
        icache_key = 'insightface-' + insightface
        if icache_key in backend_cache.cache:
            log_node_info("easy instantIDApply", f"Using InsightFaceModel {insightface} Cached")
            _, insightface_model = backend_cache.cache[icache_key][1]
        elif "InstantIDFaceAnalysis" in ALL_NODE_CLASS_MAPPINGS:
            load_insightface_cls = ALL_NODE_CLASS_MAPPINGS["InstantIDFaceAnalysis"]
            insightface_model, = load_insightface_cls().load_insight_face(insightface)
            backend_cache.update_cache(icache_key, 'insightface', (False, insightface_model))
        else:
            self.error()

        # Apply InstantID
        if "ApplyInstantID" in ALL_NODE_CLASS_MAPPINGS:
            instantid_apply = ALL_NODE_CLASS_MAPPINGS['ApplyInstantID']
            if control_net is None:
                control_net = easyCache.load_controlnet(control_net_name, cn_soft_weights)
            model, positive, negative = instantid_apply().apply_instantid(instantid_model, insightface_model, control_net, image, model, positive, negative, start_at, end_at, weight=weight, ip_weight=None, cn_strength=cn_strength, noise=noise, image_kps=image_kps, mask=mask)
        else:
            self.error()

        new_pipe = {
            "model": model,
            "positive": positive,
            "negative": negative,
            "vae": pipe['vae'],
            "clip": pipe['clip'],

            "samples": pipe["samples"],
            "images": pipe["images"],
            "seed": 0,

            "loader_settings": pipe["loader_settings"]
        }

        del pipe

        return (new_pipe, model, positive, negative)

class instantIDApply(instantID):

    def __init__(self):
        super().__init__()
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
                "required":{
                     "pipe": ("PIPE_LINE",),
                     "image": ("IMAGE",),
                     "instantid_file": (folder_paths.get_filename_list("instantid"),),
                     "insightface": (["CPU", "CUDA", "ROCM"],),
                     "control_net_name": (folder_paths.get_filename_list("controlnet"),),
                     "cn_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                     "cn_soft_weights": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},),
                     "weight": ("FLOAT", {"default": .8, "min": 0.0, "max": 5.0, "step": 0.01, }),
                     "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
                     "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
                     "noise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05, }),
                },
                "optional": {
                    "image_kps": ("IMAGE",),
                    "mask": ("MASK",),
                    "control_net": ("CONTROL_NET",),
                },
                "hidden": {
                    "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"
                },
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("pipe", "model", "positive", "negative")
    OUTPUT_NODE = True

    FUNCTION = "apply"
    CATEGORY = "EasyUse/Adapter"


    def apply(self, pipe, image, instantid_file, insightface, control_net_name, cn_strength, cn_soft_weights, weight, start_at, end_at, noise, image_kps=None, mask=None, control_net=None, prompt=None, extra_pnginfo=None, my_unique_id=None):
        positive = pipe['positive']
        negative = pipe['negative']
        return self.run(pipe, image, instantid_file, insightface, control_net_name, cn_strength, cn_soft_weights, weight, start_at, end_at, noise, image_kps, mask, control_net, positive, negative, prompt, extra_pnginfo, my_unique_id)

#Apply InstantID Advanced
class instantIDApplyAdvanced(instantID):

    def __init__(self):
        super().__init__()
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
                "required":{
                     "pipe": ("PIPE_LINE",),
                     "image": ("IMAGE",),
                     "instantid_file": (folder_paths.get_filename_list("instantid"),),
                     "insightface": (["CPU", "CUDA", "ROCM"],),
                     "control_net_name": (folder_paths.get_filename_list("controlnet"),),
                     "cn_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                     "cn_soft_weights": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},),
                     "weight": ("FLOAT", {"default": .8, "min": 0.0, "max": 5.0, "step": 0.01, }),
                     "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
                     "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
                     "noise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05, }),
                },
                "optional": {
                    "image_kps": ("IMAGE",),
                    "mask": ("MASK",),
                    "control_net": ("CONTROL_NET",),
                    "positive": ("CONDITIONING",),
                    "negative": ("CONDITIONING",),
                },
                "hidden": {
                    "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"
                },
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("pipe", "model", "positive", "negative")
    OUTPUT_NODE = True

    FUNCTION = "apply_advanced"
    CATEGORY = "EasyUse/Adapter"

    def apply_advanced(self, pipe, image, instantid_file, insightface, control_net_name, cn_strength, cn_soft_weights, weight, start_at, end_at, noise, image_kps=None, mask=None, control_net=None, positive=None, negative=None, prompt=None, extra_pnginfo=None, my_unique_id=None):

        positive = positive if positive is not None else pipe['positive']
        negative = negative if negative is not None else pipe['negative']

        return self.run(pipe, image, instantid_file, insightface, control_net_name, cn_strength, cn_soft_weights, weight, start_at, end_at, noise, image_kps, mask, control_net, positive, negative, prompt, extra_pnginfo, my_unique_id)

# ---------------------------------------------------------------适配器 结束----------------------------------------------------------------------#

#---------------------------------------------------------------预采样 开始----------------------------------------------------------------------#

# 预采样设置（基础）
class samplerSettings:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"pipe": ("PIPE_LINE",),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS + ['align_your_steps'],),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED_NUM}),
                     },
                "optional": {
                    "image_to_latent": ("IMAGE",),
                    "latent": ("LATENT",),
                },
                "hidden":
                    {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("PIPE_LINE", )
    RETURN_NAMES = ("pipe",)
    OUTPUT_NODE = True

    FUNCTION = "settings"
    CATEGORY = "EasyUse/PreSampling"

    def settings(self, pipe, steps, cfg, sampler_name, scheduler, denoise, seed, image_to_latent=None, latent=None, prompt=None, extra_pnginfo=None, my_unique_id=None):
        # 图生图转换
        vae = pipe["vae"]
        batch_size = pipe["loader_settings"]["batch_size"] if "batch_size" in pipe["loader_settings"] else 1
        if image_to_latent is not None:
            _, height, width, _ = image_to_latent.shape
            if height == 1 and width == 1:
                samples = pipe["samples"]
                images = pipe["images"]
            else:
                samples = {"samples": vae.encode(image_to_latent[:, :, :, :3])}
                samples = RepeatLatentBatch().repeat(samples, batch_size)[0]
                images = image_to_latent
        elif latent is not None:
            samples = latent
            images = pipe["images"]
        else:
            samples = pipe["samples"]
            images = pipe["images"]

        new_pipe = {
            "model": pipe['model'],
            "positive": pipe['positive'],
            "negative": pipe['negative'],
            "vae": pipe['vae'],
            "clip": pipe['clip'],

            "samples": samples,
            "images": images,
            "seed": seed,

            "loader_settings": {
                **pipe["loader_settings"],
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "add_noise": "enabled"
            }
        }

        del pipe

        return {"ui": {"value": [seed]}, "result": (new_pipe,)}

# 预采样设置（高级）
class samplerSettingsAdvanced:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"pipe": ("PIPE_LINE",),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS + ['align_your_steps'],),
                     "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                     "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                     "add_noise": (["enable", "disable"],),
                     "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED_NUM}),
                     "return_with_leftover_noise": (["disable", "enable"], ),
                     },
                "optional": {
                    "image_to_latent": ("IMAGE",),
                    "latent": ("LATENT",)
                },
                "hidden":
                    {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("PIPE_LINE", )
    RETURN_NAMES = ("pipe",)
    OUTPUT_NODE = True

    FUNCTION = "settings"
    CATEGORY = "EasyUse/PreSampling"

    def settings(self, pipe, steps, cfg, sampler_name, scheduler, start_at_step, end_at_step, add_noise, seed, return_with_leftover_noise, image_to_latent=None, latent=None, prompt=None, extra_pnginfo=None, my_unique_id=None):
        # 图生图转换
        vae = pipe["vae"]
        batch_size = pipe["loader_settings"]["batch_size"] if "batch_size" in pipe["loader_settings"] else 1
        if image_to_latent is not None:
            _, height, width, _ = image_to_latent.shape
            if height == 1 and width == 1:
                samples = pipe["samples"]
                images = pipe["images"]
            else:
                samples = {"samples": vae.encode(image_to_latent[:, :, :, :3])}
                samples = RepeatLatentBatch().repeat(samples, batch_size)[0]
                images = image_to_latent
        elif latent is not None:
            samples = latent
            images = pipe["images"]
        else:
            samples = pipe["samples"]
            images = pipe["images"]

        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False

        new_pipe = {
            "model": pipe['model'],
            "positive": pipe['positive'],
            "negative": pipe['negative'],
            "vae": pipe['vae'],
            "clip": pipe['clip'],

            "samples": samples,
            "images": images,
            "seed": seed,

            "loader_settings": {
                **pipe["loader_settings"],
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "start_step": start_at_step,
                "last_step": end_at_step,
                "denoise": 1.0,
                "add_noise": add_noise,
                "force_full_denoise": force_full_denoise
            }
        }

        del pipe

        return {"ui": {"value": [seed]}, "result": (new_pipe,)}

# 预采样设置（噪声注入）
class samplerSettingsNoiseIn:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"pipe": ("PIPE_LINE",),
                     "factor": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step":0.01, "round": 0.01}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS+['align_your_steps'],),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED_NUM}),
                     },
                "optional": {
                    "optional_noise_seed": ("INT",{"forceInput": True}),
                    "optional_latent": ("LATENT",),
                },
                "hidden":
                    {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("PIPE_LINE", )
    RETURN_NAMES = ("pipe",)
    OUTPUT_NODE = True

    FUNCTION = "settings"
    CATEGORY = "EasyUse/PreSampling"

    def slerp(self, val, low, high):
        dims = low.shape

        low = low.reshape(dims[0], -1)
        high = high.reshape(dims[0], -1)

        low_norm = low / torch.norm(low, dim=1, keepdim=True)
        high_norm = high / torch.norm(high, dim=1, keepdim=True)

        low_norm[low_norm != low_norm] = 0.0
        high_norm[high_norm != high_norm] = 0.0

        omega = torch.acos((low_norm * high_norm).sum(1))
        so = torch.sin(omega)
        res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(
            1) * high

        return res.reshape(dims)

    def prepare_mask(self, mask, shape):
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
                                               size=(shape[2], shape[3]), mode="bilinear")
        mask = mask.expand((-1, shape[1], -1, -1))
        if mask.shape[0] < shape[0]:
            mask = mask.repeat((shape[0] - 1) // mask.shape[0] + 1, 1, 1, 1)[:shape[0]]
        return mask

    def expand_mask(self, mask, expand, tapered_corners):
        try:
            import scipy

            c = 0 if tapered_corners else 1
            kernel = np.array([[c, 1, c],
                               [1, 1, 1],
                               [c, 1, c]])
            mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
            out = []
            for m in mask:
                output = m.numpy()
                for _ in range(abs(expand)):
                    if expand < 0:
                        output = scipy.ndimage.grey_erosion(output, footprint=kernel)
                    else:
                        output = scipy.ndimage.grey_dilation(output, footprint=kernel)
                output = torch.from_numpy(output)
                out.append(output)

            return torch.stack(out, dim=0)
        except:
            return None

    def settings(self, pipe, factor, steps, cfg, sampler_name, scheduler, denoise, seed, optional_noise_seed=None, optional_latent=None, prompt=None, extra_pnginfo=None, my_unique_id=None):
        latent = optional_latent if optional_latent is not None else pipe["samples"]
        model = pipe["model"]

        # generate base noise
        batch_size, _, height, width = latent["samples"].shape
        generator = torch.manual_seed(seed)
        base_noise = torch.randn((1, 4, height, width), dtype=torch.float32, device="cpu", generator=generator).repeat(batch_size, 1, 1, 1).cpu()

        # generate variation noise
        if optional_noise_seed is None or optional_noise_seed == seed:
            optional_noise_seed = seed+1
        generator = torch.manual_seed(optional_noise_seed)
        variation_noise = torch.randn((batch_size, 4, height, width), dtype=torch.float32, device="cpu",
                                      generator=generator).cpu()

        slerp_noise = self.slerp(factor, base_noise, variation_noise)

        end_at_step = steps  # min(steps, end_at_step)
        start_at_step = round(end_at_step - end_at_step * denoise)

        device = comfy.model_management.get_torch_device()
        comfy.model_management.load_model_gpu(model)
        model_patcher = comfy.model_patcher.ModelPatcher(model.model, load_device=device, offload_device=comfy.model_management.unet_offload_device())
        sampler = comfy.samplers.KSampler(model_patcher, steps=steps, device=device, sampler=sampler_name,
                                          scheduler=scheduler, denoise=1.0, model_options=model.model_options)
        sigmas = sampler.sigmas
        sigma = sigmas[start_at_step] - sigmas[end_at_step]
        sigma /= model.model.latent_format.scale_factor
        sigma = sigma.cpu().numpy()

        work_latent = latent.copy()
        work_latent["samples"] = latent["samples"].clone() + slerp_noise * sigma

        if "noise_mask" in latent:
            noise_mask = self.prepare_mask(latent["noise_mask"], latent['samples'].shape)
            work_latent["samples"] = noise_mask * work_latent["samples"] + (1-noise_mask) * latent["samples"]
            work_latent['noise_mask'] = self.expand_mask(latent["noise_mask"].clone(), 5, True)

        if pipe is None:
            pipe = {}

        new_pipe = {
            "model": pipe['model'],
            "positive": pipe['positive'],
            "negative": pipe['negative'],
            "vae": pipe['vae'],
            "clip": pipe['clip'],

            "samples": work_latent,
            "images": pipe['images'],
            "seed": seed,

            "loader_settings": {
                **pipe["loader_settings"],
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "add_noise": "disable"
            }
        }

        return (new_pipe,)

# 预采样设置（自定义）
import comfy_extras.nodes_custom_sampler as custom_samplers
from tqdm import trange
class samplerCustomSettings:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"pipe": ("PIPE_LINE",),
                     "guider": (['CFG','DualCFG','IP2P+DualCFG','Basic'],{"default":"Basic"}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "cfg_negative": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS + ['inversed_euler'],),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS + ['karrasADV','exponentialADV','polyExponential', 'sdturbo', 'vp', 'alignYourSteps'],),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "sigma_max": ("FLOAT", {"default": 14.614642, "min": 0.0, "max": 1000.0, "step": 0.01, "round": False}),
                     "sigma_min": ("FLOAT", {"default": 0.0291675, "min": 0.0, "max": 1000.0, "step": 0.01, "round": False}),
                     "rho": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": False}),
                     "beta_d": ("FLOAT", {"default": 19.9, "min": 0.0, "max": 1000.0, "step": 0.01, "round": False}),
                     "beta_min": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1000.0, "step": 0.01, "round": False}),
                     "eps_s": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.0001, "round": False}),
                     "flip_sigmas": ("BOOLEAN", {"default": False}),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "add_noise": (["enable", "disable"], {"default": "enable"}),
                     "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED_NUM}),
                     },
                "optional": {
                    "image_to_latent": ("IMAGE",),
                    "latent": ("LATENT",),
                    "optional_sampler":("SAMPLER",),
                    "optional_sigmas":("SIGMAS",),
                },
                "hidden":
                    {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("PIPE_LINE", )
    RETURN_NAMES = ("pipe",)
    OUTPUT_NODE = True

    FUNCTION = "settings"
    CATEGORY = "EasyUse/PreSampling"

    def ip2p(self, positive, negative, vae=None, pixels=None, latent=None):
        if latent is not None:
            concat_latent = latent
        else:
            x = (pixels.shape[1] // 8) * 8
            y = (pixels.shape[2] // 8) * 8

            if pixels.shape[1] != x or pixels.shape[2] != y:
                x_offset = (pixels.shape[1] % 8) // 2
                y_offset = (pixels.shape[2] % 8) // 2
                pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]

            concat_latent = vae.encode(pixels)

        out_latent = {}
        out_latent["samples"] = torch.zeros_like(concat_latent)

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()
                d["concat_latent_image"] = concat_latent
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1], out_latent)

    def get_inversed_euler_sampler(self):
        @torch.no_grad()
        def sample_inversed_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0.,
                                  s_tmax=float('inf'), s_noise=1.):
            """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
            extra_args = {} if extra_args is None else extra_args
            s_in = x.new_ones([x.shape[0]])
            for i in trange(1, len(sigmas), disable=disable):
                sigma_in = sigmas[i - 1]

                if i == 1:
                    sigma_t = sigmas[i]
                else:
                    sigma_t = sigma_in

                denoised = model(x, sigma_t * s_in, **extra_args)

                if i == 1:
                    d = (x - denoised) / (2 * sigmas[i])
                else:
                    d = (x - denoised) / sigmas[i - 1]

                dt = sigmas[i] - sigmas[i - 1]
                x = x + d * dt
                if callback is not None:
                    callback(
                        {'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
            return x / sigmas[-1]

        ksampler = comfy.samplers.KSAMPLER(sample_inversed_euler)
        return (ksampler,)

    def get_custom_cls(self, sampler_name):
        try:
            cls = custom_samplers.__dict__[sampler_name]
            return cls()
        except:
            raise Exception(f"Custom sampler {sampler_name} not found, Please updated your ComfyUI")

    def add_model_patch_option(self, model):
        if 'transformer_options' not in model.model_options:
            model.model_options['transformer_options'] = {}
        to = model.model_options['transformer_options']
        if "model_patch" not in to:
            to["model_patch"] = {}
        return to

    def settings(self, pipe, guider, cfg, cfg_negative, sampler_name, scheduler, steps, sigma_max, sigma_min, rho, beta_d, beta_min, eps_s, flip_sigmas, denoise, add_noise, seed, image_to_latent=None, latent=None, optional_sampler=None, optional_sigmas=None, prompt=None, extra_pnginfo=None, my_unique_id=None):

        # 图生图转换
        vae = pipe["vae"]
        model = pipe["model"]
        positive = pipe['positive']
        negative = pipe['negative']
        batch_size = pipe["loader_settings"]["batch_size"] if "batch_size" in pipe["loader_settings"] else 1
        _guider, sigmas = None, None

        # sigmas
        if optional_sigmas is not None:
            sigmas = optional_sigmas
        else:
            match scheduler:
                case 'vp':
                    sigmas, = self.get_custom_cls('VPScheduler').get_sigmas(steps, beta_d, beta_min, eps_s)
                case 'karrasADV':
                    sigmas, = self.get_custom_cls('KarrasScheduler').get_sigmas(steps, sigma_max, sigma_min, rho)
                case 'exponentialADV':
                    sigmas, = self.get_custom_cls('ExponentialScheduler').get_sigmas(steps, sigma_max, sigma_min)
                case 'polyExponential':
                    sigmas, = self.get_custom_cls('PolyexponentialScheduler').get_sigmas(steps, sigma_max, sigma_min,
                                                                                         rho)
                case 'sdturbo':
                    sigmas, = self.get_custom_cls('SDTurboScheduler').get_sigmas(model, steps, denoise)
                case 'alignYourSteps':
                    try:
                        model_type = get_sd_version(model)
                        if model_type == 'unknown':
                            raise Exception("This Model not supported")
                        sigmas, = alignYourStepsScheduler().get_sigmas(model_type.upper(), steps, denoise)
                    except:
                        raise Exception("Please update your ComfyUI")
                case _:
                    sigmas, = self.get_custom_cls('BasicScheduler').get_sigmas(model, scheduler, steps, denoise)

            # filp_sigmas
            if flip_sigmas:
                sigmas, = self.get_custom_cls('FlipSigmas').get_sigmas(sigmas)

        #######################################################################################
        # brushnet
        to = None
        transformer_options = model.model_options['transformer_options'] if "transformer_options" in model.model_options else {}
        if 'model_patch' in transformer_options and 'brushnet' in transformer_options['model_patch']:
            to = self.add_model_patch_option(model)
            mp = to['model_patch']
            if isinstance(model.model.model_config, comfy.supported_models.SD15):
                mp['SDXL'] = False
            elif isinstance(model.model.model_config, comfy.supported_models.SDXL):
                mp['SDXL'] = True
            else:
                print('Base model type: ', type(model.model.model_config))
                raise Exception("Unsupported model type: ", type(model.model.model_config))

            mp['all_sigmas'] = sigmas
            mp['unet'] = model.model.diffusion_model
            mp['step'] = 0
            mp['total_steps'] = 1
        #
        #######################################################################################

        if image_to_latent is not None:
            _, height, width, _ = image_to_latent.shape
            if height == 1 and width == 1:
                samples = pipe["samples"]
                images = pipe["images"]
            else:
                if guider == "IP2P+DualCFG":
                    positive, negative, latent = self.ip2p(pipe['positive'], pipe['negative'], vae, image_to_latent)
                    samples = latent
                else:
                    samples = {"samples": vae.encode(image_to_latent[:, :, :, :3])}
                    samples = RepeatLatentBatch().repeat(samples, batch_size)[0]
                images = image_to_latent
        elif latent is not None:
            if guider == "IP2P+DualCFG":
                positive, negative, latent = self.ip2p(pipe['positive'], pipe['negative'], latent=latent)
                samples = latent
            else:
                samples = latent
            images = pipe["images"]
        else:
            samples = pipe["samples"]
            images = pipe["images"]

        # guider
        if guider == 'CFG':
            _guider, = self.get_custom_cls('CFGGuider').get_guider(model, positive, negative, cfg)
        elif guider in ['DualCFG', 'IP2P+DualCFG']:
            _guider, =  self.get_custom_cls('DualCFGGuider').get_guider(model, positive, negative, pipe['negative'], cfg, cfg_negative)
        else:
            _guider, = self.get_custom_cls('BasicGuider').get_guider(model, positive)

        # sampler
        if optional_sampler:
            sampler = optional_sampler
        else:
            if sampler_name == 'inversed_euler':
                sampler, = self.get_inversed_euler_sampler()
            else:
                sampler, = self.get_custom_cls('KSamplerSelect').get_sampler(sampler_name)

        # noise
        if add_noise == 'disable':
            noise, = self.get_custom_cls('DisableNoise').get_noise()
        else:
            noise, = self.get_custom_cls('RandomNoise').get_noise(seed)

        new_pipe = {
            "model": pipe['model'],
            "positive": pipe['positive'],
            "negative": pipe['negative'],
            "vae": pipe['vae'],
            "clip": pipe['clip'],

            "samples": samples,
            "images": images,
            "seed": seed,

            "loader_settings": {
                **pipe["loader_settings"],
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "custom": {
                    "noise": noise,
                    "guider": _guider,
                    "sampler": sampler,
                    "sigmas": sigmas,
                }
            }
        }

        del pipe

        return {"ui": {"value": [seed]}, "result": (new_pipe,)}

# 预采样设置（SDTurbo）
from .libs.gradual_latent_hires_fix import sample_dpmpp_2s_ancestral, sample_dpmpp_2m_sde, sample_lcm, sample_euler_ancestral
class sdTurboSettings:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "pipe": ("PIPE_LINE",),
                    "steps": ("INT", {"default": 1, "min": 1, "max": 10}),
                    "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.SAMPLER_NAMES,),
                    "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "round": False}),
                    "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "round": False}),
                    "upscale_ratio": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 16.0, "step": 0.01, "round": False}),
                    "start_step": ("INT", {"default": 5, "min": 0, "max": 1000, "step": 1}),
                    "end_step": ("INT", {"default": 15, "min": 0, "max": 1000, "step": 1}),
                    "upscale_n_step": ("INT", {"default": 3, "min": 0, "max": 1000, "step": 1}),
                    "unsharp_kernel_size": ("INT", {"default": 3, "min": 1, "max": 21, "step": 1}),
                    "unsharp_sigma": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01, "round": False}),
                    "unsharp_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01, "round": False}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED_NUM}),
               },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    OUTPUT_NODE = True

    FUNCTION = "settings"
    CATEGORY = "EasyUse/PreSampling"

    def settings(self, pipe, steps, cfg, sampler_name, eta, s_noise, upscale_ratio, start_step, end_step, upscale_n_step, unsharp_kernel_size, unsharp_sigma, unsharp_strength, seed, prompt=None, extra_pnginfo=None, my_unique_id=None):
        model = pipe['model']
        # sigma
        timesteps = torch.flip(torch.arange(1, 11) * 100 - 1, (0,))[:steps]
        sigmas = model.model.model_sampling.sigma(timesteps)
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])

        #sampler
        sample_function = None
        extra_options = {
                "eta": eta,
                "s_noise": s_noise,
                "upscale_ratio": upscale_ratio,
                "start_step": start_step,
                "end_step": end_step,
                "upscale_n_step": upscale_n_step,
                "unsharp_kernel_size": unsharp_kernel_size,
                "unsharp_sigma": unsharp_sigma,
                "unsharp_strength": unsharp_strength,
            }
        match sampler_name:
            case "euler_ancestral":
                sample_function = sample_euler_ancestral
            case "dpmpp_2s_ancestral":
                sample_function = sample_dpmpp_2s_ancestral
            case "dpmpp_2m_sde":
                sample_function = sample_dpmpp_2m_sde
            case "lcm":
                sample_function = sample_lcm

        if sample_function is not None:
            unsharp_kernel_size = unsharp_kernel_size if unsharp_kernel_size % 2 == 1 else unsharp_kernel_size + 1
            extra_options["unsharp_kernel_size"] = unsharp_kernel_size
            _sampler = comfy.samplers.KSAMPLER(sample_function, extra_options)
        else:
            _sampler = comfy.samplers.sampler_object(sampler_name)
            extra_options = None

        new_pipe = {
            "model": pipe['model'],
            "positive": pipe['positive'],
            "negative": pipe['negative'],
            "vae": pipe['vae'],
            "clip": pipe['clip'],

            "samples": pipe["samples"],
            "images": pipe["images"],
            "seed": seed,

            "loader_settings": {
                **pipe["loader_settings"],
                "extra_options": extra_options,
                "sampler": _sampler,
                "sigmas": sigmas,
                "steps": steps,
                "cfg": cfg,
                "add_noise": "enabled"
            }
        }

        del pipe

        return {"ui": {"value": [seed]}, "result": (new_pipe,)}


# cascade预采样参数
class cascadeSettings:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {"pipe": ("PIPE_LINE",),
             "encode_vae_name": (["None"] + folder_paths.get_filename_list("vae"),),
             "decode_vae_name": (["None"] + folder_paths.get_filename_list("vae"),),
             "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
             "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0}),
             "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default":"euler_ancestral"}),
             "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default":"simple"}),
             "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
             "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED_NUM}),
             },
            "optional": {
                "image_to_latent_c": ("IMAGE",),
                "latent_c": ("LATENT",),
            },
            "hidden":{"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    OUTPUT_NODE = True

    FUNCTION = "settings"
    CATEGORY = "EasyUse/PreSampling"

    def settings(self, pipe, encode_vae_name, decode_vae_name, steps, cfg, sampler_name, scheduler, denoise, seed, model=None, image_to_latent_c=None, latent_c=None, prompt=None, extra_pnginfo=None, my_unique_id=None):
        images, samples_c = None, None
        samples = pipe['samples']
        batch_size = pipe["loader_settings"]["batch_size"] if "batch_size" in pipe["loader_settings"] else 1

        encode_vae_name = encode_vae_name if encode_vae_name is not None else pipe['loader_settings']['encode_vae_name']
        decode_vae_name = decode_vae_name if decode_vae_name is not None else pipe['loader_settings']['decode_vae_name']

        if image_to_latent_c is not None:
            if encode_vae_name != 'None':
                encode_vae = easyCache.load_vae(encode_vae_name)
            else:
                encode_vae = pipe['vae'][0]
            if "compression" not in pipe["loader_settings"]:
                raise Exception("compression is not found")
            compression = pipe["loader_settings"]['compression']
            width = image_to_latent_c.shape[-2]
            height = image_to_latent_c.shape[-3]
            out_width = (width // compression) * encode_vae.downscale_ratio
            out_height = (height // compression) * encode_vae.downscale_ratio

            s = comfy.utils.common_upscale(image_to_latent_c.movedim(-1, 1), out_width, out_height, "bicubic",
                                           "center").movedim(1,
                                                             -1)
            c_latent = encode_vae.encode(s[:, :, :, :3])
            b_latent = torch.zeros([c_latent.shape[0], 4, height // 4, width // 4])

            samples_c = {"samples": c_latent}
            samples_c = RepeatLatentBatch().repeat(samples_c, batch_size)[0]

            samples_b = {"samples": b_latent}
            samples_b = RepeatLatentBatch().repeat(samples_b, batch_size)[0]
            samples = (samples_c, samples_b)
            images = image_to_latent_c
        elif latent_c is not None:
            samples_c = latent_c
            samples = (samples_c, samples[1])
            images = pipe["images"]
        if samples_c is not None:
            samples = (samples_c, samples[1])

        new_pipe = {
            "model": pipe['model'],
            "positive": pipe['positive'],
            "negative": pipe['negative'],
            "vae": pipe['vae'],
            "clip": pipe['clip'],

            "samples": samples,
            "images": images,
            "seed": seed,

            "loader_settings": {
                **pipe["loader_settings"],
                "encode_vae_name": encode_vae_name,
                "decode_vae_name": decode_vae_name,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "add_noise": "enabled"
            }
        }

        sampler.update_value_by_id("pipe_line", my_unique_id, new_pipe)

        del pipe

        return {"ui": {"value": [seed]}, "result": (new_pipe,)}

# layerDiffusion预采样参数
class layerDiffusionSettings:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
             "pipe": ("PIPE_LINE",),
             "method": ([LayerMethod.FG_ONLY_ATTN.value, LayerMethod.FG_ONLY_CONV.value, LayerMethod.EVERYTHING.value, LayerMethod.FG_TO_BLEND.value, LayerMethod.BG_TO_BLEND.value],),
             "weight": ("FLOAT",{"default": 1.0, "min": -1, "max": 3, "step": 0.05},),
             "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
             "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
             "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
             "scheduler": (comfy.samplers.KSampler.SCHEDULERS+ ['align_your_steps'], {"default": "normal"}),
             "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
             "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED_NUM}),
             },
            "optional": {
                "image": ("IMAGE",),
                "blended_image": ("IMAGE",),
                "mask": ("MASK",),
                # "latent": ("LATENT",),
                # "blended_latent": ("LATENT",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    OUTPUT_NODE = True

    FUNCTION = "settings"
    CATEGORY = "EasyUse/PreSampling"

    def get_layer_diffusion_method(self, method, has_blend_latent):
        method = LayerMethod(method)
        if has_blend_latent:
            if method == LayerMethod.BG_TO_BLEND:
                method = LayerMethod.BG_BLEND_TO_FG
            elif method == LayerMethod.FG_TO_BLEND:
                method = LayerMethod.FG_BLEND_TO_BG
        return method

    def settings(self, pipe, method, weight, steps, cfg, sampler_name, scheduler, denoise, seed, image=None, blended_image=None, mask=None, prompt=None, extra_pnginfo=None, my_unique_id=None):
        blend_samples = pipe['blend_samples'] if "blend_samples" in pipe else None
        vae = pipe["vae"]
        batch_size = pipe["loader_settings"]["batch_size"] if "batch_size" in pipe["loader_settings"] else 1

        method = self.get_layer_diffusion_method(method, blend_samples is not None or blended_image is not None)

        if image is not None or "image" in pipe:
            image = image if image is not None else pipe['image']
            if mask is not None:
                print('inpaint')
                samples, = VAEEncodeForInpaint().encode(vae, image, mask)
            else:
                samples = {"samples": vae.encode(image[:,:,:,:3])}
            samples = RepeatLatentBatch().repeat(samples, batch_size)[0]
            images = image
        elif "samp_images" in pipe:
            samples = {"samples": vae.encode(pipe["samp_images"][:,:,:,:3])}
            samples = RepeatLatentBatch().repeat(samples, batch_size)[0]
            images = pipe["samp_images"]
        else:
            if method not in [LayerMethod.FG_ONLY_ATTN, LayerMethod.FG_ONLY_CONV, LayerMethod.EVERYTHING]:
                raise Exception("image is missing")

            samples = pipe["samples"]
            images = pipe["images"]

        if method in [LayerMethod.BG_BLEND_TO_FG, LayerMethod.FG_BLEND_TO_BG]:
            if blended_image is None and blend_samples is None:
                raise Exception("blended_image is missing")
            elif blended_image is not None:
                blend_samples = {"samples": vae.encode(blended_image[:,:,:,:3])}
                blend_samples = RepeatLatentBatch().repeat(blend_samples, batch_size)[0]

        new_pipe = {
            "model": pipe['model'],
            "positive": pipe['positive'],
            "negative": pipe['negative'],
            "vae": pipe['vae'],
            "clip": pipe['clip'],

            "samples": samples,
            "blend_samples": blend_samples,
            "images": images,
            "seed": seed,

            "loader_settings": {
                **pipe["loader_settings"],
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "add_noise": "enabled",
                "layer_diffusion_method": method,
                "layer_diffusion_weight": weight,
            }
        }

        del pipe

        return {"ui": {"value": [seed]}, "result": (new_pipe,)}

# 预采样设置（layerDiffuse附加）
class layerDiffusionSettingsADDTL:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "pipe": ("PIPE_LINE",),
                "foreground_prompt": ("STRING", {"default": "", "placeholder": "Foreground Additional Prompt", "multiline": True}),
                "background_prompt": ("STRING", {"default": "", "placeholder": "Background Additional Prompt", "multiline": True}),
                "blended_prompt": ("STRING", {"default": "", "placeholder": "Blended Additional Prompt", "multiline": True}),
            },
            "optional": {
                "optional_fg_cond": ("CONDITIONING",),
                "optional_bg_cond": ("CONDITIONING",),
                "optional_blended_cond": ("CONDITIONING",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    OUTPUT_NODE = True

    FUNCTION = "settings"
    CATEGORY = "EasyUse/PreSampling"

    def settings(self, pipe, foreground_prompt, background_prompt, blended_prompt, optional_fg_cond=None, optional_bg_cond=None, optional_blended_cond=None, prompt=None, extra_pnginfo=None, my_unique_id=None):
        fg_cond, bg_cond, blended_cond = None, None, None
        clip = pipe['clip']
        if optional_fg_cond is not None:
            fg_cond = optional_fg_cond
        elif foreground_prompt != "":
            fg_cond, = CLIPTextEncode().encode(clip, foreground_prompt)
        if optional_bg_cond is not None:
            bg_cond = optional_bg_cond
        elif background_prompt != "":
            bg_cond, = CLIPTextEncode().encode(clip, background_prompt)
        if optional_blended_cond is not None:
            blended_cond = optional_blended_cond
        elif blended_prompt != "":
            blended_cond, = CLIPTextEncode().encode(clip, blended_prompt)

        new_pipe = {
            **pipe,
            "loader_settings": {
                **pipe["loader_settings"],
                "layer_diffusion_cond": (fg_cond, bg_cond, blended_cond)
            }
        }

        del pipe

        return (new_pipe,)

# 预采样设置（动态CFG）
from .libs.dynthres_core import DynThresh
class dynamicCFGSettings:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"pipe": ("PIPE_LINE",),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "cfg_mode": (DynThresh.Modes,),
                     "cfg_scale_min": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.5}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS+['align_your_steps'],),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED_NUM}),
                     },
                "optional":{
                    "image_to_latent": ("IMAGE",),
                    "latent": ("LATENT",)
                },
                "hidden":
                    {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    OUTPUT_NODE = True

    FUNCTION = "settings"
    CATEGORY = "EasyUse/PreSampling"

    def settings(self, pipe, steps, cfg, cfg_mode, cfg_scale_min,sampler_name, scheduler, denoise, seed, image_to_latent=None, latent=None, prompt=None, extra_pnginfo=None, my_unique_id=None):


        dynamic_thresh = DynThresh(7.0, 1.0,"CONSTANT", 0, cfg_mode, cfg_scale_min, 0, 0, 999, False,
                                   "MEAN", "AD", 1)

        def sampler_dyn_thresh(args):
            input = args["input"]
            cond = input - args["cond"]
            uncond = input - args["uncond"]
            cond_scale = args["cond_scale"]
            time_step = args["timestep"]
            dynamic_thresh.step = 999 - time_step[0]

            return input - dynamic_thresh.dynthresh(cond, uncond, cond_scale, None)

        model = pipe['model']

        m = model.clone()
        m.set_model_sampler_cfg_function(sampler_dyn_thresh)

        # 图生图转换
        vae = pipe["vae"]
        batch_size = pipe["loader_settings"]["batch_size"] if "batch_size" in pipe["loader_settings"] else 1
        if image_to_latent is not None:
            samples = {"samples": vae.encode(image_to_latent[:, :, :, :3])}
            samples = RepeatLatentBatch().repeat(samples, batch_size)[0]
            images = image_to_latent
        elif latent is not None:
            samples = RepeatLatentBatch().repeat(latent, batch_size)[0]
            images = pipe["images"]
        else:
            samples = pipe["samples"]
            images = pipe["images"]

        new_pipe = {
            "model": m,
            "positive": pipe['positive'],
            "negative": pipe['negative'],
            "vae": pipe['vae'],
            "clip": pipe['clip'],

            "samples": samples,
            "images": images,
            "seed": seed,

            "loader_settings": {
                **pipe["loader_settings"],
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise
            },
        }

        del pipe

        return {"ui": {"value": [seed]}, "result": (new_pipe,)}

# 动态CFG
class dynamicThresholdingFull:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "mimic_scale": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "threshold_percentile": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mimic_mode": (DynThresh.Modes,),
                "mimic_scale_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "cfg_mode": (DynThresh.Modes,),
                "cfg_scale_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "sched_val": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "separate_feature_channels": (["enable", "disable"],),
                "scaling_startpoint": (DynThresh.Startpoints,),
                "variability_measure": (DynThresh.Variabilities,),
                "interpolate_phi": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "EasyUse/PreSampling"

    def patch(self, model, mimic_scale, threshold_percentile, mimic_mode, mimic_scale_min, cfg_mode, cfg_scale_min,
              sched_val, separate_feature_channels, scaling_startpoint, variability_measure, interpolate_phi):
        dynamic_thresh = DynThresh(mimic_scale, threshold_percentile, mimic_mode, mimic_scale_min, cfg_mode,
                                   cfg_scale_min, sched_val, 0, 999, separate_feature_channels == "enable",
                                   scaling_startpoint, variability_measure, interpolate_phi)

        def sampler_dyn_thresh(args):
            input = args["input"]
            cond = input - args["cond"]
            uncond = input - args["uncond"]
            cond_scale = args["cond_scale"]
            time_step = args["timestep"]
            dynamic_thresh.step = 999 - time_step[0]

            return input - dynamic_thresh.dynthresh(cond, uncond, cond_scale, None)

        m = model.clone()
        m.set_model_sampler_cfg_function(sampler_dyn_thresh)
        return (m,)

#---------------------------------------------------------------预采样参数 结束----------------------------------------------------------------------

#---------------------------------------------------------------采样器 开始----------------------------------------------------------------------

# 完整采样器
from .libs.chooser import ChooserMessage, ChooserCancelled
class samplerFull:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"pipe": ("PIPE_LINE",),
                 "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                 "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                 "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                 "scheduler": (comfy.samplers.KSampler.SCHEDULERS+['align_your_steps'],),
                 "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                 "image_output": (["Hide", "Preview", "Preview&Choose", "Save", "Hide&Save", "Sender", "Sender&Save"],),
                 "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"}),
                 },
                "optional": {
                    "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED_NUM}),
                    "model": ("MODEL",),
                    "positive": ("CONDITIONING",),
                    "negative": ("CONDITIONING",),
                    "latent": ("LATENT",),
                    "vae": ("VAE",),
                    "clip": ("CLIP",),
                    "xyPlot": ("XYPLOT",),
                    "image": ("IMAGE",),
                },
                "hidden":
                  {"tile_size": "INT", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                    "embeddingsList": (folder_paths.get_filename_list("embeddings"),)
                  }
                }

    RETURN_TYPES = ("PIPE_LINE", "IMAGE", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "INT",)
    RETURN_NAMES = ("pipe",  "image", "model", "positive", "negative", "latent", "vae", "clip", "seed",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "EasyUse/Sampler"

    def run(self, pipe, steps, cfg, sampler_name, scheduler, denoise, image_output, link_id, save_prefix, seed=None, model=None, positive=None, negative=None, latent=None, vae=None, clip=None, xyPlot=None, tile_size=None, prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False, downscale_options=None, image=None):

        samp_model = model if model is not None else pipe["model"]
        samp_positive = positive if positive is not None else pipe["positive"]
        samp_negative = negative if negative is not None else pipe["negative"]
        samp_samples = latent if latent is not None else pipe["samples"]
        samp_vae = vae if vae is not None else pipe["vae"]
        samp_clip = clip if clip is not None else pipe["clip"]

        samp_seed = seed if seed is not None else pipe['seed']

        samp_custom = pipe["loader_settings"]["custom"] if "custom" in pipe["loader_settings"] else None

        steps = steps if steps is not None else pipe['loader_settings']['steps']
        start_step = pipe['loader_settings']['start_step'] if 'start_step' in pipe['loader_settings'] else 0
        last_step = pipe['loader_settings']['last_step'] if 'last_step' in pipe['loader_settings'] else 10000
        cfg = cfg if cfg is not None else pipe['loader_settings']['cfg']
        sampler_name = sampler_name if sampler_name is not None else pipe['loader_settings']['sampler_name']
        scheduler = scheduler if scheduler is not None else pipe['loader_settings']['scheduler']
        denoise = denoise if denoise is not None else pipe['loader_settings']['denoise']
        add_noise = pipe['loader_settings']['add_noise'] if 'add_noise' in pipe['loader_settings'] else 'enabled'
        force_full_denoise = pipe['loader_settings']['force_full_denoise'] if 'force_full_denoise' in pipe['loader_settings'] else True

        if image is not None and latent is None:
            samp_samples = {"samples": samp_vae.encode(image[:, :, :, :3])}

        disable_noise = False
        if add_noise == "disable":
            disable_noise = True

        def downscale_model_unet(samp_model):
            # 获取Unet参数
            if "PatchModelAddDownscale" in ALL_NODE_CLASS_MAPPINGS:
                cls = ALL_NODE_CLASS_MAPPINGS['PatchModelAddDownscale']
                # 自动收缩Unet
                if downscale_options['downscale_factor'] is None:
                    unet_config = samp_model.model.model_config.unet_config
                    if unet_config is not None and "samples" in samp_samples:
                        height = samp_samples['samples'].shape[2] * 8
                        width = samp_samples['samples'].shape[3] * 8
                        context_dim = unet_config.get('context_dim')
                        longer_side = width if width > height else height
                        if context_dim is not None and longer_side > context_dim:
                            width_downscale_factor = float(width / context_dim)
                            height_downscale_factor = float(height / context_dim)
                            if width_downscale_factor > 1.75:
                                log_node_warn("正在收缩模型Unet...")
                                log_node_warn("收缩系数:" + str(width_downscale_factor))
                                (samp_model,) = cls().patch(samp_model, downscale_options['block_number'], width_downscale_factor, 0, 0.35, True, "bicubic",
                                                            "bicubic")
                            elif height_downscale_factor > 1.25:
                                log_node_warn("正在收缩模型Unet...")
                                log_node_warn("收缩系数:" + str(height_downscale_factor))
                                (samp_model,) = cls().patch(samp_model, downscale_options['block_number'], height_downscale_factor, 0, 0.35, True, "bicubic",
                                                            "bicubic")
                else:
                    cls = ALL_NODE_CLASS_MAPPINGS['PatchModelAddDownscale']
                    log_node_warn("正在收缩模型Unet...")
                    log_node_warn("收缩系数:" + str(downscale_options['downscale_factor']))
                    (samp_model,) = cls().patch(samp_model, downscale_options['block_number'], downscale_options['downscale_factor'], downscale_options['start_percent'], downscale_options['end_percent'], downscale_options['downscale_after_skip'], downscale_options['downscale_method'], downscale_options['upscale_method'])
            return samp_model

        def process_sample_state(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive,
                                 samp_negative,
                                 steps, start_step, last_step, cfg, sampler_name, scheduler, denoise,
                                 image_output, link_id, save_prefix, tile_size, prompt, extra_pnginfo, my_unique_id,
                                 preview_latent, force_full_denoise=force_full_denoise, disable_noise=disable_noise, samp_custom=None):

            # LayerDiffusion
            layerDiffuse = None
            samp_blend_samples = None
            layer_diffusion_method = pipe['loader_settings']['layer_diffusion_method'] if 'layer_diffusion_method' in pipe['loader_settings'] else None
            if layer_diffusion_method is not None:
                layerDiffuse = LayerDiffuse()
                samp_blend_samples = pipe["blend_samples"] if "blend_samples" in pipe else None
                additional_cond = pipe["loader_settings"]['layer_diffusion_cond'] if "layer_diffusion_cond" in pipe[
                    'loader_settings'] else (None, None, None)
                method = layerDiffuse.get_layer_diffusion_method(pipe['loader_settings']['layer_diffusion_method'],
                                                         samp_blend_samples is not None)

                images = pipe["images"] if "images" in pipe else None
                weight = pipe['loader_settings']['layer_diffusion_weight'] if 'layer_diffusion_weight' in pipe[
                    'loader_settings'] else 1.0
                samp_model, samp_positive, samp_negative = layerDiffuse.apply_layer_diffusion(samp_model, method, weight,
                                                                                      samp_samples, samp_blend_samples,
                                                                                      samp_positive, samp_negative,
                                                                                      images, additional_cond)
                resolution = pipe['loader_settings']['resolution'] if 'resolution' in pipe['loader_settings'] else "自定义 X 自定义"
                empty_latent_width = pipe['loader_settings']['empty_latent_width'] if 'empty_latent_width' in pipe['loader_settings'] else 512
                empty_latent_height = pipe['loader_settings']['empty_latent_height'] if 'empty_latent_height' in pipe['loader_settings'] else 512
                batch_size = pipe["loader_settings"]["batch_size"] if "batch_size" in pipe["loader_settings"] else 1
                samp_samples = sampler.emptyLatent(resolution, empty_latent_width, empty_latent_height, batch_size)

            # Downscale Model Unet
            if samp_model is not None and downscale_options is not None:
                samp_model = downscale_model_unet(samp_model)
            # 推理初始时间
            start_time = int(time.time() * 1000)
            # 开始推理
            if samp_custom is not None:
                guider = samp_custom['guider'] if 'guider' in samp_custom else None
                _sampler = samp_custom['sampler'] if 'sampler' in samp_custom else None
                sigmas = samp_custom['sigmas'] if 'sigmas' in samp_custom else None
                noise = samp_custom['noise'] if 'noise' in samp_custom else None
                samp_samples, _ = sampler.custom_advanced_ksampler(noise, guider, _sampler, sigmas, samp_samples)
            elif scheduler == 'align_your_steps':
                try:
                    model_type = get_sd_version(samp_model)
                    if model_type == 'unknown':
                        raise Exception("This Model not supported")
                    sigmas, = alignYourStepsScheduler().get_sigmas(model_type.upper(), steps, denoise)
                except:
                    raise Exception("Please update your ComfyUI")
                _sampler = comfy.samplers.sampler_object(sampler_name)
                samp_samples = sampler.custom_ksampler(samp_model, samp_seed, steps, cfg, _sampler, sigmas, samp_positive, samp_negative, samp_samples, disable_noise=disable_noise, preview_latent=preview_latent)
            else:
                samp_samples = sampler.common_ksampler(samp_model, samp_seed, steps, cfg, sampler_name, scheduler, samp_positive, samp_negative, samp_samples, denoise=denoise, preview_latent=preview_latent, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, disable_noise=disable_noise)
            # 推理结束时间
            end_time = int(time.time() * 1000)
            latent = samp_samples["samples"]

            # 解码图片
            if tile_size is not None:
                samp_images = samp_vae.decode_tiled(latent, tile_x=tile_size // 8, tile_y=tile_size // 8, )
            else:
                samp_images = samp_vae.decode(latent).cpu()

            # LayerDiffusion Decode
            if layerDiffuse is not None:
                new_images, samp_images, alpha = layerDiffuse.layer_diffusion_decode(layer_diffusion_method, latent, samp_blend_samples, samp_images, samp_model)
            else:
                new_images = samp_images
                alpha = None

            # 推理总耗时（包含解码）
            end_decode_time = int(time.time() * 1000)
            spent_time = 'Diffusion:' + str((end_time-start_time)/1000)+'″, VAEDecode:' + str((end_decode_time-end_time)/1000)+'″ '

            results = easySave(new_images, save_prefix, image_output, prompt, extra_pnginfo)

            new_pipe = {
                **pipe,
                "positive": samp_positive,
                "negative": samp_negative,
                "vae": samp_vae,
                "clip": samp_clip,

                "samples": samp_samples,
                "blend_samples": samp_blend_samples,
                "images": new_images,
                "samp_images": samp_images,
                "alpha": alpha,
                "seed": samp_seed,

                "loader_settings": {
                    **pipe["loader_settings"],
                    "spent_time": spent_time
                }
            }

            del pipe

            if image_output == 'Preview&Choose':
                if my_unique_id not in ChooserMessage.stash:
                    ChooserMessage.stash[my_unique_id] = {}
                my_stash = ChooserMessage.stash[my_unique_id]

                PromptServer.instance.send_sync("easyuse-image-choose", {"id": my_unique_id, "urls": results})
                # wait for selection
                try:
                    selections = ChooserMessage.waitForMessage(my_unique_id, asList=True)
                    samples = samp_samples['samples']
                    samples = [samples[x] for x in selections if x >= 0] if len(selections) > 1 else [samples[0]]
                    new_images = [new_images[x] for x in selections if x >= 0] if len(selections) > 1 else [new_images[0]]
                    samp_images = [samp_images[x] for x in selections if x >= 0] if len(selections) > 1 else [samp_images[0]]
                    new_images = torch.stack(new_images, dim=0)
                    samp_images = torch.stack(samp_images, dim=0)
                    samples = torch.stack(samples, dim=0)
                    samp_samples = {"samples": samples}
                    new_pipe['samples'] = samp_samples
                    new_pipe['loader_settings']['batch_size'] = len(new_images)
                except ChooserCancelled:
                    raise comfy.model_management.InterruptProcessingException()

                new_pipe['images'] = new_images
                new_pipe['samp_images'] = samp_images

                return {"ui": {"images": results},
                        "result": sampler.get_output(new_pipe,)}

            if image_output in ("Hide", "Hide&Save"):
                return {"ui": {},
                    "result": sampler.get_output(new_pipe,)}

            if image_output in ("Sender", "Sender&Save"):
                PromptServer.instance.send_sync("img-send", {"link_id": link_id, "images": results})

            if hasattr(ModelPatcher, "original_calculate_weight"):
                ModelPatcher.calculate_weight = ModelPatcher.original_calculate_weight

            return {"ui": {"images": results},
                    "result": sampler.get_output(new_pipe,)}

        def process_xyPlot(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative,
                           steps, cfg, sampler_name, scheduler, denoise,
                           image_output, link_id, save_prefix, tile_size, prompt, extra_pnginfo, my_unique_id, preview_latent, xyPlot, force_full_denoise, disable_noise, samp_custom):

            sampleXYplot = easyXYPlot(xyPlot, save_prefix, image_output, prompt, extra_pnginfo, my_unique_id, sampler, easyCache)

            if not sampleXYplot.validate_xy_plot():
                return process_sample_state(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive,
                                            samp_negative, steps, 0, 10000, cfg,
                                            sampler_name, scheduler, denoise, image_output, link_id, save_prefix, tile_size, prompt,
                                            extra_pnginfo, my_unique_id, preview_latent, samp_custom=samp_custom)

            # Downscale Model Unet
            if samp_model is not None and downscale_options is not None:
                samp_model = downscale_model_unet(samp_model)

            blend_samples = pipe['blend_samples'] if "blend_samples" in pipe else None
            layer_diffusion_method = pipe['loader_settings']['layer_diffusion_method'] if 'layer_diffusion_method' in pipe['loader_settings'] else None

            plot_image_vars = {
                "x_node_type": sampleXYplot.x_node_type, "y_node_type": sampleXYplot.y_node_type,
                "lora_name": pipe["loader_settings"]["lora_name"] if "lora_name" in pipe["loader_settings"] else None,
                "lora_model_strength": pipe["loader_settings"]["lora_model_strength"] if "model_strength" in pipe["loader_settings"] else None,
                "lora_clip_strength": pipe["loader_settings"]["lora_clip_strength"] if "clip_strength" in pipe["loader_settings"] else None,
                "lora_stack":  pipe["loader_settings"]["lora_stack"] if "lora_stack" in pipe["loader_settings"] else None,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "seed": samp_seed,
                "images": pipe['images'],

                "model": samp_model, "vae": samp_vae, "clip": samp_clip, "positive_cond": samp_positive,
                "negative_cond": samp_negative,

                "ckpt_name": pipe['loader_settings']['ckpt_name'] if "ckpt_name" in pipe["loader_settings"] else None,
                "vae_name": pipe['loader_settings']['vae_name'] if "vae_name" in pipe["loader_settings"] else None,
                "clip_skip": pipe['loader_settings']['clip_skip'] if "clip_skip" in pipe["loader_settings"] else None,
                "positive": pipe['loader_settings']['positive'] if "positive" in pipe["loader_settings"] else None,
                "positive_token_normalization": pipe['loader_settings']['positive_token_normalization'] if "positive_token_normalization" in pipe["loader_settings"] else None,
                "positive_weight_interpretation": pipe['loader_settings']['positive_weight_interpretation'] if "positive_weight_interpretation" in pipe["loader_settings"] else None,
                "negative": pipe['loader_settings']['negative'] if "negative" in pipe["loader_settings"] else None,
                "negative_token_normalization": pipe['loader_settings']['negative_token_normalization'] if "negative_token_normalization" in pipe["loader_settings"] else None,
                "negative_weight_interpretation": pipe['loader_settings']['negative_weight_interpretation'] if "negative_weight_interpretation" in pipe["loader_settings"] else None,
            }

            if "models" in pipe["loader_settings"]:
                plot_image_vars["models"] = pipe["loader_settings"]["models"]
            if "vae_use" in pipe["loader_settings"]:
                plot_image_vars["vae_use"] = pipe["loader_settings"]["vae_use"]
            if "a1111_prompt_style" in pipe["loader_settings"]:
                plot_image_vars["a1111_prompt_style"] = pipe["loader_settings"]["a1111_prompt_style"]
            if "cnet_stack" in pipe["loader_settings"]:
                plot_image_vars["cnet"] = pipe["loader_settings"]["cnet_stack"]
            if "positive_cond_stack" in pipe["loader_settings"]:
                plot_image_vars["positive_cond_stack"] = pipe["loader_settings"]["positive_cond_stack"]
            if "negative_cond_stack" in pipe["loader_settings"]:
                plot_image_vars["negative_cond_stack"] = pipe["loader_settings"]["negative_cond_stack"]
            if layer_diffusion_method:
                plot_image_vars["layer_diffusion_method"] = layer_diffusion_method
            if "layer_diffusion_weight" in pipe["loader_settings"]:
                plot_image_vars["layer_diffusion_weight"] = pipe['loader_settings']['layer_diffusion_weight']
            if "layer_diffusion_cond" in pipe["loader_settings"]:
                plot_image_vars["layer_diffusion_cond"] = pipe['loader_settings']['layer_diffusion_cond']
            if "empty_samples" in pipe["loader_settings"]:
                plot_image_vars["empty_samples"] = pipe["loader_settings"]['empty_samples']

            latent_image = sampleXYplot.get_latent(pipe["samples"])
            latents_plot = sampleXYplot.get_labels_and_sample(plot_image_vars, latent_image, preview_latent, start_step,
                                                              last_step, force_full_denoise, disable_noise)

            samp_samples = {"samples": latents_plot}

            images, image_list = sampleXYplot.plot_images_and_labels()

            # Generate output_images
            output_images = torch.stack([tensor.squeeze() for tensor in image_list])

            if layer_diffusion_method is not None:
                layerDiffuse = LayerDiffuse()
                new_images, samp_images, alpha = layerDiffuse.layer_diffusion_decode(layer_diffusion_method, latents_plot, blend_samples,
                                                                             output_images, samp_model)
            else:
                new_images = output_images
                samp_images = output_images
                alpha = None

            results = easySave(images, save_prefix, image_output, prompt, extra_pnginfo)

            new_pipe = {
                **pipe,
                "positive": samp_positive,
                "negative": samp_negative,
                "vae": samp_vae,
                "clip": samp_clip,

                "samples": samp_samples,
                "blend_samples": blend_samples,
                "samp_images": samp_images,
                "images": new_images,
                "seed": samp_seed,
                "alpha": alpha,

                "loader_settings": pipe["loader_settings"],
            }

            del pipe

            if hasattr(ModelPatcher, "original_calculate_weight"):
                ModelPatcher.calculate_weight = ModelPatcher.original_calculate_weight

            if image_output in ("Hide", "Hide&Save"):
                return sampler.get_output(new_pipe)

            return {"ui": {"images": results}, "result": (sampler.get_output(new_pipe))}

        preview_latent = True
        if image_output in ("Hide", "Hide&Save"):
            preview_latent = False

        xyplot_id = next((x for x in prompt if "XYPlot" in str(prompt[x]["class_type"])), None)
        if xyplot_id is None:
            xyPlot = None
        else:
            xyPlot = pipe["loader_settings"]["xyplot"] if "xyplot" in pipe["loader_settings"] else xyPlot
        if xyPlot is not None:
            return process_xyPlot(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative, steps, cfg, sampler_name, scheduler, denoise, image_output, link_id, save_prefix, tile_size, prompt, extra_pnginfo, my_unique_id, preview_latent, xyPlot, force_full_denoise, disable_noise, samp_custom)
        else:
            return process_sample_state(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative, steps, start_step, last_step, cfg, sampler_name, scheduler, denoise, image_output, link_id, save_prefix, tile_size, prompt, extra_pnginfo, my_unique_id, preview_latent, force_full_denoise, disable_noise, samp_custom)

# 简易采样器
class samplerSimple(samplerFull):

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"pipe": ("PIPE_LINE",),
                 "image_output": (["Hide", "Preview", "Preview&Choose", "Save", "Hide&Save", "Sender", "Sender&Save"],{"default": "Preview"}),
                 "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"}),
                 },
                "optional": {
                    "model": ("MODEL",),
                },
                "hidden":
                  {"tile_size": "INT", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                    "embeddingsList": (folder_paths.get_filename_list("embeddings"),)
                  }
                }


    RETURN_TYPES = ("PIPE_LINE", "IMAGE",)
    RETURN_NAMES = ("pipe", "image",)
    OUTPUT_NODE = True
    FUNCTION = "simple"
    CATEGORY = "EasyUse/Sampler"

    def simple(self, pipe, image_output, link_id, save_prefix, model=None, tile_size=None, prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False):

        return super().run(pipe, None, None, None, None, None, image_output, link_id, save_prefix,
                                 None, model, None, None, None, None, None, None,
                                 None, prompt, extra_pnginfo, my_unique_id, force_full_denoise, disable_noise)

# 简易采样器 (Tiled)
class samplerSimpleTiled(samplerFull):

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"pipe": ("PIPE_LINE",),
                 "tile_size": ("INT", {"default": 512, "min": 320, "max": 4096, "step": 64}),
                 "image_output": (["Hide", "Preview", "Save", "Hide&Save", "Sender", "Sender&Save"],{"default": "Preview"}),
                 "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"})
                 },
                "optional": {
                    "model": ("MODEL",),
                },
                "hidden": {
                    "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                    "embeddingsList": (folder_paths.get_filename_list("embeddings"),)
                  }
                }

    RETURN_TYPES = ("PIPE_LINE", "IMAGE",)
    RETURN_NAMES = ("pipe", "image",)
    OUTPUT_NODE = True
    FUNCTION = "tiled"
    CATEGORY = "EasyUse/Sampler"

    def tiled(self, pipe, tile_size=512, image_output='preview', link_id=0, save_prefix='ComfyUI', model=None, prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False):

        return super().run(pipe, None, None,None,None,None, image_output, link_id, save_prefix,
                               None, model, None, None, None, None, None, None,
                               tile_size, prompt, extra_pnginfo, my_unique_id, force_full_denoise, disable_noise)

# 简易采样器 (LayerDiffusion)
class samplerSimpleLayerDiffusion(samplerFull):

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"pipe": ("PIPE_LINE",),
                 "image_output": (["Hide", "Preview", "Save", "Hide&Save", "Sender", "Sender&Save"],{"default": "Preview"}),
                 "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"})
                 },
                "optional": {
                    "model": ("MODEL",),
                },
                "hidden": {
                    "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                    "embeddingsList": (folder_paths.get_filename_list("embeddings"),)
                  }
                }

    RETURN_TYPES = ("PIPE_LINE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("pipe", "final_image", "original_image", "alpha")
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (False, False, False, True)
    FUNCTION = "layerDiffusion"
    CATEGORY = "EasyUse/Sampler"

    def layerDiffusion(self, pipe, image_output='preview', link_id=0, save_prefix='ComfyUI', model=None, prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False):

        result = super().run(pipe, None, None,None,None,None, image_output, link_id, save_prefix,
                               None, model, None, None, None, None, None, None,
                               None, prompt, extra_pnginfo, my_unique_id, force_full_denoise, disable_noise)
        pipe = result["result"][0] if "result" in result else None
        return ({"ui":result['ui'], "result":(pipe, pipe["images"], pipe["samp_images"], pipe["alpha"])})

# 简易采样器(收缩Unet)
class samplerSimpleDownscaleUnet(samplerFull):

    upscale_methods = ["bicubic", "nearest-exact", "bilinear", "area", "bislerp"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"pipe": ("PIPE_LINE",),
                 "downscale_mode": (["None", "Auto", "Custom"],{"default": "Auto"}),
                 "block_number": ("INT", {"default": 3, "min": 1, "max": 32, "step": 1}),
                 "downscale_factor": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 9.0, "step": 0.001}),
                 "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                 "end_percent": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.001}),
                 "downscale_after_skip": ("BOOLEAN", {"default": True}),
                 "downscale_method": (s.upscale_methods,),
                 "upscale_method": (s.upscale_methods,),
                 "image_output": (["Hide", "Preview", "Save", "Hide&Save", "Sender", "Sender&Save"],{"default": "Preview"}),
                 "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"}),
                 },
                "optional": {
                    "model": ("MODEL",),
                },
                "hidden":
                  {"tile_size": "INT", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                    "embeddingsList": (folder_paths.get_filename_list("embeddings"),)
                  }
                }


    RETURN_TYPES = ("PIPE_LINE", "IMAGE",)
    RETURN_NAMES = ("pipe", "image",)
    OUTPUT_NODE = True
    FUNCTION = "downscale_unet"
    CATEGORY = "EasyUse/Sampler"

    def downscale_unet(self, pipe, downscale_mode, block_number, downscale_factor, start_percent, end_percent, downscale_after_skip, downscale_method, upscale_method, image_output, link_id, save_prefix, model=None, tile_size=None, prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False):
        downscale_options = None
        if downscale_mode == 'Auto':
            downscale_options = {
                "block_number": block_number,
                "downscale_factor": None,
                "start_percent": 0,
                "end_percent":0.35,
                "downscale_after_skip": True,
                "downscale_method": "bicubic",
                "upscale_method": "bicubic"
            }
        elif downscale_mode == 'Custom':
            downscale_options = {
                "block_number": block_number,
                "downscale_factor": downscale_factor,
                "start_percent": start_percent,
                "end_percent": end_percent,
                "downscale_after_skip": downscale_after_skip,
                "downscale_method": downscale_method,
                "upscale_method": upscale_method
            }

        return super().run(pipe, None, None,None,None,None, image_output, link_id, save_prefix,
                               None, model, None, None, None, None, None, None,
                               tile_size, prompt, extra_pnginfo, my_unique_id, force_full_denoise, disable_noise, downscale_options)
# 简易采样器 (内补)
class samplerSimpleInpainting(samplerFull):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"pipe": ("PIPE_LINE",),
                 "grow_mask_by": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),
                 "image_output": (["Hide", "Preview", "Save", "Hide&Save", "Sender", "Sender&Save"],{"default": "Preview"}),
                 "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"}),
                 "additional": (["None", "InpaintModelCond", "Differential Diffusion", "Fooocus Inpaint", "Fooocus Inpaint + DD", "Brushnet Random", "Brushnet Random + DD", "Brushnet Segmentation", "Brushnet Segmentation + DD"],{"default": "None"})
                 },
                "optional": {
                    "model": ("MODEL",),
                    "mask": ("MASK",),
                },
                "hidden":
                  {"tile_size": "INT", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                    "embeddingsList": (folder_paths.get_filename_list("embeddings"),)
                  }
                }

    RETURN_TYPES = ("PIPE_LINE", "IMAGE", "VAE")
    RETURN_NAMES = ("pipe", "image", "vae")
    OUTPUT_NODE = True
    FUNCTION = "inpainting"
    CATEGORY = "EasyUse/Sampler"

    def dd(self, model, positive, negative, pixels, vae, mask):
        positive, negative, latent = InpaintModelConditioning().encode(positive, negative, pixels, vae, mask)
        cls = ALL_NODE_CLASS_MAPPINGS['DifferentialDiffusion']
        if cls is not None:
            model, = cls().apply(model)
        else:
            raise Exception("Differential Diffusion not found,please update comfyui")
        return positive, negative, latent, model

    def get_brushnet_model(self, type, model):
        model_type = 'sdxl' if isinstance(model.model.model_config, comfy.supported_models.SDXL) else 'sd1'
        if type == 'random':
            brush_model = BRUSHNET_MODELS['random_mask'][model_type]['model_url']
            if model_type == 'sdxl':
                pattern = 'brushnet.random.mask.sdxl.*\.(safetensors|bin)$'
            else:
                pattern = 'brushnet.random.mask.*\.(safetensors|bin)$'
        elif type == 'segmentation':
            brush_model = BRUSHNET_MODELS['segmentation_mask'][model_type]['model_url']
            if model_type == 'sdxl':
                pattern = 'brushnet.segmentation.mask.sdxl.*\.(safetensors|bin)$'
            else:
                pattern = 'brushnet.segmentation.mask.*\.(safetensors|bin)$'


        brushfile = [e for e in folder_paths.get_filename_list('inpaint') if re.search(pattern, e, re.IGNORECASE)]
        brushname = brushfile[0] if brushfile else None
        if not brushname:
            from urllib.parse import urlparse
            get_local_filepath(brush_model, INPAINT_DIR)
            parsed_url = urlparse(brush_model)
            brushname = os.path.basename(parsed_url.path)
        return brushname

    def apply_brushnet(self, brushname, model, vae, image, mask, positive, negative, scale=1.0, start_at=0, end_at=10000):
        if "BrushNetLoader" not in ALL_NODE_CLASS_MAPPINGS:
            raise Exception("BrushNetLoader not found,please install ComfyUI-BrushNet")
        cls = ALL_NODE_CLASS_MAPPINGS['BrushNetLoader']
        brushnet, = cls().brushnet_loading(brushname, 'float16')
        cls = ALL_NODE_CLASS_MAPPINGS['BrushNet']
        m, positive, negative, latent = cls().model_update(model=model, vae=vae, image=image, mask=mask, brushnet=brushnet, positive=positive, negative=negative, scale=scale, start_at=start_at, end_at=end_at)
        return m, positive, negative, latent

    def inpainting(self, pipe, grow_mask_by, image_output, link_id, save_prefix, additional, model=None, mask=None, tile_size=None, prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False):
        _model = model if model is not None else pipe['model']
        latent = pipe['samples'] if 'samples' in pipe else None
        positive = pipe['positive']
        negative = pipe['negative']
        images = pipe["images"] if pipe and "images" in pipe else None
        vae = pipe["vae"] if pipe and "vae" in pipe else None
        if 'noise_mask' in latent and mask is None:
            mask = latent['noise_mask']
        elif mask is not None:
            if images is None:
                raise Exception("No Images found")
            if vae is None:
                raise Exception("No VAE found")

        match additional:
            case 'Differential Diffusion':
                positive, negative, latent, _model = self.dd(_model, positive, negative, images, vae, mask)
            case 'InpaintModelCond':
                if mask is not None:
                    mask, = GrowMask().expand_mask(mask, grow_mask_by, False)
                positive, negative, latent = InpaintModelConditioning().encode(positive, negative, images, vae, mask)
            case 'Fooocus Inpaint':
                head = list(FOOOCUS_INPAINT_HEAD.keys())[0]
                patch = list(FOOOCUS_INPAINT_PATCH.keys())[0]
                if mask is not None:
                    latent, = VAEEncodeForInpaint().encode(vae, images, mask, grow_mask_by)
                _model, = applyFooocusInpaint().apply(_model, latent, head, patch)
            case 'Fooocus Inpaint + DD':
                head = list(FOOOCUS_INPAINT_HEAD.keys())[0]
                patch = list(FOOOCUS_INPAINT_PATCH.keys())[0]
                if mask is not None:
                    latent, = VAEEncodeForInpaint().encode(vae, images, mask, grow_mask_by)
                _model, = applyFooocusInpaint().apply(_model, latent, head, patch)
                positive, negative, latent, _model = self.dd(_model, positive, negative, images, vae, mask)
            case 'Brushnet Random':
                mask, = GrowMask().expand_mask(mask, grow_mask_by, False)
                brush_name = self.get_brushnet_model('random', _model)
                _model, positive, negative, latent = self.apply_brushnet(brush_name, _model, vae, images, mask, positive, negative)
            case 'Brushnet Random + DD':
                mask, = GrowMask().expand_mask(mask, grow_mask_by, False)
                brush_name = self.get_brushnet_model('random', _model)
                _model, positive, negative, latent = self.apply_brushnet(brush_name, _model, vae, images, mask, positive, negative)
                positive, negative, latent, _model = self.dd(_model, positive, negative, images, vae, mask)
            case 'Brushnet Segmentation':
                mask, = GrowMask().expand_mask(mask, grow_mask_by, False)
                brush_name = self.get_brushnet_model('segmentation', _model)
                _model, positive, negative, latent = self.apply_brushnet(brush_name, _model, vae, images, mask, positive, negative)
            case 'Brushnet Segmentation + DD':
                mask, = GrowMask().expand_mask(mask, grow_mask_by, False)
                brush_name = self.get_brushnet_model('segmentation', _model)
                _model, positive, negative, latent = self.apply_brushnet(brush_name, _model, vae, images, mask, positive, negative)
                positive, negative, latent, _model = self.dd(_model, positive, negative, images, vae, mask)
            case _:
                latent, = VAEEncodeForInpaint().encode(vae, images, mask, grow_mask_by)

        results = super().run(pipe, None, None,None,None,None, image_output, link_id, save_prefix,
                               None, _model, positive, negative, latent, vae, None, None,
                               tile_size, prompt, extra_pnginfo, my_unique_id, force_full_denoise, disable_noise)

        result = results['result']

        return {"ui":results['ui'],"result":(result[0], result[1], result[0]['vae'],)}

# SDTurbo采样器
class samplerSDTurbo:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"pipe": ("PIPE_LINE",),
                     "image_output": (["Hide", "Preview", "Save", "Hide&Save", "Sender", "Sender&Save"],{"default": "Preview"}),
                     "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                     "save_prefix": ("STRING", {"default": "ComfyUI"}),
                     },
                "optional": {
                    "model": ("MODEL",),
                },
                "hidden":
                    {"tile_size": "INT", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO",
                     "my_unique_id": "UNIQUE_ID",
                     "embeddingsList": (folder_paths.get_filename_list("embeddings"),)
                     }
                }

    RETURN_TYPES = ("PIPE_LINE", "IMAGE",)
    RETURN_NAMES = ("pipe", "image",)
    OUTPUT_NODE = True
    FUNCTION = "run"

    CATEGORY = "EasyUse/Sampler"

    def run(self, pipe, image_output, link_id, save_prefix, model=None, tile_size=None, prompt=None, extra_pnginfo=None, my_unique_id=None,):
        # Clean loaded_objects
        easyCache.update_loaded_objects(prompt)

        my_unique_id = int(my_unique_id)

        samp_model = pipe["model"] if model is None else model
        samp_positive = pipe["positive"]
        samp_negative = pipe["negative"]
        samp_samples = pipe["samples"]
        samp_vae = pipe["vae"]
        samp_clip = pipe["clip"]

        samp_seed = pipe['seed']

        samp_sampler = pipe['loader_settings']['sampler']

        sigmas = pipe['loader_settings']['sigmas']
        cfg = pipe['loader_settings']['cfg']
        steps = pipe['loader_settings']['steps']

        disable_noise = False

        preview_latent = True
        if image_output in ("Hide", "Hide&Save"):
            preview_latent = False

        # 推理初始时间
        start_time = int(time.time() * 1000)
        # 开始推理
        samp_samples = sampler.custom_ksampler(samp_model, samp_seed, steps, cfg, samp_sampler, sigmas, samp_positive, samp_negative, samp_samples,
                        disable_noise, preview_latent)
        # 推理结束时间
        end_time = int(time.time() * 1000)

        latent = samp_samples['samples']

        # 解码图片
        if tile_size is not None:
            samp_images = samp_vae.decode_tiled(latent, tile_x=tile_size // 8, tile_y=tile_size // 8, )
        else:
            samp_images = samp_vae.decode(latent).cpu()

        # 推理总耗时（包含解码）
        end_decode_time = int(time.time() * 1000)
        spent_time = 'Diffusion:' + str((end_time - start_time) / 1000) + '″, VAEDecode:' + str(
            (end_decode_time - end_time) / 1000) + '″ '

        # Clean loaded_objects
        easyCache.update_loaded_objects(prompt)

        results = easySave(samp_images, save_prefix, image_output, prompt, extra_pnginfo)
        sampler.update_value_by_id("results", my_unique_id, results)

        new_pipe = {
            "model": samp_model,
            "positive": samp_positive,
            "negative": samp_negative,
            "vae": samp_vae,
            "clip": samp_clip,

            "samples": samp_samples,
            "images": samp_images,
            "seed": samp_seed,

            "loader_settings": {
                **pipe["loader_settings"],
                "spent_time": spent_time
            }
        }

        sampler.update_value_by_id("pipe_line", my_unique_id, new_pipe)

        del pipe

        if image_output in ("Hide", "Hide&Save"):
            return {"ui": {},
                    "result": sampler.get_output(new_pipe, )}

        if image_output in ("Sender", "Sender&Save"):
            PromptServer.instance.send_sync("img-send", {"link_id": link_id, "images": results})


        return {"ui": {"images": results},
                "result": sampler.get_output(new_pipe, )}


# Cascade完整采样器
class samplerCascadeFull:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"pipe": ("PIPE_LINE",),
                     "encode_vae_name": (["None"] + folder_paths.get_filename_list("vae"),),
                     "decode_vae_name": (["None"] + folder_paths.get_filename_list("vae"),),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default":"euler_ancestral"}),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default":"simple"}),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "image_output": (["Hide", "Preview", "Save", "Hide&Save", "Sender", "Sender&Save"],),
                     "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                     "save_prefix": ("STRING", {"default": "ComfyUI"}),
                     "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED_NUM}),
                 },

                "optional": {
                    "image_to_latent_c": ("IMAGE",),
                    "latent_c": ("LATENT",),
                    "model_c": ("MODEL",),
                },
                 "hidden":{"tile_size": "INT", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                    "embeddingsList": (folder_paths.get_filename_list("embeddings"),)
                  }
                }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "LATENT")
    RETURN_NAMES = ("pipe", "model_b", "latent_b")
    OUTPUT_NODE = True

    FUNCTION = "run"
    CATEGORY = "EasyUse/Sampler"

    def run(self, pipe, encode_vae_name, decode_vae_name, steps, cfg, sampler_name, scheduler, denoise, image_output, link_id, save_prefix, seed, image_to_latent_c=None, latent_c=None, model_c=None, tile_size=None, prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False):

        encode_vae_name = encode_vae_name if encode_vae_name is not None else pipe['loader_settings']['encode_vae_name']
        decode_vae_name = decode_vae_name if decode_vae_name is not None else pipe['loader_settings']['decode_vae_name']

        batch_size = pipe["loader_settings"]["batch_size"] if "batch_size" in pipe["loader_settings"] else 1
        if image_to_latent_c is not None:
            if encode_vae_name != 'None':
                encode_vae = easyCache.load_vae(encode_vae_name)
            else:
                encode_vae = pipe['vae'][0]
            if "compression" not in pipe["loader_settings"]:
                raise Exception("compression is not found")

            compression = pipe["loader_settings"]['compression']
            width = image_to_latent_c.shape[-2]
            height = image_to_latent_c.shape[-3]
            out_width = (width // compression) * encode_vae.downscale_ratio
            out_height = (height // compression) * encode_vae.downscale_ratio

            s = comfy.utils.common_upscale(image_to_latent_c.movedim(-1, 1), out_width, out_height, "bicubic",
                                           "center").movedim(1, -1)
            latent_c = encode_vae.encode(s[:, :, :, :3])
            latent_b = torch.zeros([latent_c.shape[0], 4, height // 4, width // 4])

            samples_c = {"samples": latent_c}
            samples_c = RepeatLatentBatch().repeat(samples_c, batch_size)[0]

            samples_b = {"samples": latent_b}
            samples_b = RepeatLatentBatch().repeat(samples_b, batch_size)[0]
            images = image_to_latent_c
        elif latent_c is not None:
            samples_c = latent_c
            samples_b = pipe["samples"][1]
            images = pipe["images"]
        else:
            samples_c = pipe["samples"][0]
            samples_b = pipe["samples"][1]
            images = pipe["images"]

        # Clean loaded_objects
        easyCache.update_loaded_objects(prompt)
        samp_model = model_c if model_c else pipe["model"][0]
        samp_positive = pipe["positive"]
        samp_negative = pipe["negative"]
        samp_samples = samples_c

        samp_seed = seed if seed is not None else pipe['seed']

        steps = steps if steps is not None else pipe['loader_settings']['steps']
        start_step = pipe['loader_settings']['start_step'] if 'start_step' in pipe['loader_settings'] else 0
        last_step = pipe['loader_settings']['last_step'] if 'last_step' in pipe['loader_settings'] else 10000
        cfg = cfg if cfg is not None else pipe['loader_settings']['cfg']
        sampler_name = sampler_name if sampler_name is not None else pipe['loader_settings']['sampler_name']
        scheduler = scheduler if scheduler is not None else pipe['loader_settings']['scheduler']
        denoise = denoise if denoise is not None else pipe['loader_settings']['denoise']
        # 推理初始时间
        start_time = int(time.time() * 1000)
        # 开始推理
        samp_samples = sampler.common_ksampler(samp_model, samp_seed, steps, cfg, sampler_name, scheduler,
                                               samp_positive, samp_negative, samp_samples, denoise=denoise,
                                               preview_latent=False, start_step=start_step,
                                               last_step=last_step, force_full_denoise=False,
                                               disable_noise=False)
        # 推理结束时间
        end_time = int(time.time() * 1000)
        stage_c = samp_samples["samples"]
        results = None

        if image_output not in ['Hide', 'Hide&Save']:
            if decode_vae_name != 'None':
                decode_vae = easyCache.load_vae(decode_vae_name)
            else:
                decode_vae = pipe['vae'][0]
            samp_images = decode_vae.decode(stage_c).cpu()

            results = easySave(samp_images, save_prefix, image_output, prompt, extra_pnginfo)
            sampler.update_value_by_id("results", my_unique_id, results)

        # 推理总耗时（包含解码）
        end_decode_time = int(time.time() * 1000)
        spent_time = 'Diffusion:' + str((end_time - start_time) / 1000) + '″, VAEDecode:' + str(
            (end_decode_time - end_time) / 1000) + '″ '

        # Clean loaded_objects
        easyCache.update_loaded_objects(prompt)
        # zero_out
        c1 = []
        for t in samp_positive:
            d = t[1].copy()
            if "pooled_output" in d:
                d["pooled_output"] = torch.zeros_like(d["pooled_output"])
            n = [torch.zeros_like(t[0]), d]
            c1.append(n)
        # stage_b_conditioning
        c2 = []
        for t in c1:
            d = t[1].copy()
            d['stable_cascade_prior'] = stage_c
            n = [t[0], d]
            c2.append(n)


        new_pipe = {
            "model": pipe['model'][1],
            "positive": c2,
            "negative": c1,
            "vae": pipe['vae'][1],
            "clip": pipe['clip'],

            "samples": samples_b,
            "images": images,
            "seed": seed,

            "loader_settings": {
                **pipe["loader_settings"],
                "spent_time": spent_time
            }
        }
        sampler.update_value_by_id("pipe_line", my_unique_id, new_pipe)

        del pipe

        if image_output in ("Hide", "Hide&Save"):
            return {"ui": {},
                    "result": sampler.get_output(new_pipe, )}

        if image_output in ("Sender", "Sender&Save") and results is not None:
            PromptServer.instance.send_sync("img-send", {"link_id": link_id, "images": results})

        return {"ui": {"images": results}, "result": (new_pipe, new_pipe['model'], new_pipe['samples'])}

# 简易采样器Cascade
class samplerCascadeSimple(samplerCascadeFull):

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"pipe": ("PIPE_LINE",),
                 "image_output": (["Hide", "Preview", "Save", "Hide&Save", "Sender", "Sender&Save"], {"default": "Preview"}),
                 "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"}),
                 },
                "optional": {
                    "model_c": ("MODEL",),
                },
                "hidden":
                  {"tile_size": "INT", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                    "embeddingsList": (folder_paths.get_filename_list("embeddings"),)
                  }
                }


    RETURN_TYPES = ("PIPE_LINE", "IMAGE",)
    RETURN_NAMES = ("pipe", "image",)
    OUTPUT_NODE = True
    FUNCTION = "simple"
    CATEGORY = "EasyUse/Sampler"

    def simple(self, pipe, image_output, link_id, save_prefix, model_c=None, tile_size=None, prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False):

        return super().run(pipe, None, None,None, None,None,None,None, image_output, link_id, save_prefix,
                               None, None, None, model_c, tile_size, prompt, extra_pnginfo, my_unique_id, force_full_denoise, disable_noise)

class unsampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
             "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
             "end_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
             "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
             "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
             "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
             "normalize": (["disable", "enable"],),

             },
            "optional": {
                "pipe": ("PIPE_LINE",),
                "optional_model": ("MODEL",),
                "optional_positive": ("CONDITIONING",),
                "optional_negative": ("CONDITIONING",),
                "optional_latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("PIPE_LINE", "LATENT",)
    RETURN_NAMES = ("pipe", "latent",)
    FUNCTION = "unsampler"

    CATEGORY = "EasyUse/Sampler"

    def unsampler(self, cfg, sampler_name, steps, end_at_step, scheduler, normalize, pipe=None, optional_model=None, optional_positive=None, optional_negative=None,
                  optional_latent=None):

        model = optional_model if optional_model is not None else pipe["model"]
        positive = optional_positive if optional_positive is not None else pipe["positive"]
        negative = optional_negative if optional_negative is not None else pipe["negative"]
        latent_image = optional_latent if optional_latent is not None else pipe["samples"]

        normalize = normalize == "enable"
        device = comfy.model_management.get_torch_device()
        latent = latent_image
        latent_image = latent["samples"]

        end_at_step = min(end_at_step, steps - 1)
        end_at_step = steps - end_at_step

        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = comfy.sample.prepare_mask(latent["noise_mask"], noise.shape, device)


        noise = noise.to(device)
        latent_image = latent_image.to(device)

        _positive = comfy.sampler_helpers.convert_cond(positive)
        _negative = comfy.sampler_helpers.convert_cond(negative)
        models, inference_memory = comfy.sampler_helpers.get_additional_models({"positive": _positive, "negative": _negative}, model.model_dtype())


        comfy.model_management.load_models_gpu([model] + models, model.memory_required(noise.shape) + inference_memory)

        model_patcher = comfy.model_patcher.ModelPatcher(model.model, load_device=device, offload_device=comfy.model_management.unet_offload_device())

        sampler = comfy.samplers.KSampler(model_patcher, steps=steps, device=device, sampler=sampler_name,
                                          scheduler=scheduler, denoise=1.0, model_options=model.model_options)

        sigmas = sampler.sigmas.flip(0) + 0.0001

        pbar = comfy.utils.ProgressBar(steps)

        def callback(step, x0, x, total_steps):
            pbar.update_absolute(step + 1, total_steps)

        samples = sampler.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image,
                                 force_full_denoise=False, denoise_mask=noise_mask, sigmas=sigmas, start_step=0,
                                 last_step=end_at_step, callback=callback)
        if normalize:
            # technically doesn't normalize because unsampling is not guaranteed to end at a std given by the schedule
            samples -= samples.mean()
            samples /= samples.std()
        samples = samples.cpu()

        comfy.sample.cleanup_additional_models(models)

        out = latent.copy()
        out["samples"] = samples

        if pipe is None:
            pipe = {}

        new_pipe = {
            **pipe,
            "samples": out
        }

        return (new_pipe, out,)

#---------------------------------------------------------------采样器 结束----------------------------------------------------------------------

#---------------------------------------------------------------修复 开始----------------------------------------------------------------------#

# 高清修复
class hiresFix:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos", "bislerp"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                 "model_name": (folder_paths.get_filename_list("upscale_models"),),
                 "rescale_after_model": ([False, True], {"default": True}),
                 "rescale_method": (s.upscale_methods,),
                 "rescale": (["by percentage", "to Width/Height", 'to longer side - maintain aspect'],),
                 "percent": ("INT", {"default": 50, "min": 0, "max": 1000, "step": 1}),
                 "width": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                 "height": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                 "longer_side": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                 "crop": (s.crop_methods,),
                 "image_output": (["Hide", "Preview", "Save", "Hide&Save", "Sender", "Sender&Save"],{"default": "Preview"}),
                 "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"}),
                },
                "optional": {
                    "pipe": ("PIPE_LINE",),
                    "image": ("IMAGE",),
                    "vae": ("VAE",),
                },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                           },
                }

    RETURN_TYPES = ("PIPE_LINE", "IMAGE", "LATENT", )
    RETURN_NAMES = ('pipe', 'image', "latent", )

    FUNCTION = "upscale"
    CATEGORY = "EasyUse/Fix"
    OUTPUT_NODE = True

    def vae_encode_crop_pixels(self, pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels

    def upscale(self, model_name, rescale_after_model, rescale_method, rescale, percent, width, height,
                longer_side, crop, image_output, link_id, save_prefix, pipe=None, image=None, vae=None, prompt=None,
                extra_pnginfo=None, my_unique_id=None):

        new_pipe = {}
        if pipe is not None:
            image = image if image is not None else pipe["images"]
            vae = vae if vae is not None else pipe.get("vae")
        elif image is None or vae is None:
            raise ValueError("pipe or image or vae missing.")
        # Load Model
        model_path = folder_paths.get_full_path("upscale_models", model_name)
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        upscale_model = model_loading.load_state_dict(sd).eval()

        # Model upscale
        device = comfy.model_management.get_torch_device()
        upscale_model.to(device)
        in_img = image.movedim(-1, -3).to(device)

        tile = 128 + 64
        overlap = 8
        steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile,
                                                                    tile_y=tile, overlap=overlap)
        pbar = comfy.utils.ProgressBar(steps)
        s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap,
                                    upscale_amount=upscale_model.scale, pbar=pbar)
        upscale_model.cpu()
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)

        # Post Model Rescale
        if rescale_after_model == True:
            samples = s.movedim(-1, 1)
            orig_height = samples.shape[2]
            orig_width = samples.shape[3]
            if rescale == "by percentage" and percent != 0:
                height = percent / 100 * orig_height
                width = percent / 100 * orig_width
                if (width > MAX_RESOLUTION):
                    width = MAX_RESOLUTION
                if (height > MAX_RESOLUTION):
                    height = MAX_RESOLUTION

                width = easySampler.enforce_mul_of_64(width)
                height = easySampler.enforce_mul_of_64(height)
            elif rescale == "to longer side - maintain aspect":
                longer_side = easySampler.enforce_mul_of_64(longer_side)
                if orig_width > orig_height:
                    width, height = longer_side, easySampler.enforce_mul_of_64(longer_side * orig_height / orig_width)
                else:
                    width, height = easySampler.enforce_mul_of_64(longer_side * orig_width / orig_height), longer_side

            s = comfy.utils.common_upscale(samples, width, height, rescale_method, crop)
            s = s.movedim(1, -1)

        # vae encode
        pixels = self.vae_encode_crop_pixels(s)
        t = vae.encode(pixels[:, :, :, :3])

        if pipe is not None:
            new_pipe = {
                "model": pipe['model'],
                "positive": pipe['positive'],
                "negative": pipe['negative'],
                "vae": vae,
                "clip": pipe['clip'],

                "samples": {"samples": t},
                "images": s,
                "seed": pipe['seed'],

                "loader_settings": {
                    **pipe["loader_settings"],
                }
            }
            del pipe
        else:
            new_pipe = {}

        results = easySave(s, save_prefix, image_output, prompt, extra_pnginfo)

        if image_output in ("Sender", "Sender&Save"):
            PromptServer.instance.send_sync("img-send", {"link_id": link_id, "images": results})

        if image_output in ("Hide", "Hide&Save"):
            return (new_pipe, s, {"samples": t},)

        return {"ui": {"images": results},
                "result": (new_pipe, s, {"samples": t},)}

# 预细节修复
class preDetailerFix:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
             "pipe": ("PIPE_LINE",),
             "guide_size": ("FLOAT", {"default": 256, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
             "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
             "max_size": ("FLOAT", {"default": 768, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
             "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
             "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
             "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
             "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
             "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
             "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
             "noise_mask": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
             "force_inpaint": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
             "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
             "wildcard": ("STRING", {"multiline": True, "dynamicPrompts": False}),
             "cycle": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
        },
            "optional": {
                "bbox_segm_pipe": ("PIPE_LINE",),
                "sam_pipe": ("PIPE_LINE",),
                "optional_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "doit"

    CATEGORY = "EasyUse/Fix"

    def doit(self, pipe, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler, denoise, feather, noise_mask, force_inpaint, drop_size, wildcard, cycle, bbox_segm_pipe=None, sam_pipe=None, optional_image=None):

        model = pipe["model"] if "model" in pipe else None
        if model is None:
            raise Exception(f"[ERROR] pipe['model'] is missing")
        clip = pipe["clip"] if"clip" in pipe else None
        if clip is None:
            raise Exception(f"[ERROR] pipe['clip'] is missing")
        vae = pipe["vae"] if "vae" in pipe else None
        if vae is None:
            raise Exception(f"[ERROR] pipe['vae'] is missing")
        if optional_image is not None:
            images = optional_image
        else:
            images = pipe["images"] if "images" in pipe else None
            if images is None:
                raise Exception(f"[ERROR] pipe['image'] is missing")
        positive = pipe["positive"] if "positive" in pipe else None
        if positive is None:
            raise Exception(f"[ERROR] pipe['positive'] is missing")
        negative = pipe["negative"] if "negative" in pipe else None
        if negative is None:
            raise Exception(f"[ERROR] pipe['negative'] is missing")
        bbox_segm_pipe = bbox_segm_pipe or (pipe["bbox_segm_pipe"] if pipe and "bbox_segm_pipe" in pipe else None)
        if bbox_segm_pipe is None:
            raise Exception(f"[ERROR] bbox_segm_pipe or pipe['bbox_segm_pipe'] is missing")
        sam_pipe = sam_pipe or (pipe["sam_pipe"] if pipe and "sam_pipe" in pipe else None)
        if sam_pipe is None:
            raise Exception(f"[ERROR] sam_pipe or pipe['sam_pipe'] is missing")

        loader_settings = pipe["loader_settings"] if "loader_settings" in pipe else {}

        new_pipe = {
            "images": images,
            "model": model,
            "clip": clip,
            "vae": vae,
            "positive": positive,
            "negative": negative,
            "seed": seed,

            "bbox_segm_pipe": bbox_segm_pipe,
            "sam_pipe": sam_pipe,

            "loader_settings": loader_settings,

            "detail_fix_settings": {
                "guide_size": guide_size,
                "guide_size_for": guide_size_for,
                "max_size": max_size,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "feather": feather,
                "noise_mask": noise_mask,
                "force_inpaint": force_inpaint,
                "drop_size": drop_size,
                "wildcard": wildcard,
                "cycle": cycle
            }
        }


        del bbox_segm_pipe
        del sam_pipe

        return (new_pipe,)

# 预遮罩细节修复
class preMaskDetailerFix:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
             "pipe": ("PIPE_LINE",),
             "mask": ("MASK",),

             "guide_size": ("FLOAT", {"default": 384, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
             "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
             "max_size": ("FLOAT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
             "mask_mode": ("BOOLEAN", {"default": True, "label_on": "masked only", "label_off": "whole"}),

             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
             "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
             "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
             "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
             "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
             "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),

             "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
             "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),
             "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
             "refiner_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
             "batch_size": ("INT", {"default": 1, "min": 1, "max": 100}),
             "cycle": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
        },
            "optional": {
                # "patch": ("INPAINT_PATCH",),
                "optional_image": ("IMAGE",),
                "inpaint_model": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "noise_mask_feather": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "doit"

    CATEGORY = "EasyUse/Fix"

    def doit(self, pipe, mask, guide_size, guide_size_for, max_size, mask_mode, seed, steps, cfg, sampler_name, scheduler, denoise, feather, crop_factor, drop_size,refiner_ratio, batch_size, cycle, optional_image=None, inpaint_model=False, noise_mask_feather=20):

        model = pipe["model"] if "model" in pipe else None
        if model is None:
            raise Exception(f"[ERROR] pipe['model'] is missing")
        clip = pipe["clip"] if"clip" in pipe else None
        if clip is None:
            raise Exception(f"[ERROR] pipe['clip'] is missing")
        vae = pipe["vae"] if "vae" in pipe else None
        if vae is None:
            raise Exception(f"[ERROR] pipe['vae'] is missing")
        if optional_image is not None:
            images = optional_image
        else:
            images = pipe["images"] if "images" in pipe else None
            if images is None:
                raise Exception(f"[ERROR] pipe['image'] is missing")
        positive = pipe["positive"] if "positive" in pipe else None
        if positive is None:
            raise Exception(f"[ERROR] pipe['positive'] is missing")
        negative = pipe["negative"] if "negative" in pipe else None
        if negative is None:
            raise Exception(f"[ERROR] pipe['negative'] is missing")
        latent = pipe["samples"] if "samples" in pipe else None
        if latent is None:
            raise Exception(f"[ERROR] pipe['samples'] is missing")

        if 'noise_mask' not in latent:
            if images is None:
                raise Exception("No Images found")
            if vae is None:
                raise Exception("No VAE found")
            x = (images.shape[1] // 8) * 8
            y = (images.shape[2] // 8) * 8
            mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
                                                   size=(images.shape[1], images.shape[2]), mode="bilinear")

            pixels = images.clone()
            if pixels.shape[1] != x or pixels.shape[2] != y:
                x_offset = (pixels.shape[1] % 8) // 2
                y_offset = (pixels.shape[2] % 8) // 2
                pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
                mask = mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]

            mask_erosion = mask

            m = (1.0 - mask.round()).squeeze(1)
            for i in range(3):
                pixels[:, :, :, i] -= 0.5
                pixels[:, :, :, i] *= m
                pixels[:, :, :, i] += 0.5
            t = vae.encode(pixels)

            latent = {"samples": t, "noise_mask": (mask_erosion[:, :, :x, :y].round())}
        # when patch was linked
        # if patch is not None:
        #     worker = InpaintWorker(node_name="easy kSamplerInpainting")
        #     model, = worker.patch(model, latent, patch)

        loader_settings = pipe["loader_settings"] if "loader_settings" in pipe else {}

        new_pipe = {
            "images": images,
            "model": model,
            "clip": clip,
            "vae": vae,
            "positive": positive,
            "negative": negative,
            "seed": seed,
            "mask": mask,

            "loader_settings": loader_settings,

            "detail_fix_settings": {
                "guide_size": guide_size,
                "guide_size_for": guide_size_for,
                "max_size": max_size,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "feather": feather,
                "crop_factor": crop_factor,
                "drop_size": drop_size,
                "refiner_ratio": refiner_ratio,
                "batch_size": batch_size,
                "cycle": cycle
            },

            "mask_settings": {
                "mask_mode": mask_mode,
                "inpaint_model": inpaint_model,
                "noise_mask_feather": noise_mask_feather
            }
        }

        del pipe

        return (new_pipe,)

# 细节修复
class detailerFix:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pipe": ("PIPE_LINE",),
            "image_output": (["Hide", "Preview", "Save", "Hide&Save", "Sender", "Sender&Save"],{"default": "Preview"}),
            "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
            "save_prefix": ("STRING", {"default": "ComfyUI"}),
        },
            "optional": {
                "model": ("MODEL",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID", }
        }

    RETURN_TYPES = ("PIPE_LINE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("pipe", "image", "cropped_refined", "cropped_enhanced_alpha")
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (False, False, True, True)
    FUNCTION = "doit"

    CATEGORY = "EasyUse/Fix"


    def doit(self, pipe, image_output, link_id, save_prefix, model=None, prompt=None, extra_pnginfo=None, my_unique_id=None):

        # Clean loaded_objects
        easyCache.update_loaded_objects(prompt)

        my_unique_id = int(my_unique_id)

        model = model or (pipe["model"] if "model" in pipe else None)
        if model is None:
            raise Exception(f"[ERROR] model or pipe['model'] is missing")

        detail_fix_settings = pipe["detail_fix_settings"] if "detail_fix_settings" in pipe else None
        if detail_fix_settings is None:
            raise Exception(f"[ERROR] detail_fix_settings or pipe['detail_fix_settings'] is missing")

        mask = pipe["mask"] if "mask" in pipe else None

        image = pipe["images"]
        clip = pipe["clip"]
        vae = pipe["vae"]
        seed = pipe["seed"]
        positive = pipe["positive"]
        negative = pipe["negative"]
        loader_settings = pipe["loader_settings"] if "loader_settings" in pipe else {}
        guide_size = pipe["detail_fix_settings"]["guide_size"] if "guide_size" in pipe["detail_fix_settings"] else 256
        guide_size_for = pipe["detail_fix_settings"]["guide_size_for"] if "guide_size_for" in pipe[
            "detail_fix_settings"] else True
        max_size = pipe["detail_fix_settings"]["max_size"] if "max_size" in pipe["detail_fix_settings"] else 768
        steps = pipe["detail_fix_settings"]["steps"] if "steps" in pipe["detail_fix_settings"] else 20
        cfg = pipe["detail_fix_settings"]["cfg"] if "cfg" in pipe["detail_fix_settings"] else 1.0
        sampler_name = pipe["detail_fix_settings"]["sampler_name"] if "sampler_name" in pipe[
            "detail_fix_settings"] else None
        scheduler = pipe["detail_fix_settings"]["scheduler"] if "scheduler" in pipe["detail_fix_settings"] else None
        denoise = pipe["detail_fix_settings"]["denoise"] if "denoise" in pipe["detail_fix_settings"] else 0.5
        feather = pipe["detail_fix_settings"]["feather"] if "feather" in pipe["detail_fix_settings"] else 5
        crop_factor = pipe["detail_fix_settings"]["crop_factor"] if "crop_factor" in pipe["detail_fix_settings"] else 3.0
        drop_size = pipe["detail_fix_settings"]["drop_size"] if "drop_size" in pipe["detail_fix_settings"] else 10
        refiner_ratio = pipe["detail_fix_settings"]["refiner_ratio"] if "refiner_ratio" in pipe else 0.2
        batch_size = pipe["detail_fix_settings"]["batch_size"] if "batch_size" in pipe["detail_fix_settings"] else 1
        noise_mask = pipe["detail_fix_settings"]["noise_mask"] if "noise_mask" in pipe["detail_fix_settings"] else None
        force_inpaint = pipe["detail_fix_settings"]["force_inpaint"] if "force_inpaint" in pipe["detail_fix_settings"] else False
        wildcard = pipe["detail_fix_settings"]["wildcard"] if "wildcard" in pipe["detail_fix_settings"] else ""
        cycle = pipe["detail_fix_settings"]["cycle"] if "cycle" in pipe["detail_fix_settings"] else 1

        bbox_segm_pipe = pipe["bbox_segm_pipe"] if pipe and "bbox_segm_pipe" in pipe else None
        sam_pipe = pipe["sam_pipe"] if "sam_pipe" in pipe else None

        # 细节修复初始时间
        start_time = int(time.time() * 1000)
        if "mask_settings" in pipe:
            mask_mode = pipe['mask_settings']["mask_mode"] if "inpaint_model" in pipe['mask_settings'] else True
            inpaint_model = pipe['mask_settings']["inpaint_model"] if "inpaint_model" in pipe['mask_settings'] else False
            noise_mask_feather = pipe['mask_settings']["noise_mask_feather"] if "noise_mask_feather" in pipe['mask_settings'] else 20
            cls = ALL_NODE_CLASS_MAPPINGS["MaskDetailerPipe"]
            if "MaskDetailerPipe" not in ALL_NODE_CLASS_MAPPINGS:
                raise Exception(f"[ERROR] To use MaskDetailerPipe, you need to install 'Impact Pack'")
            basic_pipe = (model, clip, vae, positive, negative)
            result_img, result_cropped_enhanced, result_cropped_enhanced_alpha, basic_pipe, refiner_basic_pipe_opt = cls().doit(image, mask, basic_pipe, guide_size, guide_size_for, max_size, mask_mode,
             seed, steps, cfg, sampler_name, scheduler, denoise,
             feather, crop_factor, drop_size, refiner_ratio, batch_size, cycle=1,
             refiner_basic_pipe_opt=None, detailer_hook=None, inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather)
            result_mask = mask
            result_cnet_images = ()
        else:
            if bbox_segm_pipe is None:
                raise Exception(f"[ERROR] bbox_segm_pipe or pipe['bbox_segm_pipe'] is missing")
            if sam_pipe is None:
                raise Exception(f"[ERROR] sam_pipe or pipe['sam_pipe'] is missing")
            bbox_detector_opt, bbox_threshold, bbox_dilation, bbox_crop_factor, segm_detector_opt = bbox_segm_pipe
            sam_model_opt, sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold, sam_mask_hint_use_negative = sam_pipe
            if "FaceDetailer" not in ALL_NODE_CLASS_MAPPINGS:
                raise Exception(f"[ERROR] To use FaceDetailer, you need to install 'Impact Pack'")
            cls = ALL_NODE_CLASS_MAPPINGS["FaceDetailer"]

            result_img, result_cropped_enhanced, result_cropped_enhanced_alpha, result_mask, pipe, result_cnet_images = cls().doit(
                image, model, clip, vae, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name,
                scheduler,
                positive, negative, denoise, feather, noise_mask, force_inpaint,
                bbox_threshold, bbox_dilation, bbox_crop_factor,
                sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                sam_mask_hint_use_negative, drop_size, bbox_detector_opt, wildcard, cycle, sam_model_opt,
                segm_detector_opt,
                detailer_hook=None)

        # 细节修复结束时间
        end_time = int(time.time() * 1000)

        spent_time = 'Fix:' + str((end_time - start_time) / 1000) + '"'

        results = easySave(result_img, save_prefix, image_output, prompt, extra_pnginfo)
        sampler.update_value_by_id("results", my_unique_id, results)

        # Clean loaded_objects
        easyCache.update_loaded_objects(prompt)

        new_pipe = {
            "samples": None,
            "images": result_img,
            "model": model,
            "clip": clip,
            "vae": vae,
            "seed": seed,
            "positive": positive,
            "negative": negative,
            "wildcard": wildcard,
            "bbox_segm_pipe": bbox_segm_pipe,
            "sam_pipe": sam_pipe,

            "loader_settings": {
                **loader_settings,
                "spent_time": spent_time
            },
            "detail_fix_settings": detail_fix_settings
        }
        if "mask_settings" in pipe:
            new_pipe["mask_settings"] = pipe["mask_settings"]

        sampler.update_value_by_id("pipe_line", my_unique_id, new_pipe)

        del bbox_segm_pipe
        del sam_pipe
        del pipe

        if image_output in ("Hide", "Hide&Save"):
            return (new_pipe, result_img, result_cropped_enhanced, result_cropped_enhanced_alpha, result_mask, result_cnet_images)

        if image_output in ("Sender", "Sender&Save"):
            PromptServer.instance.send_sync("img-send", {"link_id": link_id, "images": results})

        return {"ui": {"images": results}, "result": (new_pipe, result_img, result_cropped_enhanced, result_cropped_enhanced_alpha, result_mask, result_cnet_images )}

class ultralyticsDetectorForDetailerFix:
    @classmethod
    def INPUT_TYPES(s):
        bboxs = ["bbox/" + x for x in folder_paths.get_filename_list("ultralytics_bbox")]
        segms = ["segm/" + x for x in folder_paths.get_filename_list("ultralytics_segm")]
        return {"required":
                    {"model_name": (bboxs + segms,),
                    "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "bbox_dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                    "bbox_crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),
                    }
                }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("bbox_segm_pipe",)
    FUNCTION = "doit"

    CATEGORY = "EasyUse/Fix"

    def doit(self, model_name, bbox_threshold, bbox_dilation, bbox_crop_factor):
        if 'UltralyticsDetectorProvider' not in ALL_NODE_CLASS_MAPPINGS:
            raise Exception(f"[ERROR] To use UltralyticsDetectorProvider, you need to install 'Impact Pack'")
        cls = ALL_NODE_CLASS_MAPPINGS['UltralyticsDetectorProvider']
        bbox_detector, segm_detector = cls().doit(model_name)
        pipe = (bbox_detector, bbox_threshold, bbox_dilation, bbox_crop_factor, segm_detector)
        return (pipe,)

class samLoaderForDetailerFix:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("sams"),),
                "device_mode": (["AUTO", "Prefer GPU", "CPU"],{"default": "AUTO"}),
                "sam_detection_hint": (
                ["center-1", "horizontal-2", "vertical-2", "rect-4", "diamond-4", "mask-area", "mask-points",
                 "mask-point-bbox", "none"],),
                "sam_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "sam_threshold": ("FLOAT", {"default": 0.93, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sam_bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "sam_mask_hint_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sam_mask_hint_use_negative": (["False", "Small", "Outter"],),
            }
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("sam_pipe",)
    FUNCTION = "doit"

    CATEGORY = "EasyUse/Fix"

    def doit(self, model_name, device_mode, sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold, sam_mask_hint_use_negative):
        if 'SAMLoader' not in ALL_NODE_CLASS_MAPPINGS:
            raise Exception(f"[ERROR] To use SAMLoader, you need to install 'Impact Pack'")
        cls = ALL_NODE_CLASS_MAPPINGS['SAMLoader']
        (sam_model,) = cls().load_model(model_name, device_mode)
        pipe = (sam_model, sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold, sam_mask_hint_use_negative)
        return (pipe,)

#---------------------------------------------------------------修复 结束----------------------------------------------------------------------

#---------------------------------------------------------------节点束 开始----------------------------------------------------------------------#
# 节点束输入
class pipeIn:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
             "required": {},
             "optional": {
                "pipe": ("PIPE_LINE",),
                "model": ("MODEL",),
                "pos": ("CONDITIONING",),
                "neg": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "image": ("IMAGE",),
                "xyPlot": ("XYPLOT",),
            },
            "hidden": {"my_unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "flush"

    CATEGORY = "EasyUse/Pipe"

    def flush(self, pipe=None, model=None, pos=None, neg=None, latent=None, vae=None, clip=None, image=None, xyplot=None, my_unique_id=None):

        model = model if model is not None else pipe.get("model")
        if model is None:
            log_node_warn(f'pipeIn[{my_unique_id}]', "Model missing from pipeLine")
        pos = pos if pos is not None else pipe.get("positive")
        if pos is None:
            log_node_warn(f'pipeIn[{my_unique_id}]', "Pos Conditioning missing from pipeLine")
        neg = neg if neg is not None else pipe.get("negative")
        if neg is None:
            log_node_warn(f'pipeIn[{my_unique_id}]', "Neg Conditioning missing from pipeLine")
        vae = vae if vae is not None else pipe.get("vae")
        if vae is None:
            log_node_warn(f'pipeIn[{my_unique_id}]', "VAE missing from pipeLine")
        clip = clip if clip is not None else pipe.get("clip")
        if clip is None:
            log_node_warn(f'pipeIn[{my_unique_id}]', "Clip missing from pipeLine")
        if latent is not None:
            samples = latent
        elif image is None:
            samples = pipe.get("samples") if pipe is not None else None
            image = pipe.get("images") if pipe is not None else None
        elif image is not None:
            if pipe is None:
                batch_size = 1
            else:
                batch_size = pipe["loader_settings"]["batch_size"] if "batch_size" in pipe["loader_settings"] else 1
            samples = {"samples": vae.encode(image[:, :, :, :3])}
            samples = RepeatLatentBatch().repeat(samples, batch_size)[0]

        if pipe is None:
            pipe = {"loader_settings": {"positive": "", "negative": "", "xyplot": None}}

        xyplot = xyplot if xyplot is not None else pipe['loader_settings']['xyplot'] if xyplot in pipe['loader_settings'] else None

        new_pipe = {
            **pipe,
            "model": model,
            "positive": pos,
            "negative": neg,
            "vae": vae,
            "clip": clip,

            "samples": samples,
            "images": image,
            "seed": pipe.get('seed') if pipe is not None and "seed" in pipe else None,

            "loader_settings": {
                **pipe["loader_settings"],
                "xyplot": xyplot
            }
        }
        del pipe

        return (new_pipe,)

# 节点束输出
class pipeOut:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
             "required": {
                "pipe": ("PIPE_LINE",),
            },
            "hidden": {"my_unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "IMAGE", "INT",)
    RETURN_NAMES = ("pipe", "model", "pos", "neg", "latent", "vae", "clip", "image", "seed",)
    FUNCTION = "flush"

    CATEGORY = "EasyUse/Pipe"

    def flush(self, pipe, my_unique_id=None):
        model = pipe.get("model")
        pos = pipe.get("positive")
        neg = pipe.get("negative")
        latent = pipe.get("samples")
        vae = pipe.get("vae")
        clip = pipe.get("clip")
        image = pipe.get("images")
        seed = pipe.get("seed")

        return pipe, model, pos, neg, latent, vae, clip, image, seed

# 编辑节点束
class pipeEdit:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
             "required": {
                 "clip_skip": ("INT", {"default": -1, "min": -24, "max": 0, "step": 1}),

                 "optional_positive": ("STRING", {"default": "", "multiline": True}),
                 "positive_token_normalization": (["none", "mean", "length", "length+mean"],),
                 "positive_weight_interpretation": (["comfy", "A1111", "comfy++", "compel", "fixed attention"],),

                 "optional_negative": ("STRING", {"default": "", "multiline": True}),
                 "negative_token_normalization": (["none", "mean", "length", "length+mean"],),
                 "negative_weight_interpretation": (["comfy", "A1111", "comfy++", "compel", "fixed attention"],),

                 "a1111_prompt_style": ("BOOLEAN", {"default": False}),
                 "conditioning_mode": (['replace', 'concat', 'combine', 'average', 'timestep'], {"default": "replace"}),
                 "average_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                 "old_cond_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                 "old_cond_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                 "new_cond_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                 "new_cond_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
             },
             "optional": {
                "pipe": ("PIPE_LINE",),
                "model": ("MODEL",),
                "pos": ("CONDITIONING",),
                "neg": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "image": ("IMAGE",),
             },
            "hidden": {"my_unique_id": "UNIQUE_ID", "prompt":"PROMPT"},
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "IMAGE")
    RETURN_NAMES = ("pipe", "model", "pos", "neg", "latent", "vae", "clip", "image")
    FUNCTION = "flush"

    CATEGORY = "EasyUse/Pipe"

    def flush(self, clip_skip, optional_positive, positive_token_normalization, positive_weight_interpretation, optional_negative, negative_token_normalization, negative_weight_interpretation, a1111_prompt_style, conditioning_mode, average_strength, old_cond_start, old_cond_end, new_cond_start, new_cond_end, pipe=None, model=None, pos=None, neg=None, latent=None, vae=None, clip=None, image=None, my_unique_id=None, prompt=None):

        model = model if model is not None else pipe.get("model")
        if model is None:
            log_node_warn(f'pipeIn[{my_unique_id}]', "Model missing from pipeLine")
        vae = vae if vae is not None else pipe.get("vae")
        if vae is None:
            log_node_warn(f'pipeIn[{my_unique_id}]', "VAE missing from pipeLine")
        clip = clip if clip is not None else pipe.get("clip")
        if clip is None:
            log_node_warn(f'pipeIn[{my_unique_id}]', "Clip missing from pipeLine")
        if image is None:
            image = pipe.get("images") if pipe is not None else None
            samples = latent if latent is not None else pipe.get("samples")
            if samples is None:
                log_node_warn(f'pipeIn[{my_unique_id}]', "Latent missing from pipeLine")
        else:
            batch_size = pipe["loader_settings"]["batch_size"] if "batch_size" in pipe["loader_settings"] else 1
            samples = {"samples": vae.encode(image[:, :, :, :3])}
            samples = RepeatLatentBatch().repeat(samples, batch_size)[0]

        pipe_lora_stack = pipe.get("lora_stack") if pipe is not None and "lora_stack" in pipe else []

        steps = pipe["loader_settings"]["steps"] if "steps" in pipe["loader_settings"] else 1
        if pos is None and optional_positive != '':
            pos, positive_wildcard_prompt, model, clip = prompt_to_cond('positive', model, clip, clip_skip,
                                                                        pipe_lora_stack, optional_positive, positive_token_normalization,positive_weight_interpretation,
                                                                        a1111_prompt_style, my_unique_id, prompt, easyCache, True, steps)
            pos = set_cond(pipe['positive'], pos, conditioning_mode, average_strength, old_cond_start, old_cond_end, new_cond_start, new_cond_end)
            pipe['loader_settings']['positive'] = positive_wildcard_prompt
            pipe['loader_settings']['positive_token_normalization'] = positive_token_normalization
            pipe['loader_settings']['positive_weight_interpretation'] = positive_weight_interpretation
            if a1111_prompt_style:
                pipe['loader_settings']['a1111_prompt_style'] = True
        else:
            pos = pipe.get("positive")
            if pos is None:
                log_node_warn(f'pipeIn[{my_unique_id}]', "Pos Conditioning missing from pipeLine")

        if neg is None and optional_negative != '':
            neg, negative_wildcard_prompt, model, clip = prompt_to_cond("negative", model, clip, clip_skip, pipe_lora_stack, optional_negative,
                                                      negative_token_normalization, negative_weight_interpretation,
                                                      a1111_prompt_style, my_unique_id, prompt, easyCache, True, steps)
            neg = set_cond(pipe['negative'], neg, conditioning_mode, average_strength, old_cond_start, old_cond_end, new_cond_start, new_cond_end)
            pipe['loader_settings']['negative'] = negative_wildcard_prompt
            pipe['loader_settings']['negative_token_normalization'] = negative_token_normalization
            pipe['loader_settings']['negative_weight_interpretation'] = negative_weight_interpretation
            if a1111_prompt_style:
                pipe['loader_settings']['a1111_prompt_style'] = True
        else:
            neg = pipe.get("negative")
            if neg is None:
                log_node_warn(f'pipeIn[{my_unique_id}]', "Neg Conditioning missing from pipeLine")
        if pipe is None:
            pipe = {"loader_settings": {"positive": "", "negative": "", "xyplot": None}}

        new_pipe = {
            **pipe,
            "model": model,
            "positive": pos,
            "negative": neg,
            "vae": vae,
            "clip": clip,

            "samples": samples,
            "images": image,
            "seed": pipe.get('seed') if pipe is not None and "seed" in pipe else None,
            "loader_settings":{
                **pipe["loader_settings"]
            }
        }
        del pipe

        return (new_pipe, model,pos, neg, latent, vae, clip, image)


# 节点束到基础节点束（pipe to ComfyUI-Impack-pack's basic_pipe）
class pipeToBasicPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
            },
            "hidden": {"my_unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("BASIC_PIPE",)
    RETURN_NAMES = ("basic_pipe",)
    FUNCTION = "doit"

    CATEGORY = "EasyUse/Pipe"

    def doit(self, pipe, my_unique_id=None):
        new_pipe = (pipe.get('model'), pipe.get('clip'), pipe.get('vae'), pipe.get('positive'), pipe.get('negative'))
        del pipe
        return (new_pipe,)

# 批次索引
class pipeBatchIndex:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"pipe": ("PIPE_LINE",),
                             "batch_index": ("INT", {"default": 0, "min": 0, "max": 63}),
                             "length": ("INT", {"default": 1, "min": 1, "max": 64}),
                             },
                "hidden": {"my_unique_id": "UNIQUE_ID"},}

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "doit"

    CATEGORY = "EasyUse/Pipe"

    def doit(self, pipe, batch_index, length, my_unique_id=None):
        samples = pipe["samples"]
        new_samples, = LatentFromBatch().frombatch(samples, batch_index, length)
        new_pipe = {
            **pipe,
            "samples": new_samples
        }
        del pipe
        return (new_pipe,)

# pipeXYPlot
class pipeXYPlot:
    lora_list = ["None"] + folder_paths.get_filename_list("loras")
    lora_strengths = {"min": -4.0, "max": 4.0, "step": 0.01}
    token_normalization = ["none", "mean", "length", "length+mean"]
    weight_interpretation = ["comfy", "A1111", "compel", "comfy++"]

    loader_dict = {
        "ckpt_name": folder_paths.get_filename_list("checkpoints"),
        "vae_name": ["Baked-VAE"] + folder_paths.get_filename_list("vae"),
        "clip_skip": {"min": -24, "max": -1, "step": 1},
        "lora_name": lora_list,
        "lora_model_strength": lora_strengths,
        "lora_clip_strength": lora_strengths,
        "positive": [],
        "negative": [],
    }

    sampler_dict = {
        "steps": {"min": 1, "max": 100, "step": 1},
        "cfg": {"min": 0.0, "max": 100.0, "step": 1.0},
        "sampler_name": comfy.samplers.KSampler.SAMPLERS,
        "scheduler": comfy.samplers.KSampler.SCHEDULERS,
        "denoise": {"min": 0.0, "max": 1.0, "step": 0.01},
        "seed": {"min": 0, "max": MAX_SEED_NUM},
    }

    plot_dict = {**sampler_dict, **loader_dict}

    plot_values = ["None", ]
    plot_values.append("---------------------")
    for k in sampler_dict:
        plot_values.append(f'preSampling: {k}')
    plot_values.append("---------------------")
    for k in loader_dict:
        plot_values.append(f'loader: {k}')

    def __init__(self):
        pass

    rejected = ["None", "---------------------", "Nothing"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "grid_spacing": ("INT", {"min": 0, "max": 500, "step": 5, "default": 0, }),
                "output_individuals": (["False", "True"], {"default": "False"}),
                "flip_xy": (["False", "True"], {"default": "False"}),
                "x_axis": (pipeXYPlot.plot_values, {"default": 'None'}),
                "x_values": (
                "STRING", {"default": '', "multiline": True, "placeholder": 'insert values seperated by "; "'}),
                "y_axis": (pipeXYPlot.plot_values, {"default": 'None'}),
                "y_values": (
                "STRING", {"default": '', "multiline": True, "placeholder": 'insert values seperated by "; "'}),
            },
            "optional": {
              "pipe": ("PIPE_LINE",)
            },
            "hidden": {
                "plot_dict": (pipeXYPlot.plot_dict,),
            },
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "plot"

    CATEGORY = "EasyUse/Pipe"

    def plot(self, grid_spacing, output_individuals, flip_xy, x_axis, x_values, y_axis, y_values, pipe=None):
        def clean_values(values):
            original_values = values.split("; ")
            cleaned_values = []

            for value in original_values:
                # Strip the semi-colon
                cleaned_value = value.strip(';').strip()

                if cleaned_value == "":
                    continue

                # Try to convert the cleaned_value back to int or float if possible
                try:
                    cleaned_value = int(cleaned_value)
                except ValueError:
                    try:
                        cleaned_value = float(cleaned_value)
                    except ValueError:
                        pass

                # Append the cleaned_value to the list
                cleaned_values.append(cleaned_value)

            return cleaned_values

        if x_axis in self.rejected:
            x_axis = "None"
            x_values = []
        else:
            x_values = clean_values(x_values)

        if y_axis in self.rejected:
            y_axis = "None"
            y_values = []
        else:
            y_values = clean_values(y_values)

        if flip_xy == "True":
            x_axis, y_axis = y_axis, x_axis
            x_values, y_values = y_values, x_values


        xy_plot = {"x_axis": x_axis,
                   "x_vals": x_values,
                   "y_axis": y_axis,
                   "y_vals": y_values,
                   "grid_spacing": grid_spacing,
                   "output_individuals": output_individuals}

        if pipe is not None:
            new_pipe = pipe
            new_pipe['loader_settings'] = {
                **pipe['loader_settings'],
                "xyplot": xy_plot
            }
            del pipe
        return (new_pipe, xy_plot,)

# pipeXYPlotAdvanced
class pipeXYPlotAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "grid_spacing": ("INT", {"min": 0, "max": 500, "step": 5, "default": 0, }),
                "output_individuals": (["False", "True"], {"default": "False"}),
                "flip_xy": (["False", "True"], {"default": "False"}),
            },
            "optional": {
                "X": ("X_Y",),
                "Y": ("X_Y",),
            },
            "hidden": {"my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "plot"

    CATEGORY = "EasyUse/Pipe"

    def plot(self, pipe, grid_spacing, output_individuals, flip_xy, X=None, Y=None, my_unique_id=None):
        if X != None:
            x_axis = X.get('axis')
            x_values = X.get('values')
        else:
            x_axis = "Nothing"
            x_values = [""]
        if Y != None:
            y_axis = Y.get('axis')
            y_values = Y.get('values')
        else:
            y_axis = "Nothing"
            y_values = [""]

        if pipe is not None:
            new_pipe = pipe
            positive = pipe["loader_settings"]["positive"] if "positive" in pipe["loader_settings"] else ""
            negative = pipe["loader_settings"]["negative"] if "negative" in pipe["loader_settings"] else ""

            if x_axis == 'advanced: ModelMergeBlocks':
                models = X.get('models')
                vae_use = X.get('vae_use')
                if models is None:
                    raise Exception("models is not found")
                new_pipe['loader_settings'] = {
                    **pipe['loader_settings'],
                    "models": models,
                    "vae_use": vae_use
                }
            if y_axis == 'advanced: ModelMergeBlocks':
                models = Y.get('models')
                vae_use = Y.get('vae_use')
                if models is None:
                    raise Exception("models is not found")
                new_pipe['loader_settings'] = {
                    **pipe['loader_settings'],
                    "models": models,
                    "vae_use": vae_use
                }

            if x_axis in ['advanced: Lora', 'advanced: Checkpoint']:
                lora_stack = X.get('lora_stack')
                _lora_stack = []
                if lora_stack is not None:
                    for lora in lora_stack:
                        _lora_stack.append(
                            {"lora_name": lora[0], "model": pipe['model'], "clip": pipe['clip'], "model_strength": lora[1],
                             "clip_strength": lora[2]})
                del lora_stack
                x_values = "; ".join(x_values)
                lora_stack = pipe['lora_stack'] + _lora_stack if 'lora_stack' in pipe else _lora_stack
                new_pipe['loader_settings'] = {
                    **pipe['loader_settings'],
                    "lora_stack": lora_stack,
                }

            if y_axis in ['advanced: Lora', 'advanced: Checkpoint']:
                lora_stack = Y.get('lora_stack')
                _lora_stack = []
                if lora_stack is not None:
                    for lora in lora_stack:
                        _lora_stack.append(
                            {"lora_name": lora[0], "model": pipe['model'], "clip": pipe['clip'], "model_strength": lora[1],
                             "clip_strength": lora[2]})
                del lora_stack
                y_values = "; ".join(y_values)
                lora_stack = pipe['lora_stack'] + _lora_stack if 'lora_stack' in pipe else _lora_stack
                new_pipe['loader_settings'] = {
                    **pipe['loader_settings'],
                    "lora_stack": lora_stack,
                }

            if x_axis == 'advanced: Seeds++ Batch':
                if new_pipe['seed']:
                    value = x_values
                    x_values = []
                    for index in range(value):
                        x_values.append(str(new_pipe['seed'] + index))
                    x_values = "; ".join(x_values)
            if y_axis == 'advanced: Seeds++ Batch':
                if new_pipe['seed']:
                    value = y_values
                    y_values = []
                    for index in range(value):
                        y_values.append(str(new_pipe['seed'] + index))
                    y_values = "; ".join(y_values)

            if x_axis == 'advanced: Positive Prompt S/R':
                if positive:
                    x_value = x_values
                    x_values = []
                    for index, value in enumerate(x_value):
                        search_txt, replace_txt, replace_all = value
                        if replace_all:
                            txt = replace_txt if replace_txt is not None else positive
                            x_values.append(txt)
                        else:
                            txt = positive.replace(search_txt, replace_txt, 1) if replace_txt is not None else positive
                            x_values.append(txt)
                    x_values = "; ".join(x_values)
            if y_axis == 'advanced: Positive Prompt S/R':
                if positive:
                    y_value = y_values
                    y_values = []
                    for index, value in enumerate(y_value):
                        search_txt, replace_txt, replace_all = value
                        if replace_all:
                            txt = replace_txt if replace_txt is not None else positive
                            y_values.append(txt)
                        else:
                            txt = positive.replace(search_txt, replace_txt, 1) if replace_txt is not None else positive
                            y_values.append(txt)
                    y_values = "; ".join(y_values)

            if x_axis == 'advanced: Negative Prompt S/R':
                if negative:
                    x_value = x_values
                    x_values = []
                    for index, value in enumerate(x_value):
                        search_txt, replace_txt, replace_all = value
                        if replace_all:
                            txt = replace_txt if replace_txt is not None else negative
                            x_values.append(txt)
                        else:
                            txt = negative.replace(search_txt, replace_txt, 1) if replace_txt is not None else negative
                            x_values.append(txt)
                    x_values = "; ".join(x_values)
            if y_axis == 'advanced: Negative Prompt S/R':
                if negative:
                    y_value = y_values
                    y_values = []
                    for index, value in enumerate(y_value):
                        search_txt, replace_txt, replace_all = value
                        if replace_all:
                            txt = replace_txt if replace_txt is not None else negative
                            y_values.append(txt)
                        else:
                            txt = negative.replace(search_txt, replace_txt, 1) if replace_txt is not None else negative
                            y_values.append(txt)
                    y_values = "; ".join(y_values)

            if "advanced: ControlNet" in x_axis:
                x_value = x_values
                x_values = []
                cnet = []
                for index, value in enumerate(x_value):
                    cnet.append(value)
                    x_values.append(str(index))
                x_values = "; ".join(x_values)
                new_pipe['loader_settings'] = {
                    **pipe['loader_settings'],
                    "cnet_stack": cnet,
                }

            if "advanced: ControlNet" in y_axis:
                y_value = y_values
                y_values = []
                cnet = []
                for index, value in enumerate(y_value):
                    cnet.append(value)
                    y_values.append(str(index))
                y_values = "; ".join(y_values)
                new_pipe['loader_settings'] = {
                    **pipe['loader_settings'],
                    "cnet_stack": cnet,
                }

            if "advanced: Pos Condition" in x_axis:
                x_values = "; ".join(x_values)
                cond = X.get('cond')
                new_pipe['loader_settings'] = {
                    **pipe['loader_settings'],
                    "positive_cond_stack": cond,
                }
            if "advanced: Pos Condition" in y_axis:
                y_values = "; ".join(y_values)
                cond = Y.get('cond')
                new_pipe['loader_settings'] = {
                    **pipe['loader_settings'],
                    "positive_cond_stack": cond,
                }

            if "advanced: Neg Condition" in x_axis:
                x_values = "; ".join(x_values)
                cond = X.get('cond')
                new_pipe['loader_settings'] = {
                    **pipe['loader_settings'],
                    "negative_cond_stack": cond,
                }
            if "advanced: Neg Condition" in y_axis:
                y_values = "; ".join(y_values)
                cond = Y.get('cond')
                new_pipe['loader_settings'] = {
                    **pipe['loader_settings'],
                    "negative_cond_stack": cond,
                }

            del pipe

        return pipeXYPlot().plot(grid_spacing, output_individuals, flip_xy, x_axis, x_values, y_axis, y_values, new_pipe)

#---------------------------------------------------------------节点束 结束----------------------------------------------------------------------

# 显示推理时间
class showSpentTime:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "spent_time": ("INFO", {"default": 'Time will be displayed when reasoning is complete', "forceInput": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    FUNCTION = "notify"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    RETURN_NAMES = ()

    CATEGORY = "EasyUse/Util"

    def notify(self, pipe, spent_time=None, unique_id=None, extra_pnginfo=None):
        if unique_id and extra_pnginfo and "workflow" in extra_pnginfo:
            workflow = extra_pnginfo["workflow"]
            node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id), None)
            if node:
                spent_time = pipe['loader_settings']['spent_time'] if 'spent_time' in pipe['loader_settings'] else ''
                node["widgets_values"] = [spent_time]

        return {"ui": {"text": spent_time}, "result": {}}

# 显示加载器参数中的各种名称
class showLoaderSettingsNames:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "names": ("INFO", {"default": '', "forceInput": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("ckpt_name", "vae_name", "lora_name")

    FUNCTION = "notify"
    OUTPUT_NODE = True

    CATEGORY = "EasyUse/Util"

    def notify(self, pipe, names=None, unique_id=None, extra_pnginfo=None):
        if unique_id and extra_pnginfo and "workflow" in extra_pnginfo:
            workflow = extra_pnginfo["workflow"]
            node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id), None)
            if node:
                ckpt_name = pipe['loader_settings']['ckpt_name'] if 'ckpt_name' in pipe['loader_settings'] else ''
                vae_name = pipe['loader_settings']['vae_name'] if 'vae_name' in pipe['loader_settings'] else ''
                lora_name = pipe['loader_settings']['lora_name'] if 'lora_name' in pipe['loader_settings'] else ''

                if ckpt_name:
                    ckpt_name = os.path.basename(os.path.splitext(ckpt_name)[0])
                if vae_name:
                    vae_name = os.path.basename(os.path.splitext(vae_name)[0])
                if lora_name:
                    lora_name = os.path.basename(os.path.splitext(lora_name)[0])

                names = "ckpt_name: " + ckpt_name + '\n' + "vae_name: " + vae_name + '\n' + "lora_name: " + lora_name
                node["widgets_values"] = names

        return {"ui": {"text": names}, "result": (ckpt_name, vae_name, lora_name)}


#---------------------------------------------------------------API 开始----------------------------------------------------------------------#
from .libs.stability import stableAPI
class stableDiffusion3API:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("STRING", {"default": "", "placeholder": "Positive", "multiline": True}),
                "negative": ("STRING", {"default": "", "placeholder": "Negative", "multiline": True}),
                "model": (["sd3", "sd3-turbo"],),
                "aspect_ratio": (['16:9', '1:1', '21:9', '2:3', '3:2', '4:5', '5:4', '9:16', '9:21'],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967294}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "optional_image": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "generate"
    OUTPUT_NODE = False

    CATEGORY = "EasyUse/API"

    def generate(self, positive, negative, model, aspect_ratio, seed, denoise, optional_image=None, unique_id=None, extra_pnginfo=None):
        mode = 'text-to-image'
        if optional_image is not None:
            mode = 'image-to-image'
        output_image = stableAPI.generate_sd3_image(positive, negative, aspect_ratio, seed=seed, mode=mode, model=model, strength=denoise, image=optional_image)
        return (output_image,)

#---------------------------------------------------------------API 结束----------------------------------------------------------------------


NODE_CLASS_MAPPINGS = {
    # seed 随机种
    "easy seed": easySeed,
    "easy globalSeed": globalSeed,
    # prompt 提示词
    "easy positive": positivePrompt,
    "easy negative": negativePrompt,
    "easy wildcards": wildcardsPrompt,
    "easy prompt": prompt,
    "easy promptList": promptList,
    "easy promptLine": promptLine,
    "easy promptConcat": promptConcat,
    "easy promptReplace": promptReplace,
    "easy stylesSelector": stylesPromptSelector,
    "easy portraitMaster": portraitMaster,
    # loaders 加载器
    "easy fullLoader": fullLoader,
    "easy a1111Loader": a1111Loader,
    "easy comfyLoader": comfyLoader,
    "easy svdLoader": svdLoader,
    "easy sv3dLoader": sv3DLoader,
    "easy zero123Loader": zero123Loader,
    "easy dynamiCrafterLoader": dynamiCrafterLoader,
    "easy cascadeLoader": cascadeLoader,
    "easy loraStack": loraStack,
    "easy controlnetStack": controlnetStack,
    "easy controlnetLoader": controlnetSimple,
    "easy controlnetLoaderADV": controlnetAdvanced,
    "easy LLLiteLoader": LLLiteLoader,
    # Adapter 适配器
    "easy ipadapterApply": ipadapterApply,
    "easy ipadapterApplyADV": ipadapterApplyAdvanced,
    "easy ipadapterApplyEncoder": ipadapterApplyEncoder,
    "easy ipadapterApplyEmbeds": ipadapterApplyEmbeds,
    "easy ipadapterApplyRegional": ipadapterApplyRegional,
    "easy ipadapterApplyFromParams": ipadapterApplyFromParams,
    "easy ipadapterStyleComposition": ipadapterStyleComposition,
    "easy instantIDApply": instantIDApply,
    "easy instantIDApplyADV": instantIDApplyAdvanced,
    "easy styleAlignedBatchAlign": styleAlignedBatchAlign,
    "easy icLightApply": icLightApply,
    # Inpaint 内补
    "easy applyFooocusInpaint": applyFooocusInpaint,
    "easy applyBrushNet": applyBrushNet,
    "easy applyPowerPaint": applyPowerPaint,
    "easy applyInpaint": applyInpaint,
    # latent 潜空间
    "easy latentNoisy": latentNoisy,
    "easy latentCompositeMaskedWithCond": latentCompositeMaskedWithCond,
    "easy injectNoiseToLatent": injectNoiseToLatent,
    # preSampling 预采样处理
    "easy preSampling": samplerSettings,
    "easy preSamplingAdvanced": samplerSettingsAdvanced,
    "easy preSamplingNoiseIn": samplerSettingsNoiseIn,
    "easy preSamplingCustom": samplerCustomSettings,
    "easy preSamplingSdTurbo": sdTurboSettings,
    "easy preSamplingDynamicCFG": dynamicCFGSettings,
    "easy preSamplingCascade": cascadeSettings,
    "easy preSamplingLayerDiffusion": layerDiffusionSettings,
    "easy preSamplingLayerDiffusionADDTL": layerDiffusionSettingsADDTL,
    # kSampler k采样器
    "easy fullkSampler": samplerFull,
    "easy kSampler": samplerSimple,
    "easy kSamplerTiled": samplerSimpleTiled,
    "easy kSamplerLayerDiffusion": samplerSimpleLayerDiffusion,
    "easy kSamplerInpainting": samplerSimpleInpainting,
    "easy kSamplerDownscaleUnet": samplerSimpleDownscaleUnet,
    "easy kSamplerSDTurbo": samplerSDTurbo,
    "easy fullCascadeKSampler": samplerCascadeFull,
    "easy cascadeKSampler": samplerCascadeSimple,
    "easy unSampler": unsampler,
    # fix 修复相关
    "easy hiresFix": hiresFix,
    "easy preDetailerFix": preDetailerFix,
    "easy preMaskDetailerFix": preMaskDetailerFix,
    "easy ultralyticsDetectorPipe": ultralyticsDetectorForDetailerFix,
    "easy samLoaderPipe": samLoaderForDetailerFix,
    "easy detailerFix": detailerFix,
    # pipe 管道（节点束）
    "easy pipeIn": pipeIn,
    "easy pipeOut": pipeOut,
    "easy pipeEdit": pipeEdit,
    "easy pipeToBasicPipe": pipeToBasicPipe,
    "easy pipeBatchIndex": pipeBatchIndex,
    "easy XYPlot": pipeXYPlot,
    "easy XYPlotAdvanced": pipeXYPlotAdvanced,
    # XY Inputs
    "easy XYInputs: Seeds++ Batch": XYplot_SeedsBatch,
    "easy XYInputs: Steps": XYplot_Steps,
    "easy XYInputs: CFG Scale": XYplot_CFG,
    "easy XYInputs: Sampler/Scheduler": XYplot_Sampler_Scheduler,
    "easy XYInputs: Denoise": XYplot_Denoise,
    "easy XYInputs: Checkpoint": XYplot_Checkpoint,
    "easy XYInputs: Lora": XYplot_Lora,
    "easy XYInputs: ModelMergeBlocks": XYplot_ModelMergeBlocks,
    "easy XYInputs: PromptSR": XYplot_PromptSR,
    "easy XYInputs: ControlNet": XYplot_Control_Net,
    "easy XYInputs: PositiveCond": XYplot_Positive_Cond,
    "easy XYInputs: PositiveCondList": XYplot_Positive_Cond_List,
    "easy XYInputs: NegativeCond": XYplot_Negative_Cond,
    "easy XYInputs: NegativeCondList": XYplot_Negative_Cond_List,
    # others 其他
    "easy showSpentTime": showSpentTime,
    "easy showLoaderSettingsNames": showLoaderSettingsNames,
    "dynamicThresholdingFull": dynamicThresholdingFull,
    # api 相关
    "easy stableDiffusion3API": stableDiffusion3API,
    # utils
    "easy ckptNames": setCkptName,
    "easy controlnetNames": setControlName,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # seed 随机种
    "easy seed": "EasySeed",
    "easy globalSeed": "EasyGlobalSeed",
    # prompt 提示词
    "easy positive": "Positive",
    "easy negative": "Negative",
    "easy wildcards": "Wildcards",
    "easy prompt": "Prompt",
    "easy promptList": "PromptList",
    "easy promptLine": "PromptLine",
    "easy promptConcat": "PromptConcat",
    "easy promptReplace": "PromptReplace",
    "easy stylesSelector": "Styles Selector",
    "easy portraitMaster": "Portrait Master",
    # loaders 加载器
    "easy fullLoader": "EasyLoader (Full)",
    "easy a1111Loader": "EasyLoader (A1111)",
    "easy comfyLoader": "EasyLoader (Comfy)",
    "easy svdLoader": "EasyLoader (SVD)",
    "easy sv3dLoader": "EasyLoader (SV3D)",
    "easy zero123Loader": "EasyLoader (Zero123)",
    "easy dynamiCrafterLoader": "EasyLoader (DynamiCrafter)",
    "easy cascadeLoader": "EasyCascadeLoader",
    "easy loraStack": "EasyLoraStack",
    "easy controlnetStack": "EasyControlnetStack",
    "easy controlnetLoader": "EasyControlnet",
    "easy controlnetLoaderADV": "EasyControlnet (Advanced)",
    "easy LLLiteLoader": "EasyLLLite",
    # Adapter 适配器
    "easy ipadapterApply": "Easy Apply IPAdapter",
    "easy ipadapterApplyADV": "Easy Apply IPAdapter (Advanced)",
    "easy ipadapterStyleComposition": "Easy Apply IPAdapter (StyleComposition)",
    "easy ipadapterApplyEncoder": "Easy Apply IPAdapter (Encoder)",
    "easy ipadapterApplyRegional": "Easy Apply IPAdapter (Regional)",
    "easy ipadapterApplyEmbeds": "Easy Apply IPAdapter (Embeds)",
    "easy ipadapterApplyFromParams": "Easy Apply IPAdapter (From Params)",
    "easy instantIDApply": "Easy Apply InstantID",
    "easy instantIDApplyADV": "Easy Apply InstantID (Advanced)",
    "easy styleAlignedBatchAlign": "Easy Apply StyleAlign",
    "easy icLightApply": "Easy Apply ICLight",
    # Inpaint 内补
    "easy applyFooocusInpaint": "Easy Apply Fooocus Inpaint",
    "easy applyBrushNet": "Easy Apply BrushNet",
    "easy applyPowerPaint": "Easy Apply PowerPaint",
    "easy applyInpaint": "Easy Apply Inpaint",
    # latent 潜空间
    "easy latentNoisy": "LatentNoisy",
    "easy latentCompositeMaskedWithCond": "LatentCompositeMaskedWithCond",
    "easy injectNoiseToLatent": "InjectNoiseToLatent",
    # preSampling 预采样处理
    "easy preSampling": "PreSampling",
    "easy preSamplingAdvanced": "PreSampling (Advanced)",
    "easy preSamplingNoiseIn": "PreSampling (NoiseIn)",
    "easy preSamplingCustom": "PreSampling (Custom)",
    "easy preSamplingSdTurbo": "PreSampling (SDTurbo)",
    "easy preSamplingDynamicCFG": "PreSampling (DynamicCFG)",
    "easy preSamplingCascade": "PreSampling (Cascade)",
    "easy preSamplingLayerDiffusion": "PreSampling (LayerDiffuse)",
    "easy preSamplingLayerDiffusionADDTL": "PreSampling (LayerDiffuse ADDTL)",
    # kSampler k采样器
    "easy kSampler": "EasyKSampler",
    "easy fullkSampler": "EasyKSampler (Full)",
    "easy kSamplerTiled": "EasyKSampler (Tiled Decode)",
    "easy kSamplerLayerDiffusion": "EasyKSampler (LayerDiffuse)",
    "easy kSamplerInpainting": "EasyKSampler (Inpainting)",
    "easy kSamplerDownscaleUnet": "EasyKsampler (Downscale Unet)",
    "easy kSamplerSDTurbo": "EasyKSampler (SDTurbo)",
    "easy cascadeKSampler": "EasyCascadeKsampler",
    "easy fullCascadeKSampler": "EasyCascadeKsampler (Full)",
    "easy unSampler": "EasyUnSampler",
    # fix 修复相关
    "easy hiresFix": "HiresFix",
    "easy preDetailerFix": "PreDetailerFix",
    "easy preMaskDetailerFix": "preMaskDetailerFix",
    "easy ultralyticsDetectorPipe": "UltralyticsDetector (Pipe)",
    "easy samLoaderPipe": "SAMLoader (Pipe)",
    "easy detailerFix": "DetailerFix",
    # pipe 管道（节点束）
    "easy pipeIn": "Pipe In",
    "easy pipeOut": "Pipe Out",
    "easy pipeEdit": "Pipe Edit",
    "easy pipeBatchIndex": "Pipe Batch Index",
    "easy pipeToBasicPipe": "Pipe -> BasicPipe",
    "easy XYPlot": "XY Plot",
    "easy XYPlotAdvanced": "XY Plot Advanced",
    # XY Inputs
    "easy XYInputs: Seeds++ Batch": "XY Inputs: Seeds++ Batch //EasyUse",
    "easy XYInputs: Steps": "XY Inputs: Steps //EasyUse",
    "easy XYInputs: CFG Scale": "XY Inputs: CFG Scale //EasyUse",
    "easy XYInputs: Sampler/Scheduler": "XY Inputs: Sampler/Scheduler //EasyUse",
    "easy XYInputs: Denoise": "XY Inputs: Denoise //EasyUse",
    "easy XYInputs: Checkpoint": "XY Inputs: Checkpoint //EasyUse",
    "easy XYInputs: Lora": "XY Inputs: Lora //EasyUse",
    "easy XYInputs: ModelMergeBlocks": "XY Inputs: ModelMergeBlocks //EasyUse",
    "easy XYInputs: PromptSR": "XY Inputs: PromptSR //EasyUse",
    "easy XYInputs: ControlNet": "XY Inputs: Controlnet //EasyUse",
    "easy XYInputs: PositiveCond": "XY Inputs: PosCond //EasyUse",
    "easy XYInputs: PositiveCondList": "XY Inputs: PosCondList //EasyUse",
    "easy XYInputs: NegativeCond": "XY Inputs: NegCond //EasyUse",
    "easy XYInputs: NegativeCondList": "XY Inputs: NegCondList //EasyUse",
    # others 其他
    "easy showSpentTime": "Show Spent Time",
    "easy showLoaderSettingsNames": "Show Loader Settings Names",
    "dynamicThresholdingFull": "DynamicThresholdingFull",
    # api 相关
    "easy stableDiffusion3API": "Stable Diffusion 3 (API)",
    # utils
    "easy ckptNames": "Ckpt Names",
    "easy controlnetNames": "ControlNet Names",
}