from ..libs.stability import stableAPI
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

from ..libs.fluxai import fluxaiAPI

class fluxPromptGenAPI:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "describe": ("STRING", {"default": "", "placeholder": "Describe your image idea (you can use any language)", "multiline": True}),
            },
            "optional": {
                "cookie_override": ("STRING", {"default": "", "forceInput": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)

    FUNCTION = "generate"
    OUTPUT_NODE = False

    CATEGORY = "EasyUse/API"

    def generate(self, describe, cookie_override=None, prompt=None, unique_id=None, extra_pnginfo=None):
        prompt = fluxaiAPI.promptGenerate(describe, cookie_override)
        return (prompt,)

NODE_CLASS_MAPPINGS = {
    "easy stableDiffusion3API": stableDiffusion3API,
    "easy fluxPromptGenAPI": fluxPromptGenAPI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "easy stableDiffusion3API": "Stable Diffusion 3 (API)",
    "easy fluxPromptGenAPI": "Flux Prompt Gen (API)",
}