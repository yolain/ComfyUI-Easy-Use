

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
    "easy fluxPromptGenAPI": fluxPromptGenAPI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "easy fluxPromptGenAPI": "Flux Prompt Gen (API)",
}