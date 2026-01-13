import json
import os
from urllib.request import urlopen
import folder_paths

from .. import easyCache
from ..config import FOOOCUS_STYLES_DIR, MAX_SEED_NUM, PROMPT_TEMPLATE, RESOURCES_DIR
from ..libs.log import log_node_info
from ..libs.wildcards import WildcardProcessor, get_wildcard_list, process

from comfy_api.latest import io


# 正面提示词
class positivePrompt(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy positive",
            category="EasyUse/Prompt",
            inputs=[
                io.String.Input("positive", default="", multiline=True, placeholder="Positive"),
            ],
            outputs=[
                io.String.Output(id="output_positive", display_name="positive"),
            ],
        )

    @classmethod
    def execute(cls, positive):
        return io.NodeOutput(positive)

# 通配符提示词
class wildcardsPrompt(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        wildcard_list = get_wildcard_list()
        return io.Schema(
            node_id="easy wildcards",
            category="EasyUse/Prompt",
            inputs=[
                io.String.Input("text", default="", multiline=True, dynamic_prompts=False, placeholder="(Support wildcard)"),
                io.Combo.Input("Select to add LoRA", options=["Select the LoRA to add to the text"] + folder_paths.get_filename_list("loras")),
                io.Combo.Input("Select to add Wildcard", options=["Select the Wildcard to add to the text"] + wildcard_list),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED_NUM),
                io.Boolean.Input("multiline_mode", default=False),
            ],
            outputs=[
                io.String.Output(id="output_text", display_name="text", is_output_list=True),
                io.String.Output(id="populated_text", display_name="populated_text", is_output_list=True),
            ],
            hidden=[
                io.Hidden.prompt,
                io.Hidden.extra_pnginfo,
                io.Hidden.unique_id,
            ],
        )

    @classmethod
    def execute(cls, text, seed, multiline_mode, **kwargs):
        prompt = cls.hidden.prompt

        # Clean loaded_objects
        if prompt:
            easyCache.update_loaded_objects(prompt)

        if multiline_mode:
            populated_text = []
            _text = []
            text_lines = text.split("\n")
            for t in text_lines:
                _text.append(t)
                populated_text.append(process(t, seed))
            text = _text
        else:
            populated_text = [process(text, seed)]
            text = [text]
        return io.NodeOutput(text, populated_text, ui={"value": [seed]})

# 通配符提示词矩阵，会按顺序返回包含通配符的提示词所生成的所有可能
class wildcardsPromptMatrix(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        wildcard_list = get_wildcard_list()
        return io.Schema(
            node_id="easy wildcardsMatrix",
            category="EasyUse/Prompt",
            inputs=[
                io.String.Input("text", default="", multiline=True, dynamic_prompts=False, placeholder="(Support Lora Block Weight and wildcard)"),
                io.Combo.Input("Select to add LoRA", options=["Select the LoRA to add to the text"] + folder_paths.get_filename_list("loras")),
                io.Combo.Input("Select to add Wildcard", options=["Select the Wildcard to add to the text"] + wildcard_list),
                io.Int.Input("offset", default=0, min=0, max=MAX_SEED_NUM, step=1, control_after_generate=True),
                io.Int.Input("output_limit", default=1, min=-1, step=1, tooltip="Output All Probilities", optional=True),
            ],
            outputs=[
                io.String.Output("populated_text", is_output_list=True),
                io.Int.Output("total"),
                io.Int.Output("factors", is_output_list=True),
            ],
            hidden=[
                io.Hidden.prompt,
                io.Hidden.extra_pnginfo,
                io.Hidden.unique_id,
            ],
        )

    @classmethod
    def execute(cls, text, offset, output_limit=1, **kwargs):
        prompt = cls.hidden.prompt
        # Clean loaded_objects
        if prompt:
            easyCache.update_loaded_objects(prompt)

        p = WildcardProcessor(text)
        total = p.total()
        limit = total if output_limit > total or output_limit == -1 else output_limit
        offset = 0 if output_limit == -1 else offset
        populated_text = p.getmany(limit, offset) if output_limit != 1 else [p.getn(offset)]
        return io.NodeOutput(populated_text, p.total(), list(p.placeholder_choices.values()), ui={"value": [offset]})

# 负面提示词
class negativePrompt(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy negative",
            category="EasyUse/Prompt",
            inputs=[
                io.String.Input("negative", default="", multiline=True, placeholder="Negative"),
            ],
            outputs=[
                io.String.Output(id="output_negative", display_name="negative"),
            ],
        )

    @classmethod
    def execute(cls, negative):
        return io.NodeOutput(negative)

# 风格提示词选择器
class stylesPromptSelector(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        styles = ["fooocus_styles"]
        styles_dir = FOOOCUS_STYLES_DIR
        for file_name in os.listdir(styles_dir):
            file = os.path.join(styles_dir, file_name)
            if os.path.isfile(file) and file_name.endswith(".json"):
                if file_name != "fooocus_styles.json":
                    styles.append(file_name.split(".")[0])

        return io.Schema(
            node_id="easy stylesSelector",
            category="EasyUse/Prompt",
            inputs=[
                io.Combo.Input("styles", options=styles, default="fooocus_styles"),
                io.String.Input("positive", default="", force_input=True, optional=True),
                io.String.Input("negative", default="", force_input=True, optional=True),
                io.Custom(io_type="EASY_PROMPT_STYLES").Input("select_styles", optional=True),
            ],
            outputs=[
                io.String.Output(id="output_positive", display_name="positive"),
                io.String.Output(id="output_negative", display_name="negative"),
            ],
            hidden=[
                io.Hidden.prompt,
                io.Hidden.extra_pnginfo,
                io.Hidden.unique_id,
            ],
        )

    @classmethod
    def execute(cls, styles, positive='', negative='', select_styles=None, **kwargs):
        values = []
        all_styles = {}
        positive_prompt, negative_prompt = '', negative
        fooocus_custom_dir = os.path.join(FOOOCUS_STYLES_DIR, 'fooocus_styles.json')
        if styles == "fooocus_styles" and not os.path.exists(fooocus_custom_dir):
            file = os.path.join(RESOURCES_DIR,  styles + '.json')
        else:
            file = os.path.join(FOOOCUS_STYLES_DIR, styles + '.json')
        f = open(file, 'r', encoding='utf-8')
        data = json.load(f)
        f.close()
        for d in data:
            all_styles[d['name']] = d
        # if my_unique_id in prompt:
        #     if prompt[my_unique_id]["inputs"]['select_styles']:
        #         values = prompt[my_unique_id]["inputs"]['select_styles'].split(',')

        if isinstance(select_styles, str):
            values = select_styles.split(',')
        else:
            values = select_styles if select_styles else []

        has_prompt = False
        if len(values) == 0:
            return io.NodeOutput(positive, negative)

        for index, val in enumerate(values):
            if val not in all_styles:
                continue
            if 'prompt' in all_styles[val]:
                if "{prompt}" in all_styles[val]['prompt'] and has_prompt == False:
                    positive_prompt = all_styles[val]['prompt'].replace('{prompt}', positive)
                    has_prompt = True
                elif "{prompt}" in all_styles[val]['prompt']:
                    positive_prompt += ', ' + all_styles[val]['prompt'].replace(', {prompt}', '').replace('{prompt}', '')
                else:
                    positive_prompt = all_styles[val]['prompt'] if positive_prompt == '' else positive_prompt + ', ' + all_styles[val]['prompt']
            if 'negative_prompt' in all_styles[val]:
                negative_prompt += ', ' + all_styles[val]['negative_prompt'] if negative_prompt else all_styles[val]['negative_prompt']

        if has_prompt == False and positive:
            positive_prompt = positive + positive_prompt + ', '

        return io.NodeOutput(positive_prompt, negative_prompt)

#prompt
class prompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy prompt",
            category="EasyUse/Prompt",
            inputs=[
                io.String.Input("text", default="", multiline=True, placeholder="Prompt"),
                io.Combo.Input("prefix", options=["Select the prefix add to the text"] + PROMPT_TEMPLATE["prefix"], default="Select the prefix add to the text"),
                io.Combo.Input("subject", options=["👤Select the subject add to the text"] + PROMPT_TEMPLATE["subject"], default="👤Select the subject add to the text"),
                io.Combo.Input("action", options=["🎬Select the action add to the text"] + PROMPT_TEMPLATE["action"], default="🎬Select the action add to the text"),
                io.Combo.Input("clothes", options=["👚Select the clothes add to the text"] + PROMPT_TEMPLATE["clothes"], default="👚Select the clothes add to the text"),
                io.Combo.Input("environment", options=["☀️Select the illumination environment add to the text"] + PROMPT_TEMPLATE["environment"], default="☀️Select the illumination environment add to the text"),
                io.Combo.Input("background", options=["🎞️Select the background add to the text"] + PROMPT_TEMPLATE["background"], default="🎞️Select the background add to the text"),
                io.Combo.Input("nsfw", options=["🔞Select the nsfw add to the text"] + PROMPT_TEMPLATE["nsfw"], default="🔞️Select the nsfw add to the text"),
            ],
            outputs=[
                io.String.Output("prompt"),
            ],
            hidden=[
                io.Hidden.prompt,
                io.Hidden.extra_pnginfo,
                io.Hidden.unique_id,
            ],
        )

    @classmethod
    def execute(cls, text, **kwargs):
        return io.NodeOutput(text)

#promptList
class promptList(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy promptList",
            category="EasyUse/Prompt",
            inputs=[
                io.String.Input("prompt_1", multiline=True, default=""),
                io.String.Input("prompt_2", multiline=True, default=""),
                io.String.Input("prompt_3", multiline=True, default=""),
                io.String.Input("prompt_4", multiline=True, default=""),
                io.String.Input("prompt_5", multiline=True, default=""),
                io.Custom(io_type="LIST").Input("optional_prompt_list", optional=True),
            ],
            outputs=[
                io.Custom(io_type="LIST").Output("prompt_list"),
                io.String.Output("prompt_strings", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, prompt_1="", prompt_2="", prompt_3="", prompt_4="", prompt_5="", optional_prompt_list=None, **kwargs):
        prompts = []

        if optional_prompt_list:
            for l in optional_prompt_list:
                prompts.append(l)

        # Add individual prompts
        for p in [prompt_1, prompt_2, prompt_3, prompt_4, prompt_5]:
            if isinstance(p, str) and p != '':
                prompts.append(p)

        return io.NodeOutput(prompts, prompts)

#promptLine
class promptLine(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy promptLine",
            category="EasyUse/Prompt",
            inputs=[
                io.String.Input("prompt", multiline=True, default="text"),
                io.Int.Input("start_index", default=0, min=0, max=9999),
                io.Int.Input("max_rows", default=1000, min=1, max=9999),
                io.Boolean.Input("remove_empty_lines", default=True),
            ],
            outputs=[
                io.String.Output("STRING", is_output_list=True),
                io.Combo.Output("COMBO", is_output_list=True),
            ],
            hidden=[
                io.Hidden.prompt,
                io.Hidden.unique_id,
            ],
        )

    @classmethod
    def execute(cls, prompt, start_index, max_rows, remove_empty_lines=True, **kwargs):
        lines = prompt.split('\n')
        
        if remove_empty_lines:
            lines = [line for line in lines if line.strip()]

        start_index = max(0, min(start_index, len(lines) - 1))

        end_index = min(start_index + max_rows, len(lines))

        rows = lines[start_index:end_index]

        return io.NodeOutput(rows, rows)

import comfy.utils
from server import PromptServer
from ..libs.messages import MessageCancelled, Message
class promptAwait(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy promptAwait",
            category="EasyUse/Prompt",
            inputs=[
                io.AnyType.Input("now"),
                io.String.Input("prompt", multiline=True, default="", placeholder="Enter a prompt or use voice to enter to text"),
                io.Custom(io_type="EASY_PROMPT_AWAIT_BAR").Input("toolbar"),
                io.AnyType.Input("prev", optional=True),
            ],
            outputs=[
                io.AnyType.Output(id="output", display_name="output"),
                io.String.Output(id="output_prompt", display_name="prompt"),
                io.Boolean.Output("continue"),
                io.Int.Output("seed"),
            ],
            hidden=[
                io.Hidden.prompt,
                io.Hidden.unique_id,
                io.Hidden.extra_pnginfo,
            ],
        )

    @classmethod
    def execute(cls, now, prompt, toolbar, prev=None, **kwargs):
        id = cls.hidden.unique_id
        id = id.split('.')[len(id.split('.')) - 1] if "." in id else id
        if ":" in id:
            id = id.split(":")[0]
        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(30)
        PromptServer.instance.send_sync('easyuse_prompt_await', {"id": id})
        try:
            res = Message.waitForMessage(id, asList=False)
            if res is None or res == "-1":
                result = (now, prompt, False, 0)
            else:
                input = now if res['select'] == 'now' or prev is None else prev
                result = (input, res['prompt'], False if res['result'] == -1 else True, res['seed'] if res['unlock'] else res['last_seed'])
            pbar.update_absolute(100)
            return io.NodeOutput(*result)
        except MessageCancelled:
            pbar.update_absolute(100)
            raise comfy.model_management.InterruptProcessingException()

class promptConcat(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy promptConcat",
            category="EasyUse/Prompt",
            inputs=[
                io.String.Input("prompt1", multiline=False, default="", force_input=True, optional=True),
                io.String.Input("prompt2", multiline=False, default="", force_input=True, optional=True),
                io.String.Input("separator", multiline=False, default="", optional=True),
            ],
            outputs=[
                io.String.Output("prompt"),
            ],
        )

    @classmethod
    def execute(cls, prompt1="", prompt2="", separator=""):
        return io.NodeOutput(prompt1 + separator + prompt2)

class promptReplace(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy promptReplace",
            category="EasyUse/Prompt",
            inputs=[
                io.String.Input("prompt", multiline=True, default="", force_input=True),
                io.String.Input("find1", multiline=False, default="", optional=True),
                io.String.Input("replace1", multiline=False, default="", optional=True),
                io.String.Input("find2", multiline=False, default="", optional=True),
                io.String.Input("replace2", multiline=False, default="", optional=True),
                io.String.Input("find3", multiline=False, default="", optional=True),
                io.String.Input("replace3", multiline=False, default="", optional=True),
            ],
            outputs=[
                io.String.Output(id="output_prompt",display_name="prompt"),
            ],
        )

    @classmethod
    def execute(cls, prompt, find1="", replace1="", find2="", replace2="", find3="", replace3=""):
        prompt = prompt.replace(find1, replace1)
        prompt = prompt.replace(find2, replace2)
        prompt = prompt.replace(find3, replace3)

        return io.NodeOutput(prompt)


# 肖像大师
# Created by AI Wiz Art (Stefano Flore)
# Version: 2.2
# https://stefanoflore.it
# https://ai-wiz.art
class portraitMaster(io.ComfyNode):

    @classmethod
    def define_schema(cls):
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
            data = json.load(f)
        
        inputs = []
        # Shot
        inputs.append(io.Combo.Input("shot", options=['-'] + data['shot_list']))
        inputs.append(io.Float.Input("shot_weight", default=0, step=0.05, min=0, max=max_float_value, display_mode=io.NumberDisplay.slider))
        # Gender and age
        inputs.append(io.Combo.Input("gender", options=['-'] + data['gender_list'], default="Woman"))
        inputs.append(io.Int.Input("age", default=30, min=18, max=90, step=1, display_mode=io.NumberDisplay.slider))
        # Nationality
        inputs.append(io.Combo.Input("nationality_1", options=['-'] + data['nationality_list'], default="Chinese"))
        inputs.append(io.Combo.Input("nationality_2", options=['-'] + data['nationality_list']))
        inputs.append(io.Float.Input("nationality_mix", default=0, step=0.05, min=0, max=max_float_value, display_mode=io.NumberDisplay.slider))
        # Body
        inputs.append(io.Combo.Input("body_type", options=['-'] + data['body_type_list']))
        inputs.append(io.Float.Input("body_type_weight", default=0, step=0.05, min=0, max=max_float_value, display_mode=io.NumberDisplay.slider))
        inputs.append(io.Combo.Input("model_pose", options=['-'] + data['model_pose_list']))
        inputs.append(io.Combo.Input("eyes_color", options=['-'] + data['eyes_color_list']))
        # Face
        inputs.append(io.Combo.Input("facial_expression", options=['-'] + data['face_expression_list']))
        inputs.append(io.Float.Input("facial_expression_weight", default=0, step=0.05, min=0, max=max_float_value, display_mode=io.NumberDisplay.slider))
        inputs.append(io.Combo.Input("face_shape", options=['-'] + data['face_shape_list']))
        inputs.append(io.Float.Input("face_shape_weight", default=0, step=0.05, min=0, max=max_float_value, display_mode=io.NumberDisplay.slider))
        inputs.append(io.Float.Input("facial_asymmetry", default=0, step=0.05, min=0, max=max_float_value, display_mode=io.NumberDisplay.slider))
        # Hair
        inputs.append(io.Combo.Input("hair_style", options=['-'] + data['hair_style_list']))
        inputs.append(io.Combo.Input("hair_color", options=['-'] + data['hair_color_list']))
        inputs.append(io.Float.Input("disheveled", default=0, step=0.05, min=0, max=max_float_value, display_mode=io.NumberDisplay.slider))
        inputs.append(io.Combo.Input("beard", options=['-'] + data['beard_list']))
        # Skin details
        inputs.append(io.Float.Input("skin_details", default=0, step=0.05, min=0, max=max_float_value, display_mode=io.NumberDisplay.slider))
        inputs.append(io.Float.Input("skin_pores", default=0, step=0.05, min=0, max=max_float_value, display_mode=io.NumberDisplay.slider))
        inputs.append(io.Float.Input("dimples", default=0, step=0.05, min=0, max=max_float_value, display_mode=io.NumberDisplay.slider))
        inputs.append(io.Float.Input("freckles", default=0, step=0.05, min=0, max=max_float_value, display_mode=io.NumberDisplay.slider))
        inputs.append(io.Float.Input("moles", default=0, step=0.05, min=0, max=max_float_value, display_mode=io.NumberDisplay.slider))
        inputs.append(io.Float.Input("skin_imperfections", default=0, step=0.05, min=0, max=max_float_value, display_mode=io.NumberDisplay.slider))
        inputs.append(io.Float.Input("skin_acne", default=0, step=0.05, min=0, max=max_float_value, display_mode=io.NumberDisplay.slider))
        inputs.append(io.Float.Input("tanned_skin", default=0, step=0.05, min=0, max=max_float_value, display_mode=io.NumberDisplay.slider))
        # Eyes
        inputs.append(io.Float.Input("eyes_details", default=0, step=0.05, min=0, max=max_float_value, display_mode=io.NumberDisplay.slider))
        inputs.append(io.Float.Input("iris_details", default=0, step=0.05, min=0, max=max_float_value, display_mode=io.NumberDisplay.slider))
        inputs.append(io.Float.Input("circular_iris", default=0, step=0.05, min=0, max=max_float_value, display_mode=io.NumberDisplay.slider))
        inputs.append(io.Float.Input("circular_pupil", default=0, step=0.05, min=0, max=max_float_value, display_mode=io.NumberDisplay.slider))
        # Light
        inputs.append(io.Combo.Input("light_type", options=['-'] + data['light_type_list']))
        inputs.append(io.Combo.Input("light_direction", options=['-'] + data['light_direction_list']))
        inputs.append(io.Float.Input("light_weight", default=0, step=0.05, min=0, max=max_float_value, display_mode=io.NumberDisplay.slider))
        # Additional
        inputs.append(io.Combo.Input("photorealism_improvement", options=["enable", "disable"]))
        inputs.append(io.String.Input("prompt_start", multiline=True, default="raw photo, (realistic:1.5)"))
        inputs.append(io.String.Input("prompt_additional", multiline=True, default=""))
        inputs.append(io.String.Input("prompt_end", multiline=True, default=""))
        inputs.append(io.String.Input("negative_prompt", multiline=True, default=""))
        
        return io.Schema(
            node_id="easy portraitMaster",
            category="EasyUse/Prompt",
            inputs=inputs,
            outputs=[
                io.String.Output("positive"),
                io.String.Output("negative"),
            ],
        )

    @classmethod
    def execute(cls, shot="-", shot_weight=1, gender="-", body_type="-", body_type_weight=0, eyes_color="-",
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

        return io.NodeOutput(prompt, negative_prompt)

# 多角度
class multiAngle(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy multiAngle",
            category="EasyUse/Prompt",
            inputs=[
                io.Custom(io_type="EASY_MULTI_ANGLE").Input("multi_angle", optional=True),
            ],
            outputs=[
                io.String.Output("prompt", is_output_list=True),
                io.Custom(io_type="EASY_MULTI_ANGLE").Output("params"),
            ],
        )

    @classmethod
    def execute(cls, multi_angle=None, **kwargs):
        if multi_angle is None:
            return io.NodeOutput([""])

        if isinstance(multi_angle, str):
            try:
                multi_angle = json.loads(multi_angle)
            except:
                raise Exception(f"Invalid multi angle: {multi_angle}")

        prompts = []
        for angle_data in multi_angle:
            rotate = angle_data.get("rotate", 0)
            vertical = angle_data.get("vertical", 0)
            zoom = angle_data.get("zoom", 5)
            add_angle_prompt = angle_data.get("add_angle_prompt", True)
            
            # Validate input ranges
            rotate = max(0, min(360, int(rotate)))
            vertical = max(-90, min(90, int(vertical)))
            zoom = max(0.0, min(10.0, float(zoom)))

            h_angle = rotate % 360
            
            # Horizontal direction mapping
            h_suffix = "" if add_angle_prompt else " quarter"
            if h_angle < 22.5 or h_angle >= 337.5: h_direction = "front view"
            elif h_angle < 67.5: h_direction = f"front-right{h_suffix} view"
            elif h_angle < 112.5: h_direction = "right side view"
            elif h_angle < 157.5: h_direction = f"back-right{h_suffix} view"
            elif h_angle < 202.5: h_direction = "back view"
            elif h_angle < 247.5: h_direction = f"back-left{h_suffix} view"
            elif h_angle < 292.5: h_direction = "left side view"
            else: h_direction = f"front-left{h_suffix} view"
            
            # Vertical direction mapping
            if add_angle_prompt:
                if vertical == -90:
                    v_direction = "bottom-looking-up perspective, extreme worm's eye view, focus subject bottom"
                elif vertical < -75:
                    v_direction = "bottom-looking-up perspective, extreme worm's eye view"
                elif vertical < -45:
                    v_direction = "ultra-low angle"
                elif vertical < -15:
                    v_direction = "low angle"
                elif vertical < 15:
                    v_direction = "eye level"
                elif vertical < 45:
                    v_direction = "high angle"
                elif vertical < 75:
                    v_direction = "bird's eye view"
                elif vertical < 90:
                    v_direction = "top-down perspective, looking straight down at the top of the subject"
                else:
                    v_direction = "top-down perspective, looking straight down at the top of the subject, face not visible, focus on subject head"
            else:
                if vertical < -15:
                    v_direction = "low-angle shot"
                elif vertical < 15:
                    v_direction = "eye-level shot"
                elif vertical < 45:
                    v_direction = "elevated shot"
                elif vertical < 75:
                    v_direction = "high-angle shot"
                elif vertical < 90:
                    v_direction = "top-down perspective, looking straight down at the top of the subject"
                else:
                    v_direction = "top-down perspective, looking straight down at the top of the subject, face not visible, focus on subject head"
            
            # Distance/zoom mapping
            if add_angle_prompt:
                if zoom < 2: distance = "extreme wide shot"
                elif zoom < 4: distance = "wide shot"
                elif zoom < 6: distance = "medium shot"
                elif zoom < 8: distance = "close-up"
                else: distance = "extreme close-up"
            else:
                if zoom < 2: distance = "extreme wide shot"
                elif zoom < 4: distance = "wide shot"
                elif zoom < 6: distance = "medium shot"
                elif zoom < 8: distance = "close-up"
                else: distance = "extreme close-up"
            
            # Build prompt
            if add_angle_prompt:
                prompt = f"{h_direction}, {v_direction}, {distance} (horizontal: {rotate}, vertical: {vertical}, zoom: {zoom:.1f})"
            else:
                prompt = f"{h_direction} {v_direction} {distance}"
            
            prompts.append(prompt)
        
        return io.NodeOutput(prompts, multi_angle)


NODE_CLASS_MAPPINGS = {
    "easy positive": positivePrompt,
    "easy negative": negativePrompt,
    "easy wildcards": wildcardsPrompt,
    "easy wildcardsMatrix": wildcardsPromptMatrix,
    "easy prompt": prompt,
    "easy promptList": promptList,
    "easy promptLine": promptLine,
    "easy promptAwait": promptAwait,
    "easy promptConcat": promptConcat,
    "easy promptReplace": promptReplace,
    "easy stylesSelector": stylesPromptSelector,
    "easy portraitMaster": portraitMaster,
    "easy multiAngle": multiAngle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "easy positive": "Positive",
    "easy negative": "Negative",
    "easy wildcards": "Wildcards",
    "easy wildcardsMatrix": "Wildcards Matrix",
    "easy prompt": "Prompt",
    "easy promptList": "PromptList",
    "easy promptLine": "PromptLine",
    "easy promptAwait": "PromptAwait",
    "easy promptConcat": "PromptConcat",
    "easy promptReplace": "PromptReplace",
    "easy stylesSelector": "Styles Selector",
    "easy portraitMaster": "Portrait Master",
    "easy multiAngle": "Multi Angle",
}
