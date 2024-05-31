from typing import Iterator, List, Tuple, Dict, Any, Union, Optional
from _decimal import Context, getcontext
from decimal import Decimal
from .libs.utils import AlwaysEqualProxy, cleanGPUUsedForce
from .libs.cache import remove_cache
import numpy as np
import json

def validate_list_args(args: Dict[str, List[Any]]) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Checks that if there are multiple arguments, they are all the same length or 1
    :param args:
    :return: Tuple (Status, mismatched_key_1, mismatched_key_2)
    """
    # Only have 1 arg
    if len(args) == 1:
        return True, None, None

    len_to_match = None
    matched_arg_name = None
    for arg_name, arg in args.items():
        if arg_name == 'self':
            # self is in locals()
            continue

        if len(arg) != 1:
            if len_to_match is None:
                len_to_match = len(arg)
                matched_arg_name = arg_name
            elif len(arg) != len_to_match:
                return False, arg_name, matched_arg_name

    return True, None, None
def error_if_mismatched_list_args(args: Dict[str, List[Any]]) -> None:
    is_valid, failed_key1, failed_key2 = validate_list_args(args)
    if not is_valid:
        assert failed_key1 is not None
        assert failed_key2 is not None
        raise ValueError(
            f"Mismatched list inputs received. {failed_key1}({len(args[failed_key1])}) !== {failed_key2}({len(args[failed_key2])})"
        )

def zip_with_fill(*lists: Union[List[Any], None]) -> Iterator[Tuple[Any, ...]]:
    """
    Zips lists together, but if a list has 1 element, it will be repeated for each element in the other lists.
    If a list is None, None will be used for that element.
    (Not intended for use with lists of different lengths)
    :param lists:
    :return: Iterator of tuples of length len(lists)
    """
    max_len = max(len(lst) if lst is not None else 0 for lst in lists)
    for i in range(max_len):
        yield tuple(None if lst is None else (lst[0] if len(lst) == 1 else lst[i]) for lst in lists)

# ---------------------------------------------------------------类型 开始----------------------------------------------------------------------#

# 字符串
class String:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"value": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "execute"
    CATEGORY = "EasyUse/Logic/Type"

    def execute(self, value):
        return (value,)

# 整数
class Int:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"value": ("INT", {"default": 0})},
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "execute"
    CATEGORY = "EasyUse/Logic/Type"

    def execute(self, value):
        return (value,)

# 整数范围
class RangeInt:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "range_mode": (["step", "num_steps"], {"default": "step"}),
                "start": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "stop": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "step": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "num_steps": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "end_mode": (["Inclusive", "Exclusive"], {"default": "Inclusive"}),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("range", "range_sizes")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "build_range"

    CATEGORY = "EasyUse/Logic/Type"

    def build_range(
        self,  range_mode, start, stop, step,  num_steps, end_mode
    ) -> Tuple[List[int], List[int]]:
        error_if_mismatched_list_args(locals())

        ranges = []
        range_sizes = []
        for range_mode, e_start, e_stop, e_num_steps, e_step, e_end_mode in zip_with_fill(
            range_mode, start, stop, num_steps, step, end_mode
        ):
            if range_mode == 'step':
                if e_end_mode == "Inclusive":
                    e_stop += 1
                vals = list(range(e_start, e_stop, e_step))
                ranges.extend(vals)
                range_sizes.append(len(vals))
            elif range_mode == 'num_steps':
                direction = 1 if e_stop > e_start else -1
                if e_end_mode == "Exclusive":
                    e_stop -= direction
                vals = (np.rint(np.linspace(e_start, e_stop, e_num_steps)).astype(int).tolist())
                ranges.extend(vals)
                range_sizes.append(len(vals))
        return ranges, range_sizes



# 浮点数
class Float:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"value": ("FLOAT", {"default": 0, "step": 0.01})},
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    FUNCTION = "execute"
    CATEGORY = "EasyUse/Logic/Type"

    def execute(self, value):
        return (value,)


# 浮点数范围
class RangeFloat:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "range_mode": (["step", "num_steps"], {"default": "step"}),
                "start": ("FLOAT", {"default": 0, "min": -4096, "max": 4096, "step": 0.1}),
                "stop": ("FLOAT", {"default": 0, "min": -4096, "max": 4096, "step": 0.1}),
                "step": ("FLOAT", {"default": 0, "min": -4096, "max": 4096, "step": 0.1}),
                "num_steps": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "end_mode": (["Inclusive", "Exclusive"], {"default": "Inclusive"}),
            },
        }

    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("range", "range_sizes")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "build_range"

    CATEGORY = "EasyUse/Logic/Type"

    @staticmethod
    def _decimal_range(
           range_mode: String, start: Decimal, stop: Decimal, step: Decimal, num_steps: Int, inclusive: bool
    ) -> Iterator[float]:
        if range_mode == 'step':
            ret_val = start
            if inclusive:
                stop = stop + step
            direction = 1 if step > 0 else -1
            while (ret_val - stop) * direction < 0:
                yield float(ret_val)
                ret_val += step
        elif range_mode == 'num_steps':
            step = (stop - start) / (num_steps - 1)
            direction = 1 if step > 0 else -1

            ret_val = start
            for _ in range(num_steps):
                if (ret_val - stop) * direction > 0:  # Ensure we don't exceed the 'stop' value
                    break
                yield float(ret_val)
                ret_val += step

    def build_range(
            self,
            range_mode,
            start,
            stop,
            step,
            num_steps,
            end_mode,
    ) -> Tuple[List[float], List[int]]:
        error_if_mismatched_list_args(locals())
        getcontext().prec = 12

        start = [Decimal(s) for s in start]
        stop = [Decimal(s) for s in stop]
        step = [Decimal(s) for s in step]

        ranges = []
        range_sizes = []
        for range_mode, e_start, e_stop, e_step, e_num_steps, e_end_mode in zip_with_fill(
                range_mode, start, stop, step, num_steps, end_mode
        ):
            vals = list(
                self._decimal_range(range_mode, e_start, e_stop, e_step, e_num_steps, e_end_mode == 'Inclusive')
            )
            ranges.extend(vals)
            range_sizes.append(len(vals))

        return ranges, range_sizes


# 布尔
class Boolean:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"value": ("BOOLEAN", {"default": False})},
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "execute"
    CATEGORY = "EasyUse/Logic/Type"

    def execute(self, value):
        return (value,)

# ---------------------------------------------------------------开关 开始----------------------------------------------------------------------#
class imageSwitch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "boolean": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_switch"

    CATEGORY = "EasyUse/Logic/Switch"

    def image_switch(self, image_a, image_b, boolean):

        if boolean:
            return (image_a, )
        else:
            return (image_b, )

class textSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("INT", {"default": 1, "min": 1, "max": 2}),
            },
            "optional": {
                "text1": ("STRING", {"forceInput": True}),
                "text2": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    CATEGORY = "EasyUse/Logic/Switch"
    FUNCTION = "switch"

    def switch(self, input, text1=None, text2=None,):
        if input == 1:
            return (text1,)
        else:
            return (text2,)

# ---------------------------------------------------------------运算 开始----------------------------------------------------------------------#

COMPARE_FUNCTIONS = {
    "a == b": lambda a, b: a == b,
    "a != b": lambda a, b: a != b,
    "a < b": lambda a, b: a < b,
    "a > b": lambda a, b: a > b,
    "a <= b": lambda a, b: a <= b,
    "a >= b": lambda a, b: a >= b,
}

# 比较
class Compare:
    @classmethod
    def INPUT_TYPES(s):
        s.compare_functions = list(COMPARE_FUNCTIONS.keys())
        return {
            "required": {
                "a": (AlwaysEqualProxy("*"), {"default": 0}),
                "b": (AlwaysEqualProxy("*"), {"default": 0}),
                "comparison": (s.compare_functions, {"default": "a == b"}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "compare"
    CATEGORY = "EasyUse/Logic/Math"

    def compare(self, a, b, comparison):
        return (COMPARE_FUNCTIONS[comparison](a, b),)

# 判断
class If:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (AlwaysEqualProxy("*"),),
                "if": (AlwaysEqualProxy("*"),),
                "else": (AlwaysEqualProxy("*"),),
            },
        }

    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    RETURN_NAMES = ("?",)
    FUNCTION = "execute"
    CATEGORY = "EasyUse/Logic/Math"

    def execute(self, *args, **kwargs):
        return (kwargs['if'] if kwargs['any'] else kwargs['else'],)


#是否为SDXL
from comfy.sdxl_clip import SDXLClipModel, SDXLRefinerClipModel, SDXLClipG
class isSDXL:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "optional_pipe": ("PIPE_LINE",),
                "optional_clip": ("CLIP",),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "execute"
    CATEGORY = "EasyUse/Logic"

    def execute(self, optional_pipe=None, optional_clip=None):
        if optional_pipe is None and optional_clip is None:
            raise Exception(f"[ERROR] optional_pipe or optional_clip is missing")
        clip = optional_clip if optional_clip is not None else optional_pipe['clip']
        if isinstance(clip.cond_stage_model, (SDXLClipModel, SDXLRefinerClipModel, SDXLClipG)):
            return (True,)
        else:
            return (False,)

#xy矩阵
class xyAny:

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "X": (AlwaysEqualProxy("*"), {}),
                "Y": (AlwaysEqualProxy("*"), {}),
                "direction": (["horizontal", "vertical"], {"default": "horizontal"})
            }
        }

    RETURN_TYPES = (AlwaysEqualProxy("*"), AlwaysEqualProxy("*"))
    RETURN_NAMES = ("X", "Y")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    CATEGORY = "EasyUse/Logic"
    FUNCTION = "to_xy"

    def to_xy(self, X, Y, direction):
        new_x = list()
        new_y = list()
        if direction[0] == "horizontal":
            for y in Y:
                for x in X:
                    new_x.append(x)
                    new_y.append(y)
        else:
            for x in X:
                for y in Y:
                    new_x.append(x)
                    new_y.append(y)

        return (new_x, new_y)

# 转换所有类型
class ConvertAnything:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "anything": (AlwaysEqualProxy("*"),),
            "output_type": (["string", "int", "float", "boolean"], {"default": "string"}),
        }}

    RETURN_TYPES = (AlwaysEqualProxy("*"),),
    RETURN_NAMES = ('*',)
    OUTPUT_NODE = True
    FUNCTION = "convert"
    CATEGORY = "EasyUse/Logic"

    def convert(self, *args, **kwargs):
        print(kwargs)
        anything = kwargs['anything']
        output_type = kwargs['output_type']
        params = None
        if output_type == 'string':
            params = str(anything)
        elif output_type == 'int':
            params = int(anything)
        elif output_type == 'float':
            params = float(anything)
        elif output_type == 'boolean':
            params = bool(anything)
        return (params,)

# 将所有类型的内容都转成字符串输出
class showAnything:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}, "optional": {"anything": (AlwaysEqualProxy("*"), {}), },
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO",
            }}

    RETURN_TYPES = ()
    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    FUNCTION = "log_input"
    CATEGORY = "EasyUse/Logic"

    def log_input(self, unique_id=None, extra_pnginfo=None, **kwargs):

        values = []
        if "anything" in kwargs:
            for val in kwargs['anything']:
                try:
                    if type(val) is str:
                        values.append(val)
                    else:
                        val = json.dumps(val)
                        values.append(str(val))
                except Exception:
                    values.append(str(val))
                    pass

        if unique_id and extra_pnginfo and "workflow" in extra_pnginfo[0]:
            workflow = extra_pnginfo[0]["workflow"]
            node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id[0]), None)
            if node:
                node["widgets_values"] = [values]

        return {"ui": {"text": values}}

class showTensorShape:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"tensor": (AlwaysEqualProxy("*"),)}, "optional": {},
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"
               }}

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True
    FUNCTION = "log_input"
    CATEGORY = "EasyUse/Logic"

    def log_input(self, tensor, unique_id=None, extra_pnginfo=None):
        shapes = []

        def tensorShape(tensor):
            if isinstance(tensor, dict):
                for k in tensor:
                    tensorShape(tensor[k])
            elif isinstance(tensor, list):
                for i in range(len(tensor)):
                    tensorShape(tensor[i])
            elif hasattr(tensor, 'shape'):
                shapes.append(list(tensor.shape))

        tensorShape(tensor)

        return {"ui": {"text": shapes}}

# cleanGpuUsed
class cleanGPUUsed:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"anything": (AlwaysEqualProxy("*"), {})}, "optional": {},
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO",
                           }}

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True
    FUNCTION = "empty_cache"
    CATEGORY = "EasyUse/Logic"

    def empty_cache(self, anything, unique_id=None, extra_pnginfo=None):
        cleanGPUUsedForce()
        remove_cache('*')
        return ()

class clearCacheKey:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "anything": (AlwaysEqualProxy("*"), {}),
            "cache_key": ("STRING", {"default": "*"}),
        }, "optional": {},
            "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO",}
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True
    FUNCTION = "empty_cache"
    CATEGORY = "EasyUse/Logic"

    def empty_cache(self, anything, cache_name, unique_id=None, extra_pnginfo=None):
        remove_cache(cache_name)
        return ()

class clearCacheAll:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "anything": (AlwaysEqualProxy("*"), {}),
        }, "optional": {},
            "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO",}
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True
    FUNCTION = "empty_cache"
    CATEGORY = "EasyUse/Logic"

    def empty_cache(self, anything, unique_id=None, extra_pnginfo=None):
        remove_cache('*')
        return ()




NODE_CLASS_MAPPINGS = {
  "easy string": String,
  "easy int": Int,
  "easy rangeInt": RangeInt,
  "easy float": Float,
  "easy rangeFloat": RangeFloat,
  "easy boolean": Boolean,
  "easy compare": Compare,
  "easy imageSwitch": imageSwitch,
  "easy textSwitch": textSwitch,
  "easy if": If,
  "easy isSDXL": isSDXL,
  "easy xyAny": xyAny,
  "easy convertAnything": ConvertAnything,
  "easy showAnything": showAnything,
  "easy showTensorShape": showTensorShape,
  "easy clearCacheKey": clearCacheKey,
  "easy clearCacheAll": clearCacheAll,
  "easy cleanGpuUsed": cleanGPUUsed,
}
NODE_DISPLAY_NAME_MAPPINGS = {
  "easy string": "String",
  "easy int": "Int",
  "easy rangeInt": "Range(Int)",
  "easy float": "Float",
  "easy rangeFloat": "Range(Float)",
  "easy boolean": "Boolean",
  "easy compare": "Compare",
  "easy imageSwitch": "Image Switch",
  "easy textSwitch": "Text Switch",
  "easy if": "If",
  "easy isSDXL": "Is SDXL",
  "easy xyAny": "XYAny",
  "easy convertAnything": "Convert Any",
  "easy showAnything": "Show Any",
  "easy showTensorShape": "Show Tensor Shape",
  "easy clearCacheKey": "Clear Cache Key",
  "easy clearCacheAll": "Clear Cache All",
  "easy cleanGpuUsed": "Clean GPU Used"
}