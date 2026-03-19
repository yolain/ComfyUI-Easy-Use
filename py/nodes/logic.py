from typing import Iterator, List, Tuple, Dict, Any, Union, Optional
from _decimal import Context, getcontext
from decimal import Decimal
from nodes import PreviewImage, SaveImage, NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from PIL.PngImagePlugin import PngInfo
from ..libs.utils import AlwaysEqualProxy, ByPassTypeTuple, cleanGPUUsedForce, compare_revision
from ..libs.cache import cache, update_cache, remove_cache
from ..libs.log import log_node_info, log_node_warn
from ..libs.math import evaluate_formula
import numpy as np
import time
import os
import re
import csv
import json
import torch
import comfy.utils
import folder_paths
from comfy_api.latest import io

DEFAULT_FLOW_NUM = 2
MAX_FLOW_NUM = 20

# kept for legacy compatibility but no longer used in node definitions
any_type = AlwaysEqualProxy("*")
lazy_options = {"lazy": True}


def validate_list_args(args: Dict[str, List[Any]]) -> Tuple[bool, None, None]:
    if len(args) == 1:
        return True, None, None
    len_to_match = None
    matched_arg_name = None
    for arg_name, arg in args.items():
        if arg_name == "self":
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
    max_len = max(len(lst) if lst is not None else 0 for lst in lists)
    for i in range(max_len):
        yield tuple(None if lst is None else (lst[0] if len(lst) == 1 else lst[i]) for lst in lists)


# ---------------------------------------------------------------类型 开始----------------------------------------------------------------------#

class String(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy string",
            category="EasyUse/Logic/Type",
            inputs=[io.String.Input("value", default="")],
            outputs=[io.String.Output("string")],
        )

    @classmethod
    def execute(cls, value):
        return io.NodeOutput(value)


class Int(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy int",
            category="EasyUse/Logic/Type",
            inputs=[io.Int.Input("value", default=0, min=-999999, max=999999)],
            outputs=[io.Int.Output("int")],
        )

    @classmethod
    def execute(cls, value):
        return io.NodeOutput(value)


class RangeInt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy rangeInt",
            category="EasyUse/Logic/Type",
            is_input_list=True,
            inputs=[
                io.Combo.Input("range_mode", options=["step", "num_steps"], default="step"),
                io.Int.Input("start", default=0, min=-4096, max=4096, step=1),
                io.Int.Input("stop", default=0, min=-4096, max=4096, step=1),
                io.Int.Input("step", default=0, min=-4096, max=4096, step=1),
                io.Int.Input("num_steps", default=0, min=-4096, max=4096, step=1),
                io.Combo.Input("end_mode", options=["Inclusive", "Exclusive"], default="Inclusive"),
            ],
            outputs=[
                io.Int.Output("range", is_output_list=True),
                io.Int.Output("range_sizes", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, range_mode, start, stop, step, num_steps, end_mode):
        error_if_mismatched_list_args(locals())
        ranges = []
        range_sizes = []
        for rm, e_start, e_stop, e_num_steps, e_step, e_end_mode in zip_with_fill(
                range_mode, start, stop, num_steps, step, end_mode
        ):
            if rm == "step":
                if e_end_mode == "Inclusive":
                    e_stop += 1
                vals = list(range(e_start, e_stop, e_step))
                ranges.extend(vals)
                range_sizes.append(len(vals))
            elif rm == "num_steps":
                direction = 1 if e_stop > e_start else -1
                if e_end_mode == "Exclusive":
                    e_stop -= direction
                vals = (np.rint(np.linspace(e_start, e_stop, e_num_steps)).astype(int).tolist())
                ranges.extend(vals)
                range_sizes.append(len(vals))
        return io.NodeOutput(ranges, range_sizes)


class Float(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy float",
            category="EasyUse/Logic/Type",
            inputs=[io.Float.Input("value", default=0, step=0.01, min=-0xffffffffffffffff, max=0xffffffffffffffff)],
            outputs=[io.Float.Output("float")],
        )

    @classmethod
    def execute(cls, value):
        return io.NodeOutput(round(value, 3))


class RangeFloat(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy rangeFloat",
            category="EasyUse/Logic/Type",
            is_input_list=True,
            inputs=[
                io.Combo.Input("range_mode", options=["step", "num_steps"], default="step"),
                io.Float.Input("start", default=0, min=-4096, max=4096, step=0.1),
                io.Float.Input("stop", default=0, min=-4096, max=4096, step=0.1),
                io.Float.Input("step", default=0, min=-4096, max=4096, step=0.1),
                io.Int.Input("num_steps", default=0, min=-4096, max=4096, step=1),
                io.Combo.Input("end_mode", options=["Inclusive", "Exclusive"], default="Inclusive"),
            ],
            outputs=[
                io.Float.Output("range", is_output_list=True),
                io.Int.Output("range_sizes", is_output_list=True),
            ],
        )

    @staticmethod
    def _decimal_range(range_mode, start, stop, step, num_steps, inclusive):
        if range_mode == "step":
            ret_val = start
            if inclusive:
                stop = stop + step
            direction = 1 if step > 0 else -1
            while (ret_val - stop) * direction < 0:
                yield float(ret_val)
                ret_val += step
        elif range_mode == "num_steps":
            if num_steps is None or num_steps <= 0:
                return
            if num_steps == 1:
                yield float(start)
                return
            if inclusive:
                s = (stop - start) / Decimal(num_steps - 1)
                for i in range(num_steps):
                    yield float(stop if i == num_steps - 1 else (start + s * Decimal(i)))
            else:
                s = (stop - start) / Decimal(num_steps)
                for i in range(num_steps):
                    yield float(start + s * Decimal(i))

    @classmethod
    def execute(cls, range_mode, start, stop, step, num_steps, end_mode):
        error_if_mismatched_list_args(locals())
        getcontext().prec = 12
        start = [round(Decimal(s), 2) for s in start]
        stop = [round(Decimal(s), 2) for s in stop]
        step = [round(Decimal(s), 2) for s in step]
        ranges = []
        range_sizes = []
        for rm, e_start, e_stop, e_step, e_num_steps, e_end_mode in zip_with_fill(
                range_mode, start, stop, step, num_steps, end_mode
        ):
            vals = list(cls._decimal_range(rm, e_start, e_stop, e_step, e_num_steps, e_end_mode == "Inclusive"))
            ranges.extend(vals)
            range_sizes.append(len(vals))
        return io.NodeOutput(ranges, range_sizes)


class Boolean(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy boolean",
            category="EasyUse/Logic/Type",
            inputs=[io.Boolean.Input("value", default=False)],
            outputs=[io.Boolean.Output("boolean")],
        )

    @classmethod
    def execute(cls, value):
        return io.NodeOutput(value)


# ---------------------------------------------------------------开关 开始----------------------------------------------------------------------#

class imageSwitch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy imageSwitch",
            category="EasyUse/Logic/Switch",
            inputs=[
                io.Image.Input("image_a"),
                io.Image.Input("image_b"),
                io.Boolean.Input("boolean", default=False),
            ],
            outputs=[io.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, image_a, image_b, boolean):
        return io.NodeOutput(image_a if boolean else image_b)


class textSwitch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy textSwitch",
            category="EasyUse/Logic/Switch",
            inputs=[
                io.Int.Input("input", default=1, min=1, max=2),
                io.String.Input("text1", force_input=True, optional=True),
                io.String.Input("text2", force_input=True, optional=True),
            ],
            outputs=[io.String.Output("STRING")],
        )

    @classmethod
    def execute(cls, input, text1=None, text2=None):
        return io.NodeOutput(text1 if input == 1 else text2)


# ---------------------------------------------------------------Index Switch----------------------------------------------------------------------#

class ab(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy ab",
            category="EasyUse/Logic",
            inputs=[
                io.Boolean.Input("A or B", default=True, label_on="A", label_off="B"),
                io.AnyType.Input("in"),
            ],
            outputs=[
                io.AnyType.Output("A"),
                io.AnyType.Output("B"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def blocker(cls, value, block=False):
        from comfy_execution.graph import ExecutionBlocker
        return ExecutionBlocker(None) if block else value

    @classmethod
    def execute(cls, **kwargs):
        is_a = kwargs["A or B"]
        a = cls.blocker(kwargs["in"], not is_a)
        b = cls.blocker(kwargs["in"], is_a)
        return io.NodeOutput(a, b)


class anythingInversedSwitch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy anythingInversedSwitch",
            category="EasyUse/Logic",
            inputs=[
                io.Int.Input("index", default=0, min=0, max=9, step=1),
                io.AnyType.Input("in"),
            ],
            outputs=[io.AnyType.Output("out%d" % i) for i in range(MAX_FLOW_NUM)],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, index, **kwargs):
        from comfy_execution.graph import ExecutionBlocker
        res = [kwargs["in"] if index == i else ExecutionBlocker(None) for i in range(MAX_FLOW_NUM)]
        return io.NodeOutput(*res)


class anythingIndexSwitch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs = [io.Int.Input("index", default=0, min=0, max=9, step=1)]
        for i in range(MAX_FLOW_NUM):
            inputs.append(io.AnyType.Input("value%d" % i, optional=True, lazy=True))
        return io.Schema(
            node_id="easy anythingIndexSwitch",
            category="EasyUse/Logic/Index Switch",
            inputs=inputs,
            outputs=[io.AnyType.Output("value")],
        )

    @classmethod
    def check_lazy_status(cls, index, **kwargs):
        key = "value%d" % index
        if kwargs.get(key, None) is None:
            return [key]

    @classmethod
    def execute(cls, index, **kwargs):
        return io.NodeOutput(kwargs["value%d" % index])


class imageIndexSwitch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs = [io.Int.Input("index", default=0, min=0, max=9, step=1)]
        for i in range(MAX_FLOW_NUM):
            inputs.append(io.Image.Input("image%d" % i, optional=True, lazy=True))
        return io.Schema(
            node_id="easy imageIndexSwitch",
            category="EasyUse/Logic/Index Switch",
            inputs=inputs,
            outputs=[io.Image.Output("image")],
        )

    @classmethod
    def check_lazy_status(cls, index, **kwargs):
        key = "image%d" % index
        if kwargs.get(key, None) is None:
            return [key]

    @classmethod
    def execute(cls, index, **kwargs):
        return io.NodeOutput(kwargs["image%d" % index])


class textIndexSwitch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs = [io.Int.Input("index", default=0, min=0, max=9, step=1)]
        for i in range(MAX_FLOW_NUM):
            inputs.append(io.String.Input("text%d" % i, optional=True, lazy=True, force_input=True))
        return io.Schema(
            node_id="easy textIndexSwitch",
            category="EasyUse/Logic/Index Switch",
            inputs=inputs,
            outputs=[io.String.Output("text")],
        )

    @classmethod
    def check_lazy_status(cls, index, **kwargs):
        key = "text%d" % index
        if kwargs.get(key, None) is None:
            return [key]

    @classmethod
    def execute(cls, index, **kwargs):
        return io.NodeOutput(kwargs["text%d" % index])


class conditioningIndexSwitch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs = [io.Int.Input("index", default=0, min=0, max=9, step=1)]
        for i in range(MAX_FLOW_NUM):
            inputs.append(io.Conditioning.Input("cond%d" % i, optional=True, lazy=True))
        return io.Schema(
            node_id="easy conditioningIndexSwitch",
            category="EasyUse/Logic/Index Switch",
            inputs=inputs,
            outputs=[io.Conditioning.Output("conditioning")],
        )

    @classmethod
    def check_lazy_status(cls, index, **kwargs):
        key = "cond%d" % index
        if kwargs.get(key, None) is None:
            return [key]

    @classmethod
    def execute(cls, index, **kwargs):
        return io.NodeOutput(kwargs["cond%d" % index])


# ---------------------------------------------------------------Math----------------------------------------------------------------------#

class mathIntOperation(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy mathInt",
            category="EasyUse/Logic/Math",
            inputs=[
                io.Int.Input("a", default=0, min=-0xffffffffffffffff, max=0xffffffffffffffff, step=1),
                io.Int.Input("b", default=0, min=-0xffffffffffffffff, max=0xffffffffffffffff, step=1),
                io.Combo.Input("operation", options=["add", "subtract", "multiply", "divide", "modulo", "power"]),
            ],
            outputs=[io.Int.Output("INT")],
        )

    @classmethod
    def execute(cls, a, b, operation):
        ops = {
            "add": lambda: a + b, "subtract": lambda: a - b, "multiply": lambda: a * b,
            "divide": lambda: a // b, "modulo": lambda: a % b, "power": lambda: a ** b,
        }
        return io.NodeOutput(ops[operation]())


class mathFloatOperation(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy mathFloat",
            category="EasyUse/Logic/Math",
            inputs=[
                io.Float.Input("a", default=0, min=-999999999999.0, max=999999999999.0, step=0.01),
                io.Float.Input("b", default=0, min=-999999999999.0, max=999999999999.0, step=0.01),
                io.Combo.Input("operation", options=["add", "subtract", "multiply", "divide", "modulo", "power"]),
            ],
            outputs=[io.Float.Output("FLOAT")],
        )

    @classmethod
    def execute(cls, a, b, operation):
        ops = {
            "add": lambda: round(a + b, 3), "subtract": lambda: round(a - b, 3), "multiply": lambda: round(a * b, 3),
            "divide": lambda: round(a / b, 3), "modulo": lambda: round(a % b, 3), "power": lambda: round(a ** b, 3),
        }
        return io.NodeOutput(ops[operation]())


class mathStringOperation(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy mathString",
            category="EasyUse/Logic/Math",
            inputs=[
                io.String.Input("a", multiline=False),
                io.String.Input("b", multiline=False),
                io.Combo.Input("operation", options=["a == b", "a != b", "a IN b", "a MATCH REGEX(b)", "a BEGINSWITH b", "a ENDSWITH b"]),
                io.Boolean.Input("case_sensitive", default=True),
            ],
            outputs=[io.Boolean.Output("BOOLEAN")],
        )

    @classmethod
    def execute(cls, a, b, operation, case_sensitive):
        if not case_sensitive:
            a = a.lower()
            b = b.lower()
        if operation == "a == b":
            return io.NodeOutput(a == b)
        elif operation == "a != b":
            return io.NodeOutput(a != b)
        elif operation == "a IN b":
            return io.NodeOutput(a in b)
        elif operation == "a MATCH REGEX(b)":
            try:
                return io.NodeOutput(re.match(b, a) is not None)
            except Exception:
                return io.NodeOutput(False)
        elif operation == "a BEGINSWITH b":
            return io.NodeOutput(a.startswith(b))
        elif operation == "a ENDSWITH b":
            return io.NodeOutput(a.endswith(b))


class simpleMath(io.ComfyNode):
    """简单计算器节点，支持字符串数学公式计算"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy simpleMath",
            category="EasyUse/Logic/Math",
            inputs=[
                io.String.Input("value", default="", placeholder="输入数学公式，如: a + b, pow(a, 2), ceil(a / b), floor(a * b), round(a / b, 2)"),
                io.AnyType.Input("a", optional=True),
                io.AnyType.Input("b", optional=True),
                io.AnyType.Input("c", optional=True),
            ],
            outputs=[
                io.Int.Output("int"),
                io.Float.Output("float"),
                io.Boolean.Output("boolean"),
            ],
        )

    @classmethod
    def execute(cls, value, a=0, b=0, c=0):
        try:
            result = evaluate_formula(value, a, b, c)
            result_int = int(result)
            return io.NodeOutput(result_int, result, result_int != 0)
        except Exception as e:
            log_node_warn(f"计算错误: {str(e)}")
            return io.NodeOutput(0, 0.0, False)


class simpleMathDual(io.ComfyNode):
    """双公式计算器节点，支持两个独立的数学公式计算"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy simpleMathDual",
            category="EasyUse/Logic/Math",
            inputs=[
                io.String.Input("value1", default="", placeholder="输入数学公式1，如: a + b, pow(a, 2), ceil(a / b)"),
                io.String.Input("value2", default="", placeholder="输入数学公式2，如: c * d, sqrt(c), floor(d / 2)"),
                io.AnyType.Input("a", optional=True),
                io.AnyType.Input("b", optional=True),
                io.AnyType.Input("c", optional=True),
                io.AnyType.Input("d", optional=True),
            ],
            outputs=[
                io.Int.Output("int1"),
                io.Float.Output("float1"),
                io.Int.Output("int2"),
                io.Float.Output("float2"),
            ],
        )

    @classmethod
    def execute(cls, value1, value2, a=0, b=0, c=0, d=0):
        try:
            result1 = evaluate_formula(value1, a, b, c, d)
            result1_int = int(result1)
        except Exception as e:
            log_node_warn(f"公式1计算错误: {str(e)}")
            result1 = 0.0
            result1_int = 0
        try:
            result2 = evaluate_formula(value2, a, b, c, d)
            result2_int = int(result2)
        except Exception as e:
            log_node_warn(f"公式2计算错误: {str(e)}")
            result2 = 0.0
            result2_int = 0
        return io.NodeOutput(result1_int, result1, result2_int, result2)


# ---------------------------------------------------------------Flow----------------------------------------------------------------------#
try:
    from comfy_execution.graph_utils import GraphBuilder, is_link
    from comfy_execution.graph import ExecutionBlocker
except Exception:
    GraphBuilder = None
    ExecutionBlocker = None


class whileLoopStart:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "condition": ("BOOLEAN", {"default": True}),
            },
            "optional": {
            },
        }
        for i in range(MAX_FLOW_NUM):
            inputs["optional"]["initial_value%d" % i] = (any_type,)
        return inputs

    RETURN_TYPES = ByPassTypeTuple(tuple(["FLOW_CONTROL"] + [any_type] * MAX_FLOW_NUM))
    RETURN_NAMES = ByPassTypeTuple(tuple(["flow"] + ["value%d" % i for i in range(MAX_FLOW_NUM)]))
    FUNCTION = "while_loop_open"

    CATEGORY = "EasyUse/Logic/While Loop"

    def while_loop_open(self, condition, **kwargs):
        values = []
        for i in range(MAX_FLOW_NUM):
            values.append(kwargs.get("initial_value%d" % i, None) if condition else ExecutionBlocker(None))
        return tuple(["stub"] + values)


class whileLoopEnd:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "flow": ("FLOW_CONTROL", {"rawLink": True}),
                "condition": ("BOOLEAN", {}),
            },
            "optional": {
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }
        for i in range(MAX_FLOW_NUM):
            inputs["optional"]["initial_value%d" % i] = (any_type,)
        return inputs

    RETURN_TYPES = ByPassTypeTuple(tuple([any_type] * MAX_FLOW_NUM))
    RETURN_NAMES = ByPassTypeTuple(tuple(["value%d" % i for i in range(MAX_FLOW_NUM)]))
    FUNCTION = "while_loop_close"

    CATEGORY = "EasyUse/Logic/While Loop"

    def explore_dependencies(self, node_id, dynprompt, upstream, parent_ids):
        node_info = dynprompt.get_node(node_id)
        if "inputs" not in node_info:
            return

        for k, v in node_info["inputs"].items():
            if is_link(v):
                parent_id = v[0]
                display_id = dynprompt.get_display_node_id(parent_id)
                display_node = dynprompt.get_node(display_id)
                class_type = display_node["class_type"]
                if class_type not in ['easy forLoopEnd', 'easy whileLoopEnd']:
                    parent_ids.append(display_id)
                if parent_id not in upstream:
                    upstream[parent_id] = []
                    self.explore_dependencies(parent_id, dynprompt, upstream, parent_ids)

                upstream[parent_id].append(node_id)

    def explore_output_nodes(self, dynprompt, upstream, output_nodes, parent_ids):
        for parent_id in upstream:
            display_id = dynprompt.get_display_node_id(parent_id)
            for output_id in output_nodes:
                id = output_nodes[output_id][0]
                if id in parent_ids and display_id == id and output_id not in upstream[parent_id]:
                    if '.' in parent_id:
                        arr = parent_id.split('.')
                        arr[len(arr)-1] = output_id
                        upstream[parent_id].append('.'.join(arr))
                    else:
                        upstream[parent_id].append(output_id)

    def collect_contained(self, node_id, upstream, contained):
        if node_id not in upstream:
            return
        for child_id in upstream[node_id]:
            if child_id not in contained:
                contained[child_id] = True
                self.collect_contained(child_id, upstream, contained)

    def while_loop_close(self, flow, condition, dynprompt=None, unique_id=None,**kwargs):
        if not condition:
            # We're done with the loop
            values = []
            for i in range(MAX_FLOW_NUM):
                values.append(kwargs.get("initial_value%d" % i, None))
            return tuple(values)

        # We want to loop
        this_node = dynprompt.get_node(unique_id)
        upstream = {}
        # Get the list of all nodes between the open and close nodes
        parent_ids = []
        self.explore_dependencies(unique_id, dynprompt, upstream, parent_ids)
        parent_ids = list(set(parent_ids))
        # Get the list of all output nodes between the open and close nodes
        prompts = dynprompt.get_original_prompt()
        output_nodes = {}
        for id in prompts:
            node = prompts[id]
            if "inputs" not in node:
                continue
            class_type = node["class_type"]
            class_def = ALL_NODE_CLASS_MAPPINGS[class_type]
            if hasattr(class_def, 'OUTPUT_NODE') and class_def.OUTPUT_NODE == True:
                for k, v in node['inputs'].items():
                    if is_link(v):
                        output_nodes[id] = v

        graph = GraphBuilder()
        self.explore_output_nodes(dynprompt, upstream, output_nodes, parent_ids)
        contained = {}
        open_node = flow[0]
        self.collect_contained(open_node, upstream, contained)
        contained[unique_id] = True
        contained[open_node] = True

        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.node(original_node["class_type"], "Recurse" if node_id == unique_id else node_id)
            node.set_override_display_id(node_id)
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.lookup_node("Recurse" if node_id == unique_id else node_id)
            for k, v in original_node["inputs"].items():
                if is_link(v) and v[0] in contained:
                    parent = graph.lookup_node(v[0])
                    node.set_input(k, parent.out(v[1]))
                else:
                    node.set_input(k, v)

        new_open = graph.lookup_node(open_node)
        for i in range(MAX_FLOW_NUM):
            key = "initial_value%d" % i
            new_open.set_input(key, kwargs.get(key, None))
        my_clone = graph.lookup_node("Recurse")
        result = map(lambda x: my_clone.out(x), range(MAX_FLOW_NUM))
        return {
            "result": tuple(result),
            "expand": graph.finalize(),
        }


class forLoopStart:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1}),
            },
            "optional": {
                "initial_value%d" % i: (any_type,) for i in range(1, MAX_FLOW_NUM)
            },
            "hidden": {
                "initial_value0": (any_type,),
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ByPassTypeTuple(tuple(["FLOW_CONTROL", "INT"] + [any_type] * (MAX_FLOW_NUM - 1)))
    RETURN_NAMES = ByPassTypeTuple(tuple(["flow", "index"] + ["value%d" % i for i in range(1, MAX_FLOW_NUM)]))
    FUNCTION = "for_loop_start"

    CATEGORY = "EasyUse/Logic/For Loop"

    def for_loop_start(self, total, prompt=None, extra_pnginfo=None, unique_id=None, **kwargs):
        graph = GraphBuilder()
        i = 0
        if "initial_value0" in kwargs:
            i = kwargs["initial_value0"]

        initial_values = {("initial_value%d" % num): kwargs.get("initial_value%d" % num, None) for num in
                          range(1, MAX_FLOW_NUM)}
        while_open = graph.node("easy whileLoopStart", condition=total, initial_value0=i, **initial_values)
        outputs = [kwargs.get("initial_value%d" % num, None) for num in range(1, MAX_FLOW_NUM)]
        return {
            "result": tuple(["stub", i] + outputs),
            "expand": graph.finalize(),
        }


class forLoopEnd:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": ("FLOW_CONTROL", {"rawLink": True}),
            },
            "optional": {
                "initial_value%d" % i: (any_type, {"rawLink": True}) for i in range(1, MAX_FLOW_NUM)
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = ByPassTypeTuple(tuple([any_type] * (MAX_FLOW_NUM - 1)))
    RETURN_NAMES = ByPassTypeTuple(tuple(["value%d" % i for i in range(1, MAX_FLOW_NUM)]))
    FUNCTION = "for_loop_end"

    CATEGORY = "EasyUse/Logic/For Loop"



    def for_loop_end(self, flow, dynprompt=None, extra_pnginfo=None, unique_id=None, **kwargs):
        graph = GraphBuilder()
        while_open = flow[0]
        total = None

        # Using dynprompt to get the original node
        forstart_node = dynprompt.get_node(while_open)
        if forstart_node['class_type'] == 'easy forLoopStart':
            inputs = forstart_node['inputs']
            total = inputs['total']
        elif forstart_node['class_type'] == 'easy loadImagesForLoop':
            inputs = forstart_node['inputs']
            limit = inputs['limit']
            start_index = inputs['start_index']
            # Filter files by extension
            directory = inputs['directory']
            total = graph.node('easy imagesCountInDirectory', directory=directory, limit=limit, start_index=start_index, extension='*').out(0)

        sub = graph.node("easy mathInt", operation="add", a=[while_open, 1], b=1)
        cond = graph.node("easy compare", a=sub.out(0), b=total, comparison='a < b')
        input_values = {("initial_value%d" % i): kwargs.get("initial_value%d" % i, None) for i in
                        range(1, MAX_FLOW_NUM)}
        while_close = graph.node("easy whileLoopEnd",
                                 flow=flow,
                                 condition=cond.out(0),
                                 initial_value0=sub.out(0),
                                 **input_values)
        return {
            "result": tuple([while_close.out(i) for i in range(1, MAX_FLOW_NUM)]),
            "expand": graph.finalize(),
        }

COMPARE_FUNCTIONS = {
    "a == b": lambda a, b: a == b,
    "a != b": lambda a, b: a != b,
    "a < b": lambda a, b: a < b,
    "a > b": lambda a, b: a > b,
    "a <= b": lambda a, b: a <= b,
    "a >= b": lambda a, b: a >= b,
    "a > 0": lambda a, b: a > 0,
    "a <= 0": lambda a, b: a <= 0,
    "b > 0": lambda a, b: b > 0,
    "b <= 0": lambda a, b: b <= 0,
}

class Compare(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy compare",
            category="EasyUse/Logic/Math",
            inputs=[
                io.AnyType.Input("a", optional=True),
                io.AnyType.Input("b", optional=True),
                io.Combo.Input("comparison", options=list(COMPARE_FUNCTIONS.keys()), default="a == b", optional=True),
            ],
            outputs=[io.Boolean.Output("boolean")],
        )

    @classmethod
    def execute(cls, a=0, b=0, comparison="a == b"):
        print('a:', a, 'b:', b, 'comparison:', comparison)
        return io.NodeOutput(COMPARE_FUNCTIONS[comparison](a, b))


class IfElse(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy ifElse",
            category="EasyUse/Logic",
            inputs=[
                io.Boolean.Input("boolean"),
                io.AnyType.Input("on_true", lazy=True),
                io.AnyType.Input("on_false", lazy=True),
            ],
            outputs=[io.AnyType.Output("*")],
        )

    @classmethod
    def check_lazy_status(cls, boolean=True, on_true=None, on_false=None):
        if boolean and on_true is None:
            return ["on_true"]
        if not boolean and on_false is None:
            return ["on_false"]

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(kwargs["on_true"] if kwargs["boolean"] else kwargs["on_false"])


class Blocker(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy blocker",
            category="EasyUse/Logic",
            inputs=[
                io.Boolean.Input("continue", default=False),
                io.AnyType.Input("in",),
            ],
            outputs=[io.AnyType.Output("out")],
        )

    @classmethod
    def execute(cls, **kwargs):
        from comfy_execution.graph import ExecutionBlocker
        return io.NodeOutput(kwargs["in"] if kwargs["continue"] else ExecutionBlocker(None))


from comfy.sdxl_clip import SDXLClipModel, SDXLRefinerClipModel, SDXLClipG


class isMaskEmpty(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy isMaskEmpty",
            category="EasyUse/Logic",
            inputs=[io.Mask.Input("mask")],
            outputs=[io.Boolean.Output("boolean")],
        )

    @classmethod
    def execute(cls, mask):
        if mask is None or torch.all(mask == 0):
            return io.NodeOutput(True)
        return io.NodeOutput(False)


class isNone(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy isNone",
            category="EasyUse/Logic",
            inputs=[io.AnyType.Input("any")],
            outputs=[io.Boolean.Output("boolean")],
        )

    @classmethod
    def execute(cls, any):
        result = (isinstance(any, str) and any == "") or \
                 (isinstance(any, (int, float)) and any == 0) or \
                 any is None
        return io.NodeOutput(result)


class isSDXL(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy isSDXL",
            category="EasyUse/Logic",
            inputs=[
                io.Custom(io_type="PIPE_LINE").Input("optional_pipe", optional=True),
                io.Clip.Input("optional_clip", optional=True),
            ],
            outputs=[io.Boolean.Output("boolean")],
        )

    @classmethod
    def execute(cls, optional_pipe=None, optional_clip=None):
        if optional_pipe is None and optional_clip is None:
            raise Exception("[ERROR] optional_pipe or optional_clip is missing")
        clip = optional_clip if optional_clip is not None else optional_pipe["clip"]
        return io.NodeOutput(isinstance(clip.cond_stage_model, (SDXLClipModel, SDXLRefinerClipModel, SDXLClipG)))


class isFileExist(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy isFileExist",
            category="EasyUse/Logic",
            inputs=[
                io.String.Input("file_path", default=""),
                io.String.Input("file_name", default=""),
                io.String.Input("file_extension", default=""),
            ],
            outputs=[io.Boolean.Output("boolean")],
        )

    @classmethod
    def execute(cls, file_path, file_name, file_extension):
        if not file_path:
            raise Exception("file_path is missing")
        if file_name:
            file_path = os.path.join(file_path, file_name)
        if file_extension:
            file_path = file_path + "." + file_extension
        return io.NodeOutput(os.path.exists(file_path) and os.path.isfile(file_path))


from nodes import MAX_RESOLUTION
from ..config import BASE_RESOLUTIONS


class pixels(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        resolution_strings = [
            f"{width} x {height} (custom)" if width == "width" and height == "height" else f"{width} x {height}"
            for width, height in BASE_RESOLUTIONS
        ]
        return io.Schema(
            node_id="easy pixels",
            category="EasyUse/Logic",
            inputs=[
                io.Combo.Input("resolution", options=resolution_strings),
                io.Int.Input("width", default=512, min=64, max=MAX_RESOLUTION, step=8),
                io.Int.Input("height", default=512, min=64, max=MAX_RESOLUTION, step=8),
                io.Float.Input("scale", default=2.000, min=0.001, max=10, step=0.001),
                io.Boolean.Input("flip_w/h", default=False),
            ],
            outputs=[
                io.Int.Output("width_norm"),
                io.Int.Output("height_norm"),
                io.AnyType.Output("width"),
                io.AnyType.Output("height"),
                io.AnyType.Output("scale_factor"),
            ],
        )

    @classmethod
    def execute(cls, resolution, width, height, scale, **kwargs):
        if resolution not in ["自定义 x 自定义", "width x height (custom)"]:
            try:
                _width, _height = map(int, resolution.split(" x "))
                width = _width
                height = _height
            except ValueError:
                raise ValueError("Invalid base_resolution format.")
        width = width * scale
        height = height * scale
        width_norm = int(width - width % 8)
        height_norm = int(height - height % 8)
        if kwargs.get("flip_w/h", False):
            width, height = height, width
            width_norm, height_norm = height_norm, width_norm
        return io.NodeOutput(width_norm, height_norm, width, height, scale)


class xyAny(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy xyAny",
            category="EasyUse/Logic",
            is_input_list=True,
            inputs=[
                io.AnyType.Input("X"),
                io.AnyType.Input("Y"),
                io.Combo.Input("direction", options=["horizontal", "vertical"], default="horizontal"),
            ],
            outputs=[
                io.AnyType.Output("X", is_output_list=True),
                io.AnyType.Output("Y", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, X, Y, direction):
        new_x, new_y = [], []
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
        return io.NodeOutput(new_x, new_y)


class lengthAnything(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy lengthAnything",
            category="EasyUse/Logic",
            is_input_list=True,
            inputs=[io.AnyType.Input("any")],
            outputs=[io.Int.Output("length")],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, any, **kwargs):
        prompt = cls.hidden.prompt
        my_unique_id = cls.hidden.unique_id
        if isinstance(prompt, list):
            prompt = prompt[0]
        if isinstance(my_unique_id, list):
            my_unique_id = my_unique_id[0]
        my_unique_id = my_unique_id.split(".")[-1] if "." in my_unique_id else my_unique_id
        id, slot = prompt[my_unique_id]["inputs"]["any"]
        class_type = prompt[id]["class_type"]
        node_class = ALL_NODE_CLASS_MAPPINGS[class_type]
        output_is_list = node_class.OUTPUT_IS_LIST[slot] if hasattr(node_class, "OUTPUT_IS_LIST") else False
        return io.NodeOutput(len(any) if output_is_list or len(any) > 1 else len(any[0]))


class indexAnything(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy indexAnything",
            category="EasyUse/Logic",
            is_input_list=True,
            inputs=[
                io.AnyType.Input("any"),
                io.Int.Input("index", default=0, min=-1000000, max=1000000, step=1),
            ],
            outputs=[io.AnyType.Output("out")],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, any, index, **kwargs):
        if isinstance(index, list):
            index = index[0]
        prompt = cls.hidden.prompt
        my_unique_id = cls.hidden.unique_id
        if isinstance(prompt, list):
            prompt = prompt[0]
        if isinstance(my_unique_id, list):
            my_unique_id = my_unique_id[0]
        my_unique_id = my_unique_id.split(".")[-1] if "." in my_unique_id else my_unique_id
        id, slot = prompt[my_unique_id]["inputs"]["any"]
        class_type = prompt[id]["class_type"]
        node_class = ALL_NODE_CLASS_MAPPINGS[class_type]
        output_is_list = node_class.OUTPUT_IS_LIST[slot] if hasattr(node_class, "OUTPUT_IS_LIST") else False

        def normalize_index(idx, length):
            if idx < 0:
                idx = length + idx
            return min(max(0, idx), length - 1)

        if output_is_list or len(any) > 1:
            return io.NodeOutput(any[normalize_index(index, len(any))])
        elif isinstance(any[0], torch.Tensor):
            idx = normalize_index(index, any[0].shape[0])
            return io.NodeOutput(any[0][idx:idx+1].clone())
        else:
            if hasattr(any[0], "__len__") and len(any[0]) > 0:
                return io.NodeOutput(any[0][normalize_index(index, len(any[0]))])
            return io.NodeOutput(any[0])


class batchAnything(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy batchAnything",
            category="EasyUse/Logic",
            inputs=[
                io.AnyType.Input("any_1"),
                io.AnyType.Input("any_2"),
            ],
            outputs=[io.AnyType.Output("batch")],
        )

    @classmethod
    def latentBatch(cls, any_1, any_2):
        samples_out = any_1.copy()
        s1 = any_1["samples"]
        s2 = any_2["samples"]
        if s1.shape[1:] != s2.shape[1:]:
            s2 = comfy.utils.common_upscale(s2, s1.shape[3], s1.shape[2], "bilinear", "center")
        s = torch.cat((s1, s2), dim=0)
        samples_out["samples"] = s
        samples_out["batch_index"] = any_1.get("batch_index", list(range(s1.shape[0]))) + \
                                     any_2.get("batch_index", list(range(s2.shape[0])))
        return samples_out

    @classmethod
    def execute(cls, any_1, any_2):
        if isinstance(any_1, torch.Tensor) or isinstance(any_2, torch.Tensor):
            if any_1 is None:
                return io.NodeOutput(any_2)
            if any_2 is None:
                return io.NodeOutput(any_1)
            if any_1.shape[1:] != any_2.shape[1:]:
                any_2 = comfy.utils.common_upscale(any_2.movedim(-1, 1), any_1.shape[2], any_1.shape[1], "bilinear", "center").movedim(1, -1)
            return io.NodeOutput(torch.cat((any_1, any_2), 0))
        elif isinstance(any_1, (str, float, int)):
            if any_2 is None:
                return io.NodeOutput(any_1)
            elif isinstance(any_2, tuple):
                return io.NodeOutput(any_2 + (any_1,))
            elif isinstance(any_2, list):
                return io.NodeOutput(any_2 + [any_1])
            return io.NodeOutput([any_1, any_2])
        elif isinstance(any_2, (str, float, int)):
            if any_1 is None:
                return io.NodeOutput(any_2)
            elif isinstance(any_1, tuple):
                return io.NodeOutput(any_1 + (any_2,))
            elif isinstance(any_1, list):
                return io.NodeOutput(any_1 + [any_2])
            return io.NodeOutput([any_2, any_1])
        elif isinstance(any_1, dict) and "samples" in any_1:
            if any_2 is None:
                return io.NodeOutput(any_1)
            if isinstance(any_2, dict) and "samples" in any_2:
                return io.NodeOutput(cls.latentBatch(any_1, any_2))
        elif isinstance(any_2, dict) and "samples" in any_2:
            if any_1 is None:
                return io.NodeOutput(any_2)
            if isinstance(any_1, dict) and "samples" in any_1:
                return io.NodeOutput(cls.latentBatch(any_2, any_1))
        if any_1 is None:
            return io.NodeOutput(any_2)
        if any_2 is None:
            return io.NodeOutput(any_1)
        return io.NodeOutput(any_1 + any_2)


class convertAnything(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy convertAnything",
            category="EasyUse/Logic",
            is_output_node=True,
            inputs=[
                io.AnyType.Input("*"),
                io.Combo.Input("output_type", options=["string", "int", "float", "boolean"], default="string"),
            ],
            outputs=[io.AnyType.Output("output")],
        )

    @classmethod
    def execute(cls, **kwargs):
        anything = kwargs["*"]
        output_type = kwargs["output_type"]
        converters = {"string": str, "int": int, "float": float, "boolean": bool}
        return io.NodeOutput(converters[output_type](anything))


class showAnything(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy showAnything",
            category="EasyUse/Logic",
            is_input_list=True,
            is_output_node=True,
            inputs=[io.AnyType.Input("anything", optional=True)],
            outputs=[io.AnyType.Output("output")],
            hidden=[io.Hidden.unique_id, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def execute(cls, **kwargs):
        unique_id = cls.hidden.unique_id
        extra_pnginfo = cls.hidden.extra_pnginfo

        values = []
        if "anything" in kwargs:
            for val in kwargs["anything"]:
                if isinstance(val, str):
                    values.append(val)
                elif isinstance(val, (int, float, bool)):
                    values.append(str(val))
                elif isinstance(val, list) and len(val) <= 30:
                    values = val
                elif val is not None:
                    try:
                        values.append(json.dumps(val, indent=4, ensure_ascii=False))
                    except Exception:
                        try:
                            values.append(str(val))
                        except Exception:
                            raise Exception("source exists, but could not be serialized.")

        if extra_pnginfo and isinstance(extra_pnginfo, dict) and "workflow" in extra_pnginfo:
            _uid = unique_id[0] if isinstance(unique_id, list) else unique_id
            node = next((x for x in extra_pnginfo["workflow"]["nodes"] if str(x["id"]) == _uid), None)
            if node:
                node["widgets_values"] = [values]

        result_val = values[0] if (isinstance(values, list) and len(values) == 1) else values
        return io.NodeOutput(result_val, ui={"text": values})


class showTensorShape(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy showTensorShape",
            category="EasyUse/Logic",
            is_output_node=True,
            inputs=[io.AnyType.Input("tensor")],
            outputs=[],
            hidden=[io.Hidden.unique_id, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def execute(cls, tensor, **kwargs):
        shapes = []

        def tensorShape(t):
            if isinstance(t, dict):
                for k in t:
                    tensorShape(t[k])
            elif isinstance(t, list):
                for item in t:
                    tensorShape(item)
            elif hasattr(t, "shape"):
                shapes.append(list(t.shape))

        tensorShape(tensor)
        return io.NodeOutput(ui={"text": shapes})


class stringToIntList(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy stringToIntList",
            category="EasyUse/Logic",
            inputs=[io.String.Input("string", default="1, 2, 3", multiline=True)],
            outputs=[io.Int.Output("INT")],
        )

    @classmethod
    def execute(cls, string):
        return io.NodeOutput([int(x.strip()) for x in string.split(",")])


class stringToFloatList(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy stringToFloatList",
            category="EasyUse/Logic",
            inputs=[io.String.Input("string", default="1, 2, 3", multiline=True)],
            outputs=[io.Float.Output("FLOAT")],
        )

    @classmethod
    def execute(cls, string):
        return io.NodeOutput([float(x.strip()) for x in string.split(",")])


class stringJoinLines(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy stringJoinLines",
            category="EasyUse/Logic",
            inputs=[
                io.String.Input("string", default="", multiline=True),
                io.String.Input("delimiter", default=" | "),
            ],
            outputs=[io.String.Output("STRING")],
        )

    @classmethod
    def execute(cls, string, delimiter):
        lines = [line.strip() for line in string.split("\n") if line.strip()]
        return io.NodeOutput(delimiter.join(lines))


class outputToList(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy outputToList",
            category="EasyUse/Logic",
            inputs=[io.AnyType.Input("tuple")],
            outputs=[io.AnyType.Output("list", is_output_list=True)],
        )

    @classmethod
    def execute(cls, tuple):
        return io.NodeOutput(tuple)


class cleanGPUUsed(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy cleanGpuUsed",
            category="EasyUse/Logic",
            is_output_node=True,
            inputs=[io.AnyType.Input("anything")],
            outputs=[io.AnyType.Output("output")],
            hidden=[io.Hidden.unique_id, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def execute(cls, anything, **kwargs):
        cleanGPUUsedForce()
        remove_cache("*")
        return io.NodeOutput(anything)


class clearCacheKey(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy clearCacheKey",
            category="EasyUse/Logic",
            is_output_node=True,
            inputs=[
                io.AnyType.Input("anything"),
                io.String.Input("cache_key", default="*"),
            ],
            outputs=[io.AnyType.Output("output")],
            hidden=[io.Hidden.unique_id, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def execute(cls, anything, cache_key, **kwargs):
        remove_cache(cache_key)
        return io.NodeOutput(anything)


class clearCacheAll(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy clearCacheAll",
            category="EasyUse/Logic",
            is_output_node=True,
            inputs=[io.AnyType.Input("anything")],
            outputs=[io.AnyType.Output("output")],
            hidden=[io.Hidden.unique_id, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def execute(cls, anything, **kwargs):
        remove_cache("*")
        return io.NodeOutput(anything)


class saveText(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy saveText",
            category="EasyUse/Logic",
            is_output_node=True,
            inputs=[
                io.String.Input("text", default="", force_input=True),
                io.String.Input("output_file_path", multiline=False, default=""),
                io.String.Input("file_name", multiline=False, default=""),
                io.Combo.Input("file_extension", options=["txt", "csv"]),
                io.Boolean.Input("overwrite", default=True),
                io.Image.Input("image", optional=True),
            ],
            outputs=[
                io.String.Output("text"),
                io.Image.Output("image"),
            ],
        )

    @classmethod
    def save_image(cls, images, filename_prefix="", extension="png", quality=100, prompt=None,
                   extra_pnginfo=None, delimiter="_", filename_number_start="true", number_padding=4,
                   overwrite_mode="prefix_as_filename", output_path="", show_history="true",
                   show_previews="true", embed_workflow="true", lossless_webp=False):
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            if extension == "webp":
                img_exif = img.getexif()
                workflow_metadata = ""
                if prompt is not None:
                    img_exif[0x010f] = "Prompt:" + json.dumps(prompt)
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        workflow_metadata += json.dumps(extra_pnginfo[x])
                img_exif[0x010e] = "Workflow:" + workflow_metadata
                exif_data = img_exif.tobytes()
            else:
                metadata = PngInfo()
                if embed_workflow == "true":
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                exif_data = metadata
            file = f"{filename_prefix}.{extension}"
            try:
                output_file = os.path.abspath(os.path.join(output_path, file))
                if extension in ["jpg", "jpeg"]:
                    img.save(output_file, quality=quality, optimize=True)
                elif extension == "webp":
                    img.save(output_file, quality=quality, lossless=lossless_webp, exif=exif_data)
                elif extension == "png":
                    img.save(output_file, pnginfo=exif_data, optimize=True)
                elif extension == "bmp":
                    img.save(output_file)
                elif extension == "tiff":
                    img.save(output_file, quality=quality, optimize=True)
                else:
                    img.save(output_file, pnginfo=exif_data, optimize=True)
            except Exception as e:
                print(e)

    @classmethod
    def execute(cls, text, output_file_path, file_name, file_extension, overwrite, image=None, **kwargs):
        if isinstance(file_name, list):
            file_name = file_name[0]

        if output_file_path == "" or file_name == "":
            log_node_warn("Save Text", "No file details found. No file output.")
            return io.NodeOutput(text, None)

        filepath = os.path.join(output_file_path, file_name) + "." + file_extension
        if not os.path.exists(output_file_path):
            os.makedirs(output_file_path)

        file_mode = "w" if overwrite else "a"
        log_node_info("Save Text", f"Saving to {filepath}")

        if file_extension == "csv":
            with open(filepath, file_mode, newline="", encoding="utf-8") as csv_file:
                csv_writer = csv.writer(csv_file)
                for line in [i.strip() for i in text.split("\n")]:
                    csv_writer.writerow([line])
        else:
            with open(filepath, file_mode, newline="", encoding="utf-8") as text_file:
                for line in text:
                    text_file.write(line)

        result_image = None
        if image is not None:
            imagepath = os.path.join(output_file_path, file_name)
            index = 1
            if not overwrite:
                while os.path.exists(filepath):
                    imagepath = os.path.join(output_file_path, file_name) + "_" + str(index)
                    index += 1

            output_dir = folder_paths.output_directory
            output_path_val = "" if output_file_path in [None, "", "none", "."] else output_file_path
            if not os.path.isabs(output_file_path):
                output_path_val = os.path.join(output_dir, output_path_val)
            if output_path_val.strip():
                if not os.path.isabs(output_path_val):
                    output_path_val = os.path.join(folder_paths.output_directory, output_path_val)
                if not os.path.exists(output_path_val.strip()):
                    print(f"The path `{output_path_val.strip()}` does not exist! Creating directory.")
                    os.makedirs(output_path_val, exist_ok=True)

            images_tensor = torch.cat([image], dim=0)
            cls.save_image(images_tensor, imagepath, "png", 100, None, None,
                           filename_number_start="true", output_path=output_path_val,
                           delimiter="_", number_padding=4, lossless_webp=False)
            log_node_info("Save Text", f"Saving Image to {imagepath}")
            result_image = image

        return io.NodeOutput(text, result_image)


class sleep(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy sleep",
            category="EasyUse/Logic",
            inputs=[
                io.AnyType.Input("any"),
                io.Float.Input("delay", default=1.0, min=0.0, max=1000000, step=0.1),
            ],
            outputs=[io.AnyType.Output("out")],
        )

    @classmethod
    def execute(cls, any, delay):
        time.sleep(delay)
        return io.NodeOutput(any)


NODE_CLASS_MAPPINGS = {
    "easy string": String,
    "easy int": Int,
    "easy rangeInt": RangeInt,
    "easy float": Float,
    "easy rangeFloat": RangeFloat,
    "easy boolean": Boolean,
    "easy mathString": mathStringOperation,
    "easy mathInt": mathIntOperation,
    "easy mathFloat": mathFloatOperation,
    "easy simpleMath": simpleMath,
    "easy simpleMathDual": simpleMathDual,
    "easy compare": Compare,
    "easy imageSwitch": imageSwitch,
    "easy textSwitch": textSwitch,
    "easy imageIndexSwitch": imageIndexSwitch,
    "easy textIndexSwitch": textIndexSwitch,
    "easy conditioningIndexSwitch": conditioningIndexSwitch,
    "easy anythingIndexSwitch": anythingIndexSwitch,
    "easy ab": ab,
    "easy anythingInversedSwitch": anythingInversedSwitch,
    "easy whileLoopStart": whileLoopStart,
    "easy whileLoopEnd": whileLoopEnd,
    "easy forLoopStart": forLoopStart,
    "easy forLoopEnd": forLoopEnd,
    "easy blocker": Blocker,
    "easy ifElse": IfElse,
    "easy isMaskEmpty": isMaskEmpty,
    "easy isNone": isNone,
    "easy isSDXL": isSDXL,
    "easy isFileExist": isFileExist,
    "easy stringToIntList": stringToIntList,
    "easy stringToFloatList": stringToFloatList,
    "easy stringJoinLines": stringJoinLines,
    "easy outputToList": outputToList,
    "easy pixels": pixels,
    "easy xyAny": xyAny,
    "easy lengthAnything": lengthAnything,
    "easy indexAnything": indexAnything,
    "easy batchAnything": batchAnything,
    "easy convertAnything": convertAnything,
    "easy showAnything": showAnything,
    "easy showTensorShape": showTensorShape,
    "easy clearCacheKey": clearCacheKey,
    "easy clearCacheAll": clearCacheAll,
    "easy cleanGpuUsed": cleanGPUUsed,
    "easy saveText": saveText,
    "easy sleep": sleep,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "easy string": "String",
    "easy int": "Int",
    "easy rangeInt": "Range(Int)",
    "easy float": "Float",
    "easy rangeFloat": "Range(Float)",
    "easy boolean": "Boolean",
    "easy compare": "Compare",
    "easy mathString": "Math String",
    "easy mathInt": "Math Int",
    "easy mathFloat": "Math Float",
    "easy simpleMath": "Simple Math",
    "easy simpleMathDual": "Simple Math Dual",
    "easy imageSwitch": "Image Switch",
    "easy textSwitch": "Text Switch",
    "easy imageIndexSwitch": "Image Index Switch",
    "easy textIndexSwitch": "Text Index Switch",
    "easy conditioningIndexSwitch": "Conditioning Index Switch",
    "easy anythingIndexSwitch": "Any Index Switch",
    "easy ab": "A or B",
    "easy anythingInversedSwitch": "Any Inversed Switch",
    "easy whileLoopStart": "While Loop Start",
    "easy whileLoopEnd": "While Loop End",
    "easy forLoopStart": "For Loop Start",
    "easy forLoopEnd": "For Loop End",
    "easy ifElse": "If else",
    "easy blocker": "Blocker",
    "easy isMaskEmpty": "Is Mask Empty",
    "easy isNone": "Is None",
    "easy isSDXL": "Is SDXL",
    "easy isFileExist": "Is File Exist",
    "easy stringToIntList": "String to Int List",
    "easy stringToFloatList": "String to Float List",
    "easy stringJoinLines": "String Join Lines",
    "easy outputToList": "Output to List",
    "easy pixels": "Pixels W/H Norm",
    "easy xyAny": "XY Any",
    "easy lengthAnything": "Length Any",
    "easy indexAnything": "Index Any",
    "easy batchAnything": "Batch Any",
    "easy convertAnything": "Convert Any",
    "easy showAnything": "Show Any",
    "easy showTensorShape": "Show Tensor Shape",
    "easy clearCacheKey": "Clear Cache Key",
    "easy clearCacheAll": "Clear Cache All",
    "easy cleanGpuUsed": "Clean VRAM Used",
    "easy saveText": "Save Text",
    "easy sleep": "Sleep",
}
