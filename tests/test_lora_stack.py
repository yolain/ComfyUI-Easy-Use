import ast
from pathlib import Path
import unittest


ADAPTER_PATH = Path(__file__).parents[1] / "py" / "nodes" / "adapter.py"


def load_apply_lora_stack(easy_cache):
    tree = ast.parse(ADAPTER_PATH.read_text())
    node = next(
        item for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "applyLoraStack"
    )
    namespace = {"easyCache": easy_cache}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ADAPTER_PATH), "exec"), namespace)
    return namespace["applyLoraStack"]()


class FakeEasyCache:
    def __init__(self):
        self.calls = []

    def load_lora(self, lora, model, clip, use_cache):
        self.calls.append((lora, model, clip, use_cache))
        index = len(self.calls)
        return f"model-{index}", f"clip-{index}"


class LoraStackTests(unittest.TestCase):
    def test_each_lora_receives_the_previous_clip_result(self):
        easy_cache = FakeEasyCache()
        node = load_apply_lora_stack(easy_cache)

        result = node.apply(
            [("first.safetensors", 0.7, 0.4), ("second.safetensors", 0.5, 0.3)],
            "base-model",
            "base-clip",
        )

        self.assertEqual([call[2] for call in easy_cache.calls], ["base-clip", "clip-1"])
        self.assertEqual(result, ("model-2", "clip-2"))

    def test_empty_stack_preserves_optional_clip(self):
        node = load_apply_lora_stack(FakeEasyCache())

        self.assertEqual(node.apply([], "base-model", "base-clip"), ("base-model", "base-clip"))


if __name__ == "__main__":
    unittest.main()
