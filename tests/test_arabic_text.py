import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path


PLUGIN_ROOT = Path(__file__).parents[1]
UTIL_PATH = PLUGIN_ROOT / "py" / "nodes" / "util.py"
TEXT_PATH = PLUGIN_ROOT / "py" / "libs" / "text.py"
FONT_PATH = str(PLUGIN_ROOT / "resources" / "wenquan.ttf")


def load_module(name, path, package=None):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if package is not None:
        module.__package__ = package
    spec.loader.exec_module(module)
    return module


def make_package(name, **attrs):
    package = types.ModuleType(name)
    package.__path__ = []
    for key, value in attrs.items():
        setattr(package, key, value)
    return package


@contextmanager
def util_module():
    modules = {
        "folder_paths": make_package("folder_paths"),
        "fake_py": make_package("fake_py"),
        "fake_py.libs": make_package("fake_py.libs"),
        "fake_py.nodes": make_package("fake_py.nodes"),
        "fake_py.libs.utils": make_package("fake_py.libs.utils", AlwaysEqualProxy=str),
        "fake_py.libs.text": load_module("fake_py.libs.text", TEXT_PATH),
    }
    added = [name for name in modules if name not in sys.modules]
    sys.modules.update(modules)
    try:
        yield load_module("fake_py.nodes.util", UTIL_PATH, "fake_py.nodes")
    finally:
        for name in added:
            sys.modules.pop(name, None)


@unittest.skipIf(importlib.util.find_spec("khatt") is None, "khatt is not installed")
class ArabicTableTests(unittest.TestCase):
    def drawn(self, markdown):
        """The text the table draws, with the x it draws it at."""
        with util_module() as util:
            calls = []
            draw_text = util.draw_text

            def record(draw, xy, text, **kwargs):
                calls.append((text, xy[0]))
                draw_text(draw, xy, text, **kwargs)

            util.draw_text = record
            util._markdown_table_to_image(markdown, FONT_PATH)
        return dict(calls)

    def test_arabic_header_lays_the_table_out_right_to_left(self):
        drawn = self.drawn("| الاسم | العمر |\n|---|---|\n| ali | 30 |")
        self.assertGreater(drawn["الاسم"], drawn["العمر"])
        self.assertGreater(drawn["ali"], drawn["30"])

    def test_latin_header_keeps_left_to_right(self):
        drawn = self.drawn("| Name | Age |\n|---|---|\n| علي | 30 |")
        self.assertLess(drawn["Name"], drawn["Age"])
        self.assertLess(drawn["علي"], drawn["30"])

    def test_arabic_reaches_khatt_as_typed(self):
        drawn = self.drawn("| ملاحظة |\n|---|\n| **نص واضح** لكنه أبطأ |")
        self.assertIn("نص واضح", drawn)
        self.assertIn("لكنه أبطأ", drawn)
        # the bold part comes first, so it is drawn to the right of the rest
        self.assertGreater(drawn["نص واضح"], drawn["لكنه أبطأ"])


if __name__ == "__main__":
    unittest.main()
