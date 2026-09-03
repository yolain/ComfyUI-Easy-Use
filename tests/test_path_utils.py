import importlib.util
import os
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).parents[1] / "py" / "libs" / "path_utils.py"
SPEC = importlib.util.spec_from_file_location("easyuse_path_utils", MODULE_PATH)
path_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(path_utils)


class ResolveOutputFilePathTests(unittest.TestCase):
    def test_relative_subdirectory_is_resolved_under_output_root(self):
        with tempfile.TemporaryDirectory() as output_root:
            result = path_utils.resolve_output_file_path(
                output_root, "metadata", "prompt", "txt"
            )

            self.assertEqual(
                result,
                os.path.join(os.path.realpath(output_root), "metadata", "prompt.txt"),
            )

    def test_absolute_directory_inside_output_root_is_allowed(self):
        with tempfile.TemporaryDirectory() as output_root:
            inside = os.path.join(output_root, "metadata")

            result = path_utils.resolve_output_file_path(
                output_root, inside, "prompt", "txt"
            )

            self.assertEqual(result, os.path.join(inside, "prompt.txt"))

    def test_absolute_directory_outside_output_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as output_root:
            with tempfile.TemporaryDirectory() as outside:
                with self.assertRaises(ValueError):
                    path_utils.resolve_output_file_path(
                        output_root, outside, "marker", "txt"
                    )

    def test_output_directory_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as output_root:
            with self.assertRaises(ValueError):
                path_utils.resolve_output_file_path(
                    output_root, "../outside", "marker", "txt"
                )

    def test_file_name_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as output_root:
            with self.assertRaises(ValueError):
                path_utils.resolve_output_file_path(
                    output_root, ".", "../../marker", "txt"
                )

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as output_root:
            with tempfile.TemporaryDirectory() as outside:
                os.symlink(outside, os.path.join(output_root, "linked"))

                with self.assertRaises(ValueError):
                    path_utils.resolve_output_file_path(
                        output_root, "linked", "marker", "txt"
                    )


if __name__ == "__main__":
    unittest.main()
