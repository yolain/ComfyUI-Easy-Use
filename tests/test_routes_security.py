import ast
import asyncio
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from functools import lru_cache
from types import SimpleNamespace
from urllib.parse import urlsplit
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image, UnidentifiedImageError


ROUTES_PATH = Path(__file__).parents[1] / "py" / "routes.py"


def load_handlers(folder_paths, get_metadata):
    """Load the actual handlers without importing ComfyUI's GPU dependencies."""
    names = {
        "_same_origin_request", "get_reboot_token", "reboot", "_model_sha256",
        "load_metadata", "save_notes", "save_preview",
    }
    tree = ast.parse(ROUTES_PATH.read_text())
    functions = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            node.decorator_list = []
            functions.append(node)
    namespace = {
        "os": os, "sys": sys, "hashlib": hashlib, "hmac": __import__("hmac"),
        "json": json, "shutil": shutil, "tempfile": tempfile,
        "lru_cache": lru_cache, "urlsplit": urlsplit, "web": web,
        "Image": Image, "UnidentifiedImageError": UnidentifiedImageError,
        "folder_paths": folder_paths, "getMetadata": get_metadata,
        "_reboot_token": "test-reboot-token",
        "_PREVIEW_FORMATS": {
            ".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG",
            ".webp": "WEBP", ".gif": "GIF",
        },
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(ROUTES_PATH), "exec"), namespace)
    return SimpleNamespace(**namespace)


class SecurityRouteTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name)
        self.model_dir = self.root / "models"
        self.temp_dir = self.root / "temp"
        self.model_dir.mkdir()
        self.temp_dir.mkdir()
        self.model_path = self.model_dir / "sample.safetensors"
        self.model_path.write_bytes(b"model data")
        paths = SimpleNamespace(
            get_filename_list=lambda kind: [self.model_path.name],
            get_full_path=lambda kind, name: str(self.model_path),
            get_directory_by_type=lambda kind: str(self.temp_dir),
        )
        self.handlers = load_handlers(
            paths,
            lambda path: json.dumps({"__metadata__": {"easyuse.notes": "<img onerror=alert(1)>"}}),
        )

    def request(self, name="loras/sample.safetensors", filename="preview.png", **body):
        payload = {"type": "temp", "filename": filename, **body}
        return SimpleNamespace(
            match_info={"name": name},
            json=lambda: asyncio.sleep(0, result=payload),
            headers={}, host="localhost:8188",
        )

    def test_save_rejects_script_and_custom_node_target(self):
        (self.temp_dir / "payload.py").write_text("print('sentinel')")
        response = asyncio.run(self.handlers.save_preview(self.request(filename="payload.py")))
        self.assertEqual(response.status, 400)
        response = asyncio.run(self.handlers.save_preview(
            self.request(name="custom_nodes/package/__init__.py", filename="payload.py")
        ))
        self.assertEqual(response.status, 400)

    def test_save_accepts_real_image_and_rejects_disguised_script(self):
        Image.new("RGB", (1, 1)).save(self.temp_dir / "preview.png")
        response = asyncio.run(self.handlers.save_preview(self.request()))
        self.assertEqual(response.status, 200)
        with Image.open(self.model_dir / "sample.png") as saved:
            self.assertEqual(saved.format, "PNG")

        (self.temp_dir / "preview.png").write_text("print('sentinel')")
        response = asyncio.run(self.handlers.save_preview(self.request()))
        self.assertEqual(response.status, 400)
        with Image.open(self.model_dir / "sample.png") as saved:
            self.assertEqual(saved.format, "PNG")

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_save_does_not_follow_preview_symlink(self):
        Image.new("RGB", (1, 1)).save(self.temp_dir / "preview.png")
        protected = self.root / "protected.txt"
        protected.write_text("untouched")
        os.symlink(protected, self.model_dir / "sample.png")
        response = asyncio.run(self.handlers.save_preview(self.request()))
        self.assertEqual(response.status, 400)
        self.assertEqual(protected.read_text(), "untouched")

    def test_metadata_ignores_forged_hash_sidecar(self):
        (self.model_dir / "sample.sha256").write_text("0" * 64)
        response = asyncio.run(self.handlers.load_metadata(self.request()))
        self.assertEqual(
            json.loads(response.text)["easyuse.sha256"],
            hashlib.sha256(self.model_path.read_bytes()).hexdigest(),
        )

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_notes_reject_custom_nodes_and_do_not_follow_symlinks(self):
        request = self.request(name="custom_nodes/package/__init__.py")
        request.text = lambda: asyncio.sleep(0, result="new notes")
        self.assertEqual(asyncio.run(self.handlers.save_notes(request)).status, 400)

        protected = self.root / "protected.txt"
        protected.write_text("untouched")
        os.symlink(protected, self.model_dir / "sample.txt")
        request = self.request()
        request.text = lambda: asyncio.sleep(0, result="new notes")
        self.assertEqual(asyncio.run(self.handlers.save_notes(request)).status, 200)
        self.assertEqual(protected.read_text(), "untouched")
        self.assertEqual((self.model_dir / "sample.txt").read_text(), "new notes")

    def test_reboot_requires_token_and_same_origin(self):
        self.assertTrue(inspect.iscoroutinefunction(self.handlers.get_reboot_token))
        self.assertTrue(inspect.iscoroutinefunction(self.handlers.reboot))
        request = self.request()
        request.headers = {"Sec-Fetch-Site": "cross-site"}
        self.assertEqual(asyncio.run(self.handlers.get_reboot_token(request)).status, 403)
        request.headers = {"Sec-Fetch-Site": "same-origin"}
        self.assertEqual(json.loads(asyncio.run(self.handlers.get_reboot_token(request)).text)["token"], "test-reboot-token")
        request.headers = {}
        with patch.object(self.handlers.os, "execv", return_value="restarted") as restart:
            self.assertEqual(asyncio.run(self.handlers.reboot(request)).status, 403)
            request.headers = {"X-EasyUse-Reboot-Token": "test-reboot-token", "Origin": "http://other.test"}
            self.assertEqual(asyncio.run(self.handlers.reboot(request)).status, 403)
            restart.assert_not_called()
            request.headers["Origin"] = "http://localhost:8188"
            request.headers["Sec-Fetch-Site"] = "same-site"
            self.assertEqual(asyncio.run(self.handlers.reboot(request)).status, 403)
            request.headers["Sec-Fetch-Site"] = "same-origin"
            self.assertEqual(asyncio.run(self.handlers.reboot(request)), "restarted")
            restart.assert_called_once()

    def test_reboot_routes_return_http_responses(self):
        async def exercise_routes():
            app = web.Application()
            app.router.add_get("/easyuse/reboot-token", self.handlers.get_reboot_token)
            app.router.add_post("/easyuse/reboot", self.handlers.reboot)
            async with TestClient(TestServer(app)) as client:
                token_response = await client.get("/easyuse/reboot-token")
                self.assertEqual(token_response.status, 200)
                self.assertEqual((await token_response.json())["token"], "test-reboot-token")
                reboot_response = await client.post("/easyuse/reboot")
                self.assertEqual(reboot_response.status, 403)

        asyncio.run(exercise_routes())


if __name__ == "__main__":
    unittest.main()
