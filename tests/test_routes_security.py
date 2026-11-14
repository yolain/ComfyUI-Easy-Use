import ast
import asyncio
import hashlib
import inspect
import io
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
from unittest.mock import AsyncMock, patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image, UnidentifiedImageError


ROUTES_PATH = Path(__file__).parents[1] / "py" / "routes.py"


def load_handlers(folder_paths, get_metadata):
    """Load the actual handlers without importing ComfyUI's GPU dependencies."""
    names = {
        "_same_origin_request", "get_reboot_token", "reboot", "_model_sha256",
        "load_metadata", "save_notes", "save_preview",
        "_resolve_style_path", "getStylesImage", "getStylesList",
        "_read_model_metadata", "_read_model_notes", "download_model",
    }
    tree = ast.parse(ROUTES_PATH.read_text())
    functions = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            node.decorator_list = []
            functions.append(node)
    namespace = {
        "os": os, "sys": sys, "hashlib": hashlib, "hmac": __import__("hmac"),
        "json": json, "shutil": shutil, "tempfile": tempfile, "asyncio": asyncio,
        "lru_cache": lru_cache, "urlsplit": urlsplit, "web": web,
        "Image": Image, "UnidentifiedImageError": UnidentifiedImageError,
        "folder_paths": folder_paths, "getMetadata": get_metadata,
        "_reboot_token": "test-reboot-token",
        "FOOOCUS_STYLES_DIR": str(folder_paths.styles_dir),
        "FOOOCUS_STYLES_SAMPLES": "https://example.test/styles/",
        "RESOURCES_DIR": str(folder_paths.styles_dir / "resources"),
        "_metadata_slots": asyncio.Semaphore(2),
        "_MAX_NOTES_BYTES": 1024 * 1024,
        "_download_slots": asyncio.Semaphore(2),
        "download_model_file": AsyncMock(), "UnsafeDownload": ValueError,
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
        self.styles_dir = self.root / "styles"
        self.samples_dir = self.styles_dir / "samples"
        self.samples_dir.mkdir(parents=True)
        self.model_path = self.model_dir / "sample.safetensors"
        self.model_path.write_bytes(b"model data")
        paths = SimpleNamespace(
            styles_dir=self.styles_dir,
            models_dir=str(self.model_dir),
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

    def test_style_images_respect_directory_boundary_over_http(self):
        sample = self.samples_dir / "nested" / "preview.jpg"
        sample.parent.mkdir()
        sample.write_bytes(b"sample preview")
        (self.styles_dir / "preview.jpg").write_bytes(b"root preview")
        (self.styles_dir / "root-only.jpg").write_bytes(b"root-only preview")
        (self.samples_dir / "preview.jpg").write_bytes(b"preferred preview")
        protected = self.root / "private.jpg"
        protected.write_bytes(b"private data")
        sibling = self.root / "styles-private"
        sibling.mkdir()
        (sibling / "private.jpg").write_bytes(b"private sibling data")
        os.symlink(protected, self.samples_dir / "linked.jpg")
        os.symlink(sibling, self.styles_dir / "linked-dir")

        async def exercise_routes():
            app = web.Application()
            app.router.add_get("/easyuse/prompt/styles/image", self.handlers.getStylesImage)
            async with TestClient(TestServer(app)) as client:
                for query, expected in [
                    ({"path": "nested/preview.jpg"}, b"sample preview"),
                    ({"path": "preview.jpg"}, b"preferred preview"),
                    ({"path": "root-only.jpg"}, b"root-only preview"),
                    ({"path": "samples/nested/preview.jpg"}, b"sample preview"),
                    ({"name": "nested/preview"}, b"sample preview"),
                ]:
                    with self.subTest(query=query):
                        response = await client.get("/easyuse/prompt/styles/image", params=query)
                        self.assertEqual(response.status, 200)
                        self.assertEqual(await response.read(), expected)
                for query in [
                    {"path": "../../private.jpg"},
                    {"path": "../styles-private/private.jpg"},
                    {"path": str(protected)},
                    {"path": "linked.jpg"},
                    {"path": "linked-dir/private.jpg"},
                    {"path": "bad\u0000.jpg"},
                    {"name": "../../private"},
                    {"name": str(protected.with_suffix(""))},
                    {"name": "linked"},
                    {"name": "bad\u0000"},
                ]:
                    with self.subTest(query=query):
                        response = await client.get("/easyuse/prompt/styles/image", params=query)
                        self.assertEqual(response.status, 403)
                response = await client.get("/easyuse/prompt/styles/image", params={"path": "missing.jpg"})
                self.assertEqual(response.status, 400)
                response = await client.get("/easyuse/prompt/styles/image", params={"name": "missing", "styles_name": "fooocus_styles"})
                self.assertEqual(response.status, 200)
                self.assertEqual(await response.text(), "https://example.test/styles/missing.jpg")
                self.samples_dir.rename(self.styles_dir / "disabled-samples")
                response = await client.get("/easyuse/prompt/styles/image", params={"name": "missing", "styles_name": "fooocus_styles"})
                self.assertEqual(response.status, 200)
                self.assertEqual(await response.text(), "https://example.test/styles/missing.jpg")
                response = await client.get("/easyuse/prompt/styles/image", params={"name": "../../private", "styles_name": "fooocus_styles"})
                self.assertEqual(response.status, 403)

        asyncio.run(exercise_routes())

    def test_style_lists_reject_escapes_and_preserve_resource_fallback(self):
        resources = self.styles_dir / "resources"
        resources.mkdir()
        data = [{"name": "test-style", "prompt": "safe prompt"}]
        (resources / "fooocus_styles.json").write_text(json.dumps(data))
        (self.styles_dir / "custom.json").write_text(json.dumps(data))
        (self.root / "outside.json").write_text(json.dumps(data))
        os.symlink(self.root / "outside.json", self.styles_dir / "linked.json")
        os.symlink(self.root / "outside.json", self.styles_dir / "custom_cn.json")

        def request(name):
            return SimpleNamespace(rel_url=SimpleNamespace(query={"name": name}))

        for name in ("../outside", str(self.root / "outside"), "linked", "custom", "bad\x00"):
            with self.subTest(name=name):
                self.assertEqual(asyncio.run(self.handlers.getStylesList(request(name))).status, 403)
        (self.styles_dir / "custom_cn.json").unlink()
        for name in ("custom", "fooocus_styles"):
            response = asyncio.run(self.handlers.getStylesList(request(name)))
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.text)[0]["prompt"], "safe prompt")

    def test_metadata_runs_off_event_loop_and_limits_concurrency(self):
        import threading
        started, release = threading.Event(), threading.Event()
        namespace = self.handlers.load_metadata.__globals__
        original = namespace["_read_model_metadata"]

        def slow_metadata(path):
            started.set()
            release.wait(2)
            return original(path)

        async def exercise():
            with patch.dict(namespace, {"_read_model_metadata": slow_metadata, "_metadata_slots": asyncio.Semaphore(1)}):
                task = asyncio.create_task(self.handlers.load_metadata(self.request()))
                try:
                    for _ in range(100):
                        if started.is_set():
                            break
                        await asyncio.sleep(.005)
                    self.assertTrue(started.is_set())
                    self.assertFalse(task.done())
                    self.assertEqual((await self.handlers.load_metadata(self.request())).status, 429)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                    self.assertEqual((await self.handlers.load_metadata(self.request())).status, 429)
                finally:
                    release.set()
                for _ in range(100):
                    if not namespace["_metadata_slots"].locked():
                        break
                    await asyncio.sleep(.005)
                self.assertEqual((await self.handlers.load_metadata(self.request())).status, 200)

        asyncio.run(exercise())

    def test_download_awaits_safe_downloader_and_limits_concurrency(self):
        namespace = self.handlers.download_model.__globals__

        async def exercise():
            started, release = asyncio.Event(), asyncio.Event()

            async def downloader(url, path):
                started.set()
                await release.wait()

            request = SimpleNamespace(post=lambda: asyncio.sleep(0, result={
                "url": "https://example.test/model.bin", "local_dir": "loras",
            }))
            with patch.dict(namespace, {"download_model_file": downloader, "_download_slots": asyncio.Semaphore(1)}):
                task = asyncio.create_task(self.handlers.download_model(request))
                await asyncio.wait_for(started.wait(), 1)
                self.assertEqual((await self.handlers.download_model(request)).status, 429)
                release.set()
                self.assertEqual((await task).status, 200)
            for error, status in ((ValueError(), 400), (asyncio.TimeoutError(), 504), (OSError(), 502)):
                with patch.dict(namespace, {"download_model_file": AsyncMock(side_effect=error)}):
                    self.assertEqual((await self.handlers.download_model(request)).status, status)

        asyncio.run(exercise())

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

    def test_notes_roundtrip_and_explicit_size_errors_over_http(self):
        notes_path = self.model_dir / "sample.txt"
        limit = 1024 * 1024

        async def exercise():
            app = web.Application(client_max_size=limit * 2)
            app.router.add_post("/easyuse/metadata/notes/{name}", self.handlers.save_notes)
            app.router.add_get("/easyuse/metadata/{name}", self.handlers.load_metadata)
            async with TestClient(TestServer(app)) as client:
                for text in ("a" * limit, "备注" * (limit // 6)):
                    response = await client.post("/easyuse/metadata/notes/loras%2Fsample.safetensors", data=io.BytesIO(text.encode("utf-8")))
                    self.assertEqual(response.status, 200)
                    response = await client.get("/easyuse/metadata/loras%2Fsample.safetensors")
                    self.assertEqual(response.status, 200)
                    self.assertEqual((await response.json())["easyuse.notes"], text)
                    self.assertEqual(notes_path.read_text(encoding="utf-8"), text)
                original = notes_path.read_bytes()
                for text in ("a" * (limit + 1), "注" * (limit // 3 + 1)):
                    response = await client.post("/easyuse/metadata/notes/loras%2Fsample.safetensors", data=io.BytesIO(text.encode("utf-8")))
                    self.assertEqual(response.status, 413)
                    self.assertEqual(notes_path.read_bytes(), original)
                    self.assertEqual(list(self.model_dir.glob(".easyuse-notes-*")), [])
                oversized = b"a" * limit + b"existing-suffix"
                notes_path.write_bytes(oversized)
                response = await client.get("/easyuse/metadata/loras%2Fsample.safetensors")
                self.assertEqual(response.status, 413)
                self.assertEqual(notes_path.read_bytes(), oversized)

        asyncio.run(exercise())

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
