import ast
import asyncio
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("easyuse_download", ROOT / "py/libs/download.py")
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


class Response:
    def __init__(self, chunks=(), status=200, location=None, length=None):
        self.status = status
        self.headers = {"Location": location} if location else {}
        self.content_length = length
        self.content = self
        self.chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def raise_for_status(self):
        if self.status >= 400:
            raise OSError("HTTP error")

    async def iter_chunked(self, size):
        for chunk in self.chunks:
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.urls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def get(self, url, **kwargs):
        self.urls.append(str(url))
        assert kwargs["allow_redirects"] is False
        return next(self.responses)


class DownloadTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name)

    def session(self, responses):
        session = Session(responses)
        # DNS policy is tested separately using the real aiohttp connector.
        # Mock the network boundary only for transfer/redirect scenarios.
        self.enterContext(patch.object(download, "PublicResolver", return_value=SimpleNamespace(close=AsyncMock())))
        self.enterContext(patch.object(download, "PublicTCPConnector"))
        factory = self.enterContext(patch.object(download.aiohttp, "ClientSession", return_value=session))
        return session, factory

    async def test_rejects_unsafe_protocols_addresses_and_credentials(self):
        for url in (
            None, "", "file:///tmp/private.txt", "ftp://example.com/model.bin",
            "http://127.0.0.1/model.bin", "http://10.0.0.1/model.bin",
            "http://169.254.169.254/model.bin", "http://0.0.0.0/model.bin",
            "http://[::1]/model.bin", "http://[::ffff:127.0.0.1]/model.bin",
            "http://224.0.0.1/model.bin", "http://[ff02::1]/model.bin",
            "https://user:password@example.com/model.bin", "https://example.com:bad/model.bin",
        ):
            with self.subTest(url=url), self.assertRaises(download.UnsafeDownload):
                await download.download_model_file(url, self.root)
        self.assertEqual(list(self.root.iterdir()), [])

    async def test_dns_answers_checked_by_actual_connector(self):
        resolver = download.PublicResolver()
        self.addAsyncCleanup(resolver.close)
        public = {"hostname": "example.test", "host": "8.8.8.8", "port": 443,
                  "family": 2, "proto": 0, "flags": 0}
        private = {**public, "host": "127.0.0.1"}
        resolver._resolver.resolve = AsyncMock(side_effect=[[public], [public, private]])
        async with download.PublicTCPConnector(resolver=resolver, use_dns_cache=False) as connector:
            self.assertEqual(await connector._resolve_host("example.test", 443), [public])
            with self.assertRaises(download.UnsafeDownload):
                await connector._resolve_host("example.test", 443)
        self.assertEqual(resolver._resolver.resolve.await_count, 2)

    async def test_noncanonical_loopback_is_blocked_before_http_request(self):
        requests = []

        async def local_only(request):
            requests.append(request.path)
            return web.Response(body=b"local-only sentinel")

        app = web.Application()
        app.router.add_get("/model.bin", local_only)
        async with TestServer(app, host="127.0.0.1") as server:
            for host in ("127.000.000.001", "127.0.0.01", "0177.0.0.1", "2130706433"):
                with self.subTest(host=host), self.assertRaises(download.UnsafeDownload):
                    await download.download_model_file(f"http://{host}:{server.port}/model.bin", self.root)
        self.assertEqual(requests, [])
        self.assertEqual(list(self.root.iterdir()), [])

    async def test_download_and_public_relative_redirect(self):
        session, factory = self.session([
            Response(status=302, location="/cdn/model.bin?signature=example"),
            Response(chunks=[b"model", b" data"], length=10),
        ])
        result = await download.download_model_file("https://example.com/model.bin", self.root)
        self.assertEqual(Path(result).read_bytes(), b"model data")
        self.assertEqual(session.urls[-1], "https://example.com/cdn/model.bin?signature=example")
        self.assertFalse(factory.call_args.kwargs["trust_env"])
        self.assertFalse(factory.call_args.kwargs["auto_decompress"])
        self.assertEqual(factory.call_args.kwargs["headers"]["Accept-Encoding"], "identity")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["model.bin"])

    async def test_mirror_fallback_shares_security_policy_and_timeout(self):
        core = AsyncMock(side_effect=[aiohttp.ClientConnectionError(), str(self.root / "model.bin")])
        with patch.object(download, "_download_model_file", core):
            await download.download_model_file("https://huggingface.co/repo/model.bin", self.root)
        first, second = core.await_args_list
        self.assertEqual(first.args[0].host, "huggingface.co")
        self.assertEqual(second.args[0].host, "hf-mirror.com")
        self.assertEqual(first.args[2], second.args[2])
        core = AsyncMock(side_effect=download.UnsafeDownload())
        with patch.object(download, "_download_model_file", core), self.assertRaises(download.UnsafeDownload):
            await download.download_model_file("https://huggingface.co/repo/model.bin", self.root)
        self.assertEqual(core.await_count, 1)

    async def test_unexpected_compressed_transfer_is_rejected(self):
        response = Response(chunks=[b"compressed body"])
        response.headers["Content-Encoding"] = "gzip"
        self.session([response])
        with self.assertRaises(download.UnsafeDownload):
            await download.download_model_file("https://example.com/model.bin", self.root)
        self.assertEqual(list(self.root.iterdir()), [])

    async def test_redirect_to_private_address_or_file_is_rejected(self):
        for location in ("http://127.0.0.1/private", "file:///tmp/private"):
            with self.subTest(location=location):
                session, _ = self.session([Response(status=302, location=location)])
                with self.assertRaises(download.UnsafeDownload):
                    await download.download_model_file("https://example.com/model.bin", self.root)
                self.assertEqual(len(session.urls), 1)
                self.assertEqual(list(self.root.iterdir()), [])

    async def test_redirect_limit(self):
        self.session([Response(status=302, location="/again")] * (download.MAX_REDIRECTS + 1))
        with self.assertRaises(download.UnsafeDownload):
            await download.download_model_file("https://example.com/model.bin", self.root)

    async def test_size_limit_for_declared_and_chunked_bodies(self):
        for response in (Response(length=11), Response(chunks=[b"123456", b"123456"])):
            with self.subTest(response=response), patch.object(download, "MAX_DOWNLOAD_BYTES", 10):
                self.session([response])
                with self.assertRaises(download.UnsafeDownload):
                    await download.download_model_file("https://example.com/model.bin", self.root)
                self.assertEqual(list(self.root.iterdir()), [])

    async def test_partial_download_cleaned_on_timeout_or_cancellation(self):
        for error in (asyncio.TimeoutError(), asyncio.CancelledError()):
            with self.subTest(error=error):
                self.session([Response(chunks=[b"partial", error])])
                with self.assertRaises(type(error)):
                    await download.download_model_file("https://example.com/model.bin", self.root)
                self.assertEqual(list(self.root.iterdir()), [])

    async def test_existing_files_are_preserved_and_symlinks_rejected(self):
        model = self.root / "model.bin"
        model.write_bytes(b"existing")
        result = await download.download_model_file("https://example.com/model.bin", self.root)
        self.assertEqual(Path(result).read_bytes(), b"existing")
        model.unlink()
        target = self.root / "protected.bin"
        target.write_bytes(b"protected")
        model.symlink_to(target)
        with self.assertRaises(download.UnsafeDownload):
            await download.download_model_file("https://example.com/model.bin", self.root)
        self.assertEqual(target.read_bytes(), b"protected")

    async def test_unsafe_filenames_rejected(self):
        with patch.object(download.aiohttp, "ClientSession") as session:
            for name in ("", "%2e%2e", "a%2fb", "a%5cb", "a%00b", "a%3ab"):
                with self.subTest(name=name), self.assertRaises(download.UnsafeDownload):
                    await download.download_model_file("https://example.com/" + name, self.root)
            session.assert_not_called()


class MetadataHeaderTests(unittest.TestCase):
    def test_metadata_header_size_is_bounded_before_read(self):
        source = ROOT / "py/libs/utils.py"
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "getMetadata")
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            data = b'{"__metadata__": {}}'
            path.write_bytes(len(data).to_bytes(8, "little") + data)
            self.assertEqual(namespace["getMetadata"](path), data)
            for size in (0, len(data) + 1, 100_000_001, 2**64 - 1):
                path.write_bytes(size.to_bytes(8, "little") + data)
                with self.subTest(size=size), self.assertRaises(BufferError):
                    namespace["getMetadata"](path)


if __name__ == "__main__":
    unittest.main()
