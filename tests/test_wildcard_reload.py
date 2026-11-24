import ast
import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import yaml
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


ROOT = Path(__file__).parents[1]
WILDCARDS_PATH = ROOT / "py" / "libs" / "wildcards.py"
ROUTES_PATH = ROOT / "py" / "routes.py"
PROMPT_PATH = ROOT / "py" / "nodes" / "prompt.py"

WILDCARD_FUNCTIONS = {
    "get_wildcard_revision",
    "get_wildcard_list",
    "wildcard_normalize",
    "read_wildcard",
    "read_wildcard_dict",
    "reload_wildcard_dict",
}


def load_wildcards():
    """Load the wildcard helpers without importing ComfyUI's dependencies."""
    tree = ast.parse(WILDCARDS_PATH.read_text())
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = {getattr(target, "id", "") for target in node.targets}
            if targets & {"easy_wildcard_dict", "wildcard_revision"}:
                body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in WILDCARD_FUNCTIONS:
            body.append(node)

    namespace = {"os": os, "json": json, "yaml": yaml}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(WILDCARDS_PATH), "exec"), namespace)
    return namespace


def load_route_handlers(wildcards_dir, reload_impl):
    """Load the route handlers with a stubbed PromptServer and wildcard reload."""
    tree = ast.parse(ROUTES_PATH.read_text())
    names = {"_same_origin_request", "reloadWildcards"}
    body = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            node.decorator_list = []
            body.append(node)

    namespace = {
        "os": os,
        "web": web,
        "urlsplit": urlsplit,
        "PromptServer": SimpleNamespace(instance=SimpleNamespace(prompt_queue=None)),
        "WILDCARDS_DIR": str(wildcards_dir),
        "reload_wildcard_dict": reload_impl,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(ROUTES_PATH), "exec"), namespace)
    return namespace


def load_prompt_nodes(get_wildcard_revision):
    """Load only the wildcard node classes, stubbing the ComfyUI node API."""
    tree = ast.parse(PROMPT_PATH.read_text())
    names = {"wildcardsPrompt", "wildcardsPromptMatrix"}
    body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    namespace = {"io": SimpleNamespace(ComfyNode=object), "get_wildcard_revision": get_wildcard_revision}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(PROMPT_PATH), "exec"), namespace)
    return namespace


class WildcardReloadTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.dir = Path(self.workspace.name)

    def test_reload_replaces_the_dictionary_and_bumps_the_revision(self):
        wildcards = load_wildcards()
        self.assertEqual(wildcards["get_wildcard_revision"](), 0)

        (self.dir / "a.txt").write_text("one\ntwo\n", encoding="utf-8")
        self.assertEqual(wildcards["reload_wildcard_dict"](str(self.dir)), 1)
        self.assertEqual(wildcards["easy_wildcard_dict"]["a"], ["one", "two"])
        self.assertEqual(wildcards["get_wildcard_revision"](), 1)

        (self.dir / "b.txt").write_text("three\n", encoding="utf-8")
        self.assertEqual(wildcards["reload_wildcard_dict"](str(self.dir)), 2)
        self.assertEqual(wildcards["easy_wildcard_dict"]["b"], ["three"])
        self.assertEqual(wildcards["get_wildcard_revision"](), 2)

    def test_reload_keeps_the_previous_dictionary_when_a_file_cannot_be_parsed(self):
        wildcards = load_wildcards()
        (self.dir / "a.txt").write_text("one\n", encoding="utf-8")
        wildcards["reload_wildcard_dict"](str(self.dir))
        previous = wildcards["easy_wildcard_dict"]
        revision = wildcards["get_wildcard_revision"]

        (self.dir / "broken.yaml").write_text("broken: [1, 2\n", encoding="utf-8")
        with self.assertRaises(yaml.YAMLError):
            wildcards["reload_wildcard_dict"](str(self.dir))

        self.assertEqual(revision(), 1)
        self.assertEqual(wildcards["easy_wildcard_dict"], previous)
        self.assertEqual(wildcards["easy_wildcard_dict"]["a"], ["one"])

    def test_yaml_and_json_entries_are_loaded_into_the_replacement_dictionary(self):
        (self.dir / "nested").mkdir()
        (self.dir / "nested" / "colors.yaml").write_text("palette:\n  - red\n  - green\n", encoding="utf-8")
        (self.dir / "names.json").write_text(json.dumps({"names": ["ann", "bob"]}), encoding="utf-8")

        wildcards = load_wildcards()
        self.assertEqual(wildcards["reload_wildcard_dict"](str(self.dir)), 2)
        self.assertEqual(wildcards["easy_wildcard_dict"]["palette"], ["red", "green"])
        self.assertEqual(wildcards["easy_wildcard_dict"]["names"], ["ann", "bob"])

    def test_missing_directory_is_reported_instead_of_clearing_the_dictionary(self):
        wildcards = load_wildcards()
        (self.dir / "a.txt").write_text("one\n", encoding="utf-8")
        wildcards["reload_wildcard_dict"](str(self.dir))

        with self.assertRaises(FileNotFoundError):
            wildcards["reload_wildcard_dict"](str(self.dir / "missing"))

        self.assertEqual(wildcards["easy_wildcard_dict"]["a"], ["one"])
        self.assertEqual(wildcards["get_wildcard_revision"](), 1)


class WildcardReloadRouteTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.dir = Path(self.workspace.name)
        self.calls = []

    def reload_ok(self, path):
        self.calls.append(path)
        return 7

    def reload_failing(self, path):
        self.calls.append(path)
        raise ValueError("boom")

    def exercise(self, handler, prompt_queue=None, headers=None, params=None):
        async def run():
            handlers = load_route_handlers(self.dir, handler)
            handlers["PromptServer"].instance.prompt_queue = prompt_queue
            app = web.Application()
            app.router.add_post("/easyuse/wildcard/reload", handlers["reloadWildcards"])
            async with TestClient(TestServer(app)) as client:
                response = await client.post("/easyuse/wildcard/reload", params=params, headers=headers)
                body = await response.json() if response.content_type == "application/json" else None
                return response.status, body

        return asyncio.run(run())

    def test_reload_returns_the_number_of_entries(self):
        status, body = self.exercise(self.reload_ok)
        self.assertEqual(status, 200)
        self.assertEqual(body, {"status": "ok", "count": 7})
        self.assertEqual(self.calls, [str(self.dir)])

    def test_cross_origin_request_is_rejected(self):
        status, _ = self.exercise(self.reload_ok, headers={"Origin": "http://evil.example"})
        self.assertEqual(status, 403)
        self.assertEqual(self.calls, [])

    def test_reload_is_refused_while_the_queue_is_busy(self):
        queue = SimpleNamespace(get_tasks_remaining=lambda: 2)
        status, body = self.exercise(self.reload_ok, prompt_queue=queue)
        self.assertEqual(status, 409)
        self.assertEqual(body["status"], "busy")
        self.assertEqual(self.calls, [])

    def test_force_overrides_the_busy_guard(self):
        queue = SimpleNamespace(get_tasks_remaining=lambda: 2)
        status, body = self.exercise(self.reload_ok, prompt_queue=queue, params={"force": "1"})
        self.assertEqual(status, 200)
        self.assertEqual(body["count"], 7)

    def test_reload_failure_returns_server_error(self):
        status, body = self.exercise(self.reload_failing)
        self.assertEqual(status, 500)
        self.assertEqual(body, {"status": "error", "error": "ValueError"})

    def test_missing_queue_is_treated_as_idle(self):
        status, _ = self.exercise(self.reload_ok, prompt_queue=None)
        self.assertEqual(status, 200)


class WildcardFingerprintTests(unittest.TestCase):
    def test_wildcard_nodes_fingerprint_the_dictionary_revision(self):
        wildcards = load_wildcards()
        nodes = load_prompt_nodes(wildcards["get_wildcard_revision"])

        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.txt").write_text("one\n", encoding="utf-8")
            for name in ("wildcardsPrompt", "wildcardsPromptMatrix"):
                with self.subTest(node=name):
                    before = nodes[name].fingerprint_inputs()
                    wildcards["reload_wildcard_dict"](tmp)
                    self.assertEqual(nodes[name].fingerprint_inputs(), before + 1)


if __name__ == "__main__":
    unittest.main()
