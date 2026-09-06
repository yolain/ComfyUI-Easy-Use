import importlib.util
import os
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path

import torch


PLUGIN_ROOT = Path(__file__).parents[1]
UTILS_PATH = PLUGIN_ROOT / "py" / "libs" / "utils.py"
CONFIG_PATH = PLUGIN_ROOT / "py" / "config.py"
LOADER_PATH = PLUGIN_ROOT / "py" / "libs" / "loader.py"
XYPLOT_NODE_PATH = PLUGIN_ROOT / "py" / "nodes" / "xyplot.py"
XYPLOT_LIB_PATH = PLUGIN_ROOT / "py" / "libs" / "xyplot.py"


@contextmanager
def installed_modules(modules):
    added = []
    for name, module in modules.items():
        if name not in sys.modules:
            added.append(name)
        sys.modules[name] = module
    try:
        yield
    finally:
        for name in added:
            sys.modules.pop(name, None)


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


def comfy_stubs():
    model_management = types.ModuleType("comfy.model_management")
    model_base = types.ModuleType("comfy.model_base")
    model_base.BaseModel = object
    supported_models_base = types.ModuleType("comfy.supported_models_base")
    supported_models_base.BASE = object
    supported_models = types.ModuleType("comfy.supported_models")
    supported_models.supported_models_base = supported_models_base
    comfy = make_package(
        "comfy",
        model_management=model_management,
        model_base=model_base,
        supported_models_base=supported_models_base,
        supported_models=supported_models,
    )
    for name in (
        "SDXL", "SDXLRefiner", "SD15", "SD20", "SVD_img2vid", "SD3",
        "HunyuanDiT", "Flux", "GenmoMochi", "Anima", "Krea2",
    ):
        setattr(supported_models, name, type(name, (), {}))
    server = types.ModuleType("server")
    server.PromptServer = object
    return {
        "comfy": comfy,
        "comfy.model_management": model_management,
        "comfy.model_base": model_base,
        "comfy.supported_models_base": supported_models_base,
        "comfy.supported_models": supported_models,
        "server": server,
    }


def xyplot_node_stubs():
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_filename_list = lambda folder: []
    return {
        "comfy": make_package("comfy"),
        "folder_paths": folder_paths,
        "fake_py": make_package("fake_py"),
        "fake_py.nodes": make_package("fake_py.nodes"),
        "fake_py.libs": make_package("fake_py.libs"),
        "fake_py.config": make_package("fake_py.config", RESOURCES_DIR="resources"),
        "fake_py.libs.utils": make_package("fake_py.libs.utils", getMetadata=lambda *args, **kwargs: None),
    }


def xyplot_lib_stubs():
    fake_utils = make_package("fake_py.utils", easySave=object, get_sd_version=lambda model: "unknown")
    fake_adv_encode = make_package("fake_py.libs.adv_encode", advanced_encode=object)
    fake_controlnet = make_package("fake_py.libs.controlnet", easyControlnet=object)
    fake_log = make_package("fake_py.libs.log", log_node_warn=lambda *args, **kwargs: None)
    return {
        "nodes": make_package("nodes", CLIPTextEncode=object),
        "fake_py": make_package("fake_py"),
        "fake_py.libs": make_package("fake_py.libs"),
        "fake_py.modules": make_package("fake_py.modules"),
        "fake_py.utils": fake_utils,
        "fake_py.libs.utils": fake_utils,
        "fake_py.libs.adv_encode": fake_adv_encode,
        "fake_py.libs.controlnet": fake_controlnet,
        "fake_py.libs.log": fake_log,
        "fake_py.modules.layer_diffuse": make_package("fake_py.modules.layer_diffuse", LayerDiffuse=object),
        "fake_py.config": make_package("fake_py.config", RESOURCES_DIR="resources"),
    }


def loader_stubs():
    comfy = make_package("comfy")
    comfy.utils = make_package("comfy.utils")
    comfy.sd = make_package("comfy.sd")
    comfy.controlnet = make_package("comfy.controlnet")
    comfy.model_patcher = make_package("comfy.model_patcher", ModelPatcher=type("ModelPatcher", (), {}))
    folder_paths = make_package("folder_paths")
    folder_paths.get_full_path = lambda folder, name: None
    folder_paths.get_folder_paths = lambda folder: []
    folder_paths.get_filename_list = lambda folder: []
    fake_log = make_package("fake_py.libs.log", log_node_info=lambda *args, **kwargs: None, log_node_error=lambda *args, **kwargs: None)
    fake_utils = make_package("fake_py.libs.utils", get_sd_version=lambda model: "unknown")
    fake_config = make_package(
        "fake_py.config",
        DIFFUSION_MODEL_XY_DEFAULTS={},
        DIFFUSION_MODEL_CLIP_TYPES={"anima": "anima", "krea2": "krea2"},
    )
    fake_pixart = make_package("fake_py.modules.dit.pixArt.loader", load_pixart=object)
    return {
        "comfy": comfy,
        "comfy.utils": comfy.utils,
        "comfy.sd": comfy.sd,
        "comfy.controlnet": comfy.controlnet,
        "comfy.model_patcher": comfy.model_patcher,
        "folder_paths": folder_paths,
        "nodes": make_package("nodes", NODE_CLASS_MAPPINGS={}),
        "fake_py": make_package("fake_py"),
        "fake_py.libs": make_package("fake_py.libs"),
        "fake_py.modules": make_package("fake_py.modules"),
        "fake_py.modules.dit": make_package("fake_py.modules.dit"),
        "fake_py.modules.dit.pixArt": make_package("fake_py.modules.dit.pixArt"),
        "fake_py.libs.log": fake_log,
        "fake_py.libs.utils": fake_utils,
        "fake_py.config": fake_config,
        "fake_py.modules.dit.pixArt.loader": fake_pixart,
    }


class FakeModelPatcher:
    def __init__(self, model_config=None, latent_format=None):
        self.model = types.SimpleNamespace(model_config=model_config, latent_format=latent_format)


class FakeLatentFormat:
    latent_dimensions = 3
    latent_channels = 16


class DiffusionXYHelperTests(unittest.TestCase):
    def test_get_sd_version_anima_and_krea2(self):
        with installed_modules(comfy_stubs()):
            utils = load_module("diffusion_xy_test_utils", UTILS_PATH)

            anima_config = utils.comfy.supported_models.Anima()
            self.assertEqual(utils.get_sd_version(FakeModelPatcher(anima_config)), "anima")

            krea2_config = utils.comfy.supported_models.Krea2()
            self.assertEqual(utils.get_sd_version(FakeModelPatcher(krea2_config)), "krea2")

    def test_diffusion_model_xy_defaults_are_complete(self):
        with tempfile.TemporaryDirectory() as models_dir:
            folder_paths = types.ModuleType("folder_paths")
            folder_paths.models_dir = models_dir
            with installed_modules({"folder_paths": folder_paths}):
                config = load_module("diffusion_xy_test_config", CONFIG_PATH)

            for family in ("anima", "krea2"):
                defaults = config.DIFFUSION_MODEL_XY_DEFAULTS[family]
                self.assertTrue(defaults["clip_name"])
                self.assertTrue(defaults["clip_type"])
                self.assertTrue(defaults["vae_name"])
                self.assertEqual(config.DIFFUSION_MODEL_CLIP_TYPES[family], family)

    def test_load_diffusion_model_required_rejects_missing_clip_and_vae(self):
        with installed_modules(loader_stubs()):
            loader_module = load_module("fake_py.libs.loader", LOADER_PATH, "fake_py.libs")
            loader = loader_module.easyLoader.__new__(loader_module.easyLoader)
            loader.load_diffusion_model = lambda model_name: ("model", model_name)
            loader.load_clip = lambda clip_name, type='stable_diffusion': ("clip", clip_name, type)
            loader.load_vae = lambda vae_name: ("vae", vae_name)
            loader_module.get_sd_version = lambda model: "krea2"

            with self.assertRaisesRegex(RuntimeError, "clip_name is required"):
                loader.load_diffusion_model_required("model.safetensors", "None", "vae.safetensors")

            with self.assertRaisesRegex(RuntimeError, "vae_name is required"):
                loader.load_diffusion_model_required("model.safetensors", "clip.safetensors", None)

            model, clip, vae, family = loader.load_diffusion_model_required(
                "model.safetensors", "clip.safetensors", "vae.safetensors"
            )
            self.assertEqual(family, "krea2")
            self.assertEqual(clip[2], "krea2")

            loader_module.get_sd_version = lambda model: "flux"
            with self.assertRaisesRegex(RuntimeError, "unsupported diffusion model family: flux"):
                loader.load_diffusion_model_required("model.safetensors", "clip.safetensors", "vae.safetensors")

    def test_xyplot_diffusion_model_value_format(self):
        with installed_modules(xyplot_node_stubs()):
            node_module = load_module("fake_py.nodes.xyplot", XYPLOT_NODE_PATH, "fake_py.nodes")
            node = node_module.XYplot_DiffusionModel()

            result = node.xy_value(
                2,
                model_name_1="waiANIMA_v10Base10.safetensors",
                clip_name_1="qwen_3_06b_base.safetensors",
                vae_name_1="qwen_image_vae.safetensors",
                model_name_2="moodyKrea2Mix,v70.safetensors",
                clip_name_2="Auto",
                vae_name_2="Auto",
            )

        self.assertEqual(result[0]["axis"], "advanced: DiffusionModel")
        self.assertEqual(
            result[0]["values"],
            [
                "waiANIMA_v10Base10.safetensors,qwen_3_06b_base.safetensors,qwen_image_vae.safetensors",
                "moodyKrea2Mix*v70.safetensors,Auto,Auto",
            ],
        )

        model_name, clip_name, vae_name = result[0]["values"][0].split(",")
        self.assertEqual(model_name.replace("*", ","), "waiANIMA_v10Base10.safetensors")
        self.assertEqual(clip_name.replace("*", ","), "qwen_3_06b_base.safetensors")
        self.assertEqual(vae_name.replace("*", ","), "qwen_image_vae.safetensors")

        model_name, clip_name, vae_name = result[0]["values"][1].split(",")
        self.assertEqual(model_name.replace("*", ","), "moodyKrea2Mix,v70.safetensors")
        self.assertEqual(clip_name, "Auto")
        self.assertEqual(vae_name, "Auto")

    def test_ensure_latent_raises_for_nonempty_4d_latent_without_image(self):
        with installed_modules(xyplot_lib_stubs()):
            xyplot_module = load_module("fake_py.libs.xyplot", XYPLOT_LIB_PATH, "fake_py.libs")

            model = FakeModelPatcher(latent_format=FakeLatentFormat())
            vae = types.SimpleNamespace(encode=lambda pixels: torch.zeros([pixels.shape[0], 16, 1, pixels.shape[2], pixels.shape[3]]))
            samples = {"samples": torch.ones([1, 4, 64, 64])}

            with self.assertRaisesRegex(RuntimeError, "requires an input image"):
                xyplot_module.easyXYPlot._ensure_latent_for_model(model, vae, samples, {})

    def test_ensure_latent_expands_empty_4d_latent_to_5d(self):
        with installed_modules(xyplot_lib_stubs()):
            xyplot_module = load_module("fake_py.libs.xyplot", XYPLOT_LIB_PATH, "fake_py.libs")

            model = FakeModelPatcher(latent_format=FakeLatentFormat())
            vae = types.SimpleNamespace(encode=lambda pixels: torch.zeros([pixels.shape[0], 16, 1, pixels.shape[2], pixels.shape[3]]))
            samples = {"samples": torch.zeros([1, 4, 64, 64])}

            result = xyplot_module.easyXYPlot._ensure_latent_for_model(model, vae, samples, {})

        self.assertEqual(result["samples"].shape, torch.Size([1, 16, 1, 64, 64]))


if __name__ == "__main__":
    unittest.main()
