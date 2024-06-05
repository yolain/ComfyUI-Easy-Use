__version__ = "1.1.9"

import os
import folder_paths
import importlib
from pathlib import Path

node_list = [
    "server",
    "api",
    "easyNodes",
    "image",
    "logic"
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for module_name in node_list:
    imported_module = importlib.import_module(".py.{}".format(module_name), __name__)
    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
    NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **imported_module.NODE_DISPLAY_NAME_MAPPINGS}

cwd_path = os.path.dirname(os.path.realpath(__file__))
comfy_path = folder_paths.base_path

#Wildcards读取
from .py.libs.wildcards import read_wildcard_dict
wildcards_path = os.path.join(os.path.dirname(__file__), "wildcards")
if os.path.exists(wildcards_path):
    read_wildcard_dict(wildcards_path)
else:
    os.mkdir(wildcards_path)

#Styles
styles_path = os.path.join(os.path.dirname(__file__), "styles")
samples_path = os.path.join(os.path.dirname(__file__), "styles", "samples")
if os.path.exists(styles_path):
    if not os.path.exists(samples_path):
        os.mkdir(samples_path)
else:
    os.mkdir(styles_path)
    os.mkdir(samples_path)

# 需要把模型预览图暴露给PS读取，此处借鉴了 AIGODLIKE-ComfyUI-Studio 的部分代码
from .py.libs.add_resources import add_static_resource
from .py.libs.model import easyModelManager
model_config = easyModelManager().models_config
for model in model_config:
    paths = folder_paths.get_folder_paths(model)
    for path in paths:
        if not Path(path).exists():
            continue
        add_static_resource(path, path, limit=True)

WEB_DIRECTORY = "./web"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]


print(f'\033[34mComfy-Easy-Use v{__version__}: \033[92mLoaded\033[0m')