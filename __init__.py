__version__ = "1.2.4"

import yaml
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

#Wildcards
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

# Model thumbnails
from .py.libs.add_resources import add_static_resource
from .py.libs.model import easyModelManager
model_config = easyModelManager().models_config
for model in model_config:
    paths = folder_paths.get_folder_paths(model)
    for path in paths:
        if not Path(path).exists():
            continue
        add_static_resource(path, path, limit=True)

# get comfyui revision
from .py.libs.utils import compare_revision

new_frontend_revision = 2546
web_default_version = 'v2' if compare_revision(new_frontend_revision) else 'v1'
# web directory
config_path = os.path.join(cwd_path, "config.yaml")
if os.path.isfile(config_path):
    with open(config_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        if data and "WEB_VERSION" in data:
            directory = f"web_version/{data['WEB_VERSION']}"
            with open(config_path, 'w') as f:
                yaml.dump(data, f)
        elif web_default_version != 'v1':
            if not data:
                data = {'WEB_VERSION': web_default_version}
            elif 'WEB_VERSION' not in data:
                data = {**data, 'WEB_VERSION': web_default_version}
            with open(config_path, 'w') as f:
                yaml.dump(data, f)
            directory = f"web_version/{web_default_version}"
        else:
            directory = f"web_version/v1"
    if not os.path.exists(os.path.join(cwd_path, directory)):
        print(f"web root {data['WEB_VERSION']} not found, using default")
        directory = f"web_version/{web_default_version}"
    WEB_DIRECTORY = directory
else:
    directory = f"web_version/{web_default_version}"
    WEB_DIRECTORY =  directory

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]

print(f'\033[34m[ComfyUI-Easy-Use] server: \033[0mv{__version__} \033[92mLoaded\033[0m')
print(f'\033[34m[ComfyUI-Easy-Use] web root: \033[0m{os.path.join(cwd_path, directory)} \033[92mLoaded\033[0m')
