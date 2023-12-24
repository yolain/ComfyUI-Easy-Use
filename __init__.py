import os
import json
import folder_paths
import importlib
import shutil

node_list = [
    "server",
    "easyNodes",
    "image",
    "lllite"
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for module_name in node_list:
    imported_module = importlib.import_module(".py.{}".format(module_name), __name__)
    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
    NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **imported_module.NODE_DISPLAY_NAME_MAPPINGS}

# 复制翻译文本到多语言节点
cwd_path = os.path.dirname(os.path.realpath(__file__))
comfy_path = folder_paths.base_path
translate_path = os.path.join(comfy_path, "custom_nodes", "AIGODLIKE-COMFYUI-TRANSLATION", "zh-CN")
translate_path_old = os.path.join(comfy_path, "custom_nodes", "AIGODLIKE-COMFYUI-TRANSLATION-main", "zh-CN")
translate_file = os.path.join(cwd_path, "ComfyUI-Easy-Use.json")
def copy_file_to_nodes(path):
    nodes_path = os.path.join(path, "Nodes")
    shutil.copy(translate_file, nodes_path)
if os.path.exists(translate_path):
    copy_file_to_nodes(translate_path)
elif os.path.exists(translate_path_old):
    copy_file_to_nodes(translate_path_old)

#Wildcards读取
from .py.wildcards import read_wildcard_dict
wildcards_path = os.path.join(os.path.dirname(__file__), "wildcards")
if os.path.exists(wildcards_path):
    read_wildcard_dict(wildcards_path)
else:
    os.mkdir(wildcards_path)

WEB_DIRECTORY = "./web"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]


print('\033[34mComfy-Easy-Use (v1.0.0): \033[92mLoaded\033[0m')