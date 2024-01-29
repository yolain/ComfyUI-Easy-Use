import os
import glob
import json
import folder_paths
import importlib
import shutil

node_list = [
    "server",
    "api",
    "easyNodes",
    "image",
    "lllite",
    "logic"
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

#Styles
styles_path = os.path.join(os.path.dirname(__file__), "styles")
samples_path = os.path.join(os.path.dirname(__file__), "styles", "samples")
if os.path.exists(styles_path):
    if not os.path.exists(samples_path):
        os.mkdir(samples_path)
else:
    os.mkdir(styles_path)
    os.mkdir(samples_path)

#合并autocomplete覆盖到pyssss包
pyssss_path = os.path.join(comfy_path, "custom_nodes", "ComfyUI-Custom-Scripts", "user")
combine_folder = os.path.join(cwd_path, "autocomplete")
if os.path.exists(combine_folder):
    pass
else:
    os.mkdir(combine_folder)
if os.path.exists(pyssss_path):
    output_file = os.path.join(pyssss_path, "autocomplete.txt")
    # 遍历 combine 目录下的所有 txt 文件，读取内容并合并
    merged_content = ''
    for file_path in glob.glob(os.path.join(combine_folder, '*.txt')):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            try:
                file_content = file.read()
                merged_content += file_content + '\n'
            except UnicodeDecodeError:
                pass
    # 备份之前的autocomplete
    # bak_file = os.path.join(pyssss_path, "autocomplete.txt.bak")
    # if os.path.exists(bak_file):
    #     pass
    # elif os.path.exists(output_file):
    #     shutil.copy(output_file, bak_file)
    if merged_content != '':
        # 将合并的内容写入目标文件 autocomplete.txt，并指定编码为 utf-8
        with open(output_file, 'w', encoding='utf-8') as target_file:
            target_file.write(merged_content)

WEB_DIRECTORY = "./web"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]


print('\033[34mComfy-Easy-Use (v1.0.4): \033[92mLoaded\033[0m')