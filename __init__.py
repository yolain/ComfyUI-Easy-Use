import os
import json
import folder_paths
import importlib
import shutil

node_list = [
    "server",
    "easyNodes",
    "image",
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for module_name in node_list:
    imported_module = importlib.import_module(".py.{}".format(module_name), __name__)
    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
    NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **imported_module.NODE_DISPLAY_NAME_MAPPINGS}

# 复制翻译文本到多语言节点
# cwd_path = os.path.dirname(os.path.realpath(__file__))
# comfy_path = folder_paths.base_path
# translate_path = os.path.join(comfy_path, "custom_nodes", "AIGODLIKE-COMFYUI-TRANSLATION", "zh-CN")
# translate_path_old = os.path.join(comfy_path, "custom_nodes", "AIGODLIKE-COMFYUI-TRANSLATION-main", "zh-CN")
# translate_file = os.path.join(cwd_path, "ComfyUI-Easy-Use.json")
# def copy_file_to_nodes(path):
#     nodes_path = os.path.join(path, "Nodes")
#     shutil.copy(translate_file, nodes_path)
#     write_sth_to_category(path)
# def write_sth_to_category(path):
#     with open(path+"/NodeCategory.json", encoding="utf-8") as f:
#         try:
#             content = json.load(f)
#             if content:
#                 if "EasyUse" not in content:
#                     content['EasyUse'] = "乱乱呀优化节点"
#                 if "PreSampling" not in content:
#                     content['PreSampling'] = "预采样参数"
#                 if "Loader" not in content:
#                     content['Loader'] = "加载器"
#                 with open(path + "/NodeCategory.json", 'w', encoding="utf-8") as f:
#                     f.write(json.dumps(content, indent=4, ensure_ascii=False))
#         except:
#             print("\033[31mWrite to category Error\033[0m")
# if os.path.exists(translate_path):
#     copy_file_to_nodes(translate_path)
# elif os.path.exists(translate_path_old):
#     copy_file_to_nodes(translate_path_old)


WEB_DIRECTORY = "./web"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]


print('\033[34mComfy-Easy-Use: \033[92mLoaded\033[0m')