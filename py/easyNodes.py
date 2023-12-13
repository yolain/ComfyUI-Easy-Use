import sys
import os
import re
import json
import time
import torch
import psutil
import random
import datetime
import comfy.sd
import comfy.utils
import numpy as np
import folder_paths
import comfy.samplers
import comfy.controlnet
import latent_preview
import comfy.model_base
from pathlib import Path
import comfy.model_management
from comfy.sd import CLIP, VAE
from comfy.cli_args import args
from urllib.request import urlopen
from collections import defaultdict
from PIL.PngImagePlugin import PngInfo
from PIL import Image, ImageDraw, ImageFont
from comfy.model_patcher import ModelPatcher
from comfy_extras.chainner_models import model_loading
from typing import Dict, List, Optional, Tuple, Union, Any
from .adv_encode import advanced_encode, advanced_encode_XL

from nodes import MAX_RESOLUTION, VAEEncode, VAEEncodeTiled, VAEDecode, VAEDecodeTiled
from .config import BASE_RESOLUTIONS

from server import PromptServer

class CC:
    CLEAN = '\33[0m'
    BOLD = '\33[1m'
    ITALIC = '\33[3m'
    UNDERLINE = '\33[4m'
    BLINK = '\33[5m'
    BLINK2 = '\33[6m'
    SELECTED = '\33[7m'

    BLACK = '\33[30m'
    RED = '\33[31m'
    GREEN = '\33[32m'
    YELLOW = '\33[33m'
    BLUE = '\33[34m'
    VIOLET = '\33[35m'
    BEIGE = '\33[36m'
    WHITE = '\33[37m'

    GREY = '\33[90m'
    LIGHTRED = '\33[91m'
    LIGHTGREEN = '\33[92m'
    LIGHTYELLOW = '\33[93m'
    LIGHTBLUE = '\33[94m'
    LIGHTVIOLET = '\33[95m'
    LIGHTBEIGE = '\33[96m'
    LIGHTWHITE = '\33[97m'


class easyL:
    def __init__(self, input_string):
        self.header_value = f'{CC.LIGHTGREEN}[easy] {CC.GREEN}'
        self.label_value = ''
        self.title_value = ''
        self.input_string = f'{input_string}{CC.CLEAN}'

    def h(self, header_value):
        self.header_value = f'{CC.LIGHTGREEN}[{header_value}] {CC.GREEN}'
        return self

    def full(self):
        self.h('easyNodes')
        return self

    def success(self):
        self.label_value = f'Success: '
        return self

    def warn(self):
        self.label_value = f'{CC.RED}Warning:{CC.LIGHTRED} '
        return self

    def error(self):
        self.label_value = f'{CC.LIGHTRED}ERROR:{CC.RED} '
        return self

    def t(self, title_value):
        self.title_value = f'{title_value}:{CC.CLEAN} '
        return self

    def p(self):
        print(self.header_value + self.label_value + self.title_value + self.input_string)
        return self

    def interrupt(self, msg):
        raise Exception(msg)

class easypaths:
    ComfyUI = folder_paths.base_path
    easyNodes = Path(__file__).parent

# 加载
class easyLoader:
    def __init__(self):
        self.loaded_objects = {
            "ckpt": defaultdict(tuple),  # {ckpt_name: (model, ...)}
            "clip": defaultdict(tuple),
            "bvae": defaultdict(tuple),
            "vae": defaultdict(object),
            "lora": defaultdict(dict),  # {lora_name: {UID: (model_lora, clip_lora)}}
        }
        self.memory_threshold = self.determine_memory_threshold(0.7)

    def clean_values(self, values: str):
        original_values = values.split("; ")
        cleaned_values = []

        for value in original_values:
            cleaned_value = value.strip(';').strip()

            if cleaned_value == "":
                continue

            try:
                cleaned_value = int(cleaned_value)
            except ValueError:
                try:
                    cleaned_value = float(cleaned_value)
                except ValueError:
                    pass

            cleaned_values.append(cleaned_value)

        return cleaned_values

    def clear_unused_objects(self, desired_names: set, object_type: str):
        keys = set(self.loaded_objects[object_type].keys())
        for key in keys - desired_names:
            del self.loaded_objects[object_type][key]

    def get_input_value(self, entry, key):
        val = entry["inputs"][key]
        return val if isinstance(val, str) else val[0]

    def process_pipe_loader(self, entry,
                            desired_ckpt_names, desired_vae_names,
                            desired_lora_names, desired_lora_settings, num_loras=3, suffix=""):
        for idx in range(1, num_loras + 1):
            lora_name_key = f"{suffix}lora{idx}_name"
            desired_lora_names.add(self.get_input_value(entry, lora_name_key))
            setting = f'{self.get_input_value(entry, lora_name_key)};{entry["inputs"][f"{suffix}lora{idx}_model_strength"]};{entry["inputs"][f"{suffix}lora{idx}_clip_strength"]}'
            desired_lora_settings.add(setting)

        desired_ckpt_names.add(self.get_input_value(entry, f"{suffix}ckpt_name"))
        desired_vae_names.add(self.get_input_value(entry, f"{suffix}vae_name"))

    def update_loaded_objects(self, prompt):
        desired_ckpt_names = set()
        desired_vae_names = set()
        desired_lora_names = set()
        desired_lora_settings = set()

        for entry in prompt.values():
            class_type = entry["class_type"]

            if class_type == "easy a1111Loader" or class_type == "easy comfyLoader":
                lora_name = self.get_input_value(entry, "lora_name")
                desired_lora_names.add(lora_name)
                setting = f'{lora_name};{entry["inputs"]["lora_model_strength"]};{entry["inputs"]["lora_clip_strength"]}'
                desired_lora_settings.add(setting)

                desired_ckpt_names.add(self.get_input_value(entry, "ckpt_name"))
                desired_vae_names.add(self.get_input_value(entry, "vae_name"))
        object_types = ["ckpt", "clip", "bvae", "vae", "lora"]
        for object_type in object_types:
            desired_names = desired_ckpt_names if object_type in ["ckpt", "clip",
                                                                  "bvae"] else desired_vae_names if object_type == "vae" else desired_lora_names
            self.clear_unused_objects(desired_names, object_type)

    def add_to_cache(self, obj_type, key, value):
        """
        Add an item to the cache with the current timestamp.
        """
        timestamped_value = (value, time.time())
        self.loaded_objects[obj_type][key] = timestamped_value

    def determine_memory_threshold(self, percentage=0.8):
        """
        Determines the memory threshold as a percentage of the total available memory.

        Args:
        - percentage (float): The fraction of total memory to use as the threshold.
                              Should be a value between 0 and 1. Default is 0.8 (80%).

        Returns:
        - memory_threshold (int): Memory threshold in bytes.
        """
        total_memory = psutil.virtual_memory().total
        memory_threshold = total_memory * percentage
        return memory_threshold

    def get_memory_usage(self):
        """
        Returns the memory usage of the current process in bytes.
        """
        process = psutil.Process(os.getpid())
        return process.memory_info().rss

    def eviction_based_on_memory(self):
        """
        Evicts objects from cache based on memory usage and priority.
        """
        current_memory = self.get_memory_usage()

        if current_memory < self.memory_threshold:
            return

        eviction_order = ["vae", "lora", "bvae", "clip", "ckpt"]

        for obj_type in eviction_order:
            if current_memory < self.memory_threshold:
                break

            # Sort items based on age (using the timestamp)
            items = list(self.loaded_objects[obj_type].items())
            items.sort(key=lambda x: x[1][1])  # Sorting by timestamp

            for item in items:
                if current_memory < self.memory_threshold:
                    break

                del self.loaded_objects[obj_type][item[0]]
                current_memory = self.get_memory_usage()

    def load_checkpoint(self, ckpt_name, config_name=None):
        cache_name = ckpt_name
        if config_name not in [None, "Default"]:
            cache_name = ckpt_name + "_" + config_name
        if cache_name in self.loaded_objects["ckpt"]:
            return self.loaded_objects["ckpt"][cache_name][0], self.loaded_objects["clip"][cache_name][0], \
            self.loaded_objects["bvae"][cache_name][0]

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)

        if config_name not in [None, "Default"]:
            config_path = folder_paths.get_full_path("configs", config_name)
            loaded_ckpt = comfy.sd.load_checkpoint(config_path, ckpt_path, output_vae=True, output_clip=True,
                                                   embedding_directory=folder_paths.get_folder_paths("embeddings"))
        else:
            loaded_ckpt = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True,
                                                                embedding_directory=folder_paths.get_folder_paths(
                                                                    "embeddings"))

        self.add_to_cache("ckpt", cache_name, loaded_ckpt[0])
        self.add_to_cache("clip", cache_name, loaded_ckpt[1])
        self.add_to_cache("bvae", cache_name, loaded_ckpt[2])

        self.eviction_based_on_memory()

        return loaded_ckpt[0], loaded_ckpt[1], loaded_ckpt[2]

    def load_vae(self, vae_name):
        if vae_name in self.loaded_objects["vae"]:
            return self.loaded_objects["vae"][vae_name][0]

        vae_path = folder_paths.get_full_path("vae", vae_name)
        sd = comfy.utils.load_torch_file(vae_path)
        loaded_vae = comfy.sd.VAE(sd=sd)
        self.add_to_cache("vae", vae_name, loaded_vae)
        self.eviction_based_on_memory()

        return loaded_vae

    def load_lora(self, lora_name, model, clip, strength_model, strength_clip):
        model_hash = str(model)[44:-1]
        clip_hash = str(clip)[25:-1]

        unique_id = f'{model_hash};{clip_hash};{lora_name};{strength_model};{strength_clip}'

        if unique_id in self.loaded_objects["lora"] and unique_id in self.loaded_objects["lora"][lora_name]:
            return self.loaded_objects["lora"][unique_id][0]

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)

        self.add_to_cache("lora", unique_id, (model_lora, clip_lora))
        self.eviction_based_on_memory()

        return model_lora, clip_lora

class easySampler:
    def __init__(self):
        self.last_helds: dict[str, list] = {
            "results": [],
            "pipe_line": [],
        }

    @staticmethod
    def tensor2pil(image: torch.Tensor) -> Image.Image:
        """Convert a torch tensor to a PIL image."""
        return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    @staticmethod
    def pil2tensor(image: Image.Image) -> torch.Tensor:
        """Convert a PIL image to a torch tensor."""
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    @staticmethod
    def enforce_mul_of_64(d):
        d = int(d)
        if d <= 7:
            d = 8
        leftover = d % 8  # 8 is the number of pixels per byte
        if leftover != 0:  # if the number of pixels is not a multiple of 8
            if (leftover < 4):  # if the number of pixels is less than 4
                d -= leftover  # remove the leftover pixels
            else:  # if the number of pixels is more than 4
                d += 8 - leftover  # add the leftover pixels

        return int(d)

    @staticmethod
    def safe_split(to_split: str, delimiter: str) -> List[str]:
        """Split the input string and return a list of non-empty parts."""
        parts = to_split.split(delimiter)
        parts = [part for part in parts if part not in ('', ' ', '  ')]

        while len(parts) < 2:
            parts.append('None')
        return parts

    def common_ksampler(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0,
                        disable_noise=False, start_step=None, last_step=None, force_full_denoise=False,
                        preview_latent=True, disable_pbar=False):
        device = comfy.model_management.get_torch_device()
        latent_image = latent["samples"]

        if disable_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        preview_format = "JPEG"
        if preview_format not in ["JPEG", "PNG"]:
            preview_format = "JPEG"

        previewer = False

        if preview_latent:
            previewer = latent_preview.get_previewer(device, model.model.latent_format)

        pbar = comfy.utils.ProgressBar(steps)

        def callback(step, x0, x, total_steps):
            preview_bytes = None
            if previewer:
                preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
            pbar.update_absolute(step + 1, total_steps, preview_bytes)

        samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative,
                                      latent_image,
                                      denoise=denoise, disable_noise=disable_noise, start_step=start_step,
                                      last_step=last_step,
                                      force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback,
                                      disable_pbar=disable_pbar, seed=seed)

        out = latent.copy()
        out["samples"] = samples
        return out

    def custom_ksampler(self, model, seed, steps, cfg, _sampler, sigmas, positive, negative, latent,
                        disable_noise=False, preview_latent=True,  disable_pbar=False):

        device = comfy.model_management.get_torch_device()
        latent_image = latent["samples"]

        if disable_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        preview_format = "JPEG"
        if preview_format not in ["JPEG", "PNG"]:
            preview_format = "JPEG"

        previewer = False

        if preview_latent:
            previewer = latent_preview.get_previewer(device, model.model.latent_format)

        pbar = comfy.utils.ProgressBar(steps)

        def callback(step, x0, x, total_steps):
            preview_bytes = None
            if previewer:
                preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
            pbar.update_absolute(step + 1, total_steps, preview_bytes)

        samples = comfy.sample.sample_custom(model, noise, cfg, _sampler, sigmas, positive, negative, latent_image,
                                             noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar,
                                             seed=seed)
        out = latent.copy()
        out["samples"] = samples
        return out

    def get_value_by_id(self, key: str, my_unique_id: Any) -> Optional[Any]:
        """Retrieve value by its associated ID."""
        try:
            for value, id_ in self.last_helds[key]:
                if id_ == my_unique_id:
                    return value
        except KeyError:
            return None

    def update_value_by_id(self, key: str, my_unique_id: Any, new_value: Any) -> Union[bool, None]:
        """Update the value associated with a given ID. Return True if updated, False if appended, None if key doesn't exist."""
        try:
            for i, (value, id_) in enumerate(self.last_helds[key]):
                if id_ == my_unique_id:
                    self.last_helds[key][i] = (new_value, id_)
                    return True
            self.last_helds[key].append((new_value, my_unique_id))
            return False
        except KeyError:
            return False

    def upscale(self, samples, upscale_method, scale_by, crop):
        s = samples.copy()
        width = self.enforce_mul_of_64(round(samples["samples"].shape[3] * scale_by))
        height = self.enforce_mul_of_64(round(samples["samples"].shape[2] * scale_by))

        if (width > MAX_RESOLUTION):
            width = MAX_RESOLUTION
        if (height > MAX_RESOLUTION):
            height = MAX_RESOLUTION

        s["samples"] = comfy.utils.common_upscale(samples["samples"], width, height, upscale_method, crop)
        return (s,)

    def handle_upscale(self, samples: dict, upscale_method: str, factor: float, crop: bool) -> dict:
        """Upscale the samples if the upscale_method is not set to 'None'."""
        if upscale_method != "None":
            samples = self.upscale(samples, upscale_method, factor, crop)[0]
        return samples

    def init_state(self, my_unique_id: Any, key: str, default: Any) -> Any:
        """Initialize the state by either fetching the stored value or setting a default."""
        value = self.get_value_by_id(key, my_unique_id)
        if value is not None:
            return value
        return default

    def get_output(self, pipe: dict,) -> Tuple:
        """Return a tuple of various elements fetched from the input pipe dictionary."""
        return (
            pipe,
            pipe.get("images"),
        )

    def get_output_sdxl(self, sdxl_pipe: dict) -> Tuple:
        """Return a tuple of various elements fetched from the input sdxl_pipe dictionary."""
        return (
            sdxl_pipe,
            sdxl_pipe.get("model"),
            sdxl_pipe.get("positive"),
            sdxl_pipe.get("negative"),
            sdxl_pipe.get("vae"),
            sdxl_pipe.get("refiner_model"),
            sdxl_pipe.get("refiner_positive"),
            sdxl_pipe.get("refiner_negative"),
            sdxl_pipe.get("refiner_vae"),
            sdxl_pipe.get("samples"),
            sdxl_pipe.get("clip"),
            sdxl_pipe.get("images"),
            sdxl_pipe.get("seed")
        )

easyCache = easyLoader()
sampler = easySampler()

def nsp_parse(text, seed=0, noodle_key='__', nspterminology=None, pantry_path=None, title=None, my_unique_id=None):
    if "__" not in text:
        return text

    if nspterminology is None:
        # Fetch the NSP Pantry
        if pantry_path is None:
            pantry_path = os.path.join(easypaths.easyNodes, 'nsp_pantry.json')
        if not os.path.exists(pantry_path):
            response = urlopen('https://raw.githubusercontent.com/WASasquatch/noodle-soup-prompts/main/nsp_pantry.json')
            tmp_pantry = json.loads(response.read())
            # Dump JSON locally
            pantry_serialized = json.dumps(tmp_pantry, indent=4)
            with open(pantry_path, "w") as f:
                f.write(pantry_serialized)
            del response, tmp_pantry

        # Load local pantry
        with open(pantry_path, 'r') as f:
            nspterminology = json.load(f)

    if seed > 0 or seed < 0:
        random.seed(seed)

    # Parse Text
    new_text = text
    for term in nspterminology:
        # Target Noodle
        tkey = f'{noodle_key}{term}{noodle_key}'
        # How many occurrences?
        tcount = new_text.count(tkey)

        if tcount > 0:
            nsp_parsed = True

        # Apply random results for each noodle counted
        for _ in range(tcount):
            new_text = new_text.replace(
                tkey, random.choice(nspterminology[term]), 1)
            seed += 1
            random.seed(seed)

    easyL(new_text).t(f'{title}[{my_unique_id}]').p()

    return new_text


class easySave:
    def __init__(self, my_unique_id=0, prompt=None, extra_pnginfo=None, number_padding=5, overwrite_existing=False,
                 output_dir=folder_paths.get_temp_directory()):
        self.number_padding = int(number_padding) if number_padding not in [None, "None", 0] else None
        self.overwrite_existing = overwrite_existing
        self.my_unique_id = my_unique_id
        self.prompt = prompt
        self.extra_pnginfo = extra_pnginfo
        self.type = 'temp'
        self.output_dir = output_dir
        if self.output_dir != folder_paths.get_temp_directory():
            self.output_dir = self.folder_parser(self.output_dir, self.prompt, self.my_unique_id)
            if not os.path.exists(self.output_dir):
                self._create_directory(self.output_dir)

    @staticmethod
    def _create_directory(folder: str):
        """Try to create the directory and log the status."""
        easyL(f"Folder {folder} does not exist. Attempting to create...").warn().p()
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
                easyL(f"{folder} Created Successfully").success().p()
            except OSError:
                easyL(f"Failed to create folder {folder}").error().p()
                pass

    @staticmethod
    def _map_filename(filename: str, filename_prefix: str) -> Tuple[int, str, Optional[int]]:
        """Utility function to map filename to its parts."""

        # Get the prefix length and extract the prefix
        prefix_len = len(os.path.basename(filename_prefix))
        prefix = filename[:prefix_len]

        # Search for the primary digits
        digits = re.search(r'(\d+)', filename[prefix_len:])

        # Search for the number in brackets after the primary digits
        group_id = re.search(r'\((\d+)\)', filename[prefix_len:])

        return (int(digits.group()) if digits else 0, prefix, int(group_id.group(1)) if group_id else 0)

    @staticmethod
    def _format_date(text: str, date: datetime.datetime) -> str:
        """Format the date according to specific patterns."""
        date_formats = {
            'd': lambda d: d.day,
            'dd': lambda d: '{:02d}'.format(d.day),
            'M': lambda d: d.month,
            'MM': lambda d: '{:02d}'.format(d.month),
            'h': lambda d: d.hour,
            'hh': lambda d: '{:02d}'.format(d.hour),
            'm': lambda d: d.minute,
            'mm': lambda d: '{:02d}'.format(d.minute),
            's': lambda d: d.second,
            'ss': lambda d: '{:02d}'.format(d.second),
            'y': lambda d: d.year,
            'yy': lambda d: str(d.year)[2:],
            'yyy': lambda d: str(d.year)[1:],
            'yyyy': lambda d: d.year,
        }

        # We need to sort the keys in reverse order to ensure we match the longest formats first
        for format_str in sorted(date_formats.keys(), key=len, reverse=True):
            if format_str in text:
                text = text.replace(format_str, str(date_formats[format_str](date)))
        return text

    @staticmethod
    def _gather_all_inputs(prompt: Dict[str, dict], unique_id: str, linkInput: str = '',
                           collected_inputs: Optional[Dict[str, Union[str, List[str]]]] = None) -> Dict[
        str, Union[str, List[str]]]:
        """Recursively gather all inputs from the prompt dictionary."""
        if prompt == None:
            return None

        collected_inputs = collected_inputs or {}
        prompt_inputs = prompt[str(unique_id)]["inputs"]


        for p_input, p_input_value in prompt_inputs.items():
            a_input = f"{linkInput}>{p_input}" if linkInput else p_input
            if isinstance(p_input_value, list):
                easySave._gather_all_inputs(prompt, p_input_value[0], a_input, collected_inputs)
            else:
                existing_value = collected_inputs.get(a_input)
                if existing_value is None:
                    collected_inputs[a_input] = p_input_value
                elif p_input_value not in existing_value:
                    collected_inputs[a_input] = existing_value + "; " + p_input_value
        # if "text" in collected_inputs:
        #     del collected_inputs['text']
        # print(collected_inputs)
        return collected_inputs

    @staticmethod
    def _get_filename_with_padding(output_dir, filename, number_padding, group_id, ext):
        """Return filename with proper padding."""
        try:
            filtered = list(filter(lambda a: a[1] == filename,
                                   map(lambda x: easySave._map_filename(x, filename), os.listdir(output_dir))))
            last = max(filtered)[0]

            for f in filtered:
                if f[0] == last:
                    if f[2] == 0 or f[2] == group_id:
                        last += 1
            counter = last
        except (ValueError, FileNotFoundError):
            os.makedirs(output_dir, exist_ok=True)
            counter = 1

        if group_id == 0:
            return f"{filename}.{ext}" if number_padding is None else f"{filename}_{counter:0{number_padding}}.{ext}"
        else:
            return f"{filename}_({group_id}).{ext}" if number_padding is None else f"{filename}_{counter:0{number_padding}}_({group_id}).{ext}"

    @staticmethod
    def filename_parser(output_dir: str, filename_prefix: str, prompt: Dict[str, dict], my_unique_id: str,
                        number_padding: int, group_id: int, ext: str) -> str:
        """Parse the filename using provided patterns and replace them with actual values."""
        subfolder = os.path.dirname(os.path.normpath(filename_prefix))
        filename = os.path.basename(os.path.normpath(filename_prefix))

        filename = re.sub(r'%date:(.*?)%', lambda m: easySave._format_date(m.group(1), datetime.datetime.now()),
                          filename_prefix)
        all_inputs = easySave._gather_all_inputs(prompt, my_unique_id)

        filename = re.sub(r'%(.*?)%', lambda m: str(all_inputs.get(m.group(1), '')), filename)
        filename = re.sub(r'[/\\]+', '-', filename)

        filename = easySave._get_filename_with_padding(output_dir, filename, number_padding, group_id, ext)

        return filename, subfolder

    @staticmethod
    def folder_parser(output_dir: str, prompt: Dict[str, dict], my_unique_id: str):
        output_dir = re.sub(r'%date:(.*?)%', lambda m: easySave._format_date(m.group(1), datetime.datetime.now()),
                            output_dir)
        all_inputs = easySave._gather_all_inputs(prompt, my_unique_id)

        return re.sub(r'%(.*?)%', lambda m: str(all_inputs.get(m.group(1), '')), output_dir)

    def images(self, images, filename_prefix, output_type, embed_workflow=True, ext="png", group_id=0):
        FORMAT_MAP = {
            "png": "PNG",
            "jpg": "JPEG",
            "jpeg": "JPEG",
            "bmp": "BMP",
            "tif": "TIFF",
            "tiff": "TIFF"
        }

        if ext not in FORMAT_MAP:
            raise ValueError(f"Unsupported file extension {ext}")

        if output_type == "Hide":
            return list()
        if output_type in ("Save", "Hide/Save", "Sender/Save"):
            output_dir = self.output_dir if self.output_dir != folder_paths.get_temp_directory() else folder_paths.get_output_directory()
            self.type = "output"
        if output_type in ("Preview", "Sender"):
            output_dir = self.output_dir
            filename_prefix = 'easyPreview'

        results = list()
        for image in images:
            img = Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))

            filename = filename_prefix.replace("%width%", str(img.size[0])).replace("%height%", str(img.size[1]))

            filename, subfolder = easySave.filename_parser(output_dir, filename, self.prompt, self.my_unique_id,
                                                          self.number_padding, group_id, ext)

            file_path = os.path.join(output_dir, filename)

            if ext == "png" and embed_workflow in (True, "True"):
                metadata = PngInfo()
                if self.prompt is not None:
                    metadata.add_text("prompt", json.dumps(self.prompt))
                if hasattr(self, 'extra_pnginfo') and self.extra_pnginfo is not None:
                    for key, value in self.extra_pnginfo.items():
                        metadata.add_text(key, json.dumps(value))
                if self.overwrite_existing or not os.path.isfile(file_path):
                    img.save(file_path, pnginfo=metadata, format=FORMAT_MAP[ext])
            else:
                if self.overwrite_existing or not os.path.isfile(file_path):
                    img.save(file_path, format=FORMAT_MAP[ext])
                else:
                    easyL(f"File {file_path} already exists... Skipping").error().p()

            results.append({
                "filename": file_path,
                "subfolder": subfolder,
                "type": self.type
            })

        return results

    def textfile(self, text, filename_prefix, output_type, group_id=0, ext='txt'):
        if output_type == "Hide":
            return []
        if output_type in ("Save", "Hide/Save"):
            output_dir = self.output_dir if self.output_dir != folder_paths.get_temp_directory() else folder_paths.get_output_directory()
        if output_type == "Preview":
            filename_prefix = 'easyPreview'

        filename = easySave.filename_parser(output_dir, filename_prefix, self.prompt, self.my_unique_id,
                                           self.number_padding, group_id, ext)

        file_path = os.path.join(output_dir, filename)

        if self.overwrite_existing or not os.path.isfile(file_path):
            with open(file_path, 'w') as f:
                f.write(text)
        else:
            easyL(f"File {file_path} already exists... Skipping").error().p()

#---------------------------------------------------------------加载器 开始----------------------------------------------------------------------#

# A1111简易加载器
class a1111Loader:
    @classmethod
    def INPUT_TYPES(cls):
        resolution_strings = [f"{width} x {height}" for width, height in BASE_RESOLUTIONS]
        return {"required": {
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
            "clip_skip": ("INT", {"default": -1, "min": -24, "max": 0, "step": 1}),

            "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
            "lora_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

            "resolution": (resolution_strings,),
            "empty_latent_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "empty_latent_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),

            "positive": ("STRING", {"default": "Positive", "multiline": True}),
            "negative": ("STRING", {"default": "Negative", "multiline": True}),
        },
            "optional": {"optional_lora_stack": ("LORA_STACK",)},
            "hidden": {"prompt": "PROMPT", "positive_weight_interpretation": "A1111", "negative_weight_interpretation": "A1111"}, "my_unique_id": "UNIQUE_ID"}

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE")
    RETURN_NAMES = ("pipe", "model", "VAE")

    FUNCTION = "adv_pipeloader"
    CATEGORY = "EasyUse/Loader"

    def adv_pipeloader(self, ckpt_name, vae_name, clip_skip,
                       lora_name, lora_model_strength, lora_clip_strength,
                       resolution, empty_latent_width, empty_latent_height,
                       positive, negative, optional_lora_stack=None, prompt=None,
                       positive_weight_interpretation='A1111', negative_weight_interpretation='A1111',
                       my_unique_id=None
                       ):

        model: ModelPatcher | None = None
        clip: CLIP | None = None
        vae: VAE | None = None

        # resolution
        if resolution != "自定义 x 自定义":
            try:
                width, height = map(int, resolution.split(' x '))
                empty_latent_width = width
                empty_latent_height = height
            except ValueError:
                raise ValueError("Invalid base_resolution format.")

        # Create Empty Latent
        latent = torch.zeros([1, 4, empty_latent_height // 8, empty_latent_width // 8]).cpu()
        samples = {"samples": latent}

        # Clean models from loaded_objects
        easyCache.update_loaded_objects(prompt)

        # Load models
        model, clip, vae = easyCache.load_checkpoint(ckpt_name, "Default")

        if optional_lora_stack is not None:
            for lora in optional_lora_stack:
                model, clip = easyCache.load_lora(lora[0], model, clip, lora[1], lora[2])

        if lora_name != "None":
            model, clip = easyCache.load_lora(lora_name, model, clip, lora_model_strength, lora_clip_strength)


        # CLIP skip
        if not clip:
            raise Exception("No CLIP found")

        clipped = clip.clone()
        if clip_skip != 0:
            clipped.clip_layer(clip_skip)

        positive = nsp_parse(positive, 0, title='pipeLoader Positive', my_unique_id=my_unique_id)

        positive_embeddings_final, positive_pooled = advanced_encode(clipped, positive, "none",
                                                                    positive_weight_interpretation, w_max=1.0,
                                                                     apply_to_pooled='enable')
        positive_embeddings_final = [[positive_embeddings_final, {"pooled_output": positive_pooled}]]

        negative = nsp_parse(negative, 0, title='pipeLoader Negative', my_unique_id=my_unique_id)

        negative_embeddings_final, negative_pooled = advanced_encode(clipped, negative, "none",
                                                                     negative_weight_interpretation, w_max=1.0,
                                                                     apply_to_pooled='enable')
        negative_embeddings_final = [[negative_embeddings_final, {"pooled_output": negative_pooled}]]
        image = easySampler.pil2tensor(Image.new('RGB', (1, 1), (0, 0, 0)))

        pipe = {"model": model,
                "positive": positive_embeddings_final,
                "negative": negative_embeddings_final,
                "vae": vae,
                "clip": clip,

                "samples": samples,
                "images": image,
                "seed": 0,

                "loader_settings": {"ckpt_name": ckpt_name,
                                    "vae_name": vae_name,

                                    "lora_name": lora_name,
                                    "lora_model_strength": lora_model_strength,
                                    "lora_clip_strength": lora_clip_strength,

                                    "refiner_ckpt_name": None,
                                    "refiner_vae_name": None,
                                    "refiner_lora1_name": None,
                                    "refiner_lora1_model_strength": None,
                                    "refiner_lora1_clip_strength": None,
                                    "refiner_lora2_name": None,
                                    "refiner_lora2_model_strength": None,
                                    "refiner_lora2_clip_strength": None,

                                    "clip_skip": clip_skip,
                                    "positive": positive,
                                    "positive_l": None,
                                    "positive_g": None,
                                    "positive_token_normalization": "none",
                                    "positive_weight_interpretation": positive_weight_interpretation,
                                    "positive_balance": None,
                                    "negative": negative,
                                    "negative_l": None,
                                    "negative_g": None,
                                    "negative_token_normalization": "none",
                                    "negative_weight_interpretation": negative_weight_interpretation,
                                    "negative_balance": None,
                                    "empty_latent_width": empty_latent_width,
                                    "empty_latent_height": empty_latent_height,
                                    "batch_size": 1,
                                    "seed": 0,
                                    "empty_samples": samples, }
                }

        return (pipe, model, vae)

# Comfy简易加载器
class comfyLoader:
    @classmethod
    def INPUT_TYPES(cls):
        resolution_strings = [f"{width} x {height}" for width, height in BASE_RESOLUTIONS]
        return {"required": {
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
            "clip_skip": ("INT", {"default": -1, "min": -24, "max": 0, "step": 1}),

            "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
            "lora_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

            "resolution": (resolution_strings,),
            "empty_latent_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "empty_latent_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),

            "positive": ("STRING", {"default": "Positive", "multiline": True}),
            "negative": ("STRING", {"default": "Negative", "multiline": True}),
        },
            "optional": {"optional_lora_stack": ("LORA_STACK",)},
            "hidden": {"prompt": "PROMPT", "positive_weight_interpretation": "comfy", "negative_weight_interpretation": "comfy"}, "my_unique_id": "UNIQUE_ID"}

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE")
    RETURN_NAMES = ("pipe", "model", "vae")

    FUNCTION = "adv_pipeloader"
    CATEGORY = "EasyUse/Loader"

    def adv_pipeloader(self, ckpt_name, vae_name, clip_skip,
                       lora_name, lora_model_strength, lora_clip_strength,
                       resolution, empty_latent_width, empty_latent_height,
                       positive, negative, optional_lora_stack=None, prompt=None,
                       positive_weight_interpretation='comfy', negative_weight_interpretation='comfy',
                       my_unique_id=None
                       ):

        return a1111Loader.adv_pipeloader(self,
            ckpt_name, vae_name, clip_skip,
            lora_name, lora_model_strength, lora_clip_strength,
            resolution, empty_latent_width, empty_latent_height,
            positive, negative, optional_lora_stack, prompt,
            positive_weight_interpretation, negative_weight_interpretation,
            my_unique_id
        )


#---------------------------------------------------------------预采样 开始----------------------------------------------------------------------#

# controlnet
class controlnetSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "control_net_name": (folder_paths.get_filename_list("controlnet"),),
                "image": ("IMAGE",),
            },
            "optional": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01})
            }
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    OUTPUT_NODE = True

    FUNCTION = "controlnetApply"
    CATEGORY = "EasyUse/Loader"

    def controlnetApply(self, pipe, control_net_name, image, positive=None, negative=None, strength=1):
        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        control_net = comfy.controlnet.load_controlnet(controlnet_path)
        control_hint = image.movedim(-1, 1)

        _positive = pipe["positive"] if positive is None else positive
        _negative = pipe["negative"] if negative is None else negative

        if strength != 0:
            if _negative is None:
                p = []
                for t in positive:
                    n = [t[0], t[1].copy()]
                    c_net = control_net.copy().set_cond_hint(control_hint, strength)
                    if 'control' in t[1]:
                        c_net.set_previous_controlnet(t[1]['control'])
                    n[1]['control'] = c_net
                    n[1]['control_apply_to_uncond'] = True
                    p.append(n)
                _positive = p
            else:
                cnets = {}
                out = []
                for conditioning in [_positive, _negative]:
                    c = []
                    for t in conditioning:
                        d = t[1].copy()

                        prev_cnet = d.get('control', None)
                        if prev_cnet in cnets:
                            c_net = cnets[prev_cnet]
                        else:
                            c_net = control_net.copy().set_cond_hint(control_hint, strength)
                            c_net.set_previous_controlnet(prev_cnet)
                            cnets[prev_cnet] = c_net

                        d['control'] = c_net
                        d['control_apply_to_uncond'] = False
                        n = [t[0], d]
                        c.append(n)
                    out.append(c)
                _positive = out[0]
                _negative = out[1]

        # 拼接条件
        positive = _positive if positive is None else _positive + pipe['positive']
        negative = _negative if negative is None else _negative + pipe['negative']

        new_pipe = {
            "model": pipe['model'],
            "positive": positive,
            "negative": negative,
            "vae": pipe['vae'],
            "clip": pipe['clip'],

            "samples": pipe["samples"],
            "images": pipe["images"],
            "seed": 0,

            "loader_settings": pipe["loader_settings"]
        }

        return (new_pipe,)

# 全局Seed
class globalSeed:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": 0, "max": 1125899906842624}),
                "mode": ("BOOLEAN", {"default": True, "label_on": "control_before_generate", "label_off": "control_after_generate"}),
                "action": (["fixed", "increment", "decrement", "randomize",
                            "increment for each node", "decrement for each node", "randomize for each node"], ),
                "last_seed": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "doit"

    CATEGORY = "EasyUse/PreSampling"

    OUTPUT_NODE = True

    def doit(self, **kwargs):
        return {}

def control_seed(action, value):
    if action == 'increment':
        value += 1
        if value > 1125899906842624:
            value = 0
    elif action == 'decrement':
        value -= 1
        if value < 0:
            value = 1125899906842624
    elif action == 'randomize':
        value = random.randint(0, 1125899906842624)

    return value

# 预采样设置（基础）
class samplerSettings:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"pipe": ("PIPE_LINE",),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "seed_num": ("INT", {"default": 0, "min": 0, "max": 1125899906842624}),
                     "control_before_generate": (["fixed", "increment", "decrement", "randomize"], {"default": "randomize"}),
                     },
                "hidden":
                    {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("PIPE_LINE", )
    RETURN_NAMES = ("pipe",)
    OUTPUT_NODE = True

    FUNCTION = "settings"
    CATEGORY = "EasyUse/PreSampling"

    def settings(self, pipe, steps, cfg, sampler_name, scheduler, denoise, seed_num, control_before_generate, prompt=None, extra_pnginfo=None, my_unique_id=None):

        # seed生成
        seed_num = control_seed(control_before_generate, seed_num)
        if my_unique_id:
            workflow = extra_pnginfo["workflow"]
            node = next((x for x in workflow["nodes"] if str(x["id"]) == my_unique_id), None)
            if node:
                length = len(node["widgets_values"])
                node["widgets_values"][length-2] = seed_num

        new_pipe = {
            "model": pipe['model'],
            "positive": pipe['positive'],
            "negative": pipe['negative'],
            "vae": pipe['vae'],
            "clip": pipe['clip'],

            "samples": pipe["samples"],
            "images": pipe["images"],
            "seed": seed_num,

            "loader_settings": {
                **pipe["loader_settings"],
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "add_noise": "enabled"
            }
        }

        del pipe

        return {"ui": {"value": [seed_num]}, "result": (new_pipe,)}

# 预采样设置（高级）
class samplerSettingsAdvanced:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"pipe": ("PIPE_LINE",),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                     "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                     "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                     "add_noise": (["enable", "disable"],),
                     "seed_num": ("INT", {"default": 0, "min": 0, "max": 1125899906842624}),
                     "control_before_generate": (["fixed", "increment", "decrement", "randomize"], {"default": "randomize"}),
                     },
                "hidden":
                    {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("PIPE_LINE", )
    RETURN_NAMES = ("pipe",)
    OUTPUT_NODE = True

    FUNCTION = "settings"
    CATEGORY = "EasyUse/PreSampling"

    def settings(self, pipe, steps, cfg, sampler_name, scheduler, start_at_step, end_at_step, add_noise, seed_num, control_before_generate, prompt=None, extra_pnginfo=None, my_unique_id=None):

        # seed生成
        seed_num = control_seed(control_before_generate, seed_num)
        if my_unique_id and add_noise == 'enabled':
            workflow = extra_pnginfo["workflow"]
            node = next((x for x in workflow["nodes"] if str(x["id"]) == my_unique_id), None)
            if node:
                length = len(node["widgets_values"])
                node["widgets_values"][length-2] = seed_num

        new_pipe = {
            "model": pipe['model'],
            "positive": pipe['positive'],
            "negative": pipe['negative'],
            "vae": pipe['vae'],
            "clip": pipe['clip'],

            "samples": pipe["samples"],
            "images": pipe["images"],
            "seed": seed_num,

            "loader_settings": {
                **pipe["loader_settings"],
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "start_step": start_at_step,
                "last_step": end_at_step,
                "denoise": 1.0,
                "add_noise": add_noise
            }
        }

        del pipe

        return {"ui": {"value": [seed_num]}, "result": (new_pipe,)}

# 预采样设置（SDTurbo）
from .gradual_latent_hires_fix import sample_dpmpp_2s_ancestral, sample_dpmpp_2m_sde, sample_lcm, sample_euler_ancestral
class sdTurboSettings:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "pipe": ("PIPE_LINE",),
                    "steps": ("INT", {"default": 1, "min": 1, "max": 10}),
                    "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.SAMPLER_NAMES,),
                    "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "round": False}),
                    "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "round": False}),
                    "upscale_ratio": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 16.0, "step": 0.01, "round": False}),
                    "start_step": ("INT", {"default": 5, "min": 0, "max": 1000, "step": 1}),
                    "end_step": ("INT", {"default": 15, "min": 0, "max": 1000, "step": 1}),
                    "upscale_n_step": ("INT", {"default": 3, "min": 0, "max": 1000, "step": 1}),
                    "unsharp_kernel_size": ("INT", {"default": 3, "min": 1, "max": 21, "step": 1}),
                    "unsharp_sigma": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01, "round": False}),
                    "unsharp_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01, "round": False}),
                    "seed_num": ("INT", {"default": 0, "min": 0, "max": 1125899906842624}),
                    "control_before_generate": (["fixed", "increment", "decrement", "randomize"], {"default": "randomize"}),
               },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    OUTPUT_NODE = True

    FUNCTION = "settings"
    CATEGORY = "EasyUse/PreSampling"

    def settings(self, pipe, steps, cfg, sampler_name, eta, s_noise, upscale_ratio, start_step, end_step, upscale_n_step, unsharp_kernel_size, unsharp_sigma, unsharp_strength, seed_num, control_before_generate, prompt=None, extra_pnginfo=None, my_unique_id=None):
        model = pipe['model']
        # sigma
        timesteps = torch.flip(torch.arange(1, 11) * 100 - 1, (0,))[:steps]
        sigmas = model.model.model_sampling.sigma(timesteps)
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])

        #sampler
        sample_function = None
        extra_options = {
                "eta": eta,
                "s_noise": s_noise,
                "upscale_ratio": upscale_ratio,
                "start_step": start_step,
                "end_step": end_step,
                "upscale_n_step": upscale_n_step,
                "unsharp_kernel_size": unsharp_kernel_size,
                "unsharp_sigma": unsharp_sigma,
                "unsharp_strength": unsharp_strength,
            }
        if sampler_name == "euler_ancestral":
            sample_function = sample_euler_ancestral
        elif sampler_name == "dpmpp_2s_ancestral":
            sample_function = sample_dpmpp_2s_ancestral
        elif sampler_name == "dpmpp_2m_sde":
            sample_function = sample_dpmpp_2m_sde
        elif sampler_name == "lcm":
            sample_function = sample_lcm

        if sample_function is not None:
            unsharp_kernel_size = unsharp_kernel_size if unsharp_kernel_size % 2 == 1 else unsharp_kernel_size + 1
            extra_options["unsharp_kernel_size"] = unsharp_kernel_size
            _sampler = comfy.samplers.KSAMPLER(sample_function, extra_options)
        else:
            _sampler = comfy.samplers.sampler_object(sampler_name)
            extra_options = None

        # seed生成
        seed_num = control_seed(control_before_generate, seed_num)
        if my_unique_id:
            workflow = extra_pnginfo["workflow"]
            node = next((x for x in workflow["nodes"] if str(x["id"]) == my_unique_id), None)
            if node:
                length = len(node["widgets_values"])
                node["widgets_values"][length-2] = seed_num

        new_pipe = {
            "model": pipe['model'],
            "positive": pipe['positive'],
            "negative": pipe['negative'],
            "vae": pipe['vae'],
            "clip": pipe['clip'],

            "samples": pipe["samples"],
            "images": pipe["images"],
            "seed": seed_num,

            "loader_settings": {
                **pipe["loader_settings"],
                "extra_options": extra_options,
                "sampler": _sampler,
                "sigmas": sigmas,
                "steps": steps,
                "cfg": cfg,
                "add_noise": "enabled"
            }
        }

        del pipe

        return {"ui": {"value": [seed_num]}, "result": (new_pipe,)}

# 预采样设置（动态CFG）
from .dynthres_core import DynThresh
class dynamicCFGSettings:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"pipe": ("PIPE_LINE",),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "cfg_mode": (DynThresh.Modes,),
                     "cfg_scale_min": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.5}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "seed_num": ("INT", {"default": 0, "min": 0, "max": 1125899906842624}),
                     "control_before_generate": (["fixed", "increment", "decrement", "randomize"], {"default": "randomize"}),
                     },
                "hidden":
                    {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    OUTPUT_NODE = True

    FUNCTION = "settings"
    CATEGORY = "EasyUse/PreSampling"

    def settings(self, pipe, steps, cfg, cfg_mode, cfg_scale_min,sampler_name, scheduler, denoise, seed_num, control_before_generate, prompt=None, extra_pnginfo=None, my_unique_id=None):


        dynamic_thresh = DynThresh(7.0, 1.0,"CONSTANT", 0, cfg_mode, cfg_scale_min, 0, 0, 999, False,
                                   "MEAN", "AD", 1)

        def sampler_dyn_thresh(args):
            input = args["input"]
            cond = input - args["cond"]
            uncond = input - args["uncond"]
            cond_scale = args["cond_scale"]
            time_step = args["timestep"]
            dynamic_thresh.step = 999 - time_step[0]

            return input - dynamic_thresh.dynthresh(cond, uncond, cond_scale, None)

        model = pipe['model']

        m = model.clone()
        m.set_model_sampler_cfg_function(sampler_dyn_thresh)

        # seed生成
        seed_num = control_seed(control_before_generate, seed_num)
        if my_unique_id:
            workflow = extra_pnginfo["workflow"]
            node = next((x for x in workflow["nodes"] if str(x["id"]) == my_unique_id), None)
            if node:
                length = len(node["widgets_values"])
                node["widgets_values"][length-2] = seed_num

        new_pipe = {
            "model": m,
            "positive": pipe['positive'],
            "negative": pipe['negative'],
            "vae": pipe['vae'],
            "clip": pipe['clip'],

            "samples": pipe["samples"],
            "images": pipe["images"],
            "seed": seed_num,

            "loader_settings": {
                **pipe["loader_settings"],
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise
            },
        }

        del pipe

        return {"ui": {"value": [seed_num]}, "result": (new_pipe,)}

# 动态CFG
class dynamicThresholdingFull:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "mimic_scale": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "threshold_percentile": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mimic_mode": (DynThresh.Modes,),
                "mimic_scale_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "cfg_mode": (DynThresh.Modes,),
                "cfg_scale_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "sched_val": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "separate_feature_channels": (["enable", "disable"],),
                "scaling_startpoint": (DynThresh.Startpoints,),
                "variability_measure": (DynThresh.Variabilities,),
                "interpolate_phi": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "EasyUse/PreSampling"

    def patch(self, model, mimic_scale, threshold_percentile, mimic_mode, mimic_scale_min, cfg_mode, cfg_scale_min,
              sched_val, separate_feature_channels, scaling_startpoint, variability_measure, interpolate_phi):
        dynamic_thresh = DynThresh(mimic_scale, threshold_percentile, mimic_mode, mimic_scale_min, cfg_mode,
                                   cfg_scale_min, sched_val, 0, 999, separate_feature_channels == "enable",
                                   scaling_startpoint, variability_measure, interpolate_phi)

        def sampler_dyn_thresh(args):
            input = args["input"]
            cond = input - args["cond"]
            uncond = input - args["uncond"]
            cond_scale = args["cond_scale"]
            time_step = args["timestep"]
            dynamic_thresh.step = 999 - time_step[0]

            return input - dynamic_thresh.dynthresh(cond, uncond, cond_scale, None)

        m = model.clone()
        m.set_model_sampler_cfg_function(sampler_dyn_thresh)
        return (m,)

#---------------------------------------------------------------采样器 开始----------------------------------------------------------------------#

# 简易采样器
class samplerSimple:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"pipe": ("PIPE_LINE",),
                 "image_output": (["Hide", "Preview", "Save", "Hide/Save", "Sender", "Sender/Save"],),
                 "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"}),
                 },
                "optional": {
                    "model": ("MODEL",),
                    # "text": ("INFO", {"default": '推理完成后将显示推理时间', "multiline": False, "forceInput": False}),
                },
                "hidden":
                  {"tile_size": "INT", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                    "embeddingsList": (folder_paths.get_filename_list("embeddings"),)
                  }
                }


    RETURN_TYPES = ("PIPE_LINE", "IMAGE",)
    RETURN_NAMES = ("pipe", "image",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "EasyUse/Sampler"

    def run(self, pipe, image_output, link_id, save_prefix, model=None, tile_size=None, prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False):
        # Clean loaded_objects
        easyCache.update_loaded_objects(prompt)

        my_unique_id = int(my_unique_id)

        easy_save = easySave(my_unique_id, prompt, extra_pnginfo)

        samp_model = pipe["model"] if model is None else model
        samp_positive = pipe["positive"]
        samp_negative = pipe["negative"]
        samp_samples = pipe["samples"]
        samp_vae = pipe["vae"]
        samp_clip = pipe["clip"]

        samp_seed = pipe['seed']

        steps = pipe['loader_settings']['steps']
        start_step = pipe['loader_settings']['start_step'] if 'start_step' in pipe['loader_settings'] else 0
        last_step = pipe['loader_settings']['last_step'] if 'last_step' in pipe['loader_settings'] else 10000
        cfg = pipe['loader_settings']['cfg']
        sampler_name = pipe['loader_settings']['sampler_name']
        scheduler = pipe['loader_settings']['scheduler']
        denoise = pipe['loader_settings']['denoise']
        add_noise = pipe['loader_settings']['add_noise'] if 'add_noise' in pipe['loader_settings'] else 'enabled'

        if start_step is not None and last_step is not None:
            force_full_denoise = True
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True

        def vae_decode_latent(vae, samples, tile_size):
            return VAEDecodeTiled().decode(vae, samples, tile_size)[0] if tile_size is not None else VAEDecode().decode(vae, samples)[0]

        def process_sample_state(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive,
                                 samp_negative,
                                 steps, start_step, last_step, cfg, sampler_name, scheduler, denoise,
                                 image_output, link_id, save_prefix, tile_size, prompt, extra_pnginfo, my_unique_id,
                                 preview_latent, force_full_denoise=force_full_denoise,disable_noise=disable_noise):

            # clean spent time in prompt
            # 推理初始时间
            start_time = int(time.time() * 1000)
            # 开始推理
            samp_samples = sampler.common_ksampler(samp_model, samp_seed, steps, cfg, sampler_name, scheduler, samp_positive, samp_negative, samp_samples, denoise=denoise, preview_latent=preview_latent, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, disable_noise=disable_noise)
            # 推理结束时间
            end_time = int(time.time() * 1000)
            # 解码图片
            samp_images = vae_decode_latent(samp_vae, samp_samples, tile_size)

            # 推理总耗时（包含解码）
            end_decode_time = int(time.time() * 1000)
            spent_time = '扩散:' + str((end_time-start_time)/1000)+'秒, 解码:' + str((end_decode_time-end_time)/1000)+'秒'

            results = easy_save.images(samp_images, save_prefix, image_output)
            sampler.update_value_by_id("results", my_unique_id, results)

            # Clean loaded_objects
            easyCache.update_loaded_objects(prompt)

            new_pipe = {
                "model": samp_model,
                "positive": samp_positive,
                "negative": samp_negative,
                "vae": samp_vae,
                "clip": samp_clip,

                "samples": samp_samples,
                "images": samp_images,
                "seed": samp_seed,

                "loader_settings": {
                    **pipe["loader_settings"],
                    "spent_time": spent_time
                }
            }

            sampler.update_value_by_id("pipe_line", my_unique_id, new_pipe)

            del pipe

            if image_output in ("Hide", "Hide/Save"):
                return {"ui": {},
                    "result": sampler.get_output(new_pipe, )}

            if image_output in ("Sender", "Sender/Save"):
                PromptServer.instance.send_sync("img-send", {"link_id": link_id, "images": results})

            return {"ui": {"images": results},
                    "result": sampler.get_output(new_pipe, )}

        preview_latent = True
        if image_output in ("Hide", "Hide/Save"):
            preview_latent = False

        return process_sample_state(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative, steps, start_step, last_step, cfg, sampler_name, scheduler, denoise, image_output, link_id, save_prefix, tile_size, prompt, extra_pnginfo, my_unique_id, preview_latent)

# 简易采样器 (Tiled)
class samplerSimpleTiled:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"pipe": ("PIPE_LINE",),
                 "tile_size": ("INT", {"default": 512, "min": 320, "max": 4096, "step": 64}),
                 "image_output": (["Hide", "Preview", "Save", "Hide/Save", "Sender", "Sender/Save"],),
                 "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"})
                 },
                "optional": {
                    "model": ("MODEL",),
                },
                "hidden": {
                    "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                    "embeddingsList": (folder_paths.get_filename_list("embeddings"),)
                  }
                }

    RETURN_TYPES = ("PIPE_LINE", "IMAGE",)
    RETURN_NAMES = ("pipe", "image",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "EasyUse/Sampler"

    def run(self, pipe, tile_size=512, image_output='preview', link_id=0, save_prefix='ComfyUI', model=None, prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False):
        return samplerSimple.run(self, pipe, image_output, link_id, save_prefix, model, tile_size, prompt, extra_pnginfo, my_unique_id, force_full_denoise, disable_noise)

# SDTurbo采样器
class samplerSDTurbo:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"pipe": ("PIPE_LINE",),
                     "image_output": (["Hide", "Preview", "Save", "Hide/Save", "Sender", "Sender/Save"],),
                     "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                     "save_prefix": ("STRING", {"default": "ComfyUI"}),
                     },
                "optional": {
                    "model": ("MODEL",),
                },
                "hidden":
                    {"tile_size": "INT", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO",
                     "my_unique_id": "UNIQUE_ID",
                     "embeddingsList": (folder_paths.get_filename_list("embeddings"),)
                     }
                }

    RETURN_TYPES = ("PIPE_LINE", "IMAGE",)
    RETURN_NAMES = ("pipe", "image",)
    OUTPUT_NODE = True
    FUNCTION = "run"

    CATEGORY = "EasyUse/Sampler"

    def run(self, pipe, image_output, link_id, save_prefix, model=None, tile_size=None, prompt=None, extra_pnginfo=None, my_unique_id=None,):
        # Clean loaded_objects
        easyCache.update_loaded_objects(prompt)

        my_unique_id = int(my_unique_id)

        easy_save = easySave(my_unique_id, prompt, extra_pnginfo)

        samp_model = pipe["model"] if model is None else model
        samp_positive = pipe["positive"]
        samp_negative = pipe["negative"]
        samp_samples = pipe["samples"]
        samp_vae = pipe["vae"]
        samp_clip = pipe["clip"]

        samp_seed = pipe['seed']

        samp_sampler = pipe['loader_settings']['sampler']

        sigmas = pipe['loader_settings']['sigmas']
        cfg = pipe['loader_settings']['cfg']
        steps = pipe['loader_settings']['steps']

        disable_noise = False

        preview_latent = True
        if image_output in ("Hide", "Hide/Save"):
            preview_latent = False

        # 推理初始时间
        start_time = int(time.time() * 1000)
        # 开始推理
        samp_samples = sampler.custom_ksampler(samp_model, samp_seed, steps, cfg, samp_sampler, sigmas, samp_positive, samp_negative, samp_samples,
                        disable_noise, preview_latent)
        # 推理结束时间
        end_time = int(time.time() * 1000)

        latent = samp_samples['samples']

        # 解码图片
        if tile_size:
            samp_images = (samp_vae.decode_tiled(latent, tile_x=tile_size // 8, tile_y=tile_size // 8, ),)
        else:
            samp_images = samp_vae.decode(latent).cpu()

        # 推理总耗时（包含解码）
        end_decode_time = int(time.time() * 1000)
        spent_time = '扩散:' + str((end_time - start_time) / 1000) + '秒, 解码:' + str(
            (end_decode_time - end_time) / 1000) + '秒'

        # Clean loaded_objects
        easyCache.update_loaded_objects(prompt)

        results = easy_save.images(samp_images, save_prefix, image_output)
        sampler.update_value_by_id("results", my_unique_id, results)

        new_pipe = {
            "model": samp_model,
            "positive": samp_positive,
            "negative": samp_negative,
            "vae": samp_vae,
            "clip": samp_clip,

            "samples": samp_samples,
            "images": samp_images,
            "seed": samp_seed,

            "loader_settings": {
                **pipe["loader_settings"],
                "spent_time": spent_time
            }
        }

        sampler.update_value_by_id("pipe_line", my_unique_id, new_pipe)

        del pipe

        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                    "result": sampler.get_output(new_pipe, )}

        if image_output in ("Sender", "Sender/Save"):
            PromptServer.instance.send_sync("img-send", {"link_id": link_id, "images": results})


        return {"ui": {"images": results},
                "result": sampler.get_output(new_pipe, )}

# showSpentTime
class showSpentTime:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "spent_time": ("INFO", {"default": '推理完成后将显示推理时间', "forceInput": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    FUNCTION = "notify"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    RETURN_NAMES = ()

    CATEGORY = "EasyUse/Util"

    def notify(self, pipe, spent_time=None, unique_id=None, extra_pnginfo=None):
        if unique_id and extra_pnginfo and "workflow" in extra_pnginfo:
            workflow = extra_pnginfo["workflow"]
            node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id), None)
            if node:
                spent_time = pipe['loader_settings']['spent_time'] if 'spent_time' in pipe['loader_settings'] else ''
                node["widgets_values"] = [spent_time]

        return {"ui": {"text": spent_time}, "result": {}}

NODE_CLASS_MAPPINGS = {
    "easy a1111Loader": a1111Loader,
    "easy comfyLoader": comfyLoader,
    "easy controlnetLoader": controlnetSimple,
    "easy globalSeed": globalSeed,
    "easy preSampling": samplerSettings,
    "easy preSamplingAdvanced": samplerSettingsAdvanced,
    "easy preSamplingSdTurbo": sdTurboSettings,
    "easy preSamplingDynamicCFG": dynamicCFGSettings,
    "easy kSampler": samplerSimple,
    "easy kSamplerTiled": samplerSimpleTiled,
    "easy kSamplerSDTurbo": samplerSDTurbo,
    "easy showSpentTime": showSpentTime,
    "dynamicThresholdingFull": dynamicThresholdingFull
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "easy a1111Loader": "EasyLoader (A1111)",
    "easy comfyLoader": "EasyLoader (comfy)",
    "easy controlnetLoader": "EasyControlnet",
    "easy globalSeed": "GlobalSeed",
    "easy preSampling": "PreSampling",
    "easy preSamplingAdvanced": "PreSampling (Advanced)",
    "easy preSamplingSdTurbo": "PreSampling (SDTurbo)",
    "easy preSamplingDynamicCFG": "PreSampling (DynamicCFG)",
    "easy kSampler": "EasyKSampler",
    "easy kSamplerTiled": "EasyKSampler (Tiled Decode)",
    "easy kSamplerSDTurbo": "EasyKSampler (SDTurbo)",
    "easy showSpentTime": "ShowSpentTime",
    "dynamicThresholdingFull": "DynamicThresholdingFull"
}