import sys
import os
import re
import json
import time
import math
import torch
import psutil
import datetime
import comfy.sd
import comfy.utils
import numpy as np
import folder_paths
import comfy.samplers
import comfy.controlnet
import latent_preview
import comfy.model_base
import comfy.model_management
from comfy.sd import CLIP, VAE
from pathlib import Path
from urllib.request import urlopen
from collections import defaultdict
from PIL.PngImagePlugin import PngInfo
from PIL import Image, ImageDraw, ImageFont
from comfy.model_patcher import ModelPatcher
from comfy_extras.chainner_models import model_loading
from typing import Dict, List, Optional, Tuple, Union, Any
from .adv_encode import advanced_encode, advanced_encode_XL

from server import PromptServer
from nodes import VAELoader, MAX_RESOLUTION, RepeatLatentBatch, NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS, ConditioningSetMask
from comfy_extras.nodes_mask import LatentCompositeMasked
from .config import BASE_RESOLUTIONS, RESOURCES_DIR, INPAINT_DIR, FOOOCUS_STYLES_DIR, FOOOCUS_INPAINT_HEAD, FOOOCUS_INPAINT_PATCH
from .log import log_node_info, log_node_error, log_node_warn, log_node_success
from .wildcards import process_with_loras, get_wildcard_list

# 加载器
class easyLoader:
    def __init__(self):
        self.loaded_objects = {
            "ckpt": defaultdict(tuple),  # {ckpt_name: (model, ...)}
            "clip": defaultdict(tuple),
            "clip_vision": defaultdict(tuple),
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
            elif class_type == "easy zero123Loader" or class_type == 'easy svdLoader':
                desired_ckpt_names.add(self.get_input_value(entry, "ckpt_name"))
                desired_vae_names.add(self.get_input_value(entry, "vae_name"))

            elif class_type == "easy XYInputs: ModelMergeBlocks":
                desired_ckpt_names.add(self.get_input_value(entry, "ckpt_name_1"))
                desired_ckpt_names.add(self.get_input_value(entry, "ckpt_name_2"))
                vae_use = self.get_input_value(entry, "vae_use")
                if vae_use != 'Use Model 1' and vae_use != 'Use Model 2':
                    desired_vae_names.add(vae_use)

        object_types = ["ckpt", "clip", "bvae", "vae", "lora"]
        for object_type in object_types:
            desired_names = desired_ckpt_names if object_type in ["ckpt", "clip", "bvae"] else desired_vae_names if object_type == "vae" else desired_lora_names
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

    def load_checkpoint(self, ckpt_name, config_name=None, load_vision=False):
        cache_name = ckpt_name
        if config_name not in [None, "Default"]:
            cache_name = ckpt_name + "_" + config_name
        if cache_name in self.loaded_objects["ckpt"]:
            cache_out = self.loaded_objects["clip_vision"][cache_name][0] if load_vision else  self.loaded_objects["clip"][cache_name][0]
            return self.loaded_objects["ckpt"][cache_name][0], cache_out, self.loaded_objects["bvae"][cache_name][0]

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)

        output_clip = False if load_vision else True
        output_clipvision = True if load_vision else False
        if config_name not in [None, "Default"]:
            config_path = folder_paths.get_full_path("configs", config_name)
            loaded_ckpt = comfy.sd.load_checkpoint(config_path, ckpt_path, output_vae=True, output_clip=output_clip, output_clipvision=output_clipvision,
                                                   embedding_directory=folder_paths.get_folder_paths("embeddings"))
        else:
            loaded_ckpt = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=output_clip, output_clipvision=output_clipvision, embedding_directory=folder_paths.get_folder_paths("embeddings"))

        self.add_to_cache("ckpt", cache_name, loaded_ckpt[0])
        self.add_to_cache("bvae", cache_name, loaded_ckpt[2])
        if load_vision:
            out = loaded_ckpt[3]
            self.add_to_cache("clip_vision", cache_name, out)
        else:
            out = loaded_ckpt[1]
            self.add_to_cache("clip", cache_name, loaded_ckpt[1])

        self.eviction_based_on_memory()

        return loaded_ckpt[0], out, loaded_ckpt[2]

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

# 采样器
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
            pipe.get("model"),
            pipe.get("positive"),
            pipe.get("negative"),
            pipe.get("samples"),
            pipe.get("vae"),
            pipe.get("clip"),
            pipe.get("seed"),
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

# XY图表
class easyXYPlot:
    def __init__(self, xyPlotData, save_prefix, image_output, prompt, extra_pnginfo, my_unique_id):
        self.x_node_type, self.x_type = easySampler.safe_split(xyPlotData.get("x_axis"), ': ')
        self.y_node_type, self.y_type = easySampler.safe_split(xyPlotData.get("y_axis"), ': ')
        self.x_values = xyPlotData.get("x_vals") if self.x_type != "None" else []
        self.y_values = xyPlotData.get("y_vals") if self.y_type != "None" else []

        self.grid_spacing = xyPlotData.get("grid_spacing")
        self.latent_id = 0
        self.output_individuals = xyPlotData.get("output_individuals")

        self.x_label, self.y_label = [], []
        self.max_width, self.max_height = 0, 0
        self.latents_plot = []
        self.image_list = []

        self.num_cols = len(self.x_values) if len(self.x_values) > 0 else 1
        self.num_rows = len(self.y_values) if len(self.y_values) > 0 else 1

        self.total = self.num_cols * self.num_rows
        self.num = 0

        self.save_prefix = save_prefix
        self.image_output = image_output
        self.prompt = prompt
        self.extra_pnginfo = extra_pnginfo
        self.my_unique_id = my_unique_id

    # Helper Functions
    @staticmethod
    def define_variable(plot_image_vars, value_type, value, index):


        plot_image_vars[value_type] = value
        if value_type in ["seed", "Seeds++ Batch"]:
            value_label = f"{value}"
        else:
            value_label = f"{value_type}: {value}"

        if "ControlNet" in value_type:
            value_label = f"ControlNet {index + 1}"

        if value_type in ["ModelMergeBlocks"]:
            if ":" in value:
                line = value.split(':')
                value_label = f"{line[0]}"
            elif len(value) > 16:
                value_label = f"ModelMergeBlocks {index + 1}"
            else:
                value_label = f"MMB: {value}"

        if value_type in ["Pos Condition"]:
            value_label = f"pos cond {index + 1}" if index>0 else f"pos cond"
        if value_type in ["Neg Condition"]:
            value_label = f"neg cond {index + 1}" if index>0 else f"neg cond"

        if value_type in ["Positive Prompt S/R"]:
            value_label = f"pos prompt {index + 1}" if index>0 else f"pos prompt"
        if value_type in ["Negative Prompt S/R"]:
            value_label = f"neg prompt {index + 1}" if index>0 else f"neg prompt"

        if value_type in ["steps", "cfg", "denoise", "clip_skip",
                          "lora_model_strength", "lora_clip_strength"]:
            value_label = f"{value_type}: {value}"

        if value_type == "positive":
            value_label = f"pos prompt {index + 1}"
        elif value_type == "negative":
            value_label = f"neg prompt {index + 1}"

        return plot_image_vars, value_label

    @staticmethod
    def get_font(font_size):
        return ImageFont.truetype(str(Path(os.path.join(RESOURCES_DIR, 'OpenSans-Medium.ttf'))), font_size)

    @staticmethod
    def update_label(label, value, num_items):
        if len(label) < num_items:
            return [*label, value]
        return label

    @staticmethod
    def rearrange_tensors(latent, num_cols, num_rows):
        new_latent = []
        for i in range(num_rows):
            for j in range(num_cols):
                index = j * num_rows + i
                new_latent.append(latent[index])
        return new_latent

    def calculate_background_dimensions(self):
        border_size = int((self.max_width // 8) * 1.5) if self.y_type != "None" or self.x_type != "None" else 0
        bg_width = self.num_cols * (self.max_width + self.grid_spacing) - self.grid_spacing + border_size * (
                    self.y_type != "None")
        bg_height = self.num_rows * (self.max_height + self.grid_spacing) - self.grid_spacing + border_size * (
                    self.x_type != "None")

        x_offset_initial = border_size if self.y_type != "None" else 0
        y_offset = border_size if self.x_type != "None" else 0

        return bg_width, bg_height, x_offset_initial, y_offset

    def adjust_font_size(self, text, initial_font_size, label_width):
        font = self.get_font(initial_font_size)
        text_width, _ = font.getsize(text)

        scaling_factor = 0.9
        if text_width > (label_width * scaling_factor):
            return int(initial_font_size * (label_width / text_width) * scaling_factor)
        else:
            return initial_font_size

    def create_label(self, img, text, initial_font_size, is_x_label=True, max_font_size=70, min_font_size=10):
        label_width = img.width if is_x_label else img.height

        # Adjust font size
        font_size = self.adjust_font_size(text, initial_font_size, label_width)
        font_size = min(max_font_size, font_size)  # Ensure font isn't too large
        font_size = max(min_font_size, font_size)  # Ensure font isn't too small

        label_height = int(font_size * 1.5) if is_x_label else font_size

        label_bg = Image.new('RGBA', (label_width, label_height), color=(255, 255, 255, 0))
        d = ImageDraw.Draw(label_bg)

        font = self.get_font(font_size)

        # Check if text will fit, if not insert ellipsis and reduce text
        if d.textsize(text, font=font)[0] > label_width:
            while d.textsize(text + '...', font=font)[0] > label_width and len(text) > 0:
                text = text[:-1]
            text = text + '...'

        # Compute text width and height for multi-line text
        text_lines = text.split('\n')
        text_widths, text_heights = zip(*[d.textsize(line, font=font) for line in text_lines])
        max_text_width = max(text_widths)
        total_text_height = sum(text_heights)

        # Compute position for each line of text
        lines_positions = []
        current_y = 0
        for line, line_width, line_height in zip(text_lines, text_widths, text_heights):
            text_x = (label_width - line_width) // 2
            text_y = current_y + (label_height - total_text_height) // 2
            current_y += line_height
            lines_positions.append((line, (text_x, text_y)))

        # Draw each line of text
        for line, (text_x, text_y) in lines_positions:
            d.text((text_x, text_y), line, fill='black', font=font)

        return label_bg

    def sample_plot_image(self, plot_image_vars, samples, preview_latent, latents_plot, image_list, disable_noise,
                          start_step, last_step, force_full_denoise, x_value=None, y_value=None):
        model, clip, vae, positive, negative, seed, steps, cfg = None, None, None, None, None, None, None, None
        sampler_name, scheduler, denoise = None, None, None

        # 高级用法
        if plot_image_vars["x_node_type"] == "advanced" or plot_image_vars["y_node_type"] == "advanced":

            if self.x_type == "Seeds++ Batch" or self.y_type == "Seeds++ Batch":
                seed = int(x_value) if self.x_type == "Seeds++ Batch" else int(y_value)
            if self.x_type == "Steps" or self.y_type == "Steps":
                steps = int(x_value) if self.x_type == "Steps" else int(y_value)
            if self.x_type == "StartStep" or self.y_type == "StartStep":
                start_step = int(x_value) if self.x_type == "StartStep" else int(y_value)
            if self.x_type == "EndStep" or self.y_type == "EndStep":
                last_step = int(x_value) if self.x_type == "EndStep" else int(y_value)
            if self.x_type == "CFG Scale" or self.y_type == "CFG Scale":
                cfg = float(x_value) if self.x_type == "CFG Scale" else float(y_value)
            if self.x_type == "Sampler" or self.y_type == "Sampler":
                sampler_name = x_value if self.x_type == "Sampler" else y_value
            if self.x_type == "Scheduler" or self.y_type == "Scheduler":
                scheduler = x_value if self.x_type == "Scheduler" else y_value
            if self.x_type == "Sampler&Scheduler" or self.y_type == "Sampler&Scheduler":
                arr = x_value.split(',') if self.x_type == "Sampler&Scheduler" else y_value.split(',')
                if arr[0] and arr[0]!= 'None':
                    sampler_name = arr[0]
                if arr[1] and arr[1]!= 'None':
                    scheduler = arr[1]
            if self.x_type == "Denoise" or self.y_type == "Denoise":
                denoise = float(x_value) if self.x_type == "Denoise" else float(y_value)
            if self.x_type == "Pos Condition" or self.y_type == "Pos Condition":
                positive = plot_image_vars['positive_cond_stack'][int(x_value)] if self.x_type == "Pos Condition" else plot_image_vars['positive_cond_stack'][int(y_value)]
            if self.x_type == "Neg Condition" or self.y_type == "Neg Condition":
                negative = plot_image_vars['negative_cond_stack'][int(x_value)] if self.x_type == "Neg Condition" else plot_image_vars['negative_cond_stack'][int(y_value)]
            # 模型叠加
            if self.x_type == "ModelMergeBlocks" or self.y_type == "ModelMergeBlocks":
                ckpt_name_1, ckpt_name_2 = plot_image_vars['models']
                model1, clip1, vae1 = easyCache.load_checkpoint(ckpt_name_1)
                model2, clip2, vae2 = easyCache.load_checkpoint(ckpt_name_2)
                xy_values = x_value if self.x_type == "ModelMergeBlocks" else y_value
                if ":" in xy_values:
                    xy_line = xy_values.split(':')
                    xy_values = xy_line[1]

                xy_arrs = xy_values.split(',')
                # ModelMergeBlocks
                if len(xy_arrs) == 3:
                    input, middle, out = xy_arrs
                    kwargs = {
                        "input": input,
                        "middle": middle,
                        "out": out
                    }
                elif len(xy_arrs) == 30:
                    kwargs = {}
                    kwargs["time_embed."] = xy_arrs[0]
                    kwargs["label_emb."] = xy_arrs[1]

                    for i in range(12):
                        kwargs["input_blocks.{}.".format(i)] = xy_arrs[2+i]

                    for i in range(3):
                        kwargs["middle_block.{}.".format(i)] = xy_arrs[14+i]

                    for i in range(12):
                        kwargs["output_blocks.{}.".format(i)] = xy_arrs[17+i]

                    kwargs["out."] = xy_arrs[29]
                else:
                    raise Exception("ModelMergeBlocks weight length error")
                default_ratio = next(iter(kwargs.values()))

                m = model1.clone()
                kp = model2.get_key_patches("diffusion_model.")

                for k in kp:
                    ratio = float(default_ratio)
                    k_unet = k[len("diffusion_model."):]

                    last_arg_size = 0
                    for arg in kwargs:
                        if k_unet.startswith(arg) and last_arg_size < len(arg):
                            ratio = float(kwargs[arg])
                            last_arg_size = len(arg)

                    m.add_patches({k: kp[k]}, 1.0 - ratio, ratio)

                vae_use = plot_image_vars['vae_use']

                clip = clip2 if vae_use == 'Use Model 2' else clip1
                if vae_use == 'Use Model 2':
                    vae = vae2
                elif vae_use == 'Use Model 1':
                    vae = vae1
                else:
                    (vae,) = VAELoader().load_vae(vae_use)
                model = m

                # 如果存在lora_stack叠加lora
                optional_lora_stack = plot_image_vars['lora_stack']
                if optional_lora_stack is not None and optional_lora_stack != []:
                    for lora in optional_lora_stack:
                        lora_name = lora["lora_name"]
                        model = model if model is not None else lora["model"]
                        clip = clip if clip is not None else lora["clip"]
                        lora_model_strength = lora["lora_model_strength"]
                        lora_clip_strength = lora["lora_clip_strength"]
                        if "lbw" in lora:
                            lbw = lora["lbw"]
                            lbw_a = lora["lbw_a"]
                            lbw_b = lora["lbw_b"]
                            cls = ALL_NODE_CLASS_MAPPINGS['LoraLoaderBlockWeight //Inspire']
                            model, clip, _ = cls().doit(model, clip, lora_name, lora_model_strength, lora_clip_strength, False, 0,
                                                        lbw_a, lbw_b, "", lbw)
                        model, clip = easyCache.load_lora(lora_name, model, clip, lora_model_strength, lora_clip_strength)

                # 处理clip
                clip = clip.clone()
                if plot_image_vars['clip_skip'] != 0:
                    clip.clip_layer(plot_image_vars['clip_skip'])

            # 提示词
            if "Positive" in self.x_type or "Positive" in self.y_type:
                if self.x_type == 'Positive Prompt S/R' or self.y_type == 'Positive Prompt S/R':
                    positive = x_value if self.x_type == "Positive Prompt S/R" else y_value
                if plot_image_vars['a1111_prompt_style']:
                    if "smZ CLIPTextEncode" in ALL_NODE_CLASS_MAPPINGS:
                        cls = ALL_NODE_CLASS_MAPPINGS['smZ CLIPTextEncode']
                        steps = plot_image_vars['steps']
                        clip = clip if clip is not None else plot_image_vars["clip"]
                        positive, = cls().encode(clip, positive, "A1111", True, True, False, False, 6,
                                                 1024, 1024, 0, 0, 1024, 1024, '', '', steps)
                    else:
                        raise Exception(
                            f"[ERROR] To use clip text encode same as webui, you need to install 'smzNodes'")
                else:
                    clip = clip if clip is not None else plot_image_vars["clip"]
                    positive, positive_pooled = advanced_encode(clip, positive,
                                                                plot_image_vars['positive_token_normalization'],
                                                                plot_image_vars[
                                                                    'positive_weight_interpretation'],
                                                                w_max=1.0,
                                                                apply_to_pooled="enable")
                    positive = [[positive, {"pooled_output": positive_pooled}]]
                    if "positive_cond" in plot_image_vars:
                        positive = positive + plot_image_vars["positive_cond"]

            if "Negative" in self.x_type or "Negative" in self.y_type:
                if self.x_type == 'Negative Prompt S/R' or self.y_type == 'Negative Prompt S/R':
                    negative = x_value if self.x_type == "Negative Prompt S/R" else y_value
                if plot_image_vars['a1111_prompt_style']:
                    if "smZ CLIPTextEncode" in ALL_NODE_CLASS_MAPPINGS:
                        cls = ALL_NODE_CLASS_MAPPINGS['smZ CLIPTextEncode']
                        steps = plot_image_vars['steps']
                        clip = clip if clip is not None else plot_image_vars["clip"]
                        negative, = cls().encode(clip, negative, "A1111", True, True, False, False, 6,
                                                 1024, 1024, 0, 0, 1024, 1024, '', '', steps)
                    else:
                        raise Exception(
                            f"[ERROR] To use clip text encode same as webui, you need to install 'smzNodes'")
                else:
                    clip = clip if clip is not None else plot_image_vars["clip"]
                    negative, negative_pooled = advanced_encode(clip, negative,
                                                                plot_image_vars['negative_token_normalization'],
                                                                plot_image_vars[
                                                                    'negative_weight_interpretation'],
                                                                w_max=1.0,
                                                                apply_to_pooled="enable")
                    negative = [[negative, {"pooled_output": negative_pooled}]]
                    if "negative_cond" in plot_image_vars:
                        positive = positive + plot_image_vars["negative_cond"]

            # ControlNet
            if "ControlNet" in self.x_type or "ControlNet" in self.y_type:
                _pipe = {
                    "model": model if model is not None else plot_image_vars["model"],
                    "positive": positive if positive is not None else plot_image_vars["positive_cond"],
                    "negative": negative if negative is not None else plot_image_vars["negative_cond"],
                    "vae": vae if vae is not None else plot_image_vars['vae'],
                    "clip": clip if clip is not None else plot_image_vars['clip'],
                    "samples": None,
                    "images": None,
                    "loader_settings": {}
                }
                cnet = plot_image_vars["cnet"] if "cnet" in plot_image_vars else None
                if cnet:
                    index = x_value if "ControlNet" in self.x_type else y_value
                    controlnet = cnet[index]
                    for index, item in enumerate(controlnet):
                        control_net_name = item[0]
                        image = item[1]
                        strength = item[2]
                        start_percent = item[3]
                        end_percent = item[4]
                        _pipe, positive, negative = controlnetAdvanced().controlnetApply(_pipe, image, control_net_name, None, strength, start_percent, end_percent, 1)

                del _pipe

        # 简单用法
        if plot_image_vars["x_node_type"] == "loader" or plot_image_vars["y_node_type"] == "loader":
            model, clip, vae = easyCache.load_checkpoint(plot_image_vars['ckpt_name'])

            if plot_image_vars['lora_name'] != "None":
                model, clip = easyCache.load_lora(plot_image_vars['lora_name'], model, clip,
                                                 plot_image_vars['lora_model_strength'],
                                                 plot_image_vars['lora_clip_strength'])

            # Check for custom VAE
            if plot_image_vars['vae_name'] not in ["Baked-VAE", "Baked VAE"]:
                vae = easyCache.load_vae(plot_image_vars['vae_name'])

            # CLIP skip
            if not clip:
                raise Exception("No CLIP found")
            clip = clip.clone()
            clip.clip_layer(plot_image_vars['clip_skip'])

            if plot_image_vars['a1111_prompt_style']:
                if "smZ CLIPTextEncode" in ALL_NODE_CLASS_MAPPINGS:
                    cls = ALL_NODE_CLASS_MAPPINGS['smZ CLIPTextEncode']
                    steps = plot_image_vars['steps']
                    positive, = cls().encode(clip, plot_image_vars['positive'], "A1111", True, True, False, False, 6, 1024, 1024, 0, 0, 1024, 1024, '', '', steps)
                    negative, = cls().encode(clip, plot_image_vars['negative'], "A1111", True, True, False, False, 6, 1024, 1024, 0, 0, 1024, 1024, '', '', steps)
                else:
                    raise Exception(f"[ERROR] To use clip text encode same as webui, you need to install 'smzNodes'")
            else:
                positive, positive_pooled = advanced_encode(clip, plot_image_vars['positive'],
                                                            plot_image_vars['positive_token_normalization'],
                                                            plot_image_vars['positive_weight_interpretation'], w_max=1.0,
                                                            apply_to_pooled="enable")
                positive = [[positive, {"pooled_output": positive_pooled}]]

                negative, negative_pooled = advanced_encode(clip, plot_image_vars['negative'],
                                                            plot_image_vars['negative_token_normalization'],
                                                            plot_image_vars['negative_weight_interpretation'], w_max=1.0,
                                                            apply_to_pooled="enable")
                negative = [[negative, {"pooled_output": negative_pooled}]]

        model = model if model is not None else plot_image_vars["model"]
        clip = clip if clip is not None else plot_image_vars["clip"]
        vae = vae if vae is not None else plot_image_vars["vae"]
        positive = positive if positive is not None else plot_image_vars["positive_cond"]
        negative = negative if negative is not None else plot_image_vars["negative_cond"]

        seed = seed if seed is not None else plot_image_vars["seed"]
        steps = steps if steps is not None else plot_image_vars["steps"]
        cfg = cfg if cfg is not None else plot_image_vars["cfg"]
        sampler_name = sampler_name if sampler_name is not None else plot_image_vars["sampler_name"]
        scheduler = scheduler if scheduler is not None else plot_image_vars["scheduler"]
        denoise = denoise if denoise is not None else plot_image_vars["denoise"]
        # Sample
        samples = sampler.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, samples,
                                          denoise=denoise, disable_noise=disable_noise, preview_latent=preview_latent,
                                          start_step=start_step, last_step=last_step,
                                          force_full_denoise=force_full_denoise)

        # Decode images and store
        latent = samples["samples"]

        # Add the latent tensor to the tensors list
        latents_plot.append(latent)

        # Decode the image
        image = vae.decode(latent).cpu()

        if self.output_individuals in [True, "True"]:
            easy_save = easySave(self.my_unique_id, self.prompt, self.extra_pnginfo)
            easy_save.images(image, self.save_prefix, self.image_output, group_id=self.num)

        # Convert the image from tensor to PIL Image and add it to the list
        pil_image = easySampler.tensor2pil(image)
        image_list.append(pil_image)

        # Update max dimensions
        self.max_width = max(self.max_width, pil_image.width)
        self.max_height = max(self.max_height, pil_image.height)

        # Return the touched variables
        return image_list, self.max_width, self.max_height, latents_plot

    # Process Functions
    def validate_xy_plot(self):
        if self.x_type == 'None' and self.y_type == 'None':
            log_node_warn(f'easyKsampler[{self.my_unique_id}]','No Valid Plot Types - Reverting to default sampling...')
            return False
        else:
            return True

    def get_latent(self, samples):
        # Extract the 'samples' tensor from the dictionary
        latent_image_tensor = samples["samples"]

        # Split the tensor into individual image tensors
        image_tensors = torch.split(latent_image_tensor, 1, dim=0)

        # Create a list of dictionaries containing the individual image tensors
        latent_list = [{'samples': image} for image in image_tensors]

        # Set latent only to the first latent of batch
        if self.latent_id >= len(latent_list):
            log_node_warn(f'easy kSampler[{self.my_unique_id}]',f'The selected latent_id ({self.latent_id}) is out of range.')
            log_node_warn(f'easy kSampler[{self.my_unique_id}]', f'Automatically setting the latent_id to the last image in the list (index: {len(latent_list) - 1}).')

            self.latent_id = len(latent_list) - 1

        return latent_list[self.latent_id]

    def get_labels_and_sample(self, plot_image_vars, latent_image, preview_latent, start_step, last_step,
                              force_full_denoise, disable_noise):
        for x_index, x_value in enumerate(self.x_values):
            plot_image_vars, x_value_label = self.define_variable(plot_image_vars, self.x_type, x_value,
                                                                  x_index)
            self.x_label = self.update_label(self.x_label, x_value_label, len(self.x_values))
            if self.y_type != 'None':
                for y_index, y_value in enumerate(self.y_values):
                    plot_image_vars, y_value_label = self.define_variable(plot_image_vars, self.y_type, y_value,
                                                                          y_index)
                    self.y_label = self.update_label(self.y_label, y_value_label, len(self.y_values))
                    # ttNl(f'{CC.GREY}X: {x_value_label}, Y: {y_value_label}').t(
                    #     f'Plot Values {self.num}/{self.total} ->').p()

                    self.image_list, self.max_width, self.max_height, self.latents_plot = self.sample_plot_image(
                        plot_image_vars, latent_image, preview_latent, self.latents_plot, self.image_list,
                        disable_noise, start_step, last_step, force_full_denoise, x_value, y_value)
                    self.num += 1
            else:
                # ttNl(f'{CC.GREY}X: {x_value_label}').t(f'Plot Values {self.num}/{self.total} ->').p()
                self.image_list, self.max_width, self.max_height, self.latents_plot = self.sample_plot_image(
                    plot_image_vars, latent_image, preview_latent, self.latents_plot, self.image_list, disable_noise,
                    start_step, last_step, force_full_denoise, x_value)
                self.num += 1

        # Rearrange latent array to match preview image grid
        self.latents_plot = self.rearrange_tensors(self.latents_plot, self.num_cols, self.num_rows)

        # Concatenate the tensors along the first dimension (dim=0)
        self.latents_plot = torch.cat(self.latents_plot, dim=0)

        return self.latents_plot

    def plot_images_and_labels(self):
        # Calculate the background dimensions
        bg_width, bg_height, x_offset_initial, y_offset = self.calculate_background_dimensions()

        # Create the white background image
        background = Image.new('RGBA', (int(bg_width), int(bg_height)), color=(255, 255, 255, 255))

        output_image = []
        for row_index in range(self.num_rows):
            x_offset = x_offset_initial

            for col_index in range(self.num_cols):
                index = col_index * self.num_rows + row_index
                img = self.image_list[index]
                output_image.append(sampler.pil2tensor(img))
                background.paste(img, (x_offset, y_offset))

                # Handle X label
                if row_index == 0 and self.x_type != "None":
                    label_bg = self.create_label(img, self.x_label[col_index], int(48 * img.width / 512))
                    label_y = (y_offset - label_bg.height) // 2
                    background.alpha_composite(label_bg, (x_offset, label_y))

                # Handle Y label
                if col_index == 0 and self.y_type != "None":
                    label_bg = self.create_label(img, self.y_label[row_index], int(48 * img.height / 512), False)
                    label_bg = label_bg.rotate(90, expand=True)

                    label_x = (x_offset - label_bg.width) // 2
                    label_y = y_offset + (img.height - label_bg.height) // 2
                    background.alpha_composite(label_bg, (label_x, label_y))

                x_offset += img.width + self.grid_spacing

            y_offset += img.height + self.grid_spacing

        return (sampler.pil2tensor(background), output_image)

easyCache = easyLoader()
sampler = easySampler()


def check_link_to_clip(node_id, clip_id, visited=None, node=None):
    """Check if a given node links directly or indirectly to a loader node."""
    if visited is None:
        visited = set()

    if node_id in visited:
        return False
    visited.add(node_id)
    if "pipe" in node["inputs"]:
        link_ids = node["inputs"]["pipe"]
        for id in link_ids:
            if id != 0 and id == str(clip_id):
                return True
    return False

def find_nearest_steps(clip_id, prompt):
    """Find the nearest KSampler or preSampling node that references the given id."""
    for id in prompt:
        node = prompt[id]
        if "Sampler" in node["class_type"] or "sampler" in node["class_type"] or "Sampling" in node["class_type"]:
            # Check if this KSampler node directly or indirectly references the given CLIPTextEncode node
            if check_link_to_clip(id, clip_id, None, node):
                steps = node["inputs"]["steps"] if "steps" in node["inputs"] else 1
                return steps
    return 1
def find_wildcards_seed(clip_id, text, prompt):
    def find_link_clip_id(id, seed, wildcard_id):
        node = prompt[id]
        if "positive" in node['inputs']:
            link_ids = node["inputs"]["positive"]
            if type(link_ids) == list:
                for id in link_ids:
                    if id != 0:
                        if id == wildcard_id:
                            wildcard_node = prompt[wildcard_id]
                            seed = wildcard_node["inputs"]["seed_num"] if "seed_num" in wildcard_node["inputs"] else None
                            return seed
                        else:
                            return find_link_clip_id(id, seed, wildcard_id)
            else:
                return None
        else:
            return None
    if "__" in text:
        seed = None
        for id in prompt:
            node = prompt[id]
            if "wildcards" in node["class_type"]:
                wildcard_id = id
                return find_link_clip_id(str(clip_id), seed, wildcard_id)
        return seed
    else:
        return None

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
        log_node_warn(f"Folder {folder} does not exist. Attempting to create...")
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
                log_node_success(f"{folder} Created Successfully")
            except OSError:
                log_node_error(f"Failed to create folder {folder}")
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
        if output_type in ("Save", "Hide/Save"):
            output_dir = self.output_dir if self.output_dir != folder_paths.get_temp_directory() else folder_paths.get_output_directory()
            self.type = "output"
        if output_type == "Preview":
            output_dir = self.output_dir
            filename_prefix = 'easyPreview'
        results = list()

        filename_prefix = re.sub(r'%date:(.*?)%', lambda m: easySave._format_date(m.group(1), datetime.datetime.now()),
                          filename_prefix)
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, output_dir, images[0].shape[1], images[0].shape[0])
        for image in images:
            img = Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))
            filename = filename.replace("%width%", str(img.size[0])).replace("%height%", str(img.size[1]))
            metadata = None
            if embed_workflow in (True, "True"):
                metadata = PngInfo()
                if self.prompt is not None:
                    metadata.add_text("prompt", json.dumps(self.prompt))
                if hasattr(self, 'extra_pnginfo') and self.extra_pnginfo is not None:
                    for key, value in self.extra_pnginfo.items():
                        metadata.add_text(key, json.dumps(value))

            file = f"{filename}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return results

# ---------------------------------------------------------------提示词 开始----------------------------------------------------------------------#

# 正面提示词
class positivePrompt:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "positive": ("STRING", {"default": "", "multiline": True, "placeholder": "Positive"}),}
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "main"

    CATEGORY = "EasyUse/Prompt"

    @staticmethod
    def main(positive):
        return positive,

# 通配符提示词
class wildcardsPrompt:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        wildcard_list = get_wildcard_list()
        return {"required": {
            "text": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False, "placeholder": "(Support Lora Block Weight and wildcard)"}),
            "Select to add LoRA": (["Select the LoRA to add to the text"] + folder_paths.get_filename_list("loras"),),
            "Select to add Wildcard": (["Select the Wildcard to add to the text"] + wildcard_list,),
            "seed_num": ("INT", {"default": 0, "min": 0, "max": 1125899906842624}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_NODE = True
    FUNCTION = "main"

    CATEGORY = "EasyUse/Prompt"

    @staticmethod
    def main(*args, **kwargs):
        prompt = kwargs["prompt"] if "prompt" in kwargs else None
        seed_num = kwargs["seed_num"]

        # Clean loaded_objects
        if prompt:
            easyCache.update_loaded_objects(prompt)

        text = kwargs['text']
        return {"ui": {"value": [seed_num]}, "result": (text,)}

# 负面提示词
class negativePrompt:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "negative": ("STRING", {"default": "", "multiline": True, "placeholder": "Negative"}),}
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("negative",)
    FUNCTION = "main"

    CATEGORY = "EasyUse/Prompt"

    @staticmethod
    def main(negative):
        return negative,

# 风格提示词选择器
class stylesPromptSelector:

    @classmethod
    def INPUT_TYPES(s):
        styles = ["fooocus_styles"]
        styles_dir = FOOOCUS_STYLES_DIR
        for file_name in os.listdir(styles_dir):
            file = os.path.join(styles_dir, file_name)
            if os.path.isfile(file) and file_name.endswith(".json") and "styles" in file_name.split(".")[0]:
                styles.append(file_name.split(".")[0])
        return {
            "required": {
               "styles": (styles, {"default": "fooocus_styles"}),
            },
            "optional": {
                "positive": ("STRING", {"forceInput": True}),
                "negative": ("STRING", {"forceInput": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("positive", "negative",)

    CATEGORY = 'EasyUse/Prompt'
    FUNCTION = 'run'
    OUTPUT_NODE = True


    def replace_repeat(self, prompt):
        prompt = prompt.replace("，", ",")
        arr = prompt.split(",")
        if len(arr) != len(set(arr)):
            all_weight_prompt = re.findall(re.compile(r'[(](.*?)[)]', re.S), prompt)
            if len(all_weight_prompt) > 0:
                # others_prompt = prompt
                # for w_prompt in all_weight_prompt:
                # others_prompt = others_prompt.replace('(','').replace(')','')
                # print(others_prompt)
                return prompt
            else:
                for i in range(len(arr)):
                    arr[i] = arr[i].strip()
                arr = list(set(arr))
                return ", ".join(arr)
        else:
            return prompt

    def run(self, styles, positive='', negative='', prompt=None, extra_pnginfo=None, my_unique_id=None):
        values = []
        all_styles = {}
        positive_prompt, negative_prompt = '', negative
        if styles == "fooocus_styles":
            file = os.path.join(RESOURCES_DIR,  styles + '.json')
        else:
            file = os.path.join(RESOURCES_DIR, styles + '.json')
        f = open(file, 'r', encoding='utf-8')
        data = json.load(f)
        f.close()
        for d in data:
            all_styles[d['name']] = d
        if my_unique_id in prompt:
            if prompt[my_unique_id]["inputs"]['select_styles']:
                values = prompt[my_unique_id]["inputs"]['select_styles'].split(',')

        has_prompt = False
        for index, val in enumerate(values):
            if 'prompt' in all_styles[val]:
                if "{prompt}" in all_styles[val]['prompt'] and has_prompt == False:
                    positive_prompt = all_styles[val]['prompt'].format(prompt=positive)
                    has_prompt = True
                else:
                    positive_prompt += ', ' + all_styles[val]['prompt'].replace(', {prompt}', '').replace('{prompt}', '')
            if 'negative_prompt' in all_styles[val]:
                negative_prompt += ', ' + all_styles[val]['negative_prompt'] if negative_prompt else all_styles[val]['negative_prompt']

        if has_prompt == False and positive:
            positive_prompt = positive + ', '

        # 去重
        positive_prompt = self.replace_repeat(positive_prompt) if positive_prompt else ''
        negative_prompt = self.replace_repeat(negative_prompt) if negative_prompt else ''

        return (positive_prompt, negative_prompt)

#promptList
class promptList:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prompt_1": ("STRING", {"multiline": True, "default": ""}),
            "prompt_2": ("STRING", {"multiline": True, "default": ""}),
            "prompt_3": ("STRING", {"multiline": True, "default": ""}),
            "prompt_4": ("STRING", {"multiline": True, "default": ""}),
            "prompt_5": ("STRING", {"multiline": True, "default": ""}),
        },
            "optional": {
                "optional_prompt_list": ("LIST",)
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("prompt_list",)
    FUNCTION = "run"
    CATEGORY = "EasyUse/Prompt"

    def run(self, **kwargs):
        prompts = []

        if "optional_prompt_list" in kwargs:
            for l in kwargs["optional_prompt_list"]:
                prompts.append(l)

        # Iterate over the received inputs in sorted order.
        for k in sorted(kwargs.keys()):
            v = kwargs[k]

            # Only process string input ports.
            if isinstance(v, str) and v != '':
                prompts.append(v)

        return (prompts,)

# 肖像大师
# Created by AI Wiz Art (Stefano Flore)
# Version: 2.2
# https://stefanoflore.it
# https://ai-wiz.art
class portraitMaster:

    @classmethod
    def INPUT_TYPES(s):
        max_float_value = 1.95
        prompt_path = os.path.join(RESOURCES_DIR, 'portrait_prompt.json')
        if not os.path.exists(prompt_path):
            response = urlopen('https://raw.githubusercontent.com/yolain/ComfyUI-Easy-Use/main/resources/portrait_prompt.json')
            temp_prompt = json.loads(response.read())
            prompt_serialized = json.dumps(temp_prompt, indent=4)
            with open(prompt_path, "w") as f:
                f.write(prompt_serialized)
            del response, temp_prompt
        # Load local
        with open(prompt_path, 'r') as f:
            list = json.load(f)
        keys = [
            ['shot', 'COMBO', {"key": "shot_list"}], ['shot_weight', 'FLOAT'],
            ['gender', 'COMBO', {"default": "Woman", "key": "gender_list"}], ['age', 'INT', {"default": 30, "min": 18, "max": 90, "step": 1, "display": "slider"}],
            ['nationality_1', 'COMBO', {"default": "Chinese", "key": "nationality_list"}], ['nationality_2', 'COMBO', {"key": "nationality_list"}], ['nationality_mix', 'FLOAT'],
            ['body_type', 'COMBO', {"key": "body_type_list"}], ['body_type_weight', 'FLOAT'], ['model_pose', 'COMBO', {"key": "model_pose_list"}], ['eyes_color', 'COMBO', {"key": "eyes_color_list"}],
            ['facial_expression', 'COMBO', {"key": "face_expression_list"}], ['facial_expression_weight', 'FLOAT'], ['face_shape', 'COMBO', {"key": "face_shape_list"}], ['face_shape_weight', 'FLOAT'], ['facial_asymmetry', 'FLOAT'],
            ['hair_style', 'COMBO', {"key": "hair_style_list"}], ['hair_color', 'COMBO', {"key": "hair_color_list"}], ['disheveled', 'FLOAT'], ['beard', 'COMBO', {"key": "beard_list"}],
            ['skin_details', 'FLOAT'], ['skin_pores', 'FLOAT'], ['dimples', 'FLOAT'], ['freckles', 'FLOAT'],
            ['moles', 'FLOAT'], ['skin_imperfections', 'FLOAT'], ['skin_acne', 'FLOAT'], ['tanned_skin', 'FLOAT'],
            ['eyes_details', 'FLOAT'], ['iris_details', 'FLOAT'], ['circular_iris', 'FLOAT'], ['circular_pupil', 'FLOAT'],
            ['light_type', 'COMBO', {"key": "light_type_list"}], ['light_direction', 'COMBO', {"key": "light_direction_list"}], ['light_weight', 'FLOAT']
        ]
        widgets = {}
        for i, obj in enumerate(keys):
            if obj[1] == 'COMBO':
                key = obj[2]['key'] if obj[2] and 'key' in obj[2] else obj[0]
                _list = list[key].copy()
                _list.insert(0, '-')
                widgets[obj[0]] = (_list, {**obj[2]})
            elif obj[1] == 'FLOAT':
                widgets[obj[0]] = ("FLOAT", {"default": 0, "step": 0.05, "min": 0, "max": max_float_value, "display": "slider",})
            elif obj[1] == 'INT':
                widgets[obj[0]] = (obj[1], obj[2])
        del list
        return {
            "required": {
                **widgets,
                "photorealism_improvement": (["enable", "disable"],),
                "prompt_start": ("STRING", {"multiline": True, "default": "raw photo, (realistic:1.5)"}),
                "prompt_additional": ("STRING", {"multiline": True, "default": ""}),
                "prompt_end": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("positive", "negative",)

    FUNCTION = "pm"

    CATEGORY = "EasyUse/Prompt"

    def pm(self, shot="-", shot_weight=1, gender="-", body_type="-", body_type_weight=0, eyes_color="-",
           facial_expression="-", facial_expression_weight=0, face_shape="-", face_shape_weight=0,
           nationality_1="-", nationality_2="-", nationality_mix=0.5, age=30, hair_style="-", hair_color="-",
           disheveled=0, dimples=0, freckles=0, skin_pores=0, skin_details=0, moles=0, skin_imperfections=0,
           wrinkles=0, tanned_skin=0, eyes_details=1, iris_details=1, circular_iris=1, circular_pupil=1,
           facial_asymmetry=0, prompt_additional="", prompt_start="", prompt_end="", light_type="-",
           light_direction="-", light_weight=0, negative_prompt="", photorealism_improvement="disable", beard="-",
           model_pose="-", skin_acne=0):

        prompt = []

        if gender == "-":
            gender = ""
        else:
            if age <= 25 and gender == 'Woman':
                gender = 'girl'
            if age <= 25 and gender == 'Man':
                gender = 'boy'
            gender = " " + gender + " "

        if nationality_1 != '-' and nationality_2 != '-':
            nationality = f"[{nationality_1}:{nationality_2}:{round(nationality_mix, 2)}]"
        elif nationality_1 != '-':
            nationality = nationality_1 + " "
        elif nationality_2 != '-':
            nationality = nationality_2 + " "
        else:
            nationality = ""

        if prompt_start != "":
            prompt.append(f"{prompt_start}")

        if shot != "-" and shot_weight > 0:
            prompt.append(f"({shot}:{round(shot_weight, 2)})")

        prompt.append(f"({nationality}{gender}{round(age)}-years-old:1.5)")

        if body_type != "-" and body_type_weight > 0:
            prompt.append(f"({body_type}, {body_type} body:{round(body_type_weight, 2)})")

        if model_pose != "-":
            prompt.append(f"({model_pose}:1.5)")

        if eyes_color != "-":
            prompt.append(f"({eyes_color} eyes:1.25)")

        if facial_expression != "-" and facial_expression_weight > 0:
            prompt.append(
                f"({facial_expression}, {facial_expression} expression:{round(facial_expression_weight, 2)})")

        if face_shape != "-" and face_shape_weight > 0:
            prompt.append(f"({face_shape} shape face:{round(face_shape_weight, 2)})")

        if hair_style != "-":
            prompt.append(f"({hair_style} hairstyle:1.25)")

        if hair_color != "-":
            prompt.append(f"({hair_color} hair:1.25)")

        if beard != "-":
            prompt.append(f"({beard}:1.15)")

        if disheveled != "-" and disheveled > 0:
            prompt.append(f"(disheveled:{round(disheveled, 2)})")

        if prompt_additional != "":
            prompt.append(f"{prompt_additional}")

        if skin_details > 0:
            prompt.append(f"(skin details, skin texture:{round(skin_details, 2)})")

        if skin_pores > 0:
            prompt.append(f"(skin pores:{round(skin_pores, 2)})")

        if skin_imperfections > 0:
            prompt.append(f"(skin imperfections:{round(skin_imperfections, 2)})")

        if skin_acne > 0:
            prompt.append(f"(acne, skin with acne:{round(skin_acne, 2)})")

        if wrinkles > 0:
            prompt.append(f"(skin imperfections:{round(wrinkles, 2)})")

        if tanned_skin > 0:
            prompt.append(f"(tanned skin:{round(tanned_skin, 2)})")

        if dimples > 0:
            prompt.append(f"(dimples:{round(dimples, 2)})")

        if freckles > 0:
            prompt.append(f"(freckles:{round(freckles, 2)})")

        if moles > 0:
            prompt.append(f"(skin pores:{round(moles, 2)})")

        if eyes_details > 0:
            prompt.append(f"(eyes details:{round(eyes_details, 2)})")

        if iris_details > 0:
            prompt.append(f"(iris details:{round(iris_details, 2)})")

        if circular_iris > 0:
            prompt.append(f"(circular iris:{round(circular_iris, 2)})")

        if circular_pupil > 0:
            prompt.append(f"(circular pupil:{round(circular_pupil, 2)})")

        if facial_asymmetry > 0:
            prompt.append(f"(facial asymmetry, face asymmetry:{round(facial_asymmetry, 2)})")

        if light_type != '-' and light_weight > 0:
            if light_direction != '-':
                prompt.append(f"({light_type} {light_direction}:{round(light_weight, 2)})")
            else:
                prompt.append(f"({light_type}:{round(light_weight, 2)})")

        if prompt_end != "":
            prompt.append(f"{prompt_end}")

        prompt = ", ".join(prompt)
        prompt = prompt.lower()

        if photorealism_improvement == "enable":
            prompt = prompt + ", (professional photo, balanced photo, balanced exposure:1.2), (film grain:1.15)"

        if photorealism_improvement == "enable":
            negative_prompt = negative_prompt + ", (shinny skin, reflections on the skin, skin reflections:1.25)"

        log_node_info("Portrait Master as generate the prompt:", prompt)

        return (prompt, negative_prompt,)

# 潜空间sigma相乘
class latentNoisy:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            "steps": ("INT", {"default": 10000, "min": 0, "max": 10000}),
            "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
            "end_at_step": ("INT", {"default": 10000, "min": 1, "max": 10000}),
            "source": (["CPU", "GPU"],),
            "seed_num": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        },
        "optional": {
            "pipe": ("PIPE_LINE",),
            "optional_model": ("MODEL",),
            "optional_latent": ("LATENT",)
        }}

    RETURN_TYPES = ("PIPE_LINE", "LATENT", "FLOAT",)
    RETURN_NAMES = ("pipe", "latent", "sigma",)
    FUNCTION = "run"

    CATEGORY = "EasyUse/Latent"

    def run(self, sampler_name, scheduler, steps, start_at_step, end_at_step, source, seed_num, pipe=None, optional_model=None, optional_latent=None):
        model = optional_model if optional_model is not None else pipe["model"]
        batch_size = pipe["loader_settings"]["batch_size"]
        empty_latent_height = pipe["loader_settings"]["empty_latent_height"]
        empty_latent_width = pipe["loader_settings"]["empty_latent_width"]

        if optional_latent is not None:
            samples = optional_latent
        else:
            torch.manual_seed(seed_num)
            if source == "CPU":
                device = "cpu"
            else:
                device = comfy.model_management.get_torch_device()
            noise = torch.randn((batch_size, 4, empty_latent_height // 8, empty_latent_width // 8), dtype=torch.float32,
                                device=device).cpu()

            samples = {"samples": noise}

        device = comfy.model_management.get_torch_device()
        end_at_step = min(steps, end_at_step)
        start_at_step = min(start_at_step, end_at_step)
        real_model = None
        comfy.model_management.load_model_gpu(model)
        real_model = model.model
        sampler = comfy.samplers.KSampler(real_model, steps=steps, device=device, sampler=sampler_name,
                                          scheduler=scheduler, denoise=1.0, model_options=model.model_options)
        sigmas = sampler.sigmas
        sigma = sigmas[start_at_step] - sigmas[end_at_step]
        sigma /= model.model.latent_format.scale_factor
        sigma = sigma.cpu().numpy()

        samples_out = samples.copy()

        s1 = samples["samples"]
        samples_out["samples"] = s1 * sigma

        if pipe is None:
            pipe = {}
        new_pipe = {
            **pipe,
            "samples": samples_out
        }
        del pipe

        return (new_pipe, samples_out, sigma)


# Latent遮罩复合
class latentCompositeMaskedWithCond:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "text_combine": ("LIST",),
                "source_latent": ("LATENT",),
                "source_mask": ("MASK",),
                "destination_mask": ("MASK",),
                "text_combine_mode": (["add", "replace", "cover"], {"default": "add"}),
                "replace_text": ("STRING", {"default": ""})
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
        }

    OUTPUT_IS_LIST = (False, False, True)
    RETURN_TYPES = ("PIPE_LINE", "LATENT", "CONDITIONING")
    RETURN_NAMES = ("pipe", "latent", "conditioning",)
    FUNCTION = "run"
    OUTPUT_NODE = True

    CATEGORY = "EasyUse/Latent"

    def run(self, pipe, text_combine, source_latent, source_mask, destination_mask, text_combine_mode, replace_text, prompt=None, extra_pnginfo=None, my_unique_id=None):

        clip = pipe["clip"]
        destination_latent = pipe["samples"]

        conds = []

        for text in text_combine:
            if text_combine_mode == 'cover':
                positive = text
            elif text_combine_mode == 'replace' and replace_text != '':
                positive = pipe["loader_settings"]["positive"].replace(replace_text, text)
            else:
                positive = pipe["loader_settings"]["positive"] + ',' + text
            positive_token_normalization = pipe["loader_settings"]["positive_token_normalization"]
            positive_weight_interpretation = pipe["loader_settings"]["positive_weight_interpretation"]
            a1111_prompt_style = pipe["loader_settings"]["a1111_prompt_style"]
            positive_cond = pipe["positive"]

            log_node_warn("正在处理提示词编码...")
            # Use new clip text encode by smzNodes like same as webui, when if you installed the smzNodes
            if a1111_prompt_style:
                if "smZ CLIPTextEncode" in ALL_NODE_CLASS_MAPPINGS:
                    cls = ALL_NODE_CLASS_MAPPINGS['smZ CLIPTextEncode']
                    steps = pipe["loader_settings"]["steps"] if "steps" in pipe["loader_settings"] else 5
                    positive_embeddings_final, = cls().encode(clip, positive, "A1111", True, True, False, False, 6, 1024,
                                                              1024, 0, 0, 1024, 1024, '', '', steps)
                else:
                    raise Exception(f"[ERROR] To use clip text encode same as webui, you need to install 'smzNodes'")
            else:
                positive_embeddings_final, positive_pooled = advanced_encode(clip, positive,
                                                                             positive_token_normalization,
                                                                             positive_weight_interpretation, w_max=1.0,
                                                                             apply_to_pooled='enable')
                positive_embeddings_final = [[positive_embeddings_final, {"pooled_output": positive_pooled}]]

            # source cond
            (cond_1,) = ConditioningSetMask().append(positive_cond, source_mask, "default", 1)
            (cond_2,) = ConditioningSetMask().append(positive_embeddings_final, destination_mask, "default", 1)
            positive_cond = cond_1 + cond_2

            conds.append(positive_cond)
        # latent composite masked
        (samples,) = LatentCompositeMasked().composite(destination_latent, source_latent, 0, 0, False)

        new_pipe = {
            **pipe,
            "samples": samples,
            "loader_settings": {
                **pipe["loader_settings"],
                "positive": positive,
            }
        }

        del pipe

        return (new_pipe, samples, conds)

# 随机种
class easySeed:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed_num": ("INT", {"default": 0, "min": 0, "max": 1125899906842624}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed_num",)
    FUNCTION = "doit"

    CATEGORY = "EasyUse/Seed"

    OUTPUT_NODE = True

    def doit(self, seed_num=0, prompt=None, extra_pnginfo=None, my_unique_id=None):
        return seed_num,
# 全局随机种
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

    CATEGORY = "EasyUse/Seed"

    OUTPUT_NODE = True

    def doit(self, **kwargs):
        return {}

#---------------------------------------------------------------提示词 结束------------------------------------------------------------------------#

#---------------------------------------------------------------加载器 开始----------------------------------------------------------------------#

# 简易加载器完整
class fullLoader:
    @classmethod
    def INPUT_TYPES(cls):
        resolution_strings = [f"{width} x {height}" for width, height in BASE_RESOLUTIONS]
        a1111_prompt_style_default = False

        return {"required": {
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            "config_name": (["Default", ] + folder_paths.get_filename_list("configs"), {"default": "Default"}),
            "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
            "clip_skip": ("INT", {"default": -1, "min": -24, "max": 0, "step": 1}),

            "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
            "lora_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

            "resolution": (resolution_strings,),
            "empty_latent_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "empty_latent_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),

            "positive": ("STRING", {"default": "Positive", "multiline": True}),
            "positive_token_normalization": (["none", "mean", "length", "length+mean"],),
            "positive_weight_interpretation": (["comfy",  "A1111", "comfy++", "compel", "fixed attention"],),

            "negative": ("STRING", {"default": "Negative", "multiline": True}),
            "negative_token_normalization": (["none", "mean", "length", "length+mean"],),
            "negative_weight_interpretation": (["comfy",  "A1111", "comfy++", "compel", "fixed attention"],),

            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
        },
            "optional": {"model_override": ("MODEL",), "clip_override": ("CLIP",), "vae_override": ("VAE",), "optional_lora_stack": ("LORA_STACK",), "a1111_prompt_style": ("BOOLEAN", {"default": a1111_prompt_style_default}),},
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE", "CLIP")
    RETURN_NAMES = ("pipe", "model", "vae", "clip")

    FUNCTION = "adv_pipeloader"
    CATEGORY = "EasyUse/Loaders"

    def adv_pipeloader(self, ckpt_name, config_name, vae_name, clip_skip,
                       lora_name, lora_model_strength, lora_clip_strength,
                       resolution, empty_latent_width, empty_latent_height,
                       positive, positive_token_normalization, positive_weight_interpretation,
                       negative, negative_token_normalization, negative_weight_interpretation,
                       batch_size, model_override=None, clip_override=None, vae_override=None, optional_lora_stack=None, a1111_prompt_style=False, prompt=None,
                       my_unique_id=None
                       ):

        model: ModelPatcher | None = None
        clip: CLIP | None = None
        vae: VAE | None = None
        can_load_lora = True
        pipe_lora_stack = []

        # resolution
        if resolution != "自定义 x 自定义":
            try:
                width, height = map(int, resolution.split(' x '))
                empty_latent_width = width
                empty_latent_height = height
            except ValueError:
                raise ValueError("Invalid base_resolution format.")

        # Create Empty Latent
        latent = torch.zeros([batch_size, 4, empty_latent_height // 8, empty_latent_width // 8]).cpu()
        samples = {"samples": latent}

        # Clean models from loaded_objects
        easyCache.update_loaded_objects(prompt)

        log_node_warn("正在处理模型...")
        # 判断是否存在 模型叠加xyplot, 若存在优先缓存第一个模型
        xyinputs_id = next((x for x in prompt if str(prompt[x]["class_type"]) == "easy XYInputs: ModelMergeBlocks"), None)
        if xyinputs_id is not None:
            node = prompt[xyinputs_id]
            if "ckpt_name_1" in node["inputs"]:
                ckpt_name_1 = node["inputs"]["ckpt_name_1"]
                model, clip, vae = easyCache.load_checkpoint(ckpt_name_1)
                can_load_lora = False

        # Load models
        elif model_override is not None and clip_override is not None and vae_override is not None:
            model = model_override
            clip = clip_override
            vae = vae_override
        elif model_override is not None:
            raise Exception(f"[ERROR] clip or vae is missing")
        elif vae_override is not None:
            raise Exception(f"[ERROR] model or clip is missing")
        elif clip_override is not None:
            raise Exception(f"[ERROR] model or vae is missing")
        else:
            model, clip, vae = easyCache.load_checkpoint(ckpt_name, config_name)

        if optional_lora_stack is not None:
            for lora in optional_lora_stack:
                if can_load_lora:
                    model, clip = easyCache.load_lora(lora[0], model, clip, lora[1], lora[2])
                pipe_lora_stack.append({"lora_name": lora[0], "model": model, "clip": clip, "lora_model_strength": lora[1], "lora_clip_strength": lora[2]})

        if lora_name != "None":
            if can_load_lora:
                model, clip = easyCache.load_lora(lora_name, model, clip, lora_model_strength, lora_clip_strength)
            pipe_lora_stack.append({"lora_name": lora_name, "model": model, "clip": clip, "lora_model_strength": lora_model_strength,
                                    "lora_clip_strength": lora_clip_strength})

        # Check for custom VAE
        if vae_name not in ["Baked VAE", "Baked-VAE"]:
            vae = easyCache.load_vae(vae_name)
        # CLIP skip
        if not clip:
            raise Exception("No CLIP found")

        # 判断是否连接 styles selector
        is_positive_linked_styles_selector = False
        inputs_positive_values = prompt[my_unique_id]['inputs']['positive'] if "positive" in prompt[my_unique_id]['inputs'] else None
        if type(inputs_positive_values) == list and inputs_positive_values != 'undefined' and inputs_positive_values[0]:
             is_positive_linked_styles_selector = True if prompt[inputs_positive_values[0]] and prompt[inputs_positive_values[0]]['class_type'] == 'easy stylesSelector' else False
        is_negative_linked_styles_selector = False
        inputs_negative_values = prompt[my_unique_id]['inputs']['negative'] if "negative" in prompt[my_unique_id]['inputs'] else None
        if type(inputs_negative_values) == list and inputs_negative_values != 'undefined' and inputs_negative_values[0]:
            is_negative_linked_styles_selector = True if prompt[inputs_negative_values[0]] and prompt[inputs_negative_values[0]]['class_type'] == 'easy stylesSelector' else False

        log_node_warn("正在处理提示词...")
        positive_seed = find_wildcards_seed(my_unique_id, positive, prompt)
        model, clip, positive, positive_decode, show_positive_prompt, pipe_lora_stack = process_with_loras(positive, model, clip, "Positive", positive_seed, can_load_lora, pipe_lora_stack)
        positive_wildcard_prompt = positive_decode if show_positive_prompt or is_positive_linked_styles_selector else ""
        negative_seed = find_wildcards_seed(my_unique_id, negative, prompt)
        model, clip, negative, negative_decode, show_negative_prompt, pipe_lora_stack = process_with_loras(negative, model, clip,
                                                                                          "Negative", negative_seed, can_load_lora, pipe_lora_stack)
        negative_wildcard_prompt = negative_decode if show_negative_prompt or is_negative_linked_styles_selector else ""

        clipped = clip.clone()
        if clip_skip != 0 and can_load_lora:
            clipped.clip_layer(clip_skip)

        log_node_warn("正在处理提示词编码...")
        # Use new clip text encode by smzNodes like same as webui, when if you installed the smzNodes
        if a1111_prompt_style:
            if "smZ CLIPTextEncode" in ALL_NODE_CLASS_MAPPINGS:
                cls = ALL_NODE_CLASS_MAPPINGS['smZ CLIPTextEncode']
                steps = find_nearest_steps(my_unique_id, prompt)
                positive_embeddings_final, = cls().encode(clipped, positive, "A1111", True, True, False, False, 6, 1024, 1024, 0, 0, 1024, 1024, '', '', steps)
                negative_embeddings_final, = cls().encode(clipped, negative, "A1111", True, True, False, False, 6, 1024, 1024, 0, 0, 1024, 1024, '', '', steps)
            else:
                raise Exception(f"[ERROR] To use clip text encode same as webui, you need to install 'smzNodes'")
        else:
            positive_embeddings_final, positive_pooled = advanced_encode(clipped, positive, positive_token_normalization,
                                                                         positive_weight_interpretation, w_max=1.0,
                                                                         apply_to_pooled='enable')
            positive_embeddings_final = [[positive_embeddings_final, {"pooled_output": positive_pooled}]]

            negative_embeddings_final, negative_pooled = advanced_encode(clipped, negative, negative_token_normalization,
                                                                         negative_weight_interpretation, w_max=1.0,
                                                                         apply_to_pooled='enable')
            negative_embeddings_final = [[negative_embeddings_final, {"pooled_output": negative_pooled}]]
        image = easySampler.pil2tensor(Image.new('RGB', (1, 1), (0, 0, 0)))

        log_node_warn("处理结束...")
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

                                    "lora_stack": pipe_lora_stack,

                                    "refiner_ckpt_name": None,
                                    "refiner_vae_name": None,
                                    "refiner_lora_name": None,
                                    "refiner_lora_model_strength": None,
                                    "refiner_lora_clip_strength": None,

                                    "clip_skip": clip_skip,
                                    "a1111_prompt_style": a1111_prompt_style,
                                    "positive": positive,
                                    "positive_l": None,
                                    "positive_g": None,
                                    "positive_token_normalization": positive_token_normalization,
                                    "positive_weight_interpretation": positive_weight_interpretation,
                                    "positive_balance": None,
                                    "negative": negative,
                                    "negative_l": None,
                                    "negative_g": None,
                                    "negative_token_normalization": negative_token_normalization,
                                    "negative_weight_interpretation": negative_weight_interpretation,
                                    "negative_balance": None,
                                    "empty_latent_width": empty_latent_width,
                                    "empty_latent_height": empty_latent_height,
                                    "batch_size": batch_size,
                                    "seed": 0,
                                    "empty_samples": samples, }
                }

        return {"ui": {"positive": positive_wildcard_prompt, "negative": negative_wildcard_prompt}, "result": (pipe, model, vae, clip)}

# A1111简易加载器
class a1111Loader:
    @classmethod
    def INPUT_TYPES(cls):
        resolution_strings = [f"{width} x {height}" for width, height in BASE_RESOLUTIONS]
        a1111_prompt_style_default = False
        checkpoints = folder_paths.get_filename_list("checkpoints")
        loras = ["None"] + folder_paths.get_filename_list("loras")
        return {"required": {
            "ckpt_name": (checkpoints,),
            "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
            "clip_skip": ("INT", {"default": -1, "min": -24, "max": 0, "step": 1}),

            "lora_name": (loras,),
            "lora_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

            "resolution": (resolution_strings, {"default": "512 x 512"}),
            "empty_latent_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "empty_latent_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),

            "positive": ("STRING", {"default": "Positive", "multiline": True}),
            "negative": ("STRING", {"default": "Negative", "multiline": True}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
        },
            "optional": {"optional_lora_stack": ("LORA_STACK",), "a1111_prompt_style": ("BOOLEAN", {"default": a1111_prompt_style_default})},
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE")
    RETURN_NAMES = ("pipe", "model", "vae")

    FUNCTION = "adv_pipeloader"
    CATEGORY = "EasyUse/Loaders"

    def adv_pipeloader(self, ckpt_name, vae_name, clip_skip,
                       lora_name, lora_model_strength, lora_clip_strength,
                       resolution, empty_latent_width, empty_latent_height,
                       positive, negative, batch_size, optional_lora_stack=None, a1111_prompt_style=False, prompt=None,
                       my_unique_id=None):

        return fullLoader.adv_pipeloader(self, ckpt_name, 'Default', vae_name, clip_skip,
             lora_name, lora_model_strength, lora_clip_strength,
             resolution, empty_latent_width, empty_latent_height,
             positive, 'mean', 'A1111',
             negative,'mean','A1111',
             batch_size, None, None, None, optional_lora_stack, a1111_prompt_style, prompt,
             my_unique_id
        )

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

            "resolution": (resolution_strings, {"default": "512 x 512"}),
            "empty_latent_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "empty_latent_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),

            "positive": ("STRING", {"default": "Positive", "multiline": True}),
            "negative": ("STRING", {"default": "Negative", "multiline": True}),

            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
        },
            "optional": {"optional_lora_stack": ("LORA_STACK",)},
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE")
    RETURN_NAMES = ("pipe", "model", "vae")

    FUNCTION = "adv_pipeloader"
    CATEGORY = "EasyUse/Loaders"

    def adv_pipeloader(self, ckpt_name, vae_name, clip_skip,
                       lora_name, lora_model_strength, lora_clip_strength,
                       resolution, empty_latent_width, empty_latent_height,
                       positive, negative, batch_size, optional_lora_stack=None, prompt=None,
                      my_unique_id=None):
        return fullLoader.adv_pipeloader(self, ckpt_name, 'Default', vae_name, clip_skip,
             lora_name, lora_model_strength, lora_clip_strength,
             resolution, empty_latent_width, empty_latent_height,
             positive, 'none', 'comfy',
             negative, 'none', 'comfy',
             batch_size, None, None, None, optional_lora_stack, False, prompt,
             my_unique_id
         )

# Zero123简易加载器 (3D)
try:
    from comfy_extras.nodes_stable3d import camera_embeddings
except FileNotFoundError:
    log_node_error("EasyUse[zero123Loader]", "请更新ComfyUI到最新版本")

class zero123Loader:

    @classmethod
    def INPUT_TYPES(cls):
        def get_file_list(filenames):
            return [file for file in filenames if file != "put_models_here.txt" and "zero123" in file]

        return {"required": {
            "ckpt_name": (get_file_list(folder_paths.get_filename_list("checkpoints")),),
            "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),

            "init_image": ("IMAGE",),
            "empty_latent_width": ("INT", {"default": 256, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
            "empty_latent_height": ("INT", {"default": 256, "min": 16, "max": MAX_RESOLUTION, "step": 8}),

            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),

            "elevation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
            "azimuth": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
        },
            "hidden": {"prompt": "PROMPT"}, "my_unique_id": "UNIQUE_ID"}

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE")
    RETURN_NAMES = ("pipe", "model", "vae")

    FUNCTION = "adv_pipeloader"
    CATEGORY = "EasyUse/Loaders"

    def adv_pipeloader(self, ckpt_name, vae_name, init_image, empty_latent_width, empty_latent_height, batch_size, elevation, azimuth, prompt=None, my_unique_id=None):
        model: ModelPatcher | None = None
        vae: VAE | None = None
        clip: CLIP | None = None
        clip_vision = None

        # Clean models from loaded_objects
        easyCache.update_loaded_objects(prompt)

        model, clip_vision, vae = easyCache.load_checkpoint(ckpt_name, "Default", True)

        output = clip_vision.encode_image(init_image)
        pooled = output.image_embeds.unsqueeze(0)
        pixels = comfy.utils.common_upscale(init_image.movedim(-1, 1), empty_latent_width, empty_latent_height, "bilinear", "center").movedim(1, -1)
        encode_pixels = pixels[:, :, :, :3]
        t = vae.encode(encode_pixels)
        cam_embeds = camera_embeddings(elevation, azimuth)
        cond = torch.cat([pooled, cam_embeds.repeat((pooled.shape[0], 1, 1))], dim=-1)

        positive = [[cond, {"concat_latent_image": t}]]
        negative = [[torch.zeros_like(pooled), {"concat_latent_image": torch.zeros_like(t)}]]
        latent = torch.zeros([batch_size, 4, empty_latent_height // 8, empty_latent_width // 8])
        samples = {"samples": latent}

        image = easySampler.pil2tensor(Image.new('RGB', (1, 1), (0, 0, 0)))

        pipe = {"model": model,
                "positive": positive,
                "negative": negative,
                "vae": vae,
                "clip": clip,

                "samples": samples,
                "images": image,
                "seed": 0,

                "loader_settings": {"ckpt_name": ckpt_name,
                                    "vae_name": vae_name,

                                    "positive": positive,
                                    "positive_l": None,
                                    "positive_g": None,
                                    "positive_balance": None,
                                    "negative": negative,
                                    "negative_l": None,
                                    "negative_g": None,
                                    "negative_balance": None,
                                    "empty_latent_width": empty_latent_width,
                                    "empty_latent_height": empty_latent_height,
                                    "batch_size": batch_size,
                                    "seed": 0,
                                    "empty_samples": samples, }
                }

        return (pipe, model, vae)

#svd加载器
class svdLoader:

    @classmethod
    def INPUT_TYPES(cls):
        resolution_strings = [f"{width} x {height}" for width, height in BASE_RESOLUTIONS]
        def get_file_list(filenames):
            return [file for file in filenames if file != "put_models_here.txt" and "svd" in file]

        return {"required": {
            "ckpt_name": (get_file_list(folder_paths.get_filename_list("checkpoints")),),
            "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),

            "init_image": ("IMAGE",),
            "resolution": (resolution_strings, {"default": "1024 x 576"}),
            "empty_latent_width": ("INT", {"default": 256, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
            "empty_latent_height": ("INT", {"default": 256, "min": 16, "max": MAX_RESOLUTION, "step": 8}),

            "video_frames": ("INT", {"default": 14, "min": 1, "max": 4096}),
            "motion_bucket_id": ("INT", {"default": 127, "min": 1, "max": 1023}),
            "fps": ("INT", {"default": 6, "min": 1, "max": 1024}),
            "augmentation_level": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01})
        },
            "hidden": {"prompt": "PROMPT"}, "my_unique_id": "UNIQUE_ID"}

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE")
    RETURN_NAMES = ("pipe", "model", "vae")

    FUNCTION = "adv_pipeloader"
    CATEGORY = "EasyUse/Loaders"

    def adv_pipeloader(self, ckpt_name, vae_name, init_image, resolution, empty_latent_width, empty_latent_height, video_frames, motion_bucket_id, fps, augmentation_level, prompt=None, my_unique_id=None):
        model: ModelPatcher | None = None
        vae: VAE | None = None
        clip: CLIP | None = None
        clip_vision = None

        # resolution
        if resolution != "自定义 x 自定义":
            try:
                width, height = map(int, resolution.split(' x '))
                empty_latent_width = width
                empty_latent_height = height
            except ValueError:
                raise ValueError("Invalid base_resolution format.")

        # Clean models from loaded_objects
        easyCache.update_loaded_objects(prompt)

        model, clip_vision, vae = easyCache.load_checkpoint(ckpt_name, "Default", True)

        output = clip_vision.encode_image(init_image)
        pooled = output.image_embeds.unsqueeze(0)
        pixels = comfy.utils.common_upscale(init_image.movedim(-1, 1), empty_latent_width, empty_latent_height, "bilinear", "center").movedim(1,
                                                                                                                    -1)
        encode_pixels = pixels[:, :, :, :3]
        if augmentation_level > 0:
            encode_pixels += torch.randn_like(pixels) * augmentation_level
        t = vae.encode(encode_pixels)
        positive = [[pooled,
                     {"motion_bucket_id": motion_bucket_id, "fps": fps, "augmentation_level": augmentation_level,
                      "concat_latent_image": t}]]
        negative = [[torch.zeros_like(pooled),
                     {"motion_bucket_id": motion_bucket_id, "fps": fps, "augmentation_level": augmentation_level,
                      "concat_latent_image": torch.zeros_like(t)}]]
        latent = torch.zeros([video_frames, 4, empty_latent_height // 8, empty_latent_width // 8])
        samples = {"samples": latent}

        image = easySampler.pil2tensor(Image.new('RGB', (1, 1), (0, 0, 0)))

        pipe = {"model": model,
                "positive": positive,
                "negative": negative,
                "vae": vae,
                "clip": clip,

                "samples": samples,
                "images": image,
                "seed": 0,

                "loader_settings": {"ckpt_name": ckpt_name,
                                    "vae_name": vae_name,

                                    "positive": positive,
                                    "positive_l": None,
                                    "positive_g": None,
                                    "positive_balance": None,
                                    "negative": negative,
                                    "negative_l": None,
                                    "negative_g": None,
                                    "negative_balance": None,
                                    "empty_latent_width": empty_latent_width,
                                    "empty_latent_height": empty_latent_height,
                                    "batch_size": 1,
                                    "seed": 0,
                                    "empty_samples": samples, }
                }

        return (pipe, model, vae)


# lora
class loraStackLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        max_lora_num = 10
        inputs = {
            "required": {
                "toggle": ([True, False],),
                "mode": (["simple", "advanced"],),
                "num_loras": ("INT", {"default": 1, "min": 0, "max": max_lora_num}),
            },
            "optional": {
                "optional_lora_stack": ("LORA_STACK",),
            },
        }

        for i in range(1, max_lora_num+1):
            inputs["optional"][f"lora_{i}_name"] = (
            ["None"] + folder_paths.get_filename_list("loras"), {"default": "None"})
            inputs["optional"][f"lora_{i}_strength"] = (
            "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["optional"][f"lora_{i}_model_strength"] = (
            "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["optional"][f"lora_{i}_clip_strength"] = (
            "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})

        return inputs

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "stack"

    CATEGORY = "EasyUse/Loaders"

    def stack(self, toggle, mode, num_loras, lora_stack=None, **kwargs):
        if (toggle in [False, None, "False"]) or not kwargs:
            return (None,)

        loras = []

        # Import Stack values
        if lora_stack is not None:
            loras.extend([l for l in lora_stack if l[0] != "None"])

        # Import Lora values
        for i in range(1, num_loras + 1):
            lora_name = kwargs.get(f"lora_{i}_name")

            if not lora_name or lora_name == "None":
                continue

            if mode == "simple":
                lora_strength = float(kwargs.get(f"lora_{i}_strength"))
                loras.append((lora_name, lora_strength, lora_strength))
            elif mode == "advanced":
                model_strength = float(kwargs.get(f"lora_{i}_model_strength"))
                clip_strength = float(kwargs.get(f"lora_{i}_clip_strength"))
                loras.append((lora_name, model_strength, clip_strength))
        return (loras,)

class controlnetNameStack:

    def get_file_list(filenames):
        return [file for file in filenames if file != "put_models_here.txt" and "lllite" not in file]

    controlnets = ["None"] + get_file_list(folder_paths.get_filename_list("controlnet"))

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {},
            "optional": {
                "switch_1": (["Off", "On"],),
                "controlnet_1": (s.controlnets,),
                "controlnet_strength_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "start_percent_1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "switch_2": (["Off", "On"],),
                "controlnet_2": (s.controlnets,),
                "controlnet_strength_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "start_percent_2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "switch_3": (["Off", "On"],),
                "controlnet_3": (s.controlnets,),
                "controlnet_strength_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "start_percent_3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "controlnet_stack": ("CONTROL_NET_STACK",)
            },
        }
# controlnet
class controlnetSimple:
    @classmethod
    def INPUT_TYPES(s):
        def get_file_list(filenames):
            return [file for file in filenames if file != "put_models_here.txt" and "lllite" not in file]

        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "image": ("IMAGE",),
                "control_net_name": (get_file_list(folder_paths.get_filename_list("controlnet")),),
            },
            "optional": {
                "control_net": ("CONTROL_NET",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "scale_soft_weights": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},),
            }
        }

    RETURN_TYPES = ("PIPE_LINE", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("pipe", "positive", "negative")
    OUTPUT_NODE = True

    FUNCTION = "controlnetApply"
    CATEGORY = "EasyUse/Loaders"

    def controlnetApply(self, pipe, image, control_net_name, control_net=None, strength=1, scale_soft_weights=1):
        if control_net is None:
            if scale_soft_weights < 1:
                if "ScaledSoftControlNetWeights" in ALL_NODE_CLASS_MAPPINGS:
                    soft_weight_cls = ALL_NODE_CLASS_MAPPINGS['ScaledSoftControlNetWeights']
                    (weights, timestep_keyframe) = soft_weight_cls().load_weights(scale_soft_weights, False)
                    cn_adv_cls = ALL_NODE_CLASS_MAPPINGS['ControlNetLoaderAdvanced']
                    control_net, = cn_adv_cls().load_controlnet(control_net_name, timestep_keyframe)
                else:
                    raise Exception(f"[ERROR] To use Scale soft weight, you need to install 'COMFYUI-Advanced-ControlNet'")
            else:
                controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
                control_net = comfy.controlnet.load_controlnet(controlnet_path)

        control_hint = image.movedim(-1, 1)

        positive = pipe["positive"]
        negative = pipe["negative"]

        if strength != 0:
            if negative is None:
                p = []
                for t in positive:
                    n = [t[0], t[1].copy()]
                    c_net = control_net.copy().set_cond_hint(control_hint, strength)
                    if 'control' in t[1]:
                        c_net.set_previous_controlnet(t[1]['control'])
                    n[1]['control'] = c_net
                    n[1]['control_apply_to_uncond'] = True
                    p.append(n)
                positive = p
            else:
                cnets = {}
                out = []
                for conditioning in [positive, negative]:
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
                positive = out[0]
                negative = out[1]

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


        return (new_pipe, positive, negative)

# controlnetADV
class controlnetAdvanced:

    @classmethod
    def INPUT_TYPES(s):
        def get_file_list(filenames):
            return [file for file in filenames if file != "put_models_here.txt" and "lllite" not in file]

        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "image": ("IMAGE",),
                "control_net_name": (get_file_list(folder_paths.get_filename_list("controlnet")),),
            },
            "optional": {
                "control_net": ("CONTROL_NET",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "scale_soft_weights": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},),
            }
        }

    RETURN_TYPES = ("PIPE_LINE", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("pipe", "positive", "negative")
    OUTPUT_NODE = True

    FUNCTION = "controlnetApply"
    CATEGORY = "EasyUse/Loaders"


    def controlnetApply(self, pipe, image, control_net_name, control_net=None, strength=1, start_percent=0, end_percent=1, scale_soft_weights=1):
        if control_net is None:
            if scale_soft_weights < 1:
                if "ScaledSoftControlNetWeights" in ALL_NODE_CLASS_MAPPINGS:
                    soft_weight_cls = ALL_NODE_CLASS_MAPPINGS['ScaledSoftControlNetWeights']
                    (weights, timestep_keyframe) = soft_weight_cls().load_weights(scale_soft_weights, False)
                    cn_adv_cls = ALL_NODE_CLASS_MAPPINGS['ControlNetLoaderAdvanced']
                    control_net, = cn_adv_cls().load_controlnet(control_net_name, timestep_keyframe)
                else:
                    raise Exception(
                        f"[ERROR] To use Scale soft weight, you need to install 'COMFYUI-Advanced-ControlNet'")
            else:
                controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
                control_net = comfy.controlnet.load_controlnet(controlnet_path)

        control_hint = image.movedim(-1, 1)
        positive = pipe["positive"]
        negative = pipe["negative"]

        if strength != 0:
            if negative is None:
                p = []
                for t in positive:
                    n = [t[0], t[1].copy()]
                    c_net = control_net.copy().set_cond_hint(control_hint, strength)
                    if 'control' in t[1]:
                        c_net.set_previous_controlnet(t[1]['control'])
                    n[1]['control'] = c_net
                    n[1]['control_apply_to_uncond'] = True
                    p.append(n)
                positive = p
            else:
                cnets = {}
                out = []
                for conditioning in [positive, negative]:
                    c = []
                    for t in conditioning:
                        d = t[1].copy()

                        prev_cnet = d.get('control', None)
                        if prev_cnet in cnets:
                            c_net = cnets[prev_cnet]
                        else:
                            c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent))
                            c_net.set_previous_controlnet(prev_cnet)
                            cnets[prev_cnet] = c_net

                        d['control'] = c_net
                        d['control_apply_to_uncond'] = False
                        n = [t[0], d]
                        c.append(n)
                    out.append(c)
                positive = out[0]
                negative = out[1]

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

        del pipe

        return (new_pipe, positive, negative)

# FooocusInpaint (Testing)
from .fooocus import InpaintHead, InpaintWorker, get_local_filepath
inpaint_head_model = None
class fooocusInpaintLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "head": (list(FOOOCUS_INPAINT_HEAD.keys()),),
                "patch": (list(FOOOCUS_INPAINT_PATCH.keys()),),
            }
        }

    RETURN_TYPES = ("INPAINT_PATCH",)
    RETURN_NAMES = ("patch",)
    CATEGORY = "EasyUse/__for_testing"
    FUNCTION = "apply"

    def apply(self, head, patch):
        global inpaint_head_model

        head_file = get_local_filepath(FOOOCUS_INPAINT_HEAD[head]["model_url"], INPAINT_DIR)
        if inpaint_head_model is None:
            inpaint_head_model = InpaintHead()
            sd = torch.load(head_file, map_location='cpu')
            inpaint_head_model.load_state_dict(sd)

        patch_file = get_local_filepath(FOOOCUS_INPAINT_PATCH[patch]["model_url"], INPAINT_DIR)
        inpaint_lora = comfy.utils.load_torch_file(patch_file, safe_load=True)

        return ((inpaint_head_model, inpaint_lora),)
#---------------------------------------------------------------预采样 开始----------------------------------------------------------------------#

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
                     },
                "optional": {
                    "image_to_latent": ("IMAGE",),
                    "latent": ("LATENT",),
                },
                "hidden":
                    {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("PIPE_LINE", )
    RETURN_NAMES = ("pipe",)
    OUTPUT_NODE = True

    FUNCTION = "settings"
    CATEGORY = "EasyUse/PreSampling"

    def settings(self, pipe, steps, cfg, sampler_name, scheduler, denoise, seed_num, image_to_latent=None, latent=None, prompt=None, extra_pnginfo=None, my_unique_id=None):
        # 图生图转换
        vae = pipe["vae"]
        batch_size = pipe["loader_settings"]["batch_size"] if "batch_size" in pipe["loader_settings"] else 1
        if image_to_latent is not None:
            samples = {"samples": vae.encode(image_to_latent)}
            samples = RepeatLatentBatch().repeat(samples, batch_size)[0]
            images = image_to_latent
        elif latent is not None:
            samples = RepeatLatentBatch().repeat(latent, batch_size)[0]
            images = pipe["images"]
        else:
            samples = pipe["samples"]
            images = pipe["images"]

        new_pipe = {
            "model": pipe['model'],
            "positive": pipe['positive'],
            "negative": pipe['negative'],
            "vae": pipe['vae'],
            "clip": pipe['clip'],

            "samples": samples,
            "images": images,
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
                     },
                "optional": {
                    "image_to_latent": ("IMAGE",),
                    "latent": ("LATENT",)
                },
                "hidden":
                    {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("PIPE_LINE", )
    RETURN_NAMES = ("pipe",)
    OUTPUT_NODE = True

    FUNCTION = "settings"
    CATEGORY = "EasyUse/PreSampling"

    def settings(self, pipe, steps, cfg, sampler_name, scheduler, start_at_step, end_at_step, add_noise, seed_num, image_to_latent=None, latent=None, prompt=None, extra_pnginfo=None, my_unique_id=None):
        # 图生图转换
        vae = pipe["vae"]
        batch_size = pipe["loader_settings"]["batch_size"] if "batch_size" in pipe["loader_settings"] else 1
        if image_to_latent is not None:
            samples = {"samples": vae.encode(image_to_latent)}
            samples = RepeatLatentBatch().repeat(samples, batch_size)[0]
            images = image_to_latent
        elif latent is not None:
            samples = RepeatLatentBatch().repeat(latent, batch_size)[0]
            images = pipe["images"]
        else:
            samples = pipe["samples"]
            images = pipe["images"]

        new_pipe = {
            "model": pipe['model'],
            "positive": pipe['positive'],
            "negative": pipe['negative'],
            "vae": pipe['vae'],
            "clip": pipe['clip'],

            "samples": samples,
            "images": images,
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
               },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    OUTPUT_NODE = True

    FUNCTION = "settings"
    CATEGORY = "EasyUse/PreSampling"

    def settings(self, pipe, steps, cfg, sampler_name, eta, s_noise, upscale_ratio, start_step, end_step, upscale_n_step, unsharp_kernel_size, unsharp_sigma, unsharp_strength, seed_num, prompt=None, extra_pnginfo=None, my_unique_id=None):
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
                     },
                "optional":{
                    "image_to_latent": ("IMAGE",),
                    "latent": ("LATENT",)
                },
                "hidden":
                    {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    OUTPUT_NODE = True

    FUNCTION = "settings"
    CATEGORY = "EasyUse/PreSampling"

    def settings(self, pipe, steps, cfg, cfg_mode, cfg_scale_min,sampler_name, scheduler, denoise, seed_num, image_to_latent=None, latent=None, prompt=None, extra_pnginfo=None, my_unique_id=None):


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

        # 图生图转换
        vae = pipe["vae"]
        batch_size = pipe["loader_settings"]["batch_size"] if "batch_size" in pipe["loader_settings"] else 1
        if image_to_latent is not None:
            samples = {"samples": vae.encode(image_to_latent)}
            samples = RepeatLatentBatch().repeat(samples, batch_size)[0]
            images = image_to_latent
        elif latent is not None:
            samples = RepeatLatentBatch().repeat(latent, batch_size)[0]
            images = pipe["images"]
        else:
            samples = pipe["samples"]
            images = pipe["images"]

        new_pipe = {
            "model": m,
            "positive": pipe['positive'],
            "negative": pipe['negative'],
            "vae": pipe['vae'],
            "clip": pipe['clip'],

            "samples": samples,
            "images": images,
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

# 完整采样器
class samplerFull:

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
                 "image_output": (["Hide", "Preview", "Save", "Hide/Save", "Sender", "Sender/Save"],),
                 "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"}),
                 },
                "optional": {
                    "seed_num": ("INT", {"default": 0, "min": 0, "max": 1125899906842624}),
                    "model": ("MODEL",),
                    "positive": ("CONDITIONING",),
                    "negative": ("CONDITIONING",),
                    "latent": ("LATENT",),
                    "vae": ("VAE",),
                    "clip": ("CLIP",),
                    "xyPlot": ("XYPLOT",),
                },
                "hidden":
                  {"tile_size": "INT", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                    "embeddingsList": (folder_paths.get_filename_list("embeddings"),)
                  }
                }

    RETURN_TYPES = ("PIPE_LINE", "IMAGE", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "INT",)
    RETURN_NAMES = ("pipe",  "image", "model", "positive", "negative", "latent", "vae", "clip", "seed",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "EasyUse/Sampler"

    def run(self, pipe, steps, cfg, sampler_name, scheduler, denoise, image_output, link_id, save_prefix, seed_num=None, model=None, positive=None, negative=None, latent=None, vae=None, clip=None, xyPlot=None, tile_size=None, prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False, downscale_options=None):

        # Clean loaded_objects
        easyCache.update_loaded_objects(prompt)

        # my_unique_id = int(my_unique_id)

        # if my_unique_id:
        #     workflow = extra_pnginfo["workflow"]
        #     node = next((x for x in workflow["nodes"] if str(x["id"]) == my_unique_id), None)
        #     if node and 'seed_num' in prompt[my_unique_id]['inputs']:
        #         seed_num = prompt[my_unique_id]['inputs']['seed_num']
        #         length = len(node["widgets_values"])
        #         node["widgets_values"][length - 2] = seed_num

        easy_save = easySave(my_unique_id, prompt, extra_pnginfo)

        samp_model = model if model is not None else pipe["model"]
        samp_positive = positive if positive is not None else pipe["positive"]
        samp_negative = negative if negative is not None else pipe["negative"]
        samp_samples = latent if latent is not None else pipe["samples"]
        samp_vae = vae if vae is not None else pipe["vae"]
        samp_clip = clip if clip is not None else pipe["clip"]

        samp_seed = seed_num if seed_num is not None else pipe['seed']

        steps = steps if steps is not None else pipe['loader_settings']['steps']
        start_step = pipe['loader_settings']['start_step'] if 'start_step' in pipe['loader_settings'] else 0
        last_step = pipe['loader_settings']['last_step'] if 'last_step' in pipe['loader_settings'] else 10000
        cfg = cfg if cfg is not None else pipe['loader_settings']['cfg']
        sampler_name = sampler_name if sampler_name is not None else pipe['loader_settings']['sampler_name']
        scheduler = scheduler if scheduler is not None else pipe['loader_settings']['scheduler']
        denoise = denoise if denoise is not None else pipe['loader_settings']['denoise']
        add_noise = pipe['loader_settings']['add_noise'] if 'add_noise' in pipe['loader_settings'] else 'enabled'

        if start_step is not None and last_step is not None:
            force_full_denoise = True
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True

        def downscale_model_unet(samp_model):
            if downscale_options is None:
                return  samp_model
            # 获取Unet参数
            elif "PatchModelAddDownscale" in ALL_NODE_CLASS_MAPPINGS:
                cls = ALL_NODE_CLASS_MAPPINGS['PatchModelAddDownscale']
                # 自动收缩Unet
                if downscale_options['downscale_factor'] is None:
                    unet_config = samp_model.model.model_config.unet_config
                    if unet_config is not None and "samples" in samp_samples:
                        height = samp_samples['samples'].shape[2] * 8
                        width = samp_samples['samples'].shape[3] * 8
                        context_dim = unet_config.get('context_dim')
                        longer_side = width if width > height else height
                        if context_dim is not None and longer_side > context_dim:
                            width_downscale_factor = float(width / context_dim)
                            height_downscale_factor = float(height / context_dim)
                            if width_downscale_factor > 1.75:
                                log_node_warn("正在收缩模型Unet...")
                                log_node_warn("收缩系数:" + str(width_downscale_factor))
                                (samp_model,) = cls().patch(samp_model, downscale_options['block_number'], width_downscale_factor, 0, 0.35, True, "bicubic",
                                                            "bicubic")
                            elif height_downscale_factor > 1.25:
                                log_node_warn("正在收缩模型Unet...")
                                log_node_warn("收缩系数:" + str(height_downscale_factor))
                                (samp_model,) = cls().patch(samp_model, downscale_options['block_number'], height_downscale_factor, 0, 0.35, True, "bicubic",
                                                            "bicubic")
                else:
                    cls = ALL_NODE_CLASS_MAPPINGS['PatchModelAddDownscale']
                    log_node_warn("正在收缩模型Unet...")
                    log_node_warn("收缩系数:" + str(downscale_options['downscale_factor']))
                    (samp_model,) = cls().patch(samp_model, downscale_options['block_number'], downscale_options['downscale_factor'], downscale_options['start_percent'], downscale_options['end_percent'], downscale_options['downscale_after_skip'], downscale_options['downscale_method'], downscale_options['upscale_method'])
            return samp_model

        def process_sample_state(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive,
                                 samp_negative,
                                 steps, start_step, last_step, cfg, sampler_name, scheduler, denoise,
                                 image_output, link_id, save_prefix, tile_size, prompt, extra_pnginfo, my_unique_id,
                                 preview_latent, force_full_denoise=force_full_denoise, disable_noise=disable_noise):

            # Downscale Model Unet
            if samp_model is not None:
                samp_model = downscale_model_unet(samp_model)
            # 推理初始时间
            start_time = int(time.time() * 1000)
            # 开始推理
            samp_samples = sampler.common_ksampler(samp_model, samp_seed, steps, cfg, sampler_name, scheduler, samp_positive, samp_negative, samp_samples, denoise=denoise, preview_latent=preview_latent, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, disable_noise=disable_noise)
            # 推理结束时间
            end_time = int(time.time() * 1000)
            # 解码图片
            latent = samp_samples["samples"]

            # 解码图片
            if tile_size is not None:
                samp_images = samp_vae.decode_tiled(latent, tile_x=tile_size // 8, tile_y=tile_size // 8, )
            else:
                samp_images = samp_vae.decode(latent).cpu()

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

        def process_xyPlot(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative,
                           steps, cfg, sampler_name, scheduler, denoise,
                           image_output, link_id, save_prefix, tile_size, prompt, extra_pnginfo, my_unique_id, preview_latent, xyPlot):

            sampleXYplot = easyXYPlot(xyPlot, save_prefix, image_output, prompt, extra_pnginfo, my_unique_id)

            if not sampleXYplot.validate_xy_plot():
                return process_sample_state(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive,
                                            samp_negative, steps, 0, 10000, cfg,
                                            sampler_name, scheduler, denoise, image_output, link_id, save_prefix, tile_size, prompt,
                                            extra_pnginfo, my_unique_id, preview_latent)

            # Downscale Model Unet
            if samp_model is not None:
                samp_model = downscale_model_unet(samp_model)

            plot_image_vars = {
                "x_node_type": sampleXYplot.x_node_type, "y_node_type": sampleXYplot.y_node_type,
                "lora_name": pipe["loader_settings"]["lora_name"] if "lora_name" in pipe["loader_settings"] else None,
                "lora_model_strength": pipe["loader_settings"]["lora_model_strength"] if "lora_model_strength" in pipe["loader_settings"] else None,
                "lora_clip_strength": pipe["loader_settings"]["lora_clip_strength"] if "lora_clip_strength" in pipe["loader_settings"] else None,
                "lora_stack":  pipe["loader_settings"]["lora_stack"] if "lora_stack" in pipe["loader_settings"] else None,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "seed": samp_seed,

                "model": samp_model, "vae": samp_vae, "clip": samp_clip, "positive_cond": samp_positive,
                "negative_cond": samp_negative,

                "ckpt_name": pipe['loader_settings']['ckpt_name'] if "ckpt_name" in pipe["loader_settings"] else None,
                "vae_name": pipe['loader_settings']['vae_name'] if "vae_name" in pipe["loader_settings"] else None,
                "clip_skip": pipe['loader_settings']['clip_skip'] if "clip_skip" in pipe["loader_settings"] else None,
                "positive": pipe['loader_settings']['positive'] if "positive" in pipe["loader_settings"] else None,
                "positive_token_normalization": pipe['loader_settings']['positive_token_normalization'] if "positive_token_normalization" in pipe["loader_settings"] else None,
                "positive_weight_interpretation": pipe['loader_settings']['positive_weight_interpretation'] if "positive_weight_interpretation" in pipe["loader_settings"] else None,
                "negative": pipe['loader_settings']['negative'] if "negative" in pipe["loader_settings"] else None,
                "negative_token_normalization": pipe['loader_settings']['negative_token_normalization'] if "negative_token_normalization" in pipe["loader_settings"] else None,
                "negative_weight_interpretation": pipe['loader_settings']['negative_weight_interpretation'] if "negative_weight_interpretation" in pipe["loader_settings"] else None,
            }

            if "models" in pipe["loader_settings"]:
                plot_image_vars["models"] = pipe["loader_settings"]["models"]
            if "vae_use" in pipe["loader_settings"]:
                plot_image_vars["vae_use"] = pipe["loader_settings"]["vae_use"]
            if "a1111_prompt_style" in pipe["loader_settings"]:
                plot_image_vars["a1111_prompt_style"] = pipe["loader_settings"]["a1111_prompt_style"]
            if "cnet_stack" in pipe["loader_settings"]:
                plot_image_vars["cnet"] = pipe["loader_settings"]["cnet_stack"]
            if "positive_cond_stack" in pipe["loader_settings"]:
                plot_image_vars["positive_cond_stack"] = pipe["loader_settings"]["positive_cond_stack"]
            if "negative_cond_stack" in pipe["loader_settings"]:
                plot_image_vars["negative_cond_stack"] = pipe["loader_settings"]["negative_cond_stack"]

            latent_image = sampleXYplot.get_latent(pipe["samples"])

            latents_plot = sampleXYplot.get_labels_and_sample(plot_image_vars, latent_image, preview_latent, start_step,
                                                              last_step, force_full_denoise, disable_noise)

            samp_samples = {"samples": latents_plot}

            images, image_list = sampleXYplot.plot_images_and_labels()

            samp_images = images

            results = easy_save.images(images, save_prefix, image_output)

            # Generate output_images
            output_images = torch.stack([tensor.squeeze() for tensor in image_list])

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
                "images": output_images,
                "seed": samp_seed,

                "loader_settings": pipe["loader_settings"],
            }

            sampler.update_value_by_id("pipe_line", my_unique_id, new_pipe)

            del pipe

            if image_output in ("Hide", "Hide/Save"):
                return sampler.get_output(new_pipe)

            return {"ui": {"images": results}, "result": (sampler.get_output(new_pipe))}

        preview_latent = True
        if image_output in ("Hide", "Hide/Save"):
            preview_latent = False

        xyplot_id = next((x for x in prompt if "XYPlot" in str(prompt[x]["class_type"])), None)
        if xyplot_id is None:
            xyPlot = None
        else:
            xyPlot = pipe["loader_settings"]["xyplot"] if "xyplot" in pipe["loader_settings"] else xyPlot
        if xyPlot is not None:
            return process_xyPlot(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative, steps, cfg, sampler_name, scheduler, denoise, image_output, link_id, save_prefix, tile_size, prompt, extra_pnginfo, my_unique_id, preview_latent, xyPlot)
        else:
            return process_sample_state(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative, steps, start_step, last_step, cfg, sampler_name, scheduler, denoise, image_output, link_id, save_prefix, tile_size, prompt, extra_pnginfo, my_unique_id, preview_latent, force_full_denoise, disable_noise)

# 简易采样器
class samplerSimple:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"pipe": ("PIPE_LINE",),
                 "image_output": (["Hide", "Preview", "Save", "Hide/Save", "Sender", "Sender/Save"],{"default": "Preview"}),
                 "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"}),
                 },
                "optional": {
                    "model": ("MODEL",),
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

        return samplerFull.run(self, pipe, None, None,None,None,None, image_output, link_id, save_prefix,
                               None, model, None, None, None, None, None, None,
                               tile_size, prompt, extra_pnginfo, my_unique_id, force_full_denoise, disable_noise)

# 简易采样器 (Tiled)
class samplerSimpleTiled:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"pipe": ("PIPE_LINE",),
                 "tile_size": ("INT", {"default": 512, "min": 320, "max": 4096, "step": 64}),
                 "image_output": (["Hide", "Preview", "Save", "Hide/Save", "Sender", "Sender/Save"],{"default": "Preview"}),
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
        return samplerFull.run(self, pipe, None, None,None,None,None, image_output, link_id, save_prefix,
                               None, model, None, None, None, None, None, None,
                               tile_size, prompt, extra_pnginfo, my_unique_id, force_full_denoise, disable_noise)

# 简易采样器(收缩Unet)
class samplerSimpleDownscaleUnet:

    def __init__(self):
        pass

    upscale_methods = ["bicubic", "nearest-exact", "bilinear", "area", "bislerp"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"pipe": ("PIPE_LINE",),
                 "downscale_mode": (["None", "Auto", "Custom"],{"default": "Auto"}),
                 "block_number": ("INT", {"default": 3, "min": 1, "max": 32, "step": 1}),
                 "downscale_factor": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 9.0, "step": 0.001}),
                 "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                 "end_percent": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.001}),
                 "downscale_after_skip": ("BOOLEAN", {"default": True}),
                 "downscale_method": (s.upscale_methods,),
                 "upscale_method": (s.upscale_methods,),
                 "image_output": (["Hide", "Preview", "Save", "Hide/Save", "Sender", "Sender/Save"],{"default": "Preview"}),
                 "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"}),
                 },
                "optional": {
                    "model": ("MODEL",),
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

    def run(self, pipe, downscale_mode, block_number, downscale_factor, start_percent, end_percent, downscale_after_skip, downscale_method, upscale_method, image_output, link_id, save_prefix, model=None, tile_size=None, prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False):
        downscale_options = None
        if downscale_mode == 'Auto':
            downscale_options = {
                "block_number": block_number,
                "downscale_factor": None,
                "start_percent": 0,
                "end_percent":0.35,
                "downscale_after_skip": True,
                "downscale_method": "bicubic",
                "upscale_method": "bicubic"
            }
        elif downscale_mode == 'Custom':
            downscale_options = {
                "block_number": block_number,
                "downscale_factor": downscale_factor,
                "start_percent": start_percent,
                "end_percent": end_percent,
                "downscale_after_skip": downscale_after_skip,
                "downscale_method": downscale_method,
                "upscale_method": upscale_method
            }

        return samplerFull.run(self, pipe, None, None,None,None,None, image_output, link_id, save_prefix,
                               None, model, None, None, None, None, None, None,
                               tile_size, prompt, extra_pnginfo, my_unique_id, force_full_denoise, disable_noise, downscale_options)
# 简易采样器 (内补)
class samplerSimpleInpainting:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"pipe": ("PIPE_LINE",),
                 "grow_mask_by": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),
                 "image_output": (["Hide", "Preview", "Save", "Hide/Save", "Sender", "Sender/Save"],{"default": "Preview"}),
                 "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"}),
                 },
                "optional": {
                    "model": ("MODEL",),
                    "mask": ("MASK",),
                    "patch": ("INPAINT_PATCH",),
                },
                "hidden":
                  {"tile_size": "INT", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                    "embeddingsList": (folder_paths.get_filename_list("embeddings"),)
                  }
                }

    RETURN_TYPES = ("PIPE_LINE", "IMAGE", "VAE")
    RETURN_NAMES = ("pipe", "image", "vae")
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "EasyUse/Sampler"

    def run(self, pipe, grow_mask_by, image_output, link_id, save_prefix, model=None, mask=None, patch=None, tile_size=None, prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False):
        model = model if model is not None else pipe['model']

        if mask is not None:
            pixels = pipe["images"] if pipe and "images" in pipe else None
            if pixels is None:
                raise Exception("No Images found")
            vae = pipe["vae"] if pipe and "vae" in pipe else None
            if pixels is None:
                raise Exception("No VAE found")
            x = (pixels.shape[1] // 8) * 8
            y = (pixels.shape[2] // 8) * 8
            mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
                                                   size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

            pixels = pixels.clone()
            if pixels.shape[1] != x or pixels.shape[2] != y:
                x_offset = (pixels.shape[1] % 8) // 2
                y_offset = (pixels.shape[2] % 8) // 2
                pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
                mask = mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]

            if grow_mask_by == 0:
                mask_erosion = mask
            else:
                kernel_tensor = torch.ones((1, 1, grow_mask_by, grow_mask_by))
                padding = math.ceil((grow_mask_by - 1) / 2)

                mask_erosion = torch.clamp(torch.nn.functional.conv2d(mask.round(), kernel_tensor, padding=padding), 0,
                                           1)

            m = (1.0 - mask.round()).squeeze(1)
            for i in range(3):
                pixels[:, :, :, i] -= 0.5
                pixels[:, :, :, i] *= m
                pixels[:, :, :, i] += 0.5
            t = vae.encode(pixels)

            latent = {"samples": t, "noise_mask": (mask_erosion[:, :, :x, :y].round())}

            # when patch was linked
            if patch is not None:
                worker = InpaintWorker(node_name="easy kSamplerInpainting")
                model, = worker.patch(model, latent, patch)

            new_pipe = {
                "model": model,
                "positive": pipe['positive'],
                "negative": pipe['negative'],
                "vae": pipe['vae'],
                "clip": pipe['clip'],

                "samples": latent,
                "images": pipe['images'],
                "seed": pipe['seed'],

                "loader_settings": pipe["loader_settings"],
            }
        else:
            new_pipe = pipe
        del pipe
        return samplerFull.run(self, new_pipe, None, None,None,None,None, image_output, link_id, save_prefix,
                               None, model, None, None, None, None, None, None,
                               tile_size, prompt, extra_pnginfo, my_unique_id, force_full_denoise, disable_noise)

# SDTurbo采样器
class samplerSDTurbo:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"pipe": ("PIPE_LINE",),
                     "image_output": (["Hide", "Preview", "Save", "Hide/Save", "Sender", "Sender/Save"],{"default": "Preview"}),
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
        if tile_size is not None:
            samp_images = samp_vae.decode_tiled(latent, tile_x=tile_size // 8, tile_y=tile_size // 8, )
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


class unsampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
             "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
             "end_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
             "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
             "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
             "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
             "normalize": (["disable", "enable"],),

             },
            "optional": {
                "pipe": ("PIPE_LINE",),
                "optional_model": ("MODEL",),
                "optional_positive": ("CONDITIONING",),
                "optional_negative": ("CONDITIONING",),
                "optional_latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("PIPE_LINE", "LATENT",)
    RETURN_NAMES = ("pipe", "latent",)
    FUNCTION = "unsampler"

    CATEGORY = "EasyUse/Sampler"

    def unsampler(self, cfg, sampler_name, steps, end_at_step, scheduler, normalize, pipe=None, optional_model=None, optional_positive=None, optional_negative=None,
                  optional_latent=None):

        model = optional_model if optional_model is not None else pipe["model"]
        positive = optional_positive if optional_positive is not None else pipe["positive"]
        negative = optional_negative if optional_negative is not None else pipe["negative"]
        latent_image = optional_latent if optional_latent is not None else pipe["samples"]

        normalize = normalize == "enable"
        device = comfy.model_management.get_torch_device()
        latent = latent_image
        latent_image = latent["samples"]

        end_at_step = min(end_at_step, steps - 1)
        end_at_step = steps - end_at_step

        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = comfy.sample.prepare_mask(latent["noise_mask"], noise.shape, device)

        real_model = None
        real_model = model.model

        noise = noise.to(device)
        latent_image = latent_image.to(device)

        positive = comfy.sample.convert_cond(positive)
        negative = comfy.sample.convert_cond(negative)

        models, inference_memory = comfy.sample.get_additional_models(positive, negative, model.model_dtype())

        comfy.model_management.load_models_gpu([model] + models, model.memory_required(noise.shape) + inference_memory)

        sampler = comfy.samplers.KSampler(real_model, steps=steps, device=device, sampler=sampler_name,
                                          scheduler=scheduler, denoise=1.0, model_options=model.model_options)

        sigmas = sigmas = sampler.sigmas.flip(0) + 0.0001

        pbar = comfy.utils.ProgressBar(steps)

        def callback(step, x0, x, total_steps):
            pbar.update_absolute(step + 1, total_steps)

        samples = sampler.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image,
                                 force_full_denoise=False, denoise_mask=noise_mask, sigmas=sigmas, start_step=0,
                                 last_step=end_at_step, callback=callback)
        if normalize:
            # technically doesn't normalize because unsampling is not guaranteed to end at a std given by the schedule
            samples -= samples.mean()
            samples /= samples.std()
        samples = samples.cpu()

        comfy.sample.cleanup_additional_models(models)

        out = latent.copy()
        out["samples"] = samples

        if pipe is None:
            pipe = {}

        new_pipe = {
            **pipe,
            "samples": out
        }

        return (new_pipe, out,)

#---------------------------------------------------------------修复 开始----------------------------------------------------------------------#

# 高清修复
class hiresFix:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos", "bislerp"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                 "model_name": (folder_paths.get_filename_list("upscale_models"),),
                 "rescale_after_model": ([False, True], {"default": True}),
                 "rescale_method": (s.upscale_methods,),
                 "rescale": (["by percentage", "to Width/Height", 'to longer side - maintain aspect'],),
                 "percent": ("INT", {"default": 50, "min": 0, "max": 1000, "step": 1}),
                 "width": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                 "height": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                 "longer_side": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                 "crop": (s.crop_methods,),
                 "image_output": (["Hide", "Preview", "Save", "Hide/Save", "Sender", "Sender/Save"],{"default": "Preview"}),
                 "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"}),
                },
                "optional": {
                    "pipe": ("PIPE_LINE",),
                    "image": ("IMAGE",),
                    "vae": ("VAE",),
                },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                           },
                }

    RETURN_TYPES = ("PIPE_LINE", "IMAGE", "LATENT", )
    RETURN_NAMES = ('pipe', 'image', "latent", )

    FUNCTION = "upscale"
    CATEGORY = "EasyUse/Fix"
    OUTPUT_NODE = True

    def vae_encode_crop_pixels(self, pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels

    def upscale(self, model_name, rescale_after_model, rescale_method, rescale, percent, width, height,
                longer_side, crop, image_output, link_id, save_prefix, pipe=None, image=None, vae=None, prompt=None,
                extra_pnginfo=None, my_unique_id=None):

        new_pipe = {}
        if pipe is not None:
            image = image if image is not None else pipe["images"]
            vae = vae if vae is not None else pipe.get("vae")
        elif image is None or vae is None:
            raise ValueError("pipe or image or vae missing.")
        # Load Model
        model_path = folder_paths.get_full_path("upscale_models", model_name)
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        upscale_model = model_loading.load_state_dict(sd).eval()

        # Model upscale
        device = comfy.model_management.get_torch_device()
        upscale_model.to(device)
        in_img = image.movedim(-1, -3).to(device)

        tile = 128 + 64
        overlap = 8
        steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile,
                                                                    tile_y=tile, overlap=overlap)
        pbar = comfy.utils.ProgressBar(steps)
        s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap,
                                    upscale_amount=upscale_model.scale, pbar=pbar)
        upscale_model.cpu()
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)

        # Post Model Rescale
        if rescale_after_model == True:
            samples = s.movedim(-1, 1)
            orig_height = samples.shape[2]
            orig_width = samples.shape[3]
            if rescale == "by percentage" and percent != 0:
                height = percent / 100 * orig_height
                width = percent / 100 * orig_width
                if (width > MAX_RESOLUTION):
                    width = MAX_RESOLUTION
                if (height > MAX_RESOLUTION):
                    height = MAX_RESOLUTION

                width = easySampler.enforce_mul_of_64(width)
                height = easySampler.enforce_mul_of_64(height)
            elif rescale == "to longer side - maintain aspect":
                longer_side = easySampler.enforce_mul_of_64(longer_side)
                if orig_width > orig_height:
                    width, height = longer_side, easySampler.enforce_mul_of_64(longer_side * orig_height / orig_width)
                else:
                    width, height = easySampler.enforce_mul_of_64(longer_side * orig_width / orig_height), longer_side

            s = comfy.utils.common_upscale(samples, width, height, rescale_method, crop)
            s = s.movedim(1, -1)

        # vae encode
        pixels = self.vae_encode_crop_pixels(s)
        t = vae.encode(pixels[:, :, :, :3])

        if pipe is not None:
            new_pipe = {
                "model": pipe['model'],
                "positive": pipe['positive'],
                "negative": pipe['negative'],
                "vae": vae,
                "clip": pipe['clip'],

                "samples": {"samples": t},
                "images": s,
                "seed": pipe['seed'],

                "loader_settings": {
                    **pipe["loader_settings"],
                }
            }
            del pipe
        else:
            new_pipe = {}

        easy_save = easySave(my_unique_id, prompt, extra_pnginfo)
        results = easy_save.images(s, save_prefix, image_output)

        if image_output in ("Sender", "Sender/Save"):
            PromptServer.instance.send_sync("img-send", {"link_id": link_id, "images": results})

        if image_output in ("Hide", "Hide/Save"):
            return (new_pipe, s, {"samples": t},)

        return {"ui": {"images": results},
                "result": (new_pipe, s, {"samples": t},)}

# 预细节修复
class preDetailerFix:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
             "pipe": ("PIPE_LINE",),
             "guide_size": ("FLOAT", {"default": 256, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
             "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
             "max_size": ("FLOAT", {"default": 768, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
             "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
             "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
             "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
             "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
             "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
             "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
             "noise_mask": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
             "force_inpaint": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
             "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
             "wildcard": ("STRING", {"multiline": True, "dynamicPrompts": False}),
             "cycle": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
        },
            "optional": {
                "bbox_segm_pipe": ("PIPE_LINE",),
                "sam_pipe": ("PIPE_LINE",),
                "optional_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "doit"

    CATEGORY = "EasyUse/Fix"

    def doit(self, pipe, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler, denoise, feather, noise_mask, force_inpaint, drop_size, wildcard, cycle, bbox_segm_pipe=None, sam_pipe=None, optional_image=None):

        model = pipe["model"] if "model" in pipe else None
        if model is None:
            raise Exception(f"[ERROR] pipe['model'] is missing")
        clip = pipe["clip"] if"clip" in pipe else None
        if clip is None:
            raise Exception(f"[ERROR] pipe['clip'] is missing")
        vae = pipe["vae"] if "vae" in pipe else None
        if vae is None:
            raise Exception(f"[ERROR] pipe['vae'] is missing")
        if optional_image is not None:
            batch_size = pipe["loader_settings"]["batch_size"] if "batch_size" in pipe["loader_settings"] else 1
            samples = {"samples": vae.encode(optional_image)}
            samples = RepeatLatentBatch().repeat(samples, batch_size)[0]
            image = optional_image
        else:
            samples = pipe["samples"] if "samples" in pipe else None
            if samples is None:
                raise Exception(f"[ERROR] pipe['samples'] is missing")
            image = pipe["images"] if "images" in pipe else None
            if image is None:
                raise Exception(f"[ERROR] pipe['image'] is missing")
        positive = pipe["positive"] if "positive" in pipe else None
        if positive is None:
            raise Exception(f"[ERROR] pipe['positive'] is missing")
        negative = pipe["negative"] if "negative" in pipe else None
        if negative is None:
            raise Exception(f"[ERROR] pipe['negative'] is missing")
        bbox_segm_pipe = bbox_segm_pipe or (pipe["bbox_segm_pipe"] if pipe and "bbox_segm_pipe" in pipe else None)
        if bbox_segm_pipe is None:
            raise Exception(f"[ERROR] bbox_segm_pipe or pipe['bbox_segm_pipe'] is missing")
        sam_pipe = sam_pipe or (pipe["sam_pipe"] if pipe and "sam_pipe" in pipe else None)
        if sam_pipe is None:
            raise Exception(f"[ERROR] sam_pipe or pipe['sam_pipe'] is missing")

        loader_settings = pipe["loader_settings"] if "loader_settings" in pipe else {}

        new_pipe = {
            "samples": samples,
            "images": image,
            "model": model,
            "clip": clip,
            "vae": vae,
            "positive": positive,
            "negative": negative,
            "seed": seed,

            "bbox_segm_pipe": bbox_segm_pipe,
            "sam_pipe": sam_pipe,

            "loader_settings": loader_settings,

            "detail_fix_settings": {
                "guide_size": guide_size,
                "guide_size_for": guide_size_for,
                "max_size": max_size,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "feather": feather,
                "noise_mask": noise_mask,
                "force_inpaint": force_inpaint,
                "drop_size": drop_size,
                "wildcard": wildcard,
                "cycle": cycle
            }
        }


        del bbox_segm_pipe
        del sam_pipe

        return (new_pipe,)

# 细节修复
class detailerFix:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pipe": ("PIPE_LINE",),
            "image_output": (["Hide", "Preview", "Save", "Hide/Save", "Sender", "Sender/Save"],{"default": "Preview"}),
            "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
            "save_prefix": ("STRING", {"default": "ComfyUI"}),
        },
            "optional": {
                "model": ("MODEL",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID", }
        }

    RETURN_TYPES = ("PIPE_LINE", "IMAGE",)
    RETURN_NAMES = ("pipe", "image")
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (False, False)
    FUNCTION = "doit"

    CATEGORY = "EasyUse/Fix"


    def doit(self, pipe, image_output, link_id, save_prefix, model=None, prompt=None, extra_pnginfo=None, my_unique_id=None):

        # Clean loaded_objects
        easyCache.update_loaded_objects(prompt)

        my_unique_id = int(my_unique_id)

        easy_save = easySave(my_unique_id, prompt, extra_pnginfo)

        model = model or (pipe["model"] if "model" in pipe else None)
        if model is None:
            raise Exception(f"[ERROR] model or pipe['model'] is missing")

        bbox_segm_pipe = pipe["bbox_segm_pipe"] if pipe and "bbox_segm_pipe" in pipe else None
        if bbox_segm_pipe is None:
            raise Exception(f"[ERROR] bbox_segm_pipe or pipe['bbox_segm_pipe'] is missing")
        sam_pipe = pipe["sam_pipe"] if "sam_pipe" in pipe else None
        if sam_pipe is None:
            raise Exception(f"[ERROR] sam_pipe or pipe['sam_pipe'] is missing")
        bbox_detector_opt, bbox_threshold, bbox_dilation, bbox_crop_factor, segm_detector_opt = bbox_segm_pipe
        sam_model_opt, sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold, sam_mask_hint_use_negative = sam_pipe

        detail_fix_settings = pipe["detail_fix_settings"] if "detail_fix_settings" in pipe else None
        if detail_fix_settings is None:
            raise Exception(f"[ERROR] detail_fix_settings or pipe['detail_fix_settings'] is missing")

        image = pipe["images"]
        clip = pipe["clip"]
        vae = pipe["vae"]
        seed = pipe["seed"]
        positive = pipe["positive"]
        negative = pipe["negative"]
        loader_settings = pipe["loader_settings"] if "loader_settings" in pipe else {}
        guide_size = pipe["detail_fix_settings"]["guide_size"]
        guide_size_for = pipe["detail_fix_settings"]["guide_size_for"]
        max_size = pipe["detail_fix_settings"]["max_size"]
        steps = pipe["detail_fix_settings"]["steps"]
        cfg = pipe["detail_fix_settings"]["cfg"]
        sampler_name = pipe["detail_fix_settings"]["sampler_name"]
        scheduler = pipe["detail_fix_settings"]["scheduler"]
        denoise = pipe["detail_fix_settings"]["denoise"]
        feather = pipe["detail_fix_settings"]["feather"]
        noise_mask = pipe["detail_fix_settings"]["noise_mask"]
        force_inpaint = pipe["detail_fix_settings"]["force_inpaint"]
        drop_size = pipe["detail_fix_settings"]["drop_size"]
        wildcard = pipe["detail_fix_settings"]["wildcard"]
        cycle = pipe["detail_fix_settings"]["cycle"]

        del pipe

        # 细节修复初始时间
        start_time = int(time.time() * 1000)

        cls = ALL_NODE_CLASS_MAPPINGS["FaceDetailer"]
        enhanced_img, cropped_enhanced, cropped_enhanced_alpha, mask, cnet_pil_list = cls().enhance_face(
            image, model, clip, vae, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name,
            scheduler,
            positive, negative, denoise, feather, noise_mask, force_inpaint,
            bbox_threshold, bbox_dilation, bbox_crop_factor,
            sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
            sam_mask_hint_use_negative, drop_size, bbox_detector_opt, segm_detector_opt, sam_model_opt, wildcard,
            detailer_hook=None, cycle=cycle)

        # 细节修复结束时间
        end_time = int(time.time() * 1000)

        spent_time = '细节修复:' + str((end_time - start_time) / 1000) + '秒'

        results = easy_save.images(enhanced_img, save_prefix, image_output)
        sampler.update_value_by_id("results", my_unique_id, results)

        # Clean loaded_objects
        easyCache.update_loaded_objects(prompt)

        new_pipe = {
            "samples": None,
            "images": enhanced_img,
            "model": model,
            "clip": clip,
            "vae": vae,
            "seed": seed,
            "positive": positive,
            "negative": negative,
            "wildcard": wildcard,
            "bbox_segm_pipe": bbox_segm_pipe,
            "sam_pipe": sam_pipe,

            "loader_settings": {
                **loader_settings,
                "spent_time": spent_time
            },
            "detail_fix_settings": detail_fix_settings
        }

        sampler.update_value_by_id("pipe_line", my_unique_id, new_pipe)

        del bbox_segm_pipe
        del sam_pipe

        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                    "result": (new_pipe, enhanced_img, cropped_enhanced, cropped_enhanced_alpha, mask, cnet_pil_list)}

        if image_output in ("Sender", "Sender/Save"):
            PromptServer.instance.send_sync("img-send", {"link_id": link_id, "images": results})

        return {"ui": {"images": results}, "result": (new_pipe, enhanced_img, cropped_enhanced, cropped_enhanced_alpha, mask, cnet_pil_list)}

def add_folder_path_and_extensions(folder_name, full_folder_paths, extensions):
    for full_folder_path in full_folder_paths:
        folder_paths.add_model_folder_path(folder_name, full_folder_path)
    if folder_name in folder_paths.folder_names_and_paths:
        current_paths, current_extensions = folder_paths.folder_names_and_paths[folder_name]
        updated_extensions = current_extensions | extensions
        folder_paths.folder_names_and_paths[folder_name] = (current_paths, updated_extensions)
    else:
        folder_paths.folder_names_and_paths[folder_name] = (full_folder_paths, extensions)

model_path = folder_paths.models_dir
add_folder_path_and_extensions("ultralytics_bbox", [os.path.join(model_path, "ultralytics", "bbox")], folder_paths.supported_pt_extensions)
add_folder_path_and_extensions("ultralytics_segm", [os.path.join(model_path, "ultralytics", "segm")], folder_paths.supported_pt_extensions)
add_folder_path_and_extensions("ultralytics", [os.path.join(model_path, "ultralytics")], folder_paths.supported_pt_extensions)
add_folder_path_and_extensions("mmdets_bbox", [os.path.join(model_path, "mmdets", "bbox")], folder_paths.supported_pt_extensions)
add_folder_path_and_extensions("mmdets_segm", [os.path.join(model_path, "mmdets", "segm")], folder_paths.supported_pt_extensions)
add_folder_path_and_extensions("mmdets", [os.path.join(model_path, "mmdets")], folder_paths.supported_pt_extensions)
add_folder_path_and_extensions("sams", [os.path.join(model_path, "sams")], folder_paths.supported_pt_extensions)
add_folder_path_and_extensions("onnx", [os.path.join(model_path, "onnx")], {'.onnx'})

class ultralyticsDetectorForDetailerFix:
    @classmethod
    def INPUT_TYPES(s):
        bboxs = ["bbox/" + x for x in folder_paths.get_filename_list("ultralytics_bbox")]
        segms = ["segm/" + x for x in folder_paths.get_filename_list("ultralytics_segm")]
        return {"required":
                    {"model_name": (bboxs + segms,),
                    "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "bbox_dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                    "bbox_crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),
                    }
                }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("bbox_segm_pipe",)
    FUNCTION = "doit"

    CATEGORY = "EasyUse/Fix"

    def doit(self, model_name, bbox_threshold, bbox_dilation, bbox_crop_factor):
        if 'UltralyticsDetectorProvider' not in ALL_NODE_CLASS_MAPPINGS:
            raise Exception(f"[ERROR] To use UltralyticsDetectorProvider, you need to install 'Impact Pack'")
        cls = ALL_NODE_CLASS_MAPPINGS['UltralyticsDetectorProvider']
        bbox_detector, segm_detector = cls().doit(model_name)
        pipe = (bbox_detector, bbox_threshold, bbox_dilation, bbox_crop_factor, segm_detector)
        return (pipe,)

class samLoaderForDetailerFix:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("sams"),),
                "device_mode": (["AUTO", "Prefer GPU", "CPU"],{"default": "AUTO"}),
                "sam_detection_hint": (
                ["center-1", "horizontal-2", "vertical-2", "rect-4", "diamond-4", "mask-area", "mask-points",
                 "mask-point-bbox", "none"],),
                "sam_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "sam_threshold": ("FLOAT", {"default": 0.93, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sam_bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "sam_mask_hint_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sam_mask_hint_use_negative": (["False", "Small", "Outter"],),
            }
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("sam_pipe",)
    FUNCTION = "doit"

    CATEGORY = "EasyUse/Fix"

    def doit(self, model_name, device_mode, sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold, sam_mask_hint_use_negative):
        if 'SAMLoader' not in ALL_NODE_CLASS_MAPPINGS:
            raise Exception(f"[ERROR] To use SAMLoader, you need to install 'Impact Pack'")
        cls = ALL_NODE_CLASS_MAPPINGS['SAMLoader']
        (sam_model,) = cls().load_model(model_name, device_mode)
        pipe = (sam_model, sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold, sam_mask_hint_use_negative)
        return (pipe,)

#---------------------------------------------------------------Pipe 开始----------------------------------------------------------------------#

# pipeIn
class pipeIn:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
             "required":{
                 "pipe": ("PIPE_LINE",),
             },
             "optional": {
                "model": ("MODEL",),
                "pos": ("CONDITIONING",),
                "neg": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "image": ("IMAGE",),
                "xyPlot": ("XYPLOT",),
            },
            "hidden": {"my_unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "flush"

    CATEGORY = "EasyUse/Pipe"

    def flush(self, pipe, model=None, pos=None, neg=None, latent=None, vae=None, clip=None, image=None, xyplot=None, my_unique_id=None):

        model = model if model is not None else pipe.get("model")
        if model is None:
            log_node_warn(f'pipeIn[{my_unique_id}]', "Model missing from pipeLine")
        pos = pos if pos is not None else pipe.get("positive")
        if pos is None:
            log_node_warn(f'pipeIn[{my_unique_id}]', "Pos Conditioning missing from pipeLine")
        neg = neg if neg is not None else pipe.get("negative")
        if neg is None:
            log_node_warn(f'pipeIn[{my_unique_id}]', "Neg Conditioning missing from pipeLine")
        samples = latent if latent is not None else pipe.get("samples")
        if samples is None:
            log_node_warn(f'pipeIn[{my_unique_id}]', "Latent missing from pipeLine")
        vae = vae if vae is not None else pipe.get("vae")
        if vae is None:
            log_node_warn(f'pipeIn[{my_unique_id}]', "VAE missing from pipeLine")
        clip = clip if clip is not None else pipe.get("clip")
        if clip is None:
            log_node_warn(f'pipeIn[{my_unique_id}]', "Clip missing from pipeLine")
        if image is None:
            image = pipe.get("images")
        else:
            batch_size = pipe["loader_settings"]["batch_size"] if "batch_size" in pipe["loader_settings"] else 1
            samples = {"samples": vae.encode(image)}
            samples = RepeatLatentBatch().repeat(samples, batch_size)[0]
        seed = pipe.get("seed")
        if seed is None:
            log_node_warn(f'pipeIn[{my_unique_id}]', "Seed missing from pipeLine")
        xyplot = xyplot or pipe['loader_settings']['xyplot'] if 'xyplot' in pipe['loader_settings'] else None

        new_pipe = {
            "model": model,
            "positive": pos,
            "negative": neg,
            "vae": vae,
            "clip": clip,

            "samples": samples,
            "images": image,
            "seed": seed,

            "loader_settings": {
                **pipe["loader_settings"],
                "positive": "",
                "negative": "",
                "xyplot": xyplot
            }
        }
        del pipe

        return (new_pipe,)

# pipeOut
class pipeOut:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
             "required": {
                "pipe": ("PIPE_LINE",),
            },
            "hidden": {"my_unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "IMAGE", "INT",)
    RETURN_NAMES = ("pipe", "model", "pos", "neg", "latent", "vae", "clip", "image", "seed",)
    FUNCTION = "flush"

    CATEGORY = "EasyUse/Pipe"

    def flush(self, pipe, my_unique_id=None):
        model = pipe.get("model")
        pos = pipe.get("positive")
        neg = pipe.get("negative")
        latent = pipe.get("samples")
        vae = pipe.get("vae")
        clip = pipe.get("clip")
        image = pipe.get("images")
        seed = pipe.get("seed")

        return pipe, model, pos, neg, latent, vae, clip, image, seed

# pipeToBasicPipe
class pipeToBasicPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
            },
            "hidden": {"my_unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("BASIC_PIPE",)
    RETURN_NAMES = ("basic_pipe",)
    FUNCTION = "doit"

    CATEGORY = "EasyUse/Pipe"

    def doit(self, pipe, my_unique_id=None):
        new_pipe = (pipe.get('model'), pipe.get('clip'), pipe.get('vae'), pipe.get('positive'), pipe.get('negative'))
        del pipe
        return (new_pipe,)

# pipeXYPlot
class pipeXYPlot:
    lora_list = ["None"] + folder_paths.get_filename_list("loras")
    lora_strengths = {"min": -4.0, "max": 4.0, "step": 0.01}
    token_normalization = ["none", "mean", "length", "length+mean"]
    weight_interpretation = ["comfy", "A1111", "compel", "comfy++"]

    loader_dict = {
        "ckpt_name": folder_paths.get_filename_list("checkpoints"),
        "vae_name": ["Baked-VAE"] + folder_paths.get_filename_list("vae"),
        "clip_skip": {"min": -24, "max": -1, "step": 1},
        "lora_name": lora_list,
        "lora_model_strength": lora_strengths,
        "lora_clip_strength": lora_strengths,
        "positive": [],
        "negative": [],
    }

    sampler_dict = {
        "steps": {"min": 1, "max": 100, "step": 1},
        "cfg": {"min": 0.0, "max": 100.0, "step": 1.0},
        "sampler_name": comfy.samplers.KSampler.SAMPLERS,
        "scheduler": comfy.samplers.KSampler.SCHEDULERS,
        "denoise": {"min": 0.0, "max": 1.0, "step": 0.01},
        "seed": {"min": 0, "max": 1125899906842624},
    }

    plot_dict = {**sampler_dict, **loader_dict}

    plot_values = ["None", ]
    plot_values.append("---------------------")
    for k in sampler_dict:
        plot_values.append(f'preSampling: {k}')
    plot_values.append("---------------------")
    for k in loader_dict:
        plot_values.append(f'loader: {k}')

    def __init__(self):
        pass

    rejected = ["None", "---------------------", "Nothing"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "grid_spacing": ("INT", {"min": 0, "max": 500, "step": 5, "default": 0, }),
                "output_individuals": (["False", "True"], {"default": "False"}),
                "flip_xy": (["False", "True"], {"default": "False"}),
                "x_axis": (pipeXYPlot.plot_values, {"default": 'None'}),
                "x_values": (
                "STRING", {"default": '', "multiline": True, "placeholder": 'insert values seperated by "; "'}),
                "y_axis": (pipeXYPlot.plot_values, {"default": 'None'}),
                "y_values": (
                "STRING", {"default": '', "multiline": True, "placeholder": 'insert values seperated by "; "'}),
            },
            "optional": {
              "pipe": ("PIPE_LINE",)
            },
            "hidden": {
                "plot_dict": (pipeXYPlot.plot_dict,),
            },
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "plot"

    CATEGORY = "EasyUse/Pipe"

    def plot(self, grid_spacing, output_individuals, flip_xy, x_axis, x_values, y_axis, y_values, pipe=None):
        def clean_values(values):
            original_values = values.split("; ")
            cleaned_values = []

            for value in original_values:
                # Strip the semi-colon
                cleaned_value = value.strip(';').strip()

                if cleaned_value == "":
                    continue

                # Try to convert the cleaned_value back to int or float if possible
                try:
                    cleaned_value = int(cleaned_value)
                except ValueError:
                    try:
                        cleaned_value = float(cleaned_value)
                    except ValueError:
                        pass

                # Append the cleaned_value to the list
                cleaned_values.append(cleaned_value)

            return cleaned_values

        if x_axis in self.rejected:
            x_axis = "None"
            x_values = []
        else:
            x_values = clean_values(x_values)

        if y_axis in self.rejected:
            y_axis = "None"
            y_values = []
        else:
            y_values = clean_values(y_values)

        if flip_xy == "True":
            x_axis, y_axis = y_axis, x_axis
            x_values, y_values = y_values, x_values


        xy_plot = {"x_axis": x_axis,
                   "x_vals": x_values,
                   "y_axis": y_axis,
                   "y_vals": y_values,
                   "grid_spacing": grid_spacing,
                   "output_individuals": output_individuals}

        if pipe is not None:
            new_pipe = pipe
            new_pipe['loader_settings'] = {
                **pipe['loader_settings'],
                "xyplot": xy_plot
            }
            del pipe
        return (new_pipe, xy_plot,)

# pipeXYPlotAdvanced
class pipeXYPlotAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "grid_spacing": ("INT", {"min": 0, "max": 500, "step": 5, "default": 0, }),
                "output_individuals": (["False", "True"], {"default": "False"}),
                "flip_xy": (["False", "True"], {"default": "False"}),
            },
            "optional": {
                "X": ("X_Y",),
                "Y": ("X_Y",),
            },
            "hidden": {"my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "plot"

    CATEGORY = "EasyUse/Pipe"

    def plot(self, pipe, grid_spacing, output_individuals, flip_xy, X=None, Y=None, my_unique_id=None):
        if X != None:
            x_axis = X.get('axis')
            x_values = X.get('values')
        else:
            x_axis = "Nothing"
            x_values = [""]
        if Y != None:
            y_axis = Y.get('axis')
            y_values = Y.get('values')
        else:
            y_axis = "Nothing"
            y_values = [""]

        if pipe is not None:
            new_pipe = pipe
            positive = pipe["loader_settings"]["positive"] if "positive" in pipe["loader_settings"] else ""
            negative = pipe["loader_settings"]["negative"] if "negative" in pipe["loader_settings"] else ""

            if x_axis == 'advanced: ModelMergeBlocks':
                models = X.get('models')
                vae_use = X.get('vae_use')
                if models is None:
                    raise Exception("models is not found")
                new_pipe['loader_settings'] = {
                    **pipe['loader_settings'],
                    "models": models,
                    "vae_use": vae_use
                }
            if y_axis == 'advanced: ModelMergeBlocks':
                models = Y.get('models')
                vae_use = Y.get('vae_use')
                if models is None:
                    raise Exception("models is not found")
                new_pipe['loader_settings'] = {
                    **pipe['loader_settings'],
                    "models": models,
                    "vae_use": vae_use
                }

            if x_axis == 'advanced: Seeds++ Batch':
                if new_pipe['seed']:
                    value = x_values
                    x_values = []
                    for index in range(value):
                        x_values.append(str(new_pipe['seed'] + index))
                    x_values = "; ".join(x_values)
            if y_axis == 'advanced: Seeds++ Batch':
                if new_pipe['seed']:
                    value = y_values
                    y_values = []
                    for index in range(value):
                        y_values.append(str(new_pipe['seed'] + index))
                    y_values = "; ".join(y_values)

            if x_axis == 'advanced: Positive Prompt S/R':
                if positive:
                    x_value = x_values
                    x_values = []
                    for index, value in enumerate(x_value):
                        search_txt, replace_txt, replace_all = value
                        if replace_all:
                            txt = replace_txt if replace_txt is not None else positive
                            x_values.append(txt)
                        else:
                            txt = positive.replace(search_txt, replace_txt, 1) if replace_txt is not None else positive
                            x_values.append(txt)
                    x_values = "; ".join(x_values)
            if y_axis == 'advanced: Positive Prompt S/R':
                if positive:
                    y_value = y_values
                    y_values = []
                    for index, value in enumerate(y_value):
                        search_txt, replace_txt, replace_all = value
                        if replace_all:
                            txt = replace_txt if replace_txt is not None else positive
                            y_values.append(txt)
                        else:
                            txt = positive.replace(search_txt, replace_txt, 1) if replace_txt is not None else positive
                            y_values.append(txt)
                    y_values = "; ".join(y_values)

            if x_axis == 'advanced: Negative Prompt S/R':
                if negative:
                    x_value = x_values
                    x_values = []
                    for index, value in enumerate(x_value):
                        search_txt, replace_txt, replace_all = value
                        if replace_all:
                            txt = replace_txt if replace_txt is not None else negative
                            x_values.append(txt)
                        else:
                            txt = negative.replace(search_txt, replace_txt, 1) if replace_txt is not None else negative
                            x_values.append(txt)
                    x_values = "; ".join(x_values)
            if y_axis == 'advanced: Negative Prompt S/R':
                if negative:
                    y_value = y_values
                    y_values = []
                    for index, value in enumerate(y_value):
                        search_txt, replace_txt, replace_all = value
                        if replace_all:
                            txt = replace_txt if replace_txt is not None else negative
                            y_values.append(txt)
                        else:
                            txt = negative.replace(search_txt, replace_txt, 1) if replace_txt is not None else negative
                            y_values.append(txt)
                    y_values = "; ".join(y_values)

            if "advanced: ControlNet" in x_axis:
                x_value = x_values
                x_values = []
                cnet = []
                for index, value in enumerate(x_value):
                    cnet.append(value)
                    x_values.append(str(index))
                x_values = "; ".join(x_values)
                new_pipe['loader_settings'] = {
                    **pipe['loader_settings'],
                    "cnet_stack": cnet,
                }

            if "advanced: ControlNet" in y_axis:
                y_value = y_values
                y_values = []
                cnet = []
                for index, value in enumerate(y_value):
                    cnet.append(value)
                    y_values.append(str(index))
                y_values = "; ".join(y_values)
                new_pipe['loader_settings'] = {
                    **pipe['loader_settings'],
                    "cnet_stack": cnet,
                }

            if "advanced: Pos Condition" in x_axis:
                x_values = "; ".join(x_values)
                cond = X.get('cond')
                new_pipe['loader_settings'] = {
                    **pipe['loader_settings'],
                    "positive_cond_stack": cond,
                }
            if "advanced: Pos Condition" in y_axis:
                y_values = "; ".join(y_values)
                cond = Y.get('cond')
                new_pipe['loader_settings'] = {
                    **pipe['loader_settings'],
                    "positive_cond_stack": cond,
                }

            if "advanced: Neg Condition" in x_axis:
                x_values = "; ".join(x_values)
                cond = X.get('cond')
                new_pipe['loader_settings'] = {
                    **pipe['loader_settings'],
                    "negative_cond_stack": cond,
                }
            if "advanced: Neg Condition" in y_axis:
                y_values = "; ".join(y_values)
                cond = Y.get('cond')
                new_pipe['loader_settings'] = {
                    **pipe['loader_settings'],
                    "negative_cond_stack": cond,
                }

            del pipe

        return pipeXYPlot().plot(grid_spacing, output_individuals, flip_xy, x_axis, x_values, y_axis, y_values, new_pipe)

#---------------------------------------------------------------XY Inputs 开始----------------------------------------------------------------------#

def load_preset(filename):
    path = os.path.join(RESOURCES_DIR, filename)
    path = os.path.abspath(path)
    preset_list = []

    if os.path.exists(path):
        with open(path, 'r') as file:
            for line in file:
                preset_list.append(line.strip())

        return preset_list
    else:
        return []
def generate_floats(batch_count, first_float, last_float):
    if batch_count > 1:
        interval = (last_float - first_float) / (batch_count - 1)
        values = [str(round(first_float + i * interval, 3)) for i in range(batch_count)]
    else:
        values = [str(first_float)] if batch_count == 1 else []
    return "; ".join(values)

def generate_ints(batch_count, first_int, last_int):
    if batch_count > 1:
        interval = (last_int - first_int) / (batch_count - 1)
        values = [str(int(first_int + i * interval)) for i in range(batch_count)]
    else:
        values = [str(first_int)] if batch_count == 1 else []
    # values = list(set(values))  # Remove duplicates
    # values.sort()  # Sort in ascending order
    return "; ".join(values)

# Seed++ Batch
class XYplot_SeedsBatch:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "batch_count": ("INT", {"default": 3, "min": 1, "max": 50}), },
        }

    RETURN_TYPES = ("X_Y",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "EasyUse/XY Inputs"

    def xy_value(self, batch_count):

        axis = "advanced: Seeds++ Batch"
        xy_values = {"axis": axis, "values": batch_count}
        return (xy_values,)

# Step Values
class XYplot_Steps:
    parameters = ["steps", "start_at_step", "end_at_step",]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_parameter": (cls.parameters,),
                "batch_count": ("INT", {"default": 3, "min": 0, "max": 50}),
                "first_step": ("INT", {"default": 10, "min": 1, "max": 10000}),
                "last_step": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "first_start_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "last_start_step": ("INT", {"default": 10, "min": 0, "max": 10000}),
                "first_end_step": ("INT", {"default": 10, "min": 0, "max": 10000}),
                "last_end_step": ("INT", {"default": 20, "min": 0, "max": 10000}),
            }
        }

    RETURN_TYPES = ("X_Y",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "EasyUse/XY Inputs"

    def xy_value(self, target_parameter, batch_count, first_step, last_step, first_start_step, last_start_step,
                 first_end_step, last_end_step,):

        axis, xy_first, xy_last = None, None, None

        if target_parameter == "steps":
            axis = "advanced: Steps"
            xy_first = first_step
            xy_last = last_step
        elif target_parameter == "start_at_step":
            axis = "advanced: StartStep"
            xy_first = first_start_step
            xy_last = last_start_step
        elif target_parameter == "end_at_step":
            axis = "advanced: EndStep"
            xy_first = first_end_step
            xy_last = last_end_step

        values = generate_ints(batch_count, xy_first, xy_last)
        return ({"axis": axis, "values": values},) if values is not None else (None,)

class XYplot_CFG:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_count": ("INT", {"default": 3, "min": 0, "max": 50}),
                "first_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "last_cfg": ("FLOAT", {"default": 9.0, "min": 0.0, "max": 100.0}),
            }
        }

    RETURN_TYPES = ("X_Y",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "EasyUse/XY Inputs"

    def xy_value(self, batch_count, first_cfg, last_cfg):
        axis = "advanced: CFG Scale"
        values = generate_floats(batch_count, first_cfg, last_cfg)
        return ({"axis": axis, "values": values},) if values else (None,)

# Step Values
class XYplot_Sampler_Scheduler:
    parameters = ["sampler", "scheduler", "sampler & scheduler"]

    @classmethod
    def INPUT_TYPES(cls):
        samplers = ["None"] + comfy.samplers.KSampler.SAMPLERS
        schedulers = ["None"] + comfy.samplers.KSampler.SCHEDULERS
        inputs = {
            "required": {
                "target_parameter": (cls.parameters,),
                "input_count": ("INT", {"default": 1, "min": 1, "max": 30, "step": 1})
            }
        }
        for i in range(1, 30 + 1):
            inputs["required"][f"sampler_{i}"] = (samplers,)
            inputs["required"][f"scheduler_{i}"] = (schedulers,)

        return inputs

    RETURN_TYPES = ("X_Y",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "EasyUse/XY Inputs"

    def xy_value(self, target_parameter, input_count, **kwargs):
        axis, values, = None, None,
        if target_parameter == "scheduler":
            axis = "advanced: Scheduler"
            schedulers = [kwargs.get(f"scheduler_{i}") for i in range(1, input_count + 1)]
            values = [scheduler for scheduler in schedulers if scheduler != "None"]
        elif target_parameter == "sampler":
            axis = "advanced: Sampler"
            samplers = [kwargs.get(f"sampler_{i}") for i in range(1, input_count + 1)]
            values = [sampler for sampler in samplers if sampler != "None"]
        else:
            axis = "advanced: Sampler&Scheduler"
            samplers = [kwargs.get(f"sampler_{i}") for i in range(1, input_count + 1)]
            schedulers = [kwargs.get(f"scheduler_{i}") for i in range(1, input_count + 1)]
            values = []
            for sampler, scheduler in zip(samplers, schedulers):
                sampler = sampler if sampler else 'None'
                scheduler = scheduler if scheduler else 'None'
                values.append(sampler +', '+ scheduler)
        values = "; ".join(values)
        return ({"axis": axis, "values": values},) if values else (None,)

class XYplot_Denoise:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_count": ("INT", {"default": 3, "min": 0, "max": 50}),
                "first_denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "last_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("X_Y",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "EasyUse/XY Inputs"

    def xy_value(self, batch_count, first_denoise, last_denoise):
        axis = "advanced: Denoise"
        values = generate_floats(batch_count, first_denoise, last_denoise)
        return ({"axis": axis, "values": values},) if values else (None,)

# PromptSR
class XYplot_PromptSR:

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "target_prompt": (["positive", "negative"],),
                "search_txt": ("STRING", {"default": "", "multiline": False}),
                "replace_all_text": ("BOOLEAN", {"default": False}),
                "replace_count": ("INT", {"default": 3, "min": 1, "max": 30 - 1}),
            }
        }

        # Dynamically add replace_X inputs
        for i in range(1, 30):
            replace_key = f"replace_{i}"
            inputs["required"][replace_key] = ("STRING", {"default": "", "multiline": False, "placeholder": replace_key})

        return inputs

    RETURN_TYPES = ("X_Y",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "EasyUse/XY Inputs"

    def xy_value(self, target_prompt, search_txt, replace_all_text, replace_count, **kwargs):
        axis = None

        if target_prompt == "positive":
            axis = "advanced: Positive Prompt S/R"
        elif target_prompt == "negative":
            axis = "advanced: Negative Prompt S/R"

        # Create base entry
        values = [(search_txt, None, replace_all_text)]

        if replace_count > 0:
            # Append additional entries based on replace_count
            values.extend([(search_txt, kwargs.get(f"replace_{i+1}"), replace_all_text) for i in range(replace_count)])
        return ({"axis": axis, "values": values},) if values is not None else (None,)

# XYPlot Pos Condition
class XYplot_Positive_Cond:

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "optional": {
                "positive_1": ("CONDITIONING",),
                "positive_2": ("CONDITIONING",),
                "positive_3": ("CONDITIONING",),
                "positive_4": ("CONDITIONING",),
            }
        }

        return inputs

    RETURN_TYPES = ("X_Y",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "EasyUse/XY Inputs"

    def xy_value(self, positive_1=None, positive_2=None, positive_3=None, positive_4=None):
        axis = "advanced: Pos Condition"
        values = []
        cond = []
        # Create base entry
        if positive_1 is not None:
            values.append("0")
            cond.append(positive_1)
        if positive_2 is not None:
            values.append("1")
            cond.append(positive_2)
        if positive_3 is not None:
            values.append("2")
            cond.append(positive_3)
        if positive_4 is not None:
            values.append("3")
            cond.append(positive_4)

        return ({"axis": axis, "values": values, "cond": cond},) if values is not None else (None,)

# XYPlot Neg Condition
class XYplot_Negative_Cond:

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "optional": {
                "negative_1": ("CONDITIONING"),
                "negative_2": ("CONDITIONING"),
                "negative_3": ("CONDITIONING"),
                "negative_4": ("CONDITIONING"),
            }
        }

        return inputs

    RETURN_TYPES = ("X_Y",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "EasyUse/XY Inputs"

    def xy_value(self, negative_1=None, negative_2=None, negative_3=None, negative_4=None):
        axis = "advanced: Neg Condition"
        values = []
        cond = []
        # Create base entry
        if negative_1 is not None:
            values.append(0)
            cond.append(negative_1)
        if negative_2 is not None:
            values.append(1)
            cond.append(negative_2)
        if negative_3 is not None:
            values.append(2)
            cond.append(negative_3)
        if negative_4 is not None:
            values.append(3)
            cond.append(negative_4)

        return ({"axis": axis, "values": values, "cond": cond},) if values is not None else (None,)

# XYPlot Pos Condition List
class XYplot_Positive_Cond_List:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("X_Y",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "EasyUse/XY Inputs"

    def xy_value(self, positive):
        axis = "advanced: Pos Condition"
        values = []
        cond = []
        for index, c in enumerate(positive):
            values.append(str(index))
            cond.append(c)

        return ({"axis": axis, "values": values, "cond": cond},) if values is not None else (None,)

# XYPlot Neg Condition List
class XYplot_Negative_Cond_List:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "negative": ("CONDITIONING",),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("X_Y",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "EasyUse/XY Inputs"

    def xy_value(self, negative):
        axis = "advanced: Neg Condition"
        values = []
        cond = []
        for index, c in enumerate(negative):
            values.append(index)
            cond.append(c)

        return ({"axis": axis, "values": values, "cond": cond},) if values is not None else (None,)

# XY Plot: ControlNet
class XYplot_Control_Net:
    parameters = ["strength", "start_percent", "end_percent"]
    @classmethod
    def INPUT_TYPES(cls):
        def get_file_list(filenames):
            return [file for file in filenames if file != "put_models_here.txt" and "lllite" not in file]

        return {
            "required": {
                "control_net_name": (get_file_list(folder_paths.get_filename_list("controlnet")),),
                "image": ("IMAGE",),
                "target_parameter": (cls.parameters,),
                "batch_count": ("INT", {"default": 3, "min": 1, "max": 30}),
                "first_strength": ("FLOAT", {"default": 0.0, "min": 0.00, "max": 10.0, "step": 0.01}),
                "last_strength": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 10.0, "step": 0.01}),
                "first_start_percent": ("FLOAT", {"default": 0.0, "min": 0.00, "max": 1.0, "step": 0.01}),
                "last_start_percent": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 1.0, "step": 0.01}),
                "first_end_percent": ("FLOAT", {"default": 0.0, "min": 0.00, "max": 1.0, "step": 0.01}),
                "last_end_percent": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 1.0, "step": 0.01}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.00, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("X_Y",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "EasyUse/XY Inputs"

    def xy_value(self, control_net_name, image, target_parameter, batch_count, first_strength, last_strength, first_start_percent,
                 last_start_percent, first_end_percent, last_end_percent, strength, start_percent, end_percent):

        axis, = None,

        values = []

        if target_parameter == "strength":
            axis = "advanced: ControlNetStrength"

            values.append([(control_net_name, image, first_strength, start_percent, end_percent)])
            strength_increment = (last_strength - first_strength) / (batch_count - 1) if batch_count > 1 else 0
            for i in range(1, batch_count - 1):
                values.append([(control_net_name, image, first_strength + i * strength_increment, start_percent,
                                end_percent)])
            if batch_count > 1:
                values.append([(control_net_name, image, last_strength, start_percent, end_percent)])

        elif target_parameter == "start_percent":
            axis = "advanced: ControlNetStart%"

            percent_increment = (last_start_percent - first_start_percent) / (batch_count - 1) if batch_count > 1 else 0
            values.append([(control_net_name, image, strength, first_start_percent, end_percent)])
            for i in range(1, batch_count - 1):
                values.append([(control_net_name, image, strength, first_start_percent + i * percent_increment,
                                  end_percent)])

            # Always add the last start_percent if batch_count is more than 1.
            if batch_count > 1:
                values.append((control_net_name, image, strength, last_start_percent, end_percent))

        elif target_parameter == "end_percent":
            axis = "advanced: ControlNetEnd%"

            percent_increment = (last_end_percent - first_end_percent) / (batch_count - 1) if batch_count > 1 else 0
            values.append([(control_net_name, image, image, strength, start_percent, first_end_percent)])
            for i in range(1, batch_count - 1):
                values.append([(control_net_name, image, strength, start_percent,
                                  first_end_percent + i * percent_increment)])

            if batch_count > 1:
                values.append([(control_net_name, image, strength, start_percent, last_end_percent)])


        return ({"axis": axis, "values": values},)


# 模型叠加
class XYplot_ModelMergeBlocks:

    @classmethod
    def INPUT_TYPES(s):
        checkpoints = folder_paths.get_filename_list("checkpoints")
        vae = ["Use Model 1", "Use Model 2"] + folder_paths.get_filename_list("vae")

        preset = ["Preset"]  # 20
        preset += load_preset("mmb-preset.txt")
        preset += load_preset("mmb-preset.custom.txt")

        default_vectors = "1,0,0; \n0,1,0; \n0,0,1; \n1,1,0; \n1,0,1; \n0,1,1; "
        return {
            "required": {
                "ckpt_name_1": (checkpoints,),
                "ckpt_name_2": (checkpoints,),
                "vae_use": (vae, {"default": "Use Model 1"}),
                "preset": (preset, {"default": "preset"}),
                "values": ("STRING", {"default": default_vectors, "multiline": True, "placeholder": 'Support 2 methods:\n\n1.input, middle, out in same line and insert values seperated by "; "\n\n2.model merge block number seperated by ", " in same line and insert values seperated by "; "'}),
            },
            "hidden": {"my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("X_Y",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"

    CATEGORY = "EasyUse/XY Inputs"

    def xy_value(self, ckpt_name_1, ckpt_name_2, vae_use, preset, values, my_unique_id=None):

        axis = "advanced: ModelMergeBlocks"
        if ckpt_name_1 is None:
            raise Exception("ckpt_name_1 is not found")
        if ckpt_name_2 is None:
            raise Exception("ckpt_name_2 is not found")

        models = (ckpt_name_1, ckpt_name_2)

        xy_values = {"axis":axis, "values":values, "models":models, "vae_use": vae_use}
        return (xy_values,)

# 显示推理时间
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

# 显示加载器参数中的各种名称
class showLoaderSettingsNames:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "names": ("INFO", {"default": '', "forceInput": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("ckpt_name", "vae_name", "lora_name")

    FUNCTION = "notify"
    OUTPUT_NODE = True

    CATEGORY = "EasyUse/Util"

    def notify(self, pipe, names=None, unique_id=None, extra_pnginfo=None):
        if unique_id and extra_pnginfo and "workflow" in extra_pnginfo:
            workflow = extra_pnginfo["workflow"]
            node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id), None)
            if node:
                ckpt_name = pipe['loader_settings']['ckpt_name'] if 'ckpt_name' in pipe['loader_settings'] else ''
                vae_name = pipe['loader_settings']['vae_name'] if 'vae_name' in pipe['loader_settings'] else ''
                lora_name = pipe['loader_settings']['lora_name'] if 'lora_name' in pipe['loader_settings'] else ''

                if ckpt_name:
                    ckpt_name = os.path.basename(os.path.splitext(ckpt_name)[0])
                if vae_name:
                    vae_name = os.path.basename(os.path.splitext(vae_name)[0])
                if lora_name:
                    lora_name = os.path.basename(os.path.splitext(lora_name)[0])


                names = "ckpt_name: " + ckpt_name + '\n' + "vae_name: " + vae_name + '\n' + "lora_name: " + lora_name
                node["widgets_values"] = names

        return {"ui": {"text": names}, "result": (ckpt_name, vae_name, lora_name)}


NODE_CLASS_MAPPINGS = {
    # prompt 提示词
    "easy positive": positivePrompt,
    "easy negative": negativePrompt,
    "easy wildcards": wildcardsPrompt,
    "easy promptList": promptList,
    "easy stylesSelector": stylesPromptSelector,
    "easy portraitMaster": portraitMaster,
    # loaders 加载器
    "easy fullLoader": fullLoader,
    "easy a1111Loader": a1111Loader,
    "easy comfyLoader": comfyLoader,
    "easy zero123Loader": zero123Loader,
    "easy svdLoader": svdLoader,
    "easy loraStack": loraStackLoader,
    "easy controlnetLoader": controlnetSimple,
    "easy controlnetLoaderADV": controlnetAdvanced,
    # latent 潜空间
    "easy latentNoisy": latentNoisy,
    "easy latentCompositeMaskedWithCond": latentCompositeMaskedWithCond,
    # seed 随机种
    "easy seed": easySeed,
    "easy globalSeed": globalSeed,
    # preSampling 预采样处理
    "easy preSampling": samplerSettings,
    "easy preSamplingAdvanced": samplerSettingsAdvanced,
    "easy preSamplingSdTurbo": sdTurboSettings,
    "easy preSamplingDynamicCFG": dynamicCFGSettings,
    # kSampler k采样器
    "easy kSampler": samplerSimple,
    "easy fullkSampler": samplerFull,
    "easy kSamplerTiled": samplerSimpleTiled,
    "easy kSamplerInpainting": samplerSimpleInpainting,
    "easy kSamplerDownscaleUnet": samplerSimpleDownscaleUnet,
    "easy kSamplerSDTurbo": samplerSDTurbo,
    "easy unSampler": unsampler,
    # fix 修复相关
    "easy hiresFix": hiresFix,
    "easy preDetailerFix": preDetailerFix,
    "easy ultralyticsDetectorPipe": ultralyticsDetectorForDetailerFix,
    "easy samLoaderPipe": samLoaderForDetailerFix,
    "easy detailerFix": detailerFix,
    # pipe 管道（节点束）
    "easy pipeIn": pipeIn,
    "easy pipeOut": pipeOut,
    "easy pipeToBasicPipe": pipeToBasicPipe,
    "easy XYPlot": pipeXYPlot,
    "easy XYPlotAdvanced": pipeXYPlotAdvanced,
    # XY Inputs
    "easy XYInputs: Seeds++ Batch": XYplot_SeedsBatch,
    "easy XYInputs: Steps": XYplot_Steps,
    "easy XYInputs: CFG Scale": XYplot_CFG,
    "easy XYInputs: Sampler/Scheduler": XYplot_Sampler_Scheduler,
    "easy XYInputs: Denoise": XYplot_Denoise,
    "easy XYInputs: ModelMergeBlocks": XYplot_ModelMergeBlocks,
    "easy XYInputs: PromptSR": XYplot_PromptSR,
    "easy XYInputs: ControlNet": XYplot_Control_Net,
    "easy XYInputs: PositiveCond": XYplot_Positive_Cond,
    "easy XYInputs: PositiveCondList": XYplot_Positive_Cond_List,
    "easy XYInputs: NegativeCond": XYplot_Negative_Cond,
    "easy XYInputs: NegativeCondList": XYplot_Negative_Cond_List,
    # others 其他
    "easy showSpentTime": showSpentTime,
    "easy showLoaderSettingsNames": showLoaderSettingsNames,
    # "easy imageRemoveBG": imageREMBG,
    "dynamicThresholdingFull": dynamicThresholdingFull,
    # __for_testing 测试
    "easy fooocusInpaintLoader": fooocusInpaintLoader
}
NODE_DISPLAY_NAME_MAPPINGS = {
    # prompt 提示词
    "easy positive": "Positive",
    "easy negative": "Negative",
    "easy wildcards": "Wildcards",
    "easy promptList": "PromptList",
    "easy stylesSelector": "Styles Selector",
    "easy portraitMaster": "Portrait Master",
    # loaders 加载器
    "easy fullLoader": "EasyLoader (Full)",
    "easy a1111Loader": "EasyLoader (A1111)",
    "easy comfyLoader": "EasyLoader (Comfy)",
    "easy zero123Loader": "EasyLoader (Zero123)",
    "easy svdLoader": "EasyLoader (SVD)",
    "easy loraStack": "EasyLoraStack",
    "easy controlnetLoader": "EasyControlnet",
    "easy controlnetLoaderADV": "EasyControlnet (Advanced)",
    "easy photoMakerApply": "Apply PhotoMaker",
    # latent 潜空间
    "easy latentNoisy": "LatentNoisy",
    "easy latentCompositeMaskedWithCond": "LatentCompositeMaskedWithCond",
    # seed 随机种
    "easy seed": "EasySeed",
    "easy globalSeed": "EasyGlobalSeed",
    # preSampling 预采样处理
    "easy preSampling": "PreSampling",
    "easy preSamplingAdvanced": "PreSampling (Advanced)",
    "easy preSamplingSdTurbo": "PreSampling (SDTurbo)",
    "easy preSamplingDynamicCFG": "PreSampling (DynamicCFG)",
    # kSampler k采样器
    "easy kSampler": "EasyKSampler",
    "easy fullkSampler": "EasyKSampler (Full)",
    "easy kSamplerTiled": "EasyKSampler (Tiled Decode)",
    "easy kSamplerInpainting": "EasyKSampler (Inpainting)",
    "easy kSamplerDownscaleUnet": "EasyKsampler (Downscale Unet)",
    "easy kSamplerSDTurbo": "EasyKSampler (SDTurbo)",
    "easy unSampler": "EasyUnSampler",
    # fix 修复相关
    "easy hiresFix": "HiresFix",
    "easy preDetailerFix": "PreDetailerFix",
    "easy ultralyticsDetectorPipe": "UltralyticsDetector (Pipe)",
    "easy samLoaderPipe": "SAMLoader (Pipe)",
    "easy detailerFix": "DetailerFix",
    # pipe 管道（节点束）
    "easy pipeIn": "Pipe In",
    "easy pipeOut": "Pipe Out",
    "easy pipeToBasicPipe": "Pipe -> BasicPipe",
    "easy XYPlot": "XY Plot",
    "easy XYPlotAdvanced": "XY Plot Advanced",
    # XY Inputs
    "easy XYInputs: Seeds++ Batch": "XY Inputs: Seeds++ Batch //EasyUse",
    "easy XYInputs: Steps": "XY Inputs: Steps //EasyUse",
    "easy XYInputs: CFG Scale": "XY Inputs: CFG Scale //EasyUse",
    "easy XYInputs: Sampler/Scheduler": "XY Inputs: Sampler/Scheduler //EasyUse",
    "easy XYInputs: Denoise": "XY Inputs: Denoise //EasyUse",
    "easy XYInputs: ModelMergeBlocks": "XY Inputs: ModelMergeBlocks //EasyUse",
    "easy XYInputs: PromptSR": "XY Inputs: PromptSR //EasyUse",
    "easy XYInputs: ControlNet": "XY Inputs: Controlnet //EasyUse",
    "easy XYInputs: PositiveCond": "XY Inputs: PosCond //EasyUse",
    "easy XYInputs: PositiveCondList": "XY Inputs: PosCondList //EasyUse",
    "easy XYInputs: NegativeCond": "XY Inputs: NegCond //EasyUse",
    "easy XYInputs: NegativeCondList": "XY Inputs: NegCondList //EasyUse",
    # others 其他
    "easy showSpentTime": "ShowSpentTime",
    "easy showLoaderSettingsNames": "ShowLoaderSettingsNames",
    "easy imageRemoveBG": "ImageRemoveBG",
    "dynamicThresholdingFull": "DynamicThresholdingFull",
    # __for_testing 测试
    "easy fooocusInpaintLoader": "Load Fooocus Inpaint"
}