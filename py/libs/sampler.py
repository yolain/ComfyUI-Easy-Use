import comfy
import torch
import numpy as np
import latent_preview
from nodes import MAX_RESOLUTION
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union, Any
from .utils import get_sd_version
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

    def add_model_patch_option(self, model):
        if 'transformer_options' not in model.model_options:
            model.model_options['transformer_options'] = {}
        to = model.model_options['transformer_options']
        if "model_patch" not in to:
            to["model_patch"] = {}
        return to

    def common_ksampler(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0,
                        disable_noise=False, start_step=None, last_step=None, force_full_denoise=False,
                        preview_latent=True, disable_pbar=False, custom=None):
        device = comfy.model_management.get_torch_device()
        latent_image = latent["samples"]

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

        if custom is not None:
            guider = custom['guider'] if 'guider' in custom else None
            sampler = custom['sampler'] if 'sampler' in custom else None
            sigmas = custom['sigmas'] if 'sigmas' in custom else None
            noise = custom['noise'] if 'noise' in custom else None
            samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas,
                                    denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar,
                                    seed=noise.seed)
            samples = samples.to(comfy.model_management.intermediate_device())
        else:
            if disable_noise:
                noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout,
                                    device="cpu")
            else:
                batch_inds = latent["batch_index"] if "batch_index" in latent else None
                noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

            #######################################################################################
            # brushnet
            transformer_options = model.model_options['transformer_options'] if "transformer_options" in model.model_options else {}
            if 'model_patch' in transformer_options and 'brushnet_model' in transformer_options['model_patch']:
                to = self.add_model_patch_option(model)
                to['model_patch']['step'] = 0
                to['model_patch']['total_steps'] = steps
                to['model_patch']['cfg'] = cfg

                def callback(step, x0, x, total_steps):
                    if to is not None and "model_patch" in to:
                        to['model_patch']['step'] = step + 1
                    preview_bytes = None
                    if previewer:
                        preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
                    pbar.update_absolute(step + 1, total_steps, preview_bytes)
            #
            #######################################################################################

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

        #######################################################################################
        # brushnet
        to = None
        transformer_options = model.model_options['transformer_options'] if "transformer_options" in model.model_options else {}
        if 'model_patch' in transformer_options and 'brushnet_model' in transformer_options['model_patch']:
            to = self.add_model_patch_option(model)
            to['model_patch']['step'] = 0
            to['model_patch']['total_steps'] = steps
            to['model_patch']['cfg'] = cfg
        #
        #######################################################################################

        def callback(step, x0, x, total_steps):
            preview_bytes = None
            if to is not None and "model_patch" in to:
                to['model_patch']['step'] = step + 1
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

class alignYourStepsScheduler:

    NOISE_LEVELS = {
        "SD1": [14.6146412293, 6.4745760956, 3.8636745985, 2.6946151520, 1.8841921177, 1.3943805092, 0.9642583904,
                0.6523686016, 0.3977456272, 0.1515232662, 0.0291671582],
        "SDXL": [14.6146412293, 6.3184485287, 3.7681790315, 2.1811480769, 1.3405244945, 0.8620721141, 0.5550693289,
                 0.3798540708, 0.2332364134, 0.1114188177, 0.0291671582],
        "SVD": [700.00, 54.5, 15.886, 7.977, 4.248, 1.789, 0.981, 0.403, 0.173, 0.034, 0.002]}


    def loglinear_interp(self, t_steps, num_steps):
        """
        Performs log-linear interpolation of a given array of decreasing numbers.
        """
        xs = np.linspace(0, 1, len(t_steps))
        ys = np.log(t_steps[::-1])

        new_xs = np.linspace(0, 1, num_steps)
        new_ys = np.interp(new_xs, xs, ys)

        interped_ys = np.exp(new_ys)[::-1].copy()
        return interped_ys

    def get_sigmas(self, model_type, steps, denoise):

        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0:
                return (torch.FloatTensor([]),)
            total_steps = round(steps * denoise)

        sigmas = self.NOISE_LEVELS[model_type][:]
        if (steps + 1) != len(sigmas):
            sigmas = self.loglinear_interp(sigmas, steps + 1)

        sigmas = sigmas[-(total_steps + 1):]
        sigmas[-1] = 0
        return (torch.FloatTensor(sigmas),)