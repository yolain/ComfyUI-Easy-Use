import os
import base64
import torch
import numpy as np
from enum import Enum
from PIL import Image
from io import BytesIO

import folder_paths
from .utils import install_package

# PIL to Tensor
def pil2tensor(image):
  return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2byte(pil_image, format='PNG'):
  byte_arr = BytesIO()
  pil_image.save(byte_arr, format=format)
  byte_arr.seek(0)
  return byte_arr

def image2base64(image_base64):
    image_bytes = base64.b64decode(image_base64)
    image_data = Image.open(BytesIO(image_bytes))
    return image_data

# Get new bounds
def get_new_bounds(width, height, left, right, top, bottom):
  """Returns the new bounds for an image with inset crop data."""
  left = 0 + left
  right = width - right
  top = 0 + top
  bottom = height - bottom
  return (left, right, top, bottom)


class ResizeMode(Enum):
  RESIZE = "Just Resize"
  INNER_FIT = "Crop and Resize"
  OUTER_FIT = "Resize and Fill"
  def int_value(self):
    if self == ResizeMode.RESIZE:
      return 0
    elif self == ResizeMode.INNER_FIT:
      return 1
    elif self == ResizeMode.OUTER_FIT:
      return 2
    assert False, "NOTREACHED"



# CLIP反推
import comfy.utils
from torchvision import transforms
Config, Interrogator = None, None
class CI_Inference:
  ci_model = None
  cache_path: str

  def __init__(self):
    self.ci_model = None
    self.low_vram = False
    self.cache_path = os.path.join(folder_paths.models_dir, "clip_interrogator")

  def _load_model(self, model_name, low_vram=False):
    if not (self.ci_model and model_name == self.ci_model.config.clip_model_name and self.low_vram == low_vram):
      self.low_vram = low_vram
      print(f"Load model: {model_name}")

      config = Config(
        device="cuda" if torch.cuda.is_available() else "cpu",
        download_cache=True,
        clip_model_name=model_name,
        clip_model_path=self.cache_path,
        cache_path=self.cache_path,
        caption_model_name='blip-large'
      )

      if low_vram:
        config.apply_low_vram_defaults()

      self.ci_model = Interrogator(config)

  def _interrogate(self, image, mode, caption=None):
    if mode == 'best':
      prompt = self.ci_model.interrogate(image, caption=caption)
    elif mode == 'classic':
      prompt = self.ci_model.interrogate_classic(image, caption=caption)
    elif mode == 'fast':
      prompt = self.ci_model.interrogate_fast(image, caption=caption)
    elif mode == 'negative':
      prompt = self.ci_model.interrogate_negative(image)
    else:
      raise Exception(f"Unknown mode {mode}")
    return prompt

  def image_to_prompt(self, image, mode, model_name='ViT-L-14/openai', low_vram=False):
    try:
      install_package("clip_interrogator", "0.6.0")
      from clip_interrogator import Config, Interrogator
      global Config, Interrogator

      pbar = comfy.utils.ProgressBar(len(image))

      self._load_model(model_name, low_vram)
      prompt = []
      for i in range(len(image)):
        im = image[i]

        im = tensor2pil(im)
        im = im.convert('RGB')

        _prompt = self._interrogate(im, mode)
        pbar.update(1)
        prompt.append(_prompt)

      return prompt
    except Exception as e:
      print(e)
      return [""]

ci = CI_Inference()
