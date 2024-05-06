import os
import base64
import torch
import numpy as np
from enum import Enum
from PIL import Image
from io import BytesIO
from typing import List, Union

import folder_paths
from .utils import install_package

# PIL to Tensor
def pil2tensor(image):
  return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
# np to Tensor
def np2tensor(img_np: Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor:
  if isinstance(img_np, list):
    return torch.cat([np2tensor(img) for img in img_np], dim=0)
  return torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0)
# Tensor to np
def tensor2np(tensor: torch.Tensor) -> List[np.ndarray]:
  if len(tensor.shape) == 3:  # Single image
    return np.clip(255.0 * tensor.cpu().numpy(), 0, 255).astype(np.uint8)
  else:  # Batch of images
    return [np.clip(255.0 * t.cpu().numpy(), 0, 255).astype(np.uint8) for t in tensor]

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

def RGB2RGBA(image: Image, mask: Image) -> Image:
  (R, G, B) = image.convert('RGB').split()
  return Image.merge('RGBA', (R, G, B, mask.convert('L')))

def image2mask(image: Image) -> torch.Tensor:
  _image = image.convert('RGBA')
  alpha = _image.split()[0]
  bg = Image.new("L", _image.size)
  _image = Image.merge('RGBA', (bg, bg, bg, alpha))
  ret_mask = torch.tensor([pil2tensor(_image)[0, :, :, 3].tolist()])
  return ret_mask

def mask2image(mask: torch.Tensor) -> Image:
  masks = tensor2np(mask)
  for m in masks:
    _mask = Image.fromarray(m).convert("L")
    _image = Image.new("RGBA", _mask.size, color='white')
    _image = Image.composite(
      _image, Image.new("RGBA", _mask.size, color='black'), _mask)
  return _image

# 图像融合
class blendImage:
  def g(self, x):
    return torch.where(x <= 0.25, ((16 * x - 12) * x + 4) * x, torch.sqrt(x))

  def blend_mode(self, img1, img2, mode):
    if mode == "normal":
      return img2
    elif mode == "multiply":
      return img1 * img2
    elif mode == "screen":
      return 1 - (1 - img1) * (1 - img2)
    elif mode == "overlay":
      return torch.where(img1 <= 0.5, 2 * img1 * img2, 1 - 2 * (1 - img1) * (1 - img2))
    elif mode == "soft_light":
      return torch.where(img2 <= 0.5, img1 - (1 - 2 * img2) * img1 * (1 - img1),
                         img1 + (2 * img2 - 1) * (self.g(img1) - img1))
    elif mode == "difference":
      return img1 - img2
    else:
      raise ValueError(f"Unsupported blend mode: {mode}")

  def blend_images(self, image1: torch.Tensor, image2: torch.Tensor, blend_factor: float, blend_mode: str = 'normal'):
    image2 = image2.to(image1.device)
    if image1.shape != image2.shape:
      image2 = image2.permute(0, 3, 1, 2)
      image2 = comfy.utils.common_upscale(image2, image1.shape[2], image1.shape[1], upscale_method='bicubic',
                                          crop='center')
      image2 = image2.permute(0, 2, 3, 1)

    blended_image = self.blend_mode(image1, image2, blend_mode)
    blended_image = image1 * (1 - blend_factor) + blended_image * blend_factor
    blended_image = torch.clamp(blended_image, 0, 1)
    return blended_image




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
      from clip_interrogator import Config, Interrogator
      global Config, Interrogator
    except:
      install_package("clip_interrogator", "0.6.0")
      from clip_interrogator import Config, Interrogator

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

ci = CI_Inference()
