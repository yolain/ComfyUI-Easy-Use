from PIL import Image
from enum import Enum
import os
import hashlib
import folder_paths
import torch
import numpy as np
from nodes import MAX_RESOLUTION
from .log import log_node_info


def pil2tensor(image):
  return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

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

RESIZE_MODES = [ResizeMode.RESIZE.value, ResizeMode.INNER_FIT.value, ResizeMode.OUTER_FIT.value]

def get_new_bounds(width, height, left, right, top, bottom):
  """Returns the new bounds for an image with inset crop data."""
  left = 0 + left
  right = width - right
  top = 0 + top
  bottom = height - bottom
  return (left, right, top, bottom)

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# 图像裁切
class imageInsetCrop:

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
    return {
      "required": {
        "image": ("IMAGE",),
        "measurement": (['Pixels', 'Percentage'],),
        "left": ("INT", {
          "default": 0,
          "min": 0,
          "max": MAX_RESOLUTION,
          "step": 8
        }),
        "right": ("INT", {
          "default": 0,
          "min": 0,
          "max": MAX_RESOLUTION,
          "step": 8
        }),
        "top": ("INT", {
          "default": 0,
          "min": 0,
          "max": MAX_RESOLUTION,
          "step": 8
        }),
        "bottom": ("INT", {
          "default": 0,
          "min": 0,
          "max": MAX_RESOLUTION,
          "step": 8
        }),
      },
    }

  RETURN_TYPES = ("IMAGE",)
  FUNCTION = "crop"

  CATEGORY = "EasyUse/Image"

  # pylint: disable = too-many-arguments
  def crop(self, measurement, left, right, top, bottom, image=None):
    """Does the crop."""

    _, height, width, _ = image.shape

    if measurement == 'Percentage':
      left = int(width - (width * (100 - left) / 100))
      right = int(width - (width * (100 - right) / 100))
      top = int(height - (height * (100 - top) / 100))
      bottom = int(height - (height * (100 - bottom) / 100))

    # Snap to 8 pixels
    left = left // 8 * 8
    right = right // 8 * 8
    top = top // 8 * 8
    bottom = bottom // 8 * 8

    if left == 0 and right == 0 and bottom == 0 and top == 0:
      return (image,)

    inset_left, inset_right, inset_top, inset_bottom = get_new_bounds(width, height, left, right,
                                                                      top, bottom)
    if inset_top > inset_bottom:
      raise ValueError(
        f"Invalid cropping dimensions top ({inset_top}) exceeds bottom ({inset_bottom})")
    if inset_left > inset_right:
      raise ValueError(
        f"Invalid cropping dimensions left ({inset_left}) exceeds right ({inset_right})")

    log_node_info("Image Inset Crop", f'Cropping image {width}x{height} width inset by {inset_left},{inset_right}, ' +
                 f'and height inset by {inset_top}, {inset_bottom}')
    image = image[:, inset_top:inset_bottom, inset_left:inset_right, :]

    return (image,)

# 图像尺寸
class imageSize:
  def __init__(self):
    pass

  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "image": ("IMAGE",),
      }
    }

  RETURN_TYPES = ("INT", "INT")
  RETURN_NAMES = ("width_int", "height_int")
  OUTPUT_NODE = True
  FUNCTION = "image_width_height"

  CATEGORY = "EasyUse/Image"

  def image_width_height(self, image):
    _, raw_H, raw_W, _ = image.shape

    width = raw_W
    height = raw_H

    if width is not None and height is not None:
      result = (width, height)
    else:
      result = (0, 0)
    return {"ui": {"text": "Width: "+str(width)+" , Height: "+str(height)}, "result": result}

# 图像尺寸（最长边）
class imageSizeBySide:
  def __init__(self):
    pass

  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "image": ("IMAGE",),
        "side": (["Longest", "Shortest"],)
      }
    }

  RETURN_TYPES = ("INT",)
  RETURN_NAMES = ("resolution",)
  OUTPUT_NODE = True
  FUNCTION = "image_side"

  CATEGORY = "EasyUse/Image"

  def image_side(self, image, side):
    _, raw_H, raw_W, _ = image.shape

    width = raw_W
    height = raw_H
    if width is not None and height is not None:
      if side == "Longest":
         result = (width,) if width > height else (height,)
      elif side == 'Shortest':
         result = (width,) if width < height else (height,)
    else:
      result = (0,)
    return {"ui": {"text": str(result[0])}, "result": result}

# 图像尺寸（最长边）
class imageSizeByLongerSide:
  def __init__(self):
    pass

  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "image": ("IMAGE",),
      }
    }

  RETURN_TYPES = ("INT",)
  RETURN_NAMES = ("resolution",)
  OUTPUT_NODE = True
  FUNCTION = "image_longer_side"

  CATEGORY = "EasyUse/Image"

  def image_longer_side(self, image):
    _, raw_H, raw_W, _ = image.shape

    width = raw_W
    height = raw_H
    if width is not None and height is not None:
      if width > height:
         result = (width,)
      else:
         result = (height,)
    else:
      result = (0,)
    return {"ui": {"text": str(result[0])}, "result": result}

# 图像缩放
class imageScaleDown:
  crop_methods = ["disabled", "center"]

  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "images": ("IMAGE",),
        "width": (
          "INT",
          {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1},
        ),
        "height": (
          "INT",
          {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1},
        ),
        "crop": (s.crop_methods,),
      }
    }

  RETURN_TYPES = ("IMAGE",)
  CATEGORY = "EasyUse/Image"
  FUNCTION = "image_scale_down"

  def image_scale_down(self, images, width, height, crop):
    if crop == "center":
      old_width = images.shape[2]
      old_height = images.shape[1]
      old_aspect = old_width / old_height
      new_aspect = width / height
      x = 0
      y = 0
      if old_aspect > new_aspect:
        x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
      elif old_aspect < new_aspect:
        y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
      s = images[:, y: old_height - y, x: old_width - x, :]
    else:
      s = images

    results = []
    for image in s:
      img = tensor2pil(image).convert("RGB")
      img = img.resize((width, height), Image.LANCZOS)
      results.append(pil2tensor(img))

    return (torch.cat(results, dim=0),)

# 图像缩放比例
class imageScaleDownBy(imageScaleDown):
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "images": ("IMAGE",),
        "scale_by": (
          "FLOAT",
          {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01},
        ),
      }
    }

  RETURN_TYPES = ("IMAGE",)
  CATEGORY = "EasyUse/Image"
  FUNCTION = "image_scale_down_by"

  def image_scale_down_by(self, images, scale_by):
    width = images.shape[2]
    height = images.shape[1]
    new_width = int(width * scale_by)
    new_height = int(height * scale_by)
    return self.image_scale_down(images, new_width, new_height, "center")

# 图像缩放尺寸
class imageScaleDownToSize(imageScaleDownBy):
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "images": ("IMAGE",),
        "size": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
        "mode": ("BOOLEAN", {"default": True, "label_on": "max", "label_off": "min"}),
      }
    }

  RETURN_TYPES = ("IMAGE",)
  CATEGORY = "EasyUse/Image"
  FUNCTION = "image_scale_down_to_size"

  def image_scale_down_to_size(self, images, size, mode):
    width = images.shape[2]
    height = images.shape[1]

    if mode:
      scale_by = size / max(width, height)
    else:
      scale_by = size / min(width, height)

    scale_by = min(scale_by, 1.0)
    return self.image_scale_down_by(images, scale_by)


# 图像完美像素
class imagePixelPerfect:
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "image": ("IMAGE",),
        "resize_mode": (RESIZE_MODES, {"default": ResizeMode.RESIZE.value})
      }
    }

  RETURN_TYPES = ("INT",)
  RETURN_NAMES = ("resolution",)
  OUTPUT_NODE = True
  FUNCTION = "execute"

  CATEGORY = "EasyUse/Image"

  def execute(self, image, resize_mode):

    _, raw_H, raw_W, _ = image.shape

    width = raw_W
    height = raw_H

    k0 = float(height) / float(raw_H)
    k1 = float(width) / float(raw_W)

    if resize_mode == ResizeMode.OUTER_FIT.value:
      estimation = min(k0, k1) * float(min(raw_H, raw_W))
    else:
      estimation = max(k0, k1) * float(min(raw_H, raw_W))

    result = int(np.round(estimation))
    text = f"Width:{str(width)}\nHeight:{str(height)}\nPixelPerfect:{str(result)}"

    return {"ui": {"text": text}, "result": (result,)}

# 图片到遮罩
class imageToMask:
  @classmethod
  def INPUT_TYPES(s):
    return {"required": {
        "image": ("IMAGE",),
        "channel": (['red', 'green', 'blue'],),
       }
    }

  RETURN_TYPES = ("MASK",)
  FUNCTION = "convert"
  CATEGORY = "EasyUse/Image"

  def convert_to_single_channel(self, image, channel='red'):
    # Convert to RGB mode to access individual channels
    image = image.convert('RGB')

    # Extract the desired channel and convert to greyscale
    if channel == 'red':
      channel_img = image.split()[0].convert('L')
    elif channel == 'green':
      channel_img = image.split()[1].convert('L')
    elif channel == 'blue':
      channel_img = image.split()[2].convert('L')
    else:
      raise ValueError(
        "Invalid channel option. Please choose 'red', 'green', or 'blue'.")

    # Convert the greyscale channel back to RGB mode
    channel_img = Image.merge(
      'RGB', (channel_img, channel_img, channel_img))

    return channel_img

  def convert(self, image, channel='red'):
    image = self.convert_to_single_channel(tensor2pil(image), channel)
    image = pil2tensor(image)
    return (image.squeeze().mean(2),)

# 图像保存 (简易)
from nodes import PreviewImage, SaveImage
class imageSaveSimple:

  def __init__(self):
    self.output_dir = folder_paths.get_output_directory()
    self.type = "output"
    self.prefix_append = ""
    self.compress_level = 4

  @classmethod
  def INPUT_TYPES(s):
    return {"required":
              {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "only_preview": ("BOOLEAN", {"default": False}),
              },
              "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
            }

  RETURN_TYPES = ()
  FUNCTION = "save"
  OUTPUT_NODE = True
  CATEGORY = "EasyUse/Image"

  def save(self, images, filename_prefix="ComfyUI", only_preview=False, prompt=None, extra_pnginfo=None):
    if only_preview:
      PreviewImage().save_images(images, filename_prefix, prompt, extra_pnginfo)
      return ()
    else:
      return SaveImage().save_images(images, filename_prefix, prompt, extra_pnginfo)


# 图像批次合并
class JoinImageBatch:
  """Turns an image batch into one big image."""

  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "images": ("IMAGE",),
        "mode": (("horizontal", "vertical"), {"default": "horizontal"}),
      },
    }

  RETURN_TYPES = ("IMAGE",)
  RETURN_NAMES = ("image",)
  FUNCTION = "join"
  CATEGORY = "EasyUse/Image"

  def join(self, images, mode):
    n, h, w, c = images.shape
    image = None
    if mode == "vertical":
      # for vertical we can just reshape
      image = images.reshape(1, n * h, w, c)
    elif mode == "horizontal":
      # for horizontal we have to swap axes
      image = torch.transpose(torch.transpose(images, 1, 2).reshape(1, n * w, h, c), 1, 2)
    return (image,)

# 图像拆分
class imageSplitList:
  @classmethod

  def INPUT_TYPES(s):
    return {
      "required": {
        "images": ("IMAGE",),
      },
    }

  RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE",)
  RETURN_NAMES = ("images", "images", "images",)
  FUNCTION = "doit"
  CATEGORY = "EasyUse/Image"

  def doit(self, images):
    length = len(images)
    new_images = ([], [], [])
    if length % 3 == 0:
      for index, img in enumerate(images):
        if index % 3 == 0:
          new_images[0].append(img)
        elif (index+1) % 3 == 0:
          new_images[2].append(img)
        else:
          new_images[1].append(img)
    elif length % 2 == 0:
      for index, img in enumerate(images):
        if index % 2 == 0:
          new_images[0].append(img)
        else:
          new_images[1].append(img)
    return new_images

# 图片背景移除
from .briaai.rembg import BriaRMBG, preprocess_image, postprocess_image
from .libs.utils import get_local_filepath, easySave
from .config import REMBG_DIR, REMBG_MODELS
class imageRemBg:
  @classmethod
  def INPUT_TYPES(self):
    return {
      "required": {
        "images": ("IMAGE",),
        "rem_mode": (("RMBG-1.4",),),
        "image_output": (["Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
        "save_prefix": ("STRING", {"default": "ComfyUI"}),
      },
      "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
    }

  RETURN_TYPES = ("IMAGE", "MASK")
  RETURN_NAMES = ("image", "mask")
  FUNCTION = "remove"
  OUTPUT_NODE = True

  CATEGORY = "EasyUse/Image"

  def remove(self, rem_mode, images, image_output, save_prefix, prompt=None, extra_pnginfo=None):
    if rem_mode == "RMBG-1.4":
      # load model
      model_url = REMBG_MODELS[rem_mode]['model_url']
      suffix = model_url.split(".")[-1]
      model_path = get_local_filepath(model_url, REMBG_DIR, rem_mode+'.'+suffix)

      net = BriaRMBG()
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      net.load_state_dict(torch.load(model_path, map_location=device))
      net.to(device)
      net.eval()
      # prepare input
      model_input_size = [1024, 1024]
      new_images = list()
      masks = list()
      for image in images:
        orig_im = tensor2pil(image)
        w, h = orig_im.size
        image = preprocess_image(orig_im, model_input_size).to(device)
        # inference
        result = net(image)
        result_image = postprocess_image(result[0][0], (h, w))
        mask_im = Image.fromarray(result_image)
        new_im = Image.new("RGBA", mask_im.size, (0,0,0,0))
        new_im.paste(orig_im, mask=mask_im)

        new_images.append(pil2tensor(new_im))
        masks.append(pil2tensor(mask_im))

      new_images = torch.cat(new_images, dim=0)
      masks = torch.cat(masks, dim=0)


      results = easySave(new_images, save_prefix, image_output, prompt, extra_pnginfo)

      if image_output in ("Hide", "Hide/Save"):
        return {"ui": {},
                "result": (new_images, masks)}

      return {"ui": {"images": results},
              "result": (new_images, masks)}

    else:
      return (None, None)

    # 姿势编辑器
class poseEditor:
  @classmethod
  def INPUT_TYPES(self):
    temp_dir = folder_paths.get_temp_directory()

    if not os.path.isdir(temp_dir):
      os.makedirs(temp_dir)

    temp_dir = folder_paths.get_temp_directory()

    return {"required":
              {"image": (sorted(os.listdir(temp_dir)),)},
            }

  RETURN_TYPES = ("IMAGE",)
  FUNCTION = "output_pose"

  CATEGORY = "EasyUse/Image"

  def output_pose(self, image):
    image_path = os.path.join(folder_paths.get_temp_directory(), image)
    # print(f"Create: {image_path}")

    i = Image.open(image_path)
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]

    return (image,)

  @classmethod
  def IS_CHANGED(self, image):
    image_path = os.path.join(
      folder_paths.get_temp_directory(), image)
    # print(f'Change: {image_path}')

    m = hashlib.sha256()
    with open(image_path, 'rb') as f:
      m.update(f.read())
    return m.digest().hex()

NODE_CLASS_MAPPINGS = {
  "easy imageInsetCrop": imageInsetCrop,
  "easy imageSize": imageSize,
  "easy imageSizeBySide": imageSizeBySide,
  "easy imageSizeByLongerSide": imageSizeByLongerSide,
  "easy imagePixelPerfect": imagePixelPerfect,
  "easy imageScaleDown": imageScaleDown,
  "easy imageScaleDownBy": imageScaleDownBy,
  "easy imageScaleDownToSize": imageScaleDownToSize,
  "easy imageToMask": imageToMask,
  "easy imageSplitList": imageSplitList,
  "easy imageSave": imageSaveSimple,
  "easy imageRemBg": imageRemBg,
  "easy joinImageBatch": JoinImageBatch,
  "easy poseEditor": poseEditor
}

NODE_DISPLAY_NAME_MAPPINGS = {
  "easy imageInsetCrop": "ImageInsetCrop",
  "easy imageSize": "ImageSize",
  "easy imageSizeBySide": "ImageSize (Side)",
  "easy imageSizeByLongerSide": "ImageSize (LongerSide)",
  "easy imagePixelPerfect": "ImagePixelPerfect",
  "easy imageScaleDown": "Image Scale Down",
  "easy imageScaleDownBy": "Image Scale Down By",
  "easy imageScaleDownToSize": "Image Scale Down To Size",
  "easy imageToMask": "ImageToMask",
  "easy imageHSVMask": "ImageHSVMask",
  "easy imageSplitList": "imageSplitList",
  "easy imageSave": "SaveImage (Simple)",
  "easy imageRemBg": "Image Remove Bg",
  "easy joinImageBatch": "JoinImageBatch",
  "easy poseEditor": "PoseEditor"
}