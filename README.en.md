<p align="right">
   <a href="./README.md">中文</a> | <strong>English</strong>
</p>

<div align="center">

# ComfyUI Easy Use
</div>

**ComfyUI-Easy-Use** is a simplified node integration package, which is extended on the basis of [tinyterraNodes](https://github.com/TinyTerra/ComfyUI_tinyterraNodes), and has been integrated and optimized for many mainstream node packages to achieve the purpose of faster and more convenient use of ComfyUI. While ensuring the degree of freedom, it restores the ultimate smooth image production experience that belongs to Stable Diffusion.

## Introduce

### Random seed control before generate
<img src="https://raw.githubusercontent.com/yolain/yolain-comfyui-workflow/main/Docs/seed_generate_compare.jpg">

### Separate sampling parameters from sample preview

<img src="https://raw.githubusercontent.com/yolain/yolain-comfyui-workflow/main/Docs/workflow_node_compare.png">

### Wildcard prompt nodes are supported

Support `.yaml`, `.txt`, `.json` format wildcard files, just place them in the 'wildcards' folder of the node package, and update the file to run ComfyUI again. <br>
To use the Lora Block Weight usage, make sure that [ComfyUI-Inspire-Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack) is installed in the custom node package.

### UI interface is beautified

After installing the node package, the UI interface will be automatically switched, if you need to change other themes, please switch and refresh the page in Settings -> Color Palette.

### Stable Cascade

[WorkFlow Example](https://github.com/yolain/ComfyUI-Easy-Use?tab=readme-ov-file#StableCascade) <br><br>
Currently, txt2img and img2img are supported，Lora and Controlnet are comming soon!<br><br>

Usage：<br>
1.There is no need to load the rest of the VAE and clips when you are choose [checkpoints](https://huggingface.co/stabilityai/stable-cascade/tree/main/comfyui_checkpoints) models.<br> 
2.You need to load it extra [stage_a](https://huggingface.co/stabilityai/stable-cascade/blob/main/stage_a.safetensors)、[clip](https://huggingface.co/stabilityai/stable-cascade/resolve/main/text_encoder/model.safetensors) and [effnet_encoder](https://huggingface.co/stabilityai/stable-cascade/resolve/main/effnet_encoder.safetensors?download=true)、[previewer](https://huggingface.co/stabilityai/stable-cascade/resolve/main/previewer.safetensors) for img2img when you are choose unet models.<br>
<br>

## Changelog

**2024-02-29**

- Fixed `easy svdLoader` error when the positive or negative is empty

**2024-02-28**

- Fixed the issue that 'easy preSampling' and other similar node, latent could not be generated based on the batch index after passing in

**2024-02-26**

- `easy fullLoader` **positive**、**negative**、**latent** added to the output items
- Fixed the error of SDXLClipModel in ComfyUI revision 2016[c2cb8e88] and above (the revision number was judged to be compatible with the old revision)
- Fixed `easy detailerFix` generation error when batch size is greater than 1

**v1.0.8(2024-02-25)**

- `easy cascadeLoader` stage_c and stage_b support the checkpoint model (Download [checkpoints](https://huggingface.co/stabilityai/stable-cascade/tree/main/comfyui_checkpoints) models) 
- `easy styleSelector` The search box is modified to be case-insensitive
- Optimize the code, reduce a lot of redundant code and improve the running speed

**v1.0.7 (2024-02-19)**

- Added `easy cascadeLoader` - stable cascade Loader
- Added `easy preSamplingCascade` - stable cascade preSampling Settings
- Added `easy fullCascadeKSampler` - stable cascade stage-c ksampler full
- Added `easy cascadeKSampler` - stable cascade stage-c ksampler simple
- 
- Optimize the image to image[Example](https://github.com/yolain/ComfyUI-Easy-Use?tab=readme-ov-file#image-to-image)

**v1.0.6**

- Added `easy XYInputs: Checkpoint`
- Added `easy XYInputs: Lora`
- `easy seed` can manually switch the random seed when increasing the fixed seed value
- Fixed `easy fullLoader` and all loaders to automatically adjust the node size when switching LoRa
- Removed the original ttn image saving logic and adapted to the default image saving format extension of ComfyUI

- **v1.0.5**

- Added `easy isSDXL` 
- Added prompt word control on `easy svdLoader`, which can be used with open_clip model
- Added **populated_text** on `easy wildcards`, wildcard populated text can be output

**v1.0.4**

- `easy showAnything` added support for converting other types (e.g., tensor conditions, images, etc.)
- Added `easy showLoaderSettingsNames` can display the model and VAE name in the output loader assembly
- Added `easy promptList`
- Added `easy fooocusInpaintLoader` （only the process of SDXLModel is supported）
- Added **Logic** nodes
- Added `easy imageSave` - Image saving node with date conversion and aspect and height formatting
- Added `easy joinImageBatch`
- `easy kSamplerInpainting` Added the **patch** input value to be used with the FooocusInpaintLoader node

- Fixed xyplot error when with Pillow>9.5
- Fixed `easy wildcards` An error is reported when running with the PS extension
- Fixed `easy XYInputs: ControlNet` Error
- Fixed `easy loraStack` error when **toggle** is disabled


- Changing the first-time install node package no longer automatically replaces the theme, you need to manually adjust and refresh the page
- `easy imageSave` added **only_preivew**
- Adjust the `easy latentCompositeMaskedWithCond` node


**v1.0.3**

- Added `easy stylesSelector`
- Added **scale_soft_weights** in `easy controlnetLoader` and `easy controlnetLoaderADV` 
- Added the queue progress bar setting item, which is not enabled by default


- Fixed `easy XYInputs: Sampler/Scheduler` Error
- Fixed the right menu has a problem when clicking the button
- Fixed `easy comfyLoader` error
- Fixed xyPlot error when connecting to zero123
- Fixed the error message in the loader when the prompt word was component
- Fixed `easy getNode` and `easy setNode` the title does not change when loading
- Fixed all samplers using subdirectories to store images


- Adjust the UI theme, divided into two sets of styles: the official default background and the dark black background, which can be switched in the color palette in the settings
- Modify the styles path to be compatible with other environments


**v1.0.2**

- Added `easy XYPlotAdvanced` and some nodes about `easy XYInputs`
- Added **Alt+1-Alt+9** Shortcut keys to quickly paste node presets for Node templates (corresponding to 1~9 sequences)
- Added a `📜Groups Map(EasyUse)` to the context menu.
- An `autocomplete` folder has been added, If you have [ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts) installed, the txt files in that folder will be merged and overwritten to the autocomplete .txt file of the pyssss package at startup.


- Fixed XYPlot is not working when `a1111_prompt_style` is True
- Fixed UI loading failure in the new version of ComfyUI
- `easy XYInputs ModelMergeBlocks` Values can be imported from CSV files
- Fixed `easy pipeToBasicPipe` Bug


- Removed `easy imageRemBg`
- Remove the introductory diagram and workflow files from the package to reduce the package size
- Replaced the font file used in the generation of XY diagrams


<details>
<summary><b>v1.0.1</b></summary>

- Fixed `easy comfyLoader` error
- Fixed All nodes that contain the value of the image size
- Added `easy kSamplerInpainting`
- Added `easy pipeToBasicPipe`
- Fixed `width` and `height` can not customize in `easy svdLoader`
- Fixed all preview image path (Previously, it was not possible to preview the image on the Mac system)
- Fixed `vae_name` is not working in `easy fullLoader` and `easy a1111Loader` and `easy comfyLoader`
- Fixed `easy fullkSampler` outputs error
- Fixed `model_override` is not working in `easy fullLoader`
- Fixed `easy hiresFix` error
- Fixed `easy xyplot` font file path error
- Fixed seed that cannot be fixed when you convert `seed_num` to `easy seed` 
- Fixed `easy pipeIn` inputs bug
- `easy preDetailerFix` have added a new parameter `optional_image`
- Fixed `easy zero123Loader` and `easy svdLoader` model into cache.
- Added `easy seed`
- Fixed `image_output` default value is "Preview"
- `easy fullLoader` and `easy a1111Loader` have added a new parameter `a1111_prompt_style`,that can reproduce the same image generated from stable-diffusion-webui on comfyui, but you need to install [ComfyUI_smZNodes](https://github.com/shiimizu/ComfyUI_smZNodes) to use this feature in the current version
</details>

<details>
<summary><b>v1.0.0</b></summary>

- Added `easy positive` - simple positive prompt text
- Added `easy negative` - simple negative prompt text
- Added `easy wildcards` - support for wildcards and hint text selected by Lora
- Added `easy portraitMaster` - PortraitMaster v2.2
- Added `easy loraStack` - Lora stack
- Added `easy fullLoader` - full version of the loader
- Added `easy zero123Loader` - simple zero123 loader
- Added `easy svdLoader` - easy svd loader
- Added `easy fullkSampler` - full version of the sampler (no separation)
- Added `easy hiresFix` - support for HD repair of Pipe
- Added `easy predetailerFix` and `easy DetailerFix` - support for Pipe detail fixing
- Added `easy ultralyticsDetectorPipe` and `easy samLoaderPipe` - Detect loader (detail fixed input)
- Added `easy pipein` `easy pipeout` - Pipe input and output
- Added `easy xyPlot` - simple xyplot (more controllable parameters will be updated in the future)
- Added `easy imageRemoveBG` - image to remove background
- Added `easy imagePixelPerfect` - image pixel perfect
- Added `easy poseEditor` - Pose editor
- New UI Theme (Obsidian) - Auto-load UI by default, which can also be changed in the settings 

- Fixed `easy globalSeed` is not working
- Fixed an issue where all `seed_num` values were out of order due to [cg-use-everywhere](https://github.com/chrisgoringe/cg-use-everywhere) updating the chart in real time
- Fixed `easy imageSize`, `easy imageSizeBySide`, `easy imageSizeByLongerSide` as end nodes
- Fixed the bug that `seed_num` (random seed value) could not be read consistently in history
</details>

<details>
<summary><b>Updated at 12/14/2023</b></summary>

- `easy a1111Loader` and `easy comfyLoader` added `batch_size` of required input parameters 
- Added the `easy controlnetLoaderADV` node
- `easy controlnetLoaderADV` and `easy controlnetLoader` added `control_net ` of optional input parameters
- `easy preSampling` and `easy preSamplingAdvanced` added `image_to_latent` optional input parameters
- Added the `easy imageSizeBySide` node, which can be output as a long side or a short side
</details>

<details>
<summary><b>Updated at 12/13/2023</b></summary>

-  Added the `easy LLLiteLoader` node, if you have pre-installed the kohya-ss/ControlNet-LLLite-ComfyUI package, please move the model files in the models to `ComfyUI\models\controlnet\` (i.e. in the default controlnet path of comfy, please do not change the file name of the model, otherwise it will not be read).
-  Modify `easy controlnetLoader` to the bottom of the loader category.
-  Added size display for `easy imageSize` and `easy imageSizeByLongerSize` outputs.
</details>

<details>
<summary><b>Updated at 12/11/2023</b></summary>
-  Added the `showSpentTime` node to display the time spent on image diffusion and the time spent on VAE decoding images
</details>

## The relevant node package involved

Disclaimer: Opened source was not easy. I have a lot of respect for the contributions of these original authors. I just did some integration and optimization.

| Nodes Name(Search Name)    | Related libraries                                                                        | Library-related node              |
|:---------------------------|:----------------------------------------------------------------------------|:------------------------|
| easy setNode               | [ComfyUI-extensions](https://github.com/diffus3/ComfyUI-extensions) | diffus3.SetNode         |
| easy getNode               | [ComfyUI-extensions](https://github.com/diffus3/ComfyUI-extensions) | diffus3.GetNode         |
| easy bookmark              | [rgthree-comfy](https://github.com/rgthree/rgthree-comfy) | Bookmark 🔖             |
| easy portraitMarker        | [comfyui-portrait-master](https://github.com/florestefano1975/comfyui-portrait-master) | Portrait Master         |
| easy LLLiteLoader          | [ControlNet-LLLite-ComfyUI](https://github.com/kohya-ss/ControlNet-LLLite-ComfyUI) | LLLiteLoader            |
| easy globalSeed            | [ComfyUI-Inspire-Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack) | Global Seed (Inspire)   | 
| easy preSamplingDynamicCFG | [sd-dynamic-thresholding](https://github.com/mcmonkeyprojects/sd-dynamic-thresholding) | DynamicThresholdingFull | 
| dynamicThresholdingFull    | [sd-dynamic-thresholding](https://github.com/mcmonkeyprojects/sd-dynamic-thresholding) | DynamicThresholdingFull | 
| easy imageInsetCrop        | [rgthree-comfy](https://github.com/rgthree/rgthree-comfy) | ImageInsetCrop          | 
| easy poseEditor            | [ComfyUI_Custom_Nodes_AlekPet](https://github.com/AlekPet/ComfyUI_Custom_Nodes_AlekPet) | poseNode                | 

## Workflow Examples

### Text to image

<img src="https://raw.githubusercontent.com/yolain/yolain-comfyui-workflow/main/Workflows/Simple/text_to_image.png">

### Image to image + controlnet

<img src="https://raw.githubusercontent.com/yolain/yolain-comfyui-workflow/main/Workflows/Simple/image_to_image_controlnet.png">

### SDTurbo + HiresFix + SVD

<img src="https://raw.githubusercontent.com/yolain/yolain-comfyui-workflow/main/Workflows/Simple/sdturbo_hiresfix_svd.png">


### StableCascade
#### Text to image
<img src="https://raw.githubusercontent.com/yolain/yolain-comfyui-workflow/main/Workflows/StableCascade/text_to_image.png">

#### Image to image
<img src="https://raw.githubusercontent.com/yolain/yolain-comfyui-workflow/main/Workflows/StableCascade/image_to_image.png">

## Credits

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) - Powerful and modular Stable Diffusion GUI

[ComfyUI-ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) - ComfyUI Manager

[tinyterraNodes](https://github.com/TinyTerra/ComfyUI_tinyterraNodes) - Pipe nodes (node bundles) allow users to reduce unnecessary connections

[ComfyUI-extensions](https://github.com/diffus3/ComfyUI-extensions) - Diffus3 gets and sets points that allow the user to detach the composition of the workflow 

[ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) - General modpack 1

[ComfyUI-Inspire-Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack) - General Modpack 2