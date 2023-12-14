<p align="right">
   <a href="./README.md">中文</a> | <strong>English</strong>
</p>

<div align="center">

# ComfyUI Easy Use

In order to make it easier to use the ComfyUI, I have made some optimizations and integrations to some commonly used nodes.

[//]: # ([![Bilibili Badge]&#40;https://img.shields.io/badge/使用说明视频-00A1D6?style=for-the-badge&logo=bilibili&logoColor=white&link=https://space.bilibili.com/1840885116&#41;]&#40;https://space.bilibili.com/1840885116&#41;)
</div>

## Workflow comparison

<img src="./docs/workflow_node_compare.png">

EasyUse is simplified on the basis of [tinyterraNodes](https://github.com/TinyTerra/ComfyUI_tinyterraNodes), and it is recommended to use it with the original [tinyterraNodes](https://github.com/TinyTerra/ComfyUI_tinyterraNodes) package.

### Updated

**2023-12-14**

- `easy a1111Loader` and `easy comfyLoader` added `batch_size` of required input parameters 
- Added the `easy controlnetLoaderADV` node
- `easy controlnetLoaderADV` and `easy controlnetLoader` added `control_net ` of optional input parameters
- `easy preSampling` and `easy preSamplingAdvanced` added 'image_to_latent' optional input parameters
- Added the `easy imageSizeBySide` node, which can be output as a long side or a short side

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

### Major optimizations

- The **preSampling** node has been added to separate the sampling parameter configuration from the real-time preview image at the time of sampling。
- Adjust the default generation timing of the seed, change **Control After Generate** to **Control Before Generate**.

### The relevant node package involved

Disclaimer: Opened source was not easy. I have a lot of respect for the contributions of these original authors. I just did some integration and optimization.

| Nodes Name                 | Related libraries                                                                        | Library-related node              |
|:---------------------------|:----------------------------------------------------------------------------|:----------------------------------|
| easy SetNode               | [diffus3/ComfyUI-extensions](https://github.com/diffus3/ComfyUI-extensions) | diffus3.SetNode                   |
| easy GetNode               | [diffus3/ComfyUI-extensions](https://github.com/diffus3/ComfyUI-extensions) | diffus3.GetNode                   |
| easy LLLiteLoader          | [kohya-ss/ControlNet-LLLite-ComfyUI](https://github.com/kohya-ss/ControlNet-LLLite-ComfyUI) | LLLiteLoader            |
| easy GlobalSeed            | [ltdrdata/ComfyUI-Inspire-Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack) | Global Seed (Inspire)             | 
| easy PreSamplingDynamicCFG | [sd-dynamic-thresholding](https://github.com/mcmonkeyprojects/sd-dynamic-thresholding) | DynamicThresholdingFull           | 
| DynamicThresholdingFull    | [sd-dynamic-thresholding](https://github.com/mcmonkeyprojects/sd-dynamic-thresholding) | DynamicThresholdingFull           | 
| easy ImageInsetCrop        | [rgthree/rgthree-comfy](https://github.com/rgthree/rgthree-comfy) | ImageInsetCrop                    | 


## Workflow Examples

### Text to image

<img src="./docs/text_to_image.png">

### Image to image + controlnet

<img src="./docs/image_to_image_controlnet.png">

### SDTurbo + HiresFix + SVD

<img src="./docs/sdturbo_hiresfix_svd.png">