#credit to ExponentialML for this module
#from https://github.com/ExponentialML/ComfyUI_Native_DynamiCrafter
import os
import torch
import comfy

from einops import rearrange
from comfy import model_base, model_management
from .lvdm.modules.networks.openaimodel3d import UNetModel as DynamiCrafterUNetModel

from .utils.model_utils import DynamiCrafterBase, DYNAMICRAFTER_CONFIG, load_image_proj_dict, load_dynamicrafter_dict, get_image_proj_model

class DynamiCrafter:

    def __init__(self):
        self.model_patcher = None

    # There is probably a better way to do this, but with the apply_model callback, this seems necessary.
    # The model gets wrapped around a CFG Denoiser class, and handles the conditioning parts there.
    # We cannot access it, so we must find the conditioning according to how ComfyUI handles it.
    def get_conditioning_pair(self, c_crossattn, use_cfg: bool):
        if not use_cfg:
            return c_crossattn

        conditioning_group = []

        for i in range(c_crossattn.shape[0]):
            # Get the positive and negative conditioning.
            positive_idx = i + 1
            negative_idx = i

            if positive_idx >= c_crossattn.shape[0]:
                break

            if not torch.equal(c_crossattn[[positive_idx]], c_crossattn[[negative_idx]]):
                conditioning_group = [
                    c_crossattn[[positive_idx]],
                    c_crossattn[[negative_idx]]
                ]
                break

        if len(conditioning_group) == 0:
            raise ValueError("Could not get the appropriate conditioning group.")

        return torch.cat(conditioning_group)

    # apply_model, {"input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": cond_or_uncond}
    def _forward(self, *args):
        transformer_options = self.model_patcher.model_options['transformer_options']
        conditioning = transformer_options['conditioning']

        apply_model = args[0]

        # forward_dict
        fd = args[1]

        x, t, model_in_kwargs, _ = fd['input'], fd['timestep'], fd['c'], fd['cond_or_uncond']

        c_crossattn = model_in_kwargs.pop("c_crossattn")
        c_concat = conditioning['c_concat']
        num_video_frames = conditioning['num_video_frames']
        fs = conditioning['fs']

        original_num_frames = num_video_frames

        # Better way to determine if we're using CFG
        # The cond batch will always be num_frames >= 2 since we're doing video,
        # so we need get this condition differently here.
        if x.shape[0] > num_video_frames:
            num_video_frames *= 2
            batch_size = 2
            use_cfg = True
        else:
            use_cfg = False
            batch_size = 1

        if use_cfg:
            c_concat = torch.cat([c_concat] * 2)

        self.validate_forwardable_latent(x, c_concat, num_video_frames, use_cfg)

        x_in, c_concat = map(lambda xc: rearrange(xc, '(b t) c h w -> b c t h w', b=batch_size), (x, c_concat))

        # We always assume video, so there will always be batched conditionings.
        c_crossattn = self.get_conditioning_pair(c_crossattn, use_cfg)
        c_crossattn = c_crossattn[:2] if use_cfg else c_crossattn[:1]
        context_in = c_crossattn

        img_embs = conditioning['image_emb']

        if use_cfg:
            img_emb_uncond = conditioning['image_emb_uncond']
            img_embs = torch.cat([img_embs, img_emb_uncond])

        fs = torch.cat([fs] * x_in.shape[0])

        outs = []
        for i in range(batch_size):
            model_in_kwargs['transformer_options']['cond_idx'] = i
            x_out = apply_model(
                x_in[[i]],
                t=torch.cat([t[:1]]),
                context_in=context_in[[i]],
                c_crossattn=c_crossattn,
                cc_concat=c_concat[[i]],  # "cc" is to handle naming conflict with apply_model wrapper.
                # We want to handle this in the UNet forward.
                num_video_frames=num_video_frames // 2 if batch_size > 1 else num_video_frames,
                img_emb=img_embs[[i]],
                fs=fs[[i]],
                **model_in_kwargs
            )
            outs.append(x_out)

        x_out = torch.cat(list(reversed(outs)))
        x_out = rearrange(x_out, 'b c t h w -> (b t) c h w')

        return x_out

    def assign_forward_args(
            self,
            model,
            c_concat,
            image_emb,
            image_emb_uncond,
            fs,
            frames,
    ):
        model.model_options['transformer_options']['conditioning'] = {
            "c_concat": c_concat,
            "image_emb": image_emb,
            'image_emb_uncond': image_emb_uncond,
            "fs": fs,
            "num_video_frames": frames,
        }

    def validate_forwardable_latent(self, latent, c_concat, num_video_frames, use_cfg):
        check_no_cfg = latent.shape[0] != num_video_frames
        check_with_cfg = latent.shape[0] != (num_video_frames * 2)

        latent_batch_size = latent.shape[0] if not use_cfg else latent.shape[0] // 2
        num_frames = num_video_frames if not use_cfg else num_video_frames // 2

        if all([check_no_cfg, check_with_cfg]):
            raise ValueError(
                "Please make sure your latent inputs match the number of frames in the DynamiCrafter Processor."
                f"Got a latent batch size of ({latent_batch_size}) with number of frames being ({num_frames})."
            )

        latent_h, latent_w = latent.shape[-2:]
        c_concat_h, c_concat_w = c_concat.shape[-2:]

        if not all([latent_h == c_concat_h, latent_w == c_concat_w]):
            raise ValueError(
                "Please make sure that your input latent and image frames are the same height and width.",
                f"Image Size: {c_concat_w * 8}, {c_concat_h * 8}, Latent Size: {latent_h * 8}, {latent_w * 8}"
            )

    def process_image_conditioning(
            self,
            model,
            clip_vision,
            vae,
            image_proj_model,
            images,
            use_interpolate,
            fps: int,
            frames: int,
            scale_latents: bool
    ):
        self.model_patcher = model
        encoded_latent = vae.encode(images[:, :, :, :3])

        encoded_image = clip_vision.encode_image(images[:1])['last_hidden_state']
        image_emb = image_proj_model(encoded_image)

        encoded_image_uncond = clip_vision.encode_image(torch.zeros_like(images)[:1])['last_hidden_state']
        image_emb_uncond = image_proj_model(encoded_image_uncond)

        c_concat = encoded_latent

        if scale_latents:
            vae_process_input = vae.process_input
            vae.process_input = lambda image: (image - .5) * 2
            c_concat = vae.encode(images[:, :, :, :3])
            vae.process_input = vae_process_input
            c_concat = model.model.process_latent_in(c_concat) * 1.3
        else:
            c_concat = model.model.process_latent_in(c_concat)

        fs = torch.tensor([fps], dtype=torch.long, device=model_management.intermediate_device())

        model.set_model_unet_function_wrapper(self._forward)

        used_interpolate_processing = False

        if use_interpolate and frames > 16:
            raise ValueError(
                "When using interpolation mode, the maximum amount of frames are 16."
                "If you're doing long video generation, consider using the last frame\
                     from the first generation for the next one (autoregressive)."
            )
        if encoded_latent.shape[0] == 1:
            c_concat = torch.cat([c_concat] * frames, dim=0)[:frames]

            if use_interpolate:
                mask = torch.zeros_like(c_concat)
                mask[:1] = c_concat[:1]
                c_concat = mask

                used_interpolate_processing = True
        else:
            if use_interpolate and c_concat.shape[0] in [2, 3]:
                input_frame_count = c_concat.shape[0]

                # We're just padding to the same type an size of the concat
                masked_frames = torch.zeros_like(torch.cat([c_concat[:1]] * frames))[:frames]

                # Start frame
                masked_frames[:1] = c_concat[:1]

                end_frame_idx = -1

                # TODO
                speed = 1.0
                if speed < 1.0:
                    possible_speeds = list(torch.linspace(0, 1.0, c_concat.shape[0]))
                    speed_from_frames = enumerate(possible_speeds)
                    speed_idx = min(speed_from_frames, key=lambda n: n[1] - speed)[0]
                    end_frame_idx = speed_idx

                # End frame
                masked_frames[-1:] = c_concat[[end_frame_idx]]

                # Possible middle frame, but not working at the moment.
                if input_frame_count == 3:
                    middle_idx = masked_frames.shape[0] // 2
                    middle_idx_frame = c_concat.shape[0] // 2
                    masked_frames[[middle_idx]] = c_concat[[middle_idx_frame]]

                c_concat = masked_frames
                used_interpolate_processing = True

                print(f"Using interpolation mode with {input_frame_count} frames.")

            if c_concat.shape[0] < frames and not used_interpolate_processing:
                print(
                    "Multiple images found, but interpolation mode is unset. Using the first frame as condition.",
                )
                c_concat = torch.cat([c_concat[:1]] * frames)

        c_concat = c_concat[:frames]

        if encoded_latent.shape[0] == 1:
            encoded_latent = torch.cat([encoded_latent] * frames)[:frames]

        if encoded_latent.shape[0] < frames and encoded_latent.shape[0] != 1:
            encoded_latent = torch.cat(
                [encoded_latent] + [encoded_latent[-1:]] * abs(encoded_latent.shape[0] - frames)
            )[:frames]

        # We could store this as a state in this Node Class Instance, but to prevent any weird edge cases,
        # this should always be passed through the 'stateless' way, and let ComfyUI handle the transformer_options state.
        self.assign_forward_args(model, c_concat, image_emb, image_emb_uncond, fs, frames)

        return (model, {"samples": torch.zeros_like(c_concat)}, {"samples": encoded_latent},)


    # Loader for the DynamiCrafter model.
    def load_model_sicts(self, model_path: str):
        model_state_dict = comfy.utils.load_torch_file(model_path)
        dynamicrafter_dict = load_dynamicrafter_dict(model_state_dict)
        image_proj_dict = load_image_proj_dict(model_state_dict)

        return dynamicrafter_dict, image_proj_dict

    def get_prediction_type(self, is_eps: bool, model_config):
        if not is_eps and "image_cross_attention_scale_learnable" in model_config.unet_config.keys():
            model_config.unet_config["image_cross_attention_scale_learnable"] = False

        return model_base.ModelType.EPS if is_eps else model_base.ModelType.V_PREDICTION

    def handle_model_management(self, dynamicrafter_dict: dict, model_config):
        parameters = comfy.utils.calculate_parameters(dynamicrafter_dict, "model.diffusion_model.")
        load_device = model_management.get_torch_device()
        unet_dtype = model_management.unet_dtype(
            model_params=parameters,
            supported_dtypes=model_config.supported_inference_dtypes
        )
        manual_cast_dtype = model_management.unet_manual_cast(
            unet_dtype,
            load_device,
            model_config.supported_inference_dtypes
        )
        model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
        inital_load_device = model_management.unet_inital_load_device(parameters, unet_dtype)
        offload_device = model_management.unet_offload_device()

        return load_device, inital_load_device

    def check_leftover_keys(self, state_dict: dict):
        left_over = state_dict.keys()
        if len(left_over) > 0:
            print("left over keys:", left_over)

    def load_dynamicrafter(self, model_path):

        if os.path.exists(model_path):
            dynamicrafter_dict, image_proj_dict = self.load_model_sicts(model_path)
            model_config = DynamiCrafterBase(DYNAMICRAFTER_CONFIG)

            dynamicrafter_dict, is_eps = model_config.process_dict_version(state_dict=dynamicrafter_dict)

            MODEL_TYPE = self.get_prediction_type(is_eps, model_config)
            load_device, inital_load_device = self.handle_model_management(dynamicrafter_dict, model_config)

            model = model_base.BaseModel(
                model_config,
                model_type=MODEL_TYPE,
                device=inital_load_device,
                unet_model=DynamiCrafterUNetModel
            )

            image_proj_model = get_image_proj_model(image_proj_dict)
            model.load_model_weights(dynamicrafter_dict, "model.diffusion_model.")
            self.check_leftover_keys(dynamicrafter_dict)

            model_patcher = comfy.model_patcher.ModelPatcher(
                model,
                load_device=load_device,
                offload_device=model_management.unet_offload_device(),
                current_device=inital_load_device
            )

        return (model_patcher, image_proj_model,)