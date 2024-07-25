import comfy.model_management
import comfy.samplers
import torch
from torch.nn import Linear
from types import MethodType

# def add_model_patch(model, sd):
#     load_device = comfy.model_management.get_torch_device()
#     encoder_hid_proj_weight = sd.pop("encoder_hid_proj.weight")
#     encoder_hid_proj_bias = sd.pop("encoder_hid_proj.bias")
#     hid_proj = Linear(encoder_hid_proj_weight.shape[1], encoder_hid_proj_weight.shape[0])
#     hid_proj.weight.data = encoder_hid_proj_weight
#     hid_proj.bias.data = encoder_hid_proj_bias
#     hid_proj = hid_proj.to(load_device)
#     model.model_options["model_patch"] = {
#         "hid_proj": hid_proj
#     }
#
# def patched_set_conds(model, positive, negative=None, middle=None):
#     model_attrs = dir(model)
#
#     if "model_patch" in model.model_options:
#         mp = model.model_options["model_patch"]
#         if "hid_proj" in mp:
#             import copy
#             hid_proj = mp["hid_proj"]
#             positive = copy.deepcopy(positive)
#             negative = copy.deepcopy(negative)
#
#             if hid_proj is not None:
#                 for i in range(len(positive)):
#                     positive[i][0] = hid_proj(positive[i][0])
#                     if "control" in positive[i][1]:
#                         if hasattr(positive[i][1]["control"], "control_model"):
#                             positive[i][1]["control"].control_model.label_emb = model.model_patcher.model.diffusion_model.label_emb if "model_patcher" in model_attrs else model.model.diffusion_model.label_emb
#
#                 if negative is not None:
#                     for i in range(len(negative)):
#                         negative[i][0] = hid_proj(negative[i][0])
#                         if "control" in negative[i][1]:
#                             if hasattr(negative[i][1]["control"], "control_model"):
#                                 negative[i][1]["control"].control_model.label_emb = model.model_patcher.model.diffusion_model.label_emb if "model_patcher" in model_attrs else model.model.diffusion_model.label_emb
#                 if middle is not None:
#                     for i in range(len(middle)):
#                         middle[i][0] = hid_proj(middle[i][0])
#                         if "control" in middle[i][1]:
#                             if hasattr(middle[i][1]["control"], "control_model"):
#                                 middle[i][1]["control"].control_model.label_emb = model.model_patcher.model.diffusion_model.label_emb if "model_patcher" in model_attrs else model.model.diffusion_model.label_emb
#
#     return model, positive, negative, middle

from comfy.cldm.cldm import ControlNet
from comfy.controlnet import ControlLora
def patch_controlnet(model, control_net):
    import comfy.controlnet
    if isinstance(control_net, ControlLora):
        del_keys = []
        for k in control_net.control_weights:
            if k.startswith("label_emb.0.0."):
                del_keys.append(k)

        for k in del_keys:
            control_net.control_weights.pop(k)

        super_pre_run = ControlLora.pre_run
        super_copy = ControlLora.copy

        super_forward = ControlNet.forward

        def KolorsControlNet_forward(self, x, hint, timesteps, context, **kwargs):
            with torch.cuda.amp.autocast(enabled=True):
                context = model.model.diffusion_model.encoder_hid_proj(context)
                return super_forward(self, x, hint, timesteps, context, **kwargs)

        def KolorsControlLora_pre_run(self, *args, **kwargs):
            result = super_pre_run(self, *args, **kwargs)

            if hasattr(self, "control_model"):
                self.control_model.forward = MethodType(
                    KolorsControlNet_forward, self.control_model)
            return result

        control_net.pre_run = MethodType(
            KolorsControlLora_pre_run, control_net)

        def KolorsControlLora_copy(self, *args, **kwargs):
            c = super_copy(self, *args, **kwargs)
            c.pre_run = MethodType(
                KolorsControlLora_pre_run, c)
            return c

        control_net.copy = MethodType(KolorsControlLora_copy, control_net)

    elif isinstance(control_net, comfy.controlnet.ControlNet):
        model_label_emb = model.model.diffusion_model.label_emb
        control_net.control_model.label_emb = model_label_emb
        control_net.control_model_wrapped.model.label_emb = model_label_emb
        super_forward = ControlNet.forward

        def KolorsControlNet_forward(self, x, hint, timesteps, context, **kwargs):
            with torch.cuda.amp.autocast(enabled=True):
                context = model.model.diffusion_model.encoder_hid_proj(context)
                return super_forward(self, x, hint, timesteps, context, **kwargs)

        control_net.control_model.forward = MethodType(
            KolorsControlNet_forward, control_net.control_model)

    else:
        raise NotImplementedError(f"Type {control_net} not supported for KolorsControlNetPatch")

    return control_net
