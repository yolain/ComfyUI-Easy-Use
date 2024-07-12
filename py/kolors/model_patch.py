import torch.nn
import comfy.model_management
import comfy.samplers

if "original_CFGGuider_inner_set_conds" not in globals():
    original_CFGGuider_inner_set_conds = comfy.samplers.CFGGuider.set_conds

def add_model_patch(model, sd):
    load_device = comfy.model_management.get_torch_device()
    encoder_hid_proj_weight = sd.pop("encoder_hid_proj.weight")
    encoder_hid_proj_bias = sd.pop("encoder_hid_proj.bias")
    hid_proj = torch.nn.Linear(encoder_hid_proj_weight.shape[1], encoder_hid_proj_weight.shape[0])
    hid_proj.weight.data = encoder_hid_proj_weight
    hid_proj.bias.data = encoder_hid_proj_bias
    hid_proj = hid_proj.to(load_device)
    model.model_options["model_patch"] = {
        "hid_proj": hid_proj
    }

def patched_kolors_conds(self, positive, negative):
    if "model_patch" in self.model_options:
        mp = self.model_options["model_patch"]
        if "hid_proj" in mp:
            import copy
            hid_proj = mp["hid_proj"]
            positive = copy.deepcopy(positive)
            negative = copy.deepcopy(negative)

            if hid_proj is not None:
                for i in range(len(positive)):
                    positive[i][0] = hid_proj(positive[i][0])
                    if "control" in positive[i][1]:
                        if hasattr(positive[i][1]["control"], "control_model"):
                            positive[i][1]["control"].control_model.label_emb = self.model_patcher.model.diffusion_model.label_emb

                for i in range(len(negative)):
                    negative[i][0] = hid_proj(negative[i][0])
                    if "control" in negative[i][1]:
                        if hasattr(negative[i][1]["control"], "control_model"):
                            negative[i][1]["control"].control_model.label_emb = self.model_patcher.model.diffusion_model.label_emb

    return original_CFGGuider_inner_set_conds(self, positive, negative)

comfy.samplers.CFGGuider.set_conds = patched_kolors_conds


