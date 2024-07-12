import torch.nn
import comfy.model_management
import comfy.samplers
from comfy_extras.nodes_custom_sampler import Guider_Basic, Guider_DualCFG

_globals = globals()
if "original_CFGGuider_inner_set_conds" not in _globals:
    original_CFGGuider_inner_set_conds = comfy.samplers.CFGGuider.set_conds
if "original_BasicGuider_inner_set_conds" not in _globals:
    original_BasicGuider_inner_set_conds = Guider_Basic.set_conds
if "original_DualCFGGuider_inner_set_conds" not in _globals:
    original_DualCFGGuider_inner_set_conds = Guider_DualCFG.set_conds

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

def patched_set_conds(self, positive, negative=None, middle=None):
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

                if negative is not None:
                    for i in range(len(negative)):
                        negative[i][0] = hid_proj(negative[i][0])
                        if "control" in negative[i][1]:
                            if hasattr(negative[i][1]["control"], "control_model"):
                                negative[i][1]["control"].control_model.label_emb = self.model_patcher.model.diffusion_model.label_emb
                if middle is not None:
                    for i in range(len(middle)):
                        middle[i][0] = hid_proj(middle[i][0])
                        if "control" in middle[i][1]:
                            if hasattr(middle[i][1]["control"], "control_model"):
                                middle[i][1]["control"].control_model.label_emb = self.model_patcher.model.diffusion_model.label_emb

    return self, positive, negative, middle

def patched_cfgguider_set_conds(self, positive, negative):
    self, positive, negative, _ = patched_set_conds(self, positive, negative)
    return original_CFGGuider_inner_set_conds(self, positive, negative)
def patched_basicguider_set_conds(self, positive):
    self, positive, _, _ = patched_set_conds(self, positive)
    return original_BasicGuider_inner_set_conds(self, positive)
def patched_dualcfgguider_set_conds(self, positive, midele, negative):
    self, positive, negative, middle = patched_set_conds(self, positive, midele, negative)
    return original_DualCFGGuider_inner_set_conds(self, positive, midele, negative)

comfy.samplers.CFGGuider.set_conds = patched_cfgguider_set_conds
Guider_Basic.set_conds = patched_basicguider_set_conds
Guider_DualCFG.set_conds = patched_dualcfgguider_set_conds


