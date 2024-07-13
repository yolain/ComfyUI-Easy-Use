import comfy.model_management
import comfy.samplers

def add_model_patch(model, sd):
    from torch.nn import Linear
    load_device = comfy.model_management.get_torch_device()
    encoder_hid_proj_weight = sd.pop("encoder_hid_proj.weight")
    encoder_hid_proj_bias = sd.pop("encoder_hid_proj.bias")
    hid_proj = Linear(encoder_hid_proj_weight.shape[1], encoder_hid_proj_weight.shape[0])
    hid_proj.weight.data = encoder_hid_proj_weight
    hid_proj.bias.data = encoder_hid_proj_bias
    hid_proj = hid_proj.to(load_device)
    model.model_options["model_patch"] = {
        "hid_proj": hid_proj
    }

def patched_set_conds(model, positive, negative=None, middle=None):
    model_attrs = dir(model)

    if "model_patch" in model.model_options:
        mp = model.model_options["model_patch"]
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
                            positive[i][1]["control"].control_model.label_emb = model.model_patcher.model.diffusion_model.label_emb if "model_patcher" in model_attrs else model.model.diffusion_model.label_emb

                if negative is not None:
                    for i in range(len(negative)):
                        negative[i][0] = hid_proj(negative[i][0])
                        if "control" in negative[i][1]:
                            if hasattr(negative[i][1]["control"], "control_model"):
                                negative[i][1]["control"].control_model.label_emb = model.model_patcher.model.diffusion_model.label_emb if "model_patcher" in model_attrs else model.model.diffusion_model.label_emb
                if middle is not None:
                    for i in range(len(middle)):
                        middle[i][0] = hid_proj(middle[i][0])
                        if "control" in middle[i][1]:
                            if hasattr(middle[i][1]["control"], "control_model"):
                                middle[i][1]["control"].control_model.label_emb = model.model_patcher.model.diffusion_model.label_emb if "model_patcher" in model_attrs else model.model.diffusion_model.label_emb

    return model, positive, negative, middle


