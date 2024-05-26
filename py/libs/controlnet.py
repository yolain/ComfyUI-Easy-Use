import folder_paths
import comfy.controlnet
import comfy.model_management
from nodes import NODE_CLASS_MAPPINGS

class easyControlnet:
    def __init__(self):
        pass

    def apply(self, control_net_name, image, positive, negative, strength, start_percent=0, end_percent=1, control_net=None, scale_soft_weights=1, mask=None, easyCache=None, use_cache=True):
        if strength == 0:
            return (positive, negative)

        if control_net is None:
            control_net = easyCache.load_controlnet(control_net_name, scale_soft_weights, use_cache)

        if mask is not None:
            mask = mask.to(self.device)

        if mask is not None and len(mask.shape) < 3:
            mask = mask.unsqueeze(0)

        control_hint = image.movedim(-1, 1)

        is_cond = True
        if negative is None:
            p = []
            for t in positive:
                n = [t[0], t[1].copy()]
                c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent))
                if 'control' in t[1]:
                    c_net.set_previous_controlnet(t[1]['control'])
                n[1]['control'] = c_net
                n[1]['control_apply_to_uncond'] = True
                if mask is not None:
                    n[1]['mask'] = mask
                    n[1]['set_area_to_bounds'] = False
                p.append(n)
            positive = p
        else:
            cnets = {}
            out = []
            for conditioning in [positive, negative]:
                c = []
                for t in conditioning:
                    d = t[1].copy()

                    prev_cnet = d.get('control', None)
                    if prev_cnet in cnets:
                        c_net = cnets[prev_cnet]
                    else:
                        c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent))
                        c_net.set_previous_controlnet(prev_cnet)
                        cnets[prev_cnet] = c_net

                    d['control'] = c_net
                    d['control_apply_to_uncond'] = False

                    if mask is not None:
                        d['mask'] = mask
                        d['set_area_to_bounds'] = False

                    n = [t[0], d]
                    c.append(n)
                out.append(c)
            positive = out[0]
            negative = out[1]

        return (positive, negative)