import folder_paths
import comfy.controlnet
from nodes import NODE_CLASS_MAPPINGS

class easyControlnet:
    def __init__(self):
        pass

    def apply(self, control_net_name, image, positive, negative, strength, start_percent=0, end_percent=1, control_net=None, scale_soft_weights=1):
        if control_net is None:
            if scale_soft_weights < 1:
                if "ScaledSoftControlNetWeights" in NODE_CLASS_MAPPINGS:
                    soft_weight_cls = NODE_CLASS_MAPPINGS['ScaledSoftControlNetWeights']
                    (weights, timestep_keyframe) = soft_weight_cls().load_weights(scale_soft_weights, False)
                    cn_adv_cls = NODE_CLASS_MAPPINGS['ControlNetLoaderAdvanced']
                    control_net, = cn_adv_cls().load_controlnet(control_net_name, timestep_keyframe)
                else:
                    raise Exception(f"[Advanced-ControlNet Not Found] you need to install 'COMFYUI-Advanced-ControlNet'")
            else:
                controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
                control_net = comfy.controlnet.load_controlnet(controlnet_path)

        control_hint = image.movedim(-1, 1)

        if strength != 0:
            if negative is None:
                p = []
                for t in positive:
                    n = [t[0], t[1].copy()]
                    c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent))
                    if 'control' in t[1]:
                        c_net.set_previous_controlnet(t[1]['control'])
                    n[1]['control'] = c_net
                    n[1]['control_apply_to_uncond'] = True
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
                        n = [t[0], d]
                        c.append(n)
                    out.append(c)
                positive = out[0]
                negative = out[1]

        return (positive, negative)