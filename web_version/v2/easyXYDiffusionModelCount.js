// 让「XY输入: Diffusion Model」的模型数量按行折叠，与 ckpt_count / lora_count 行为一致。
import { app } from "../../scripts/app.js";

const NODE = "easy XYInputs: DiffusionModel";
const ROWS = 10;
const COLUMNS = ["model_name_", "clip_name_", "vae_name_"];
const origProps = {};

function toggle(node, widget, show) {
    if (!widget) return;
    if (!origProps[widget.name]) origProps[widget.name] = { type: widget.type, computeSize: widget.computeSize };
    widget.hidden = !show;
    widget.type = show ? origProps[widget.name].type : "easyHidden";
    widget.computeSize = show ? origProps[widget.name].computeSize : () => [0, -4];
}

function applyCount(node) {
    const count = node.widgets.find((w) => w.name === "model_count");
    const last = Number(count.value) || 0;
    for (let i = 0; i <= ROWS; i++) {
        for (const prefix of COLUMNS) {
            toggle(node, node.widgets.find((w) => w.name === prefix + i), i > 0 && i <= last);
        }
    }
    node.setSize([node.size[0], node.computeSize()[1]]);
}

app.registerExtension({
    name: "Comfy.EasyUse.XYDiffusionModelCount",
    nodeCreated(node) {
        if (node.comfyClass !== NODE) return;
        const count = node.widgets.find((w) => w.name === "model_count");
        let value = count.value;
        Object.defineProperty(count, "value", {
            get: () => value,
            set: (v) => {
                if (v === value) return;
                value = v;
                requestAnimationFrame(() => applyCount(node));
            },
        });
        applyCount(node);
    },
});
