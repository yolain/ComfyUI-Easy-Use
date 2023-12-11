import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

let origProps = {};

const findWidgetByName = (node, name) => node.widgets.find((w) => w.name === name);

const doesInputWithNameExist = (node, name) => node.inputs ? node.inputs.some((input) => input.name === name) : false;

function updateNodeHeight(node) {
	node.setSize([node.size[0], node.computeSize()[1]]);
}

function toggleWidget(node, widget, show = false, suffix = "") {
	if (!widget || doesInputWithNameExist(node, widget.name)) return;
	if (!origProps[widget.name]) {
		origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize };	
	}
	const origSize = node.size;

	widget.type = show ? origProps[widget.name].origType : "esayHidden" + suffix;
	widget.computeSize = show ? origProps[widget.name].origComputeSize : () => [0, -4];

	widget.linkedWidgets?.forEach(w => toggleWidget(node, w, ":" + widget.name, show));	

	const height = show ? Math.max(node.computeSize()[1], origSize[1]) : node.size[1];
	node.setSize([node.size[0], height]);
	
}

function widgetLogic(node, widget) {
	if (widget.name === 'lora_name') {
		if (widget.value === "None") {
			toggleWidget(node, findWidgetByName(node, 'lora_model_strength'))
			toggleWidget(node, findWidgetByName(node, 'lora_clip_strength'))
		} else {
			toggleWidget(node, findWidgetByName(node, 'lora_model_strength'), true)
			toggleWidget(node, findWidgetByName(node, 'lora_clip_strength'), true)
		}
	}
	if (widget.name === 'lora1_name') {
		if (widget.value === "None") {
			toggleWidget(node, findWidgetByName(node, 'lora1_model_strength'))
			toggleWidget(node, findWidgetByName(node, 'lora1_clip_strength'))
		} else {
			toggleWidget(node, findWidgetByName(node, 'lora1_model_strength'), true)
			toggleWidget(node, findWidgetByName(node, 'lora1_clip_strength'), true)
		}
	}
	if (widget.name === 'lora2_name') {
		if (widget.value === "None") {
			toggleWidget(node, findWidgetByName(node, 'lora2_model_strength'))
			toggleWidget(node, findWidgetByName(node, 'lora2_clip_strength'))
		} else {
			toggleWidget(node, findWidgetByName(node, 'lora2_model_strength'), true)
			toggleWidget(node, findWidgetByName(node, 'lora2_clip_strength'), true)
		}
	}
	if (widget.name === 'lora3_name') {
		if (widget.value === "None") {
			toggleWidget(node, findWidgetByName(node, 'lora3_model_strength'))
			toggleWidget(node, findWidgetByName(node, 'lora3_clip_strength'))
		} else {
			toggleWidget(node, findWidgetByName(node, 'lora3_model_strength'), true)
			toggleWidget(node, findWidgetByName(node, 'lora3_clip_strength'), true)
		}
	}
	if (widget.name === 'refiner_ckpt_name') {
		let refiner_lora1 = findWidgetByName(node, 'refiner_lora1_name').value
		let refiner_lora2 = findWidgetByName(node, 'refiner_lora2_name').value
		if (widget.value === "None") {
			toggleWidget(node, findWidgetByName(node, 'refiner_vae_name'))
			toggleWidget(node, findWidgetByName(node, 'refiner_lora1_name'))
			toggleWidget(node, findWidgetByName(node, 'refiner_lora1_model_strength'))
			toggleWidget(node, findWidgetByName(node, 'refiner_lora1_clip_strength'))
			toggleWidget(node, findWidgetByName(node, 'refiner_lora2_name'))
			toggleWidget(node, findWidgetByName(node, 'refiner_lora2_model_strength'))
			toggleWidget(node, findWidgetByName(node, 'refiner_lora2_clip_strength'))
		} else {
			toggleWidget(node, findWidgetByName(node, 'refiner_vae_name'), true)
			toggleWidget(node, findWidgetByName(node, 'refiner_lora1_name'), true)
			if (refiner_lora1 !== "None") {
				toggleWidget(node, findWidgetByName(node, 'refiner_lora1_model_strength'), true)
				toggleWidget(node, findWidgetByName(node, 'refiner_lora1_clip_strength'), true)
			}
			toggleWidget(node, findWidgetByName(node, 'refiner_lora2_name'), true)
			if (refiner_lora2 !== "None") {
				toggleWidget(node, findWidgetByName(node, 'refiner_lora2_model_strength'), true)
				toggleWidget(node, findWidgetByName(node, 'refiner_lora2_clip_strength'), true)
			}
		}
	}
	if (widget.name === 'refiner_lora1_name') {
		let refiner_ckpt = findWidgetByName(node, 'refiner_ckpt_name').value

		if (widget.value === "None" || refiner_ckpt === "None") {
			toggleWidget(node, findWidgetByName(node, 'refiner_lora1_model_strength'))
			toggleWidget(node, findWidgetByName(node, 'refiner_lora1_clip_strength'))
		} else {
			toggleWidget(node, findWidgetByName(node, 'refiner_lora1_model_strength'), true)
			toggleWidget(node, findWidgetByName(node, 'refiner_lora1_clip_strength'), true)
		}
	}
	if (widget.name === 'refiner_lora2_name') {
		let refiner_ckpt = findWidgetByName(node, 'refiner_ckpt_name').value

		if (widget.value === "None" || refiner_ckpt === "None") {
			toggleWidget(node, findWidgetByName(node, 'refiner_lora2_model_strength'))
			toggleWidget(node, findWidgetByName(node, 'refiner_lora2_clip_strength'))
		} else {
			toggleWidget(node, findWidgetByName(node, 'refiner_lora2_model_strength'), true)
			toggleWidget(node, findWidgetByName(node, 'refiner_lora2_clip_strength'), true)
		}
	}
	if (widget.name === 'rescale_after_model') {
		if (widget.value === false) {
			toggleWidget(node, findWidgetByName(node, 'rescale_method'))
			toggleWidget(node, findWidgetByName(node, 'rescale'))
			toggleWidget(node, findWidgetByName(node, 'percent'))
			toggleWidget(node, findWidgetByName(node, 'width'))
			toggleWidget(node, findWidgetByName(node, 'height'))
			toggleWidget(node, findWidgetByName(node, 'longer_side'))
			toggleWidget(node, findWidgetByName(node, 'crop'))
		} else {
			toggleWidget(node, findWidgetByName(node, 'rescale_method'), true)
			toggleWidget(node, findWidgetByName(node, 'rescale'), true)
			
			let rescale_value = findWidgetByName(node, 'rescale').value

			if (rescale_value === 'by percentage') {
				toggleWidget(node, findWidgetByName(node, 'percent'), true)
			} else if (rescale_value === 'to Width/Height') {
				toggleWidget(node, findWidgetByName(node, 'width'), true)
				toggleWidget(node, findWidgetByName(node, 'height'), true)
			} else {
				toggleWidget(node, findWidgetByName(node, 'longer_side'), true)
			}
			toggleWidget(node, findWidgetByName(node, 'crop'), true)
		}
	}
	if (widget.name === 'rescale') {
		let rescale_after_model = findWidgetByName(node, 'rescale_after_model').value
		if (widget.value === 'by percentage' && rescale_after_model) {
			toggleWidget(node, findWidgetByName(node, 'width'))
			toggleWidget(node, findWidgetByName(node, 'height'))
			toggleWidget(node, findWidgetByName(node, 'longer_side'))
			toggleWidget(node, findWidgetByName(node, 'percent'), true)
		} else if (widget.value === 'to Width/Height' && rescale_after_model) {
			toggleWidget(node, findWidgetByName(node, 'width'), true)
			toggleWidget(node, findWidgetByName(node, 'height'), true)
			toggleWidget(node, findWidgetByName(node, 'percent'))
			toggleWidget(node, findWidgetByName(node, 'longer_side'))
		} else if (rescale_after_model) {
			toggleWidget(node, findWidgetByName(node, 'longer_side'), true)
			toggleWidget(node, findWidgetByName(node, 'width'))
			toggleWidget(node, findWidgetByName(node, 'height'))
			toggleWidget(node, findWidgetByName(node, 'percent'))
		}
	}
	if (widget.name === 'upscale_method') {
		if (widget.value === "None") {
			toggleWidget(node, findWidgetByName(node, 'factor'))
			toggleWidget(node, findWidgetByName(node, 'crop'))
		} else {
			toggleWidget(node, findWidgetByName(node, 'factor'), true)
			toggleWidget(node, findWidgetByName(node, 'crop'), true)
		}
	}
	if (widget.name === 'image_output') {
	    if (widget.value === 'Sender' || widget.value === 'Sender/Save'){
	        toggleWidget(node, findWidgetByName(node, 'link_id'), true)
	    }else {
	        toggleWidget(node, findWidgetByName(node, 'link_id'))
	    }
		if (widget.value === 'Hide' || widget.value === 'Preview' || widget.value === 'Sender') {
			toggleWidget(node, findWidgetByName(node, 'save_prefix'))
			toggleWidget(node, findWidgetByName(node, 'output_path'))
			toggleWidget(node, findWidgetByName(node, 'embed_workflow'))
			toggleWidget(node, findWidgetByName(node, 'number_padding'))
			toggleWidget(node, findWidgetByName(node, 'overwrite_existing'))
		} else if (widget.value === 'Save' || widget.value === 'Hide/Save' || widget.value === 'Sender/Save') {
			toggleWidget(node, findWidgetByName(node, 'save_prefix'), true)
			toggleWidget(node, findWidgetByName(node, 'output_path'), true)
			toggleWidget(node, findWidgetByName(node, 'embed_workflow'), true)
			toggleWidget(node, findWidgetByName(node, 'number_padding'), true)
			toggleWidget(node, findWidgetByName(node, 'overwrite_existing'), true)
		}
	}
	if (widget.name === 'add_noise') {
		if (widget.value === "disable") {
			toggleWidget(node, findWidgetByName(node, 'seed_num'))
			toggleWidget(node, findWidgetByName(node, 'control_before_generate'))
		} else {
			toggleWidget(node, findWidgetByName(node, 'seed_num'), true)
			toggleWidget(node, findWidgetByName(node, 'control_before_generate'), true)
		}
	}
	if (widget.name === 'ckpt_B_name') {
		if (widget.value === "None") {
			toggleWidget(node, findWidgetByName(node, 'config_B_name'))
		} else {
			toggleWidget(node, findWidgetByName(node, 'config_B_name'), true)
		}
	}
	if (widget.name === 'ckpt_C_name') {
		if (widget.value === "None") {
			toggleWidget(node, findWidgetByName(node, 'config_C_name'))
		} else {
			toggleWidget(node, findWidgetByName(node, 'config_C_name'), true)
		}
	}
	if (widget.name === 'save_model') {
		if (widget.value === "True") {
			toggleWidget(node, findWidgetByName(node, 'save_prefix'), true)

		} else {
			toggleWidget(node, findWidgetByName(node, 'save_prefix'))
		}
	}
	if (widget.name === 'num_loras') {
		let number_to_show = widget.value + 1
		for (let i = 0; i < number_to_show; i++) {
			toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_name'), true)
			if (findWidgetByName(node, 'mode').value === "simple") {
				toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_strength'), true)
				toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_model_strength'))
				toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_clip_strength'))
			} else {
				toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_strength'))
				toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_model_strength'), true)
				toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_clip_strength'), true)
			}
		}
		for (let i = number_to_show; i < 21; i++) {
			toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_name'))
			toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_strength'))
			toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_model_strength'))
			toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_clip_strength'))
		}
		updateNodeHeight(node)
	}
	if (widget.name === 'mode') {
		let number_to_show = findWidgetByName(node, 'num_loras').value + 1
		for (let i = 0; i < number_to_show; i++) {
			if (widget.value === "simple") {
				toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_strength'), true)
				toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_model_strength'))
				toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_clip_strength'))
			} else {
				toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_strength'))
				toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_model_strength'), true)
				toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_clip_strength'), true)}
		}
		updateNodeHeight(node)
	}
	if (widget.name === 'resolution') {
		if (widget.value === "自定义 x 自定义") {
			toggleWidget(node, findWidgetByName(node, 'empty_latent_width'), true)
			toggleWidget(node, findWidgetByName(node, 'empty_latent_height'), true)
		} else {
			toggleWidget(node, findWidgetByName(node, 'empty_latent_width'), false)
			toggleWidget(node, findWidgetByName(node, 'empty_latent_height'), false)
		}
	}

	if (widget.name === 'toggle') {
		widget.type = 'toggle'
		widget.options = {on: 'Enabled', off: 'Disabled'}
	}

}

function widgetLogic2(node, widget) {
	if (widget.name === 'sampler_name') {
		if (["euler_ancestral", "dpmpp_2s_ancestral", "dpmpp_2m_sde", "lcm"].includes(widget.value)) {
			toggleWidget(node, findWidgetByName(node, 'eta'), true)
			toggleWidget(node, findWidgetByName(node, 's_noise'), true)
			toggleWidget(node, findWidgetByName(node, 'upscale_ratio'), true)
			toggleWidget(node, findWidgetByName(node, 'start_step'), true)
			toggleWidget(node, findWidgetByName(node, 'end_step'), true)
			toggleWidget(node, findWidgetByName(node, 'upscale_n_step'), true)
			toggleWidget(node, findWidgetByName(node, 'unsharp_kernel_size'), true)
			toggleWidget(node, findWidgetByName(node, 'unsharp_sigma'), true)
			toggleWidget(node, findWidgetByName(node, 'unsharp_strength'), true)
		} else {
			toggleWidget(node, findWidgetByName(node, 'eta'))
			toggleWidget(node, findWidgetByName(node, 's_noise'))
			toggleWidget(node, findWidgetByName(node, 'upscale_ratio'))
			toggleWidget(node, findWidgetByName(node, 'start_step'))
			toggleWidget(node, findWidgetByName(node, 'end_step'))
			toggleWidget(node, findWidgetByName(node, 'upscale_n_step'))
			toggleWidget(node, findWidgetByName(node, 'unsharp_kernel_size'))
			toggleWidget(node, findWidgetByName(node, 'unsharp_sigma'))
			toggleWidget(node, findWidgetByName(node, 'unsharp_strength'))
		}
	}
}

app.registerExtension({
	name: "comfy.easyUse.dynamicWidgets",

	nodeCreated(node) {
		if (["easy a1111Loader","easy comfyLoader","easy preSamplingAdvanced","easy preSamplingSdTurbo", "easy kSampler","easy kSamplerSDTurbo","easy kSamplerTiled"].includes(node.comfyClass)) {
			getSetters(node)
		}
	},
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (["easy showSpentTime"].includes(nodeData.name)) {
			function populate(arr_text) {
				var text = '';
				for (let i = 0; i < arr_text.length; i++){
					text += arr_text[i];
				}
				if (this.widgets) {
					const pos = this.widgets.findIndex((w) => w.name === "spent_time");
					if (pos !== -1 && this.widgets[pos]) {
						const w = this.widgets[pos]
						w.value = text;
					}
				}
				requestAnimationFrame(() => {
					const sz = this.computeSize();
					if (sz[0] < this.size[0]) {
						sz[0] = this.size[0];
					}
					if (sz[1] < this.size[1]) {
						sz[1] = this.size[1];
					}
					this.onResize?.(sz);
					app.graph.setDirtyCanvas(true, false);
				});
			}

			// When the node is executed we will be sent the input text, display this in the widget
			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);
				populate.call(this, message.text);
			};
		}

		if (["easy preSampling", "easy preSamplingAdvanced", "easy preSamplingSdTurbo", "easy preSamplingDynamicCFG"].includes(nodeData.name)) {
			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);
				const seed_changed = message.value[0]
				const seed_index = this.widgets.findIndex((w) => w.name === "seed_num")
				const w = this.widgets[seed_index]
				w.value = seed_changed;
			};
		}
	},
});


const getSetWidgets = ['rescale_after_model', 'rescale', 'image_output', 
						'lora_name', 'lora1_name', 'lora2_name', 'lora3_name', 
						'refiner_lora1_name', 'refiner_lora2_name', 'upscale_method', 
						'image_output', 'add_noise', 'info', 'sampler_name',
						'ckpt_B_name', 'ckpt_C_name', 'save_model', 'refiner_ckpt_name',
						'num_loras', 'mode', 'toggle', "resolution"]

function getSetters(node) {
	if (node.widgets)
		for (const w of node.widgets) {
			if (getSetWidgets.includes(w.name)) {
				widgetLogic(node, w);
				if(w.name == 'sampler_name' && node.comfyClass == 'easy preSamplingSdTurbo') widgetLogic2(node, w);
				let widgetValue = w.value;

				// Define getters and setters for widget values
				Object.defineProperty(w, 'value', {
					get() {
						return widgetValue;
					},
					set(newVal) {
						if (newVal !== widgetValue) {
							widgetValue = newVal;
							widgetLogic(node, w);
							if(w.name == 'sampler_name' && node.comfyClass == 'easy preSamplingSdTurbo') widgetLogic2(node, w);
						}
					}
				});
			}
		}
}