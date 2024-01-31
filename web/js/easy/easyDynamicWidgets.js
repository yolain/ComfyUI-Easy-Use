import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { ComfyWidgets } from "/scripts/widgets.js";


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
		updateNodeHeight(node)
	}
	if (widget.name === 'downscale_mode') {
		const widget_names = ['block_number', 'downscale_factor', 'start_percent', 'end_percent', 'downscale_after_skip', 'downscale_method', 'upscale_method']
		if (widget.value === "None") widget_names.map(name=> toggleWidget(node, findWidgetByName(node, name)))
		else if(widget.value == 'Auto') widget_names.map(name =>toggleWidget(node, findWidgetByName(node, name),name == 'block_number' ? true : false))
		else widget_names.map(name=> toggleWidget(node, findWidgetByName(node, name), true))
		updateNodeHeight(node)
	}

	if (widget.name == 'range_mode'){
		if(widget.value == 'step'){
			toggleWidget(node, findWidgetByName(node, 'step'), true)
			toggleWidget(node, findWidgetByName(node, 'num_steps'))
		}else if(widget.value == 'num_steps'){
			toggleWidget(node, findWidgetByName(node, 'step'))
			toggleWidget(node, findWidgetByName(node, 'num_steps'), true)
		}
		updateNodeHeight(node)
	}

	if (widget.name === 'toggle') {
		widget.type = 'toggle'
		widget.options = {on: 'Enabled', off: 'Disabled'}
	}

	if(widget.name == 'text_combine_mode'){
		if(widget.value == 'replace'){
			toggleWidget(node, findWidgetByName(node, 'replace_text'), true)
		}else{
			toggleWidget(node, findWidgetByName(node, 'replace_text'))
		}
		updateNodeHeight(node)
	}
}

function widgetLogic2(node, widget) {
	if (widget.name === 'sampler_name') {
		const widget_names = ['eta','s_noise','upscale_ratio','start_step','end_step','upscale_n_step','unsharp_kernel_size','unsharp_sigma','unsharp_strength']
		if (["euler_ancestral", "dpmpp_2s_ancestral", "dpmpp_2m_sde", "lcm"].includes(widget.value)) {
			widget_names.map(name=> toggleWidget(node, findWidgetByName(node, name)), true)
		} else {
			widget_names.map(name=> toggleWidget(node, findWidgetByName(node, name)))
		}
		updateNodeHeight(node)
	}
}

function widgetLogic3(node, widget){
	if (widget.name === 'target_parameter') {
		if (node.comfyClass == 'easy XYInputs: Steps'){
			switch (widget.value){
				case "steps":
					toggleWidget(node, findWidgetByName(node, 'first_step'), true)
					toggleWidget(node, findWidgetByName(node, 'last_step'), true)
					toggleWidget(node, findWidgetByName(node, 'first_start_step'))
					toggleWidget(node, findWidgetByName(node, 'last_start_step'))
					toggleWidget(node, findWidgetByName(node, 'first_end_step'))
					toggleWidget(node, findWidgetByName(node, 'last_end_step'))
					break
				case "start_at_step":
					toggleWidget(node, findWidgetByName(node, 'first_step'))
					toggleWidget(node, findWidgetByName(node, 'last_step'))
					toggleWidget(node, findWidgetByName(node, 'first_start_step'), true)
					toggleWidget(node, findWidgetByName(node, 'last_start_step'), true)
					toggleWidget(node, findWidgetByName(node, 'first_end_step'))
					toggleWidget(node, findWidgetByName(node, 'last_end_step'))
					break
				case "end_at_step":
					toggleWidget(node, findWidgetByName(node, 'first_step'))
					toggleWidget(node, findWidgetByName(node, 'last_step'))
					toggleWidget(node, findWidgetByName(node, 'first_start_step'))
					toggleWidget(node, findWidgetByName(node, 'last_start_step'))
					toggleWidget(node, findWidgetByName(node, 'first_end_step'),true)
					toggleWidget(node, findWidgetByName(node, 'last_end_step'),true)
					break
			}
		}
		if (node.comfyClass == 'easy XYInputs: Sampler/Scheduler'){
			let number_to_show = findWidgetByName(node, 'input_count').value + 1
			for (let i = 0; i < number_to_show; i++) {
				switch (widget.value) {
					case "sampler":
						toggleWidget(node, findWidgetByName(node, 'sampler_'+i), true)
						toggleWidget(node, findWidgetByName(node, 'scheduler_'+i))
						break
					case "scheduler":
						toggleWidget(node, findWidgetByName(node, 'scheduler_'+i), true)
						toggleWidget(node, findWidgetByName(node, 'sampler_'+i))
						break
					default:
						toggleWidget(node, findWidgetByName(node, 'sampler_'+i), true)
						toggleWidget(node, findWidgetByName(node, 'scheduler_'+i), true)
						break
				}
			}
			updateNodeHeight(node)
		}
		if (node.comfyClass == 'easy XYInputs: ControlNet'){
			switch (widget.value){
				case "strength":
					toggleWidget(node, findWidgetByName(node, 'first_strength'), true)
					toggleWidget(node, findWidgetByName(node, 'last_strength'), true)
					toggleWidget(node, findWidgetByName(node, 'strength'))
					toggleWidget(node, findWidgetByName(node, 'start_percent'), true)
					toggleWidget(node, findWidgetByName(node, 'end_percent'), true)
					toggleWidget(node, findWidgetByName(node, 'first_start_percent'))
					toggleWidget(node, findWidgetByName(node, 'last_start_percent'))
					toggleWidget(node, findWidgetByName(node, 'first_end_percent'))
					toggleWidget(node, findWidgetByName(node, 'last_end_percent'))
					break
				case "start_percent":
					toggleWidget(node, findWidgetByName(node, 'first_strength'))
					toggleWidget(node, findWidgetByName(node, 'last_strength'))
					toggleWidget(node, findWidgetByName(node, 'strength'), true)
					toggleWidget(node, findWidgetByName(node, 'start_percent'))
					toggleWidget(node, findWidgetByName(node, 'end_percent'), true)
					toggleWidget(node, findWidgetByName(node, 'first_start_percent'), true)
					toggleWidget(node, findWidgetByName(node, 'last_start_percent'), true)
					toggleWidget(node, findWidgetByName(node, 'first_end_percent'))
					toggleWidget(node, findWidgetByName(node, 'last_end_percent'))
					break
				case "end_percent":
					toggleWidget(node, findWidgetByName(node, 'first_strength'))
					toggleWidget(node, findWidgetByName(node, 'last_strength'))
					toggleWidget(node, findWidgetByName(node, 'strength'), true)
					toggleWidget(node, findWidgetByName(node, 'start_percent'), true)
					toggleWidget(node, findWidgetByName(node, 'end_percent'))
					toggleWidget(node, findWidgetByName(node, 'first_start_percent'))
					toggleWidget(node, findWidgetByName(node, 'last_start_percent'))
					toggleWidget(node, findWidgetByName(node, 'first_end_percent'), true)
					toggleWidget(node, findWidgetByName(node, 'last_end_percent'), true)
					break
			}
			updateNodeHeight(node)
		}

	}
	if (node.comfyClass == 'easy XYInputs: PromptSR'){
		let number_to_show = findWidgetByName(node, 'replace_count').value + 1
		for (let i = 0; i < number_to_show; i++) {
			toggleWidget(node, findWidgetByName(node, 'replace_'+i), true)
		}
		for (let i = number_to_show; i < 31; i++) {
			toggleWidget(node, findWidgetByName(node, 'replace_'+i))
		}
		updateNodeHeight(node)
	}

	if(widget.name == 'input_count'){
		let number_to_show = widget.value + 1
		for (let i = 0; i < number_to_show; i++) {
			if (findWidgetByName(node, 'target_parameter').value === "sampler") {
				toggleWidget(node, findWidgetByName(node, 'sampler_'+i), true)
				toggleWidget(node, findWidgetByName(node, 'scheduler_'+i))
			}
			else if (findWidgetByName(node, 'target_parameter').value === "scheduler") {
				toggleWidget(node, findWidgetByName(node, 'scheduler_'+i), true)
				toggleWidget(node, findWidgetByName(node, 'sampler_'+i))
			} else {
				toggleWidget(node, findWidgetByName(node, 'sampler_'+i), true)
				toggleWidget(node, findWidgetByName(node, 'scheduler_'+i), true)
			}
		}
		for (let i = number_to_show; i < 31; i++) {
			toggleWidget(node, findWidgetByName(node, 'sampler_'+i))
			toggleWidget(node, findWidgetByName(node, 'scheduler_'+i))
		}
		updateNodeHeight(node)
	}

	// if(widget.name == 'replace_count'){
	// 	let number_to_show = widget.value + 1
	// 	for (let i = 0; i < number_to_show; i++) {
	// 		toggleWidget(node, findWidgetByName(node, 'replace_'+i), true)
	// 	}
	// 	for (let i = number_to_show; i < 31; i++) {
	// 		toggleWidget(node, findWidgetByName(node, 'replace_'+i))
	// 	}
	// 	updateNodeHeight(node)
	// }
}

app.registerExtension({
	name: "comfy.easyUse.dynamicWidgets",

	nodeCreated(node) {
		switch (node.comfyClass){
			case "easy fullLoader":
			case "easy a1111Loader":
			case "easy comfyLoader":
			case "easy svdLoader":
			case "easy loraStack":
			case "easy latentNoisy":
			case "easy preSamplingAdvanced":
			case "easy preSamplingSdTurbo":
			case "easy fullkSampler":
			case "easy kSampler":
			case "easy kSamplerSDTurbo":
			case "easy kSamplerTiled":
			case "easy kSamplerInpainting":
			case "easy kSamplerDownscaleUnet":
			case "easy hiresFix":
			case "easy detailerFix":
			case "easy imageRemoveBG":
			case "easy XYInputs: Steps":
			case "easy XYInputs: Sampler/Scheduler":
			case "easy XYInputs: PromptSR":
			case "easy XYInputs: ControlNet":
			case "easy rangeInt":
			case "easy rangeFloat":
			case 'easy latentCompositeMaskedWithCond':
				getSetters(node)
				break
			case "easy wildcards":
				const wildcard_text_widget_index = node.widgets.findIndex((w) => w.name == 'text');
				const wildcard_text_widget = node.widgets[wildcard_text_widget_index];

				// lora selector, wildcard selector
				let combo_id = 1;

				Object.defineProperty(node.widgets[combo_id], "value", {
					set: (value) => {
							const stackTrace = new Error().stack;
							if(stackTrace.includes('inner_value_change')) {
								if(value != "Select the LoRA to add to the text") {
									let lora_name = value;
									if (lora_name.endsWith('.safetensors')) {
										lora_name = lora_name.slice(0, -12);
									}

									wildcard_text_widget.value += `<lora:${lora_name}>`;
								}
							}
						},
					get: () => { return "Select the LoRA to add to the text"; }
				});

				Object.defineProperty(node.widgets[combo_id+1], "value", {
					set: (value) => {
							const stackTrace = new Error().stack;
							if(stackTrace.includes('inner_value_change')) {
								if(value != "Select the Wildcard to add to the text") {
									if(wildcard_text_widget.value != '')
										wildcard_text_widget.value += ', '

									wildcard_text_widget.value += value;
								}
							}
						},
					get: () => { return "Select the Wildcard to add to the text"; }
				});

				// Preventing validation errors from occurring in any situation.
				node.widgets[combo_id].serializeValue = () => { return "Select the LoRA to add to the text"; }
				node.widgets[combo_id+1].serializeValue = () => { return "Select the Wildcard to add to the text"; }
				break
			case "easy detailerFix":
				const textarea_widget_index = node.widgets.findIndex((w) => w.type === "customtext");
				if(textarea_widget_index == -1) return
				node.widgets[textarea_widget_index].dynamicPrompts = false
				node.widgets[textarea_widget_index].inputEl.placeholder = "wildcard spec: if kept empty, this option will be ignored";
				node.widgets[textarea_widget_index].serializeValue = () => {return node.widgets[textarea_widget_index].value};
				break
			case "easy XYInputs: ModelMergeBlocks":
 				let preset_i = 3;
		    	let vector_i = 4;
		    	let file_i = 5;
				node._value = "Preset";

				let valuesWidget = node.widgets[vector_i]
				Object.defineProperty(node.widgets[preset_i], "value", {
					set: (value) => {
							const stackTrace = new Error().stack;
							if(stackTrace.includes('inner_value_change')) {
								if(value != "Preset") {
									if(!value.startsWith('@') && valuesWidget.value != "")
										valuesWidget.value += "\n";
									if(value.startsWith('@')) {
										let spec = value.split(':')[1];
										var n;
										var sub_n = null;
										var block = null;

										if(isNaN(spec)) {
											let sub_spec = spec.split(',');

											if(sub_spec.length != 3) {
												valuesWidget = '!! SPEC ERROR !!';
												node._value = '';
												return;
											}

											n = parseInt(sub_spec[0].trim());
											sub_n = parseInt(sub_spec[1].trim());
											block = parseInt(sub_spec[2].trim());
										}
										else {
											n = parseInt(spec.trim());
										}

										valuesWidget.value = "";
										if(sub_n == null) {
											for(let i=1; i<=n; i++) {
												var temp = "1,1";
												for(let j=1; j<=n; j++) {
													if(temp!='')
														temp += ',';
													if(j==i)
														temp += '1';
													else
														temp += '0';
												}
												temp += ',1; ';

												valuesWidget.value += `B${i}:${temp}\n`;
											}
										}
										else {
											for(let i=1; i<=sub_n; i++) {
												var temp = "";
												for(let j=1; j<=n; j++) {
													if(temp!='')
														temp += ',';

													if(block!=j)
														temp += '0';
													else {
														temp += ' ';
														for(let k=1; k<=sub_n; k++) {
															if(k==i)
																temp += '1 ';
															else
																temp += '0 ';
														}
													}
												}

												valuesWidget.value += `B${block}.SUB${i}:${temp}\n`;
											}
										}
									}
									else {
										valuesWidget.value += `${value}; `;
									}
									if(node.widgets_values) {
										valuesWidget = node.widgets[preset_i].value+ `; `;
									}
								}
							}

							node._value = value;
						},
					get: () => {
						return node._value;
				 	}
				});

				const cb = node.callback;
				valuesWidget.callback = function () {
					if (cb) {
						return cb.apply(this, arguments);
					}
				};

				// upload .csv
				async function uploadFile(file) {
					try {
						const body = new FormData();
						body.append("csv", file);
						const resp = await api.fetchApi("/easyuse/upload/csv", {
							method: "POST",
							body,
						});

						if (resp.status === 200) {
							const data = await resp.json();
							node.widgets[vector_i].value = data
						} else {
							alert(resp.status + " - " + resp.statusText);
						}
					} catch (error) {
						alert(error);
					}
				}

				const fileInput = document.createElement("input");
				Object.assign(fileInput, {
					type: "file",
					accept: "text/csv",
					style: "display: none",
					onchange: async (event) => {
						if (fileInput.files.length) {
							await uploadFile(fileInput.files[0], true);
							event.target.value = ''
						}
					},
				});
				document.body.append(fileInput);

				const name = "choose .csv file into values"
				let uploadWidget = node.addWidget("button", name, "csv", () => {
					fileInput.click();
				});
				uploadWidget.label = name;
				uploadWidget.serialize = false;

				break
		}

	},
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		function addText(arr_text) {
			var text = '';
			for (let i = 0; i < arr_text.length; i++) {
				text += arr_text[i];
			}
			return text
		}

		if (["easy showSpentTime"].includes(nodeData.name)) {
			function populate(text) {
				if (this.widgets) {
					const pos = this.widgets.findIndex((w) => w.name === "spent_time");
					if (pos !== -1 && this.widgets[pos]) {
						const w = this.widgets[pos]
						w.value = text;
					}
				}
			}

			// When the node is executed we will be sent the input text, display this in the widget
			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);
				const text = addText(message.text)
				populate.call(this, text);
			};
		}

		if (["easy showLoaderSettingsNames"].includes(nodeData.name)) {
			function populate(text) {
				if (this.widgets) {
					const pos = this.widgets.findIndex((w) => w.name === "names");
					if (pos !== -1 && this.widgets[pos]) {
						const w = this.widgets[pos]
						w.value = text;
					}
				}
			}

			// When the node is executed we will be sent the input text, display this in the widget
			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);
				const text = addText(message.text)
				populate.call(this, text);
			};
		}

		if (["easy fullLoader", "easy a1111Loader", "easy comfyLoader"].includes(nodeData.name)) {
			function populate(text, type = 'positive') {
				if (this.widgets) {
					const pos = this.widgets.findIndex((w) => w.name === type + "_prompt");
					const className = "comfy-multiline-input wildcard_" + type + '_' + this.id.toString()
					if (pos == -1 && text) {
						const inputEl = document.createElement("textarea");
						inputEl.className = className;
						inputEl.placeholder = "Wildcard Prompt (" + type + ")"
						const widget = this.addDOMWidget(type + "_prompt", "customtext", inputEl, {
							getValue() {
								return inputEl.value;
							},
							setValue(v) {
								inputEl.value = v;
							},
							serialize: false,
						});
						widget.inputEl = inputEl;
						widget.inputEl.readOnly = true
						inputEl.addEventListener("input", () => {
							widget.callback?.(widget.value);
						});
						widget.value = text;
					} else if (this.widgets[pos]) {
						if (text) {
							const w = this.widgets[pos]
							w.value = text;
						} else {
							this.widgets.splice(pos, 1);
							const element = document.getElementsByClassName(className)
							if (element && element[0]) element[0].remove()
						}
					}
				}
			}

			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);
				const positive = addText(message.positive)
				const negative = addText(message.negative)
				populate.call(this, positive, "positive");
				populate.call(this, negative, "negative");
			};
		}

		if (["easy seed", "easy latentNoisy", "easy wildcards", "easy preSampling", "easy preSamplingAdvanced", "easy preSamplingSdTurbo", "easy preSamplingDynamicCFG", "easy fullkSampler"].includes(nodeData.name)) {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = async function () {
				onNodeCreated ? onNodeCreated.apply(this, []) : undefined;
				const values = ["randomize", "fixed", "increment", "decrement"]
				const seed_widget = this.widgets.find(w => w.name == 'seed_num')
				const seed_control = this.addWidget("combo", "control_before_generate", values[0], () => {
				}, {
					values,
					serialize: false
				})
				seed_widget.linkedWidgets = [seed_control]
			}
		}

		if (nodeData.name == 'easy imageInsetCrop') {
			function setWidgetStep(a) {
				const measurementWidget = a.widgets[0]
				for (let i = 1; i <= 4; i++) {
					if (measurementWidget.value === 'Pixels') {
						a.widgets[i].options.step = 80;
						a.widgets[i].options.max = 8192;
					} else {
						a.widgets[i].options.step = 10;
						a.widgets[i].options.max = 99;
					}
				}
			}

			nodeType.prototype.onAdded = async function (graph) {
				const measurementWidget = this.widgets[0];
				let callback = measurementWidget.callback;
				measurementWidget.callback = (...args) => {
					setWidgetStep(this);
					callback && callback.apply(measurementWidget, [...args]);
				};
				setTimeout(_=>{
					setWidgetStep(this);
				},1)
			}
		}

		if(nodeData.name == 'easy showAnything'){
			function populate(text) {
				if (this.widgets) {
					const pos = this.widgets.findIndex((w) => w.name === "text");
					if (pos !== -1) {
						for (let i = pos; i < this.widgets.length; i++) {
							this.widgets[i].onRemove?.();
						}
						this.widgets.length = pos;
					}
				}

				for (const list of text) {
					const w = ComfyWidgets["STRING"](this, "text", ["STRING", { multiline: true }], app).widget;
					w.inputEl.readOnly = true;
					w.inputEl.style.opacity = 0.6;
					w.value = list;
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

			const onConfigure = nodeType.prototype.onConfigure;
			nodeType.prototype.onConfigure = function () {
				onConfigure?.apply(this, arguments);
				if (this.widgets_values?.length) {
					populate.call(this, this.widgets_values);
				}
			};
		}

		if(nodeData.name == 'easy convertAnything'){
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = async function () {
				onNodeCreated ? onNodeCreated.apply(this, []) : undefined;
				const type_control = this.widgets[this.widgets.findIndex((w) => w.name === "output_type")]
				let _this = this
				type_control.callback = _ => {
					_this.outputs[0].type = (type_control.value).toUpperCase()
					_this.outputs[0].name = type_control.value
				}

			}
		}
	}
});


const getSetWidgets = ['rescale_after_model', 'rescale', 'image_output', 
						'lora_name', 'lora1_name', 'lora2_name', 'lora3_name', 
						'refiner_lora1_name', 'refiner_lora2_name', 'upscale_method', 
						'image_output', 'add_noise', 'info', 'sampler_name',
						'ckpt_B_name', 'ckpt_C_name', 'save_model', 'refiner_ckpt_name',
						'num_loras', 'mode', 'toggle', 'resolution', 'target_parameter', 'input_count', 'replace_count', 'downscale_mode', 'range_mode','text_combine_mode']

function getSetters(node) {
	if (node.widgets)
		for (const w of node.widgets) {
			if (getSetWidgets.includes(w.name)) {
				if(node.comfyClass.indexOf("easy XYInputs:") != -1) widgetLogic3(node, w)
				else if(w.name == 'sampler_name' && node.comfyClass == 'easy preSamplingSdTurbo') widgetLogic2(node, w);
				else widgetLogic(node, w);
				let widgetValue = w.value;

				// Define getters and setters for widget values
				Object.defineProperty(w, 'value', {
					get() {
						return widgetValue;
					},
					set(newVal) {
						if (newVal !== widgetValue) {
							widgetValue = newVal;
							if(node.comfyClass.indexOf("easy XYInputs:") != -1) widgetLogic3(node, w)
							else if(w.name == 'sampler_name' && node.comfyClass == 'easy preSamplingSdTurbo') widgetLogic2(node, w);
							else widgetLogic(node, w);
						}
					}
				});
			}
		}
}