import { app } from "/scripts/app.js";
import { ComfyWidgets } from '/scripts/widgets.js'

// Node that allows you to tunnel connections for cleaner graphs

app.registerExtension({
	name: "easy setNode",
	registerCustomNodes() {
		class SetNode {
			defaultVisibility = true;
			serialize_widgets = true;
			constructor() {
				if (!this.properties) {
					this.properties = {
						"previousName": ""
					};
				}
				this.properties.showOutputText = SetNode.defaultVisibility;

				const node = this;

				
				this.addWidget(
					"text", 
					"Constant", 
					'', 
					(s, t, u, v, x) => {
						node.validateName(node.graph);
						this.update();
						this.properties.previousName = this.widgets[0].value;
					}, 
					{}
				)
				
				this.addInput("*", "*");


				this.onConnectionsChange = function(
					slotType,	//1 = input, 2 = output
					slot,
					isChangeConnect,
                    link_info,
                    output
				) {
					console.log("onConnectionsChange");
					//On Disconnect
					if (slotType == 1 && !isChangeConnect) {
						this.inputs[slot].type = '*';
						this.inputs[slot].name = '*';
					}

					//On Connect
					if (link_info && node.graph && slotType == 1 && isChangeConnect) {
						const fromNode = node.graph._nodes.find((otherNode) => otherNode.id == link_info.origin_id);
						const type = fromNode.outputs[link_info.origin_slot].type;

						this.inputs[0].type = type;
						this.inputs[0].name = type;
					}

					//Update either way
					this.update();
				}

				this.validateName = function(graph) {
					let widgetValue = node.widgets[0].value;
					
					if (widgetValue != '') {
						let tries = 0;
						let collisions = [];
						
						do {
							collisions = graph._nodes.filter((otherNode) => {
								if (otherNode == this) {
									return false;
								}
								if (otherNode.type == 'easy setNode' && otherNode.widgets[0].value === widgetValue) {
									return true;
								}
								return false;
							})
							if (collisions.length > 0) {
								widgetValue = node.widgets[0].value + "_" + tries;
							}
							tries++;
						} while (collisions.length > 0)
						node.widgets[0].value = widgetValue;
						this.update();
					}
				}

				this.clone = function () {
					console.log("CLONE");
					const cloned = SetNode.prototype.clone.apply(this);
					//cloned.inputs = [];
					cloned.inputs[0].name = '*';
					cloned.inputs[0].type = '*';
					cloned.properties.previousName = '';
					cloned.size = cloned.computeSize();
					return cloned;
				};

				this.onAdded = function(graph) {
					this.validateName(graph);
				}


				this.update = function() {
					console.log("SetNode.update()");
					console.log(this.widgets[0].value);
					if (node.graph) {
						this.findGetters(node.graph).forEach((getter) => {
							getter.setType(this.inputs[0].type);
						});
						if (this.widgets[0].value) {
							this.findGetters(node.graph, true).forEach((getter) => {
								getter.setName(this.widgets[0].value)
							});
						}

						const allGetters = node.graph._nodes.filter((otherNode) => otherNode.type == "easy getNode");
						allGetters.forEach((otherNode) => {
							if (otherNode.setComboValues) {
								otherNode.setComboValues();
							}
						})
					}
				}


				this.findGetters = function(graph, checkForPreviousName) {
					const name = checkForPreviousName ? this.properties.previousName : this.widgets[0].value;
					return graph._nodes.filter((otherNode) => {
						//console.log("otherNode.type:");
						//console.log(otherNode.type)
						if (otherNode.type == 'easy getNode' && otherNode.widgets[0].value === name && name != '') {
							return true;
						}
						return false;
					})
				}


				// This node is purely frontend and does not impact the resulting prompt so should not be serialized
				this.isVirtualNode = true;
			}

			onRemoved() {
				console.log("onRemove");
				console.log(this);
				console.log(this.flags);
				const allGetters = this.graph._nodes.filter((otherNode) => otherNode.type == "easy getNode");
				allGetters.forEach((otherNode) => {
					if (otherNode.setComboValues) {
						otherNode.setComboValues([this]);
					}
				})
			}
		}


		LiteGraph.registerNodeType(
			"easy setNode",
			Object.assign(SetNode, {
				title: "Set",
			})
		);

		SetNode.category = "utils";
	},
});


app.registerExtension({
	name: "easy getNode",
	registerCustomNodes() {
		class GetNode {

			defaultVisibility = true;
			serialize_widgets = true;

			constructor() {
				if (!this.properties) {
					this.properties = {};
				}
				this.properties.showOutputText = GetNode.defaultVisibility;
				
				const node = this;
				this.addWidget(
					"combo",
					"Constant",
					"",
					(e) => {
						this.onRename();
					},
					{
						values: () => {
                            const setterNodes = graph._nodes.filter((otherNode) => otherNode.type == 'easy setNode');
                            //console.log("setting combo values");
                            /*setterNodes.forEach((otherNode) => {
                                console.log(otherNode.widgets[0].value)
                            })*/
                            return setterNodes.map((otherNode) => otherNode.widgets[0].value).sort();
                        }
					}
				)


				this.addOutput("*", '*');

				
				this.onConnectionsChange = function(
					slotType,	//0 = output, 1 = input
					slot,	//self-explanatory
					isChangeConnect,
                    link_info,
                    output
				) {
					this.validateLinks();	
				}

				
				this.setName = function(name) {
					console.log("renaming getter: ");
					console.log(node.widgets[0].value + " -> " + name);
					node.widgets[0].value = name;
					node.onRename();
					node.serialize();
				}
				

				this.onRename = function() {
					console.log("onRename");

					const setter = this.findSetter(node.graph);
					if (setter) {
						this.setType(setter.inputs[0].type);
					} else {
						this.setType('*');
					}
				}

				this.clone = function () {
					const cloned = GetNode.prototype.clone.apply(this);
					cloned.size = cloned.computeSize();
					//this.update();
					return cloned;
				};

				this.validateLinks = function() {
					console.log("validating links");
					if (this.outputs[0].type != '*' && this.outputs[0].links) {
						console.log("in");
						this.outputs[0].links.forEach((linkId) => {
							const link = node.graph.links[linkId];
							if (link && link.type != this.outputs[0].type && link.type != '*') {
								console.log("removing link");
								node.graph.removeLink(linkId)
							}
						})
					} 
				}

				this.setType = function(type) {
					this.outputs[0].name = type;
					this.outputs[0].type = type;
					this.validateLinks();
				}

				this.findSetter = function(graph) {
					const name = this.widgets[0].value;
					return graph._nodes.find((otherNode) => {
						//console.log("findSetter");
						//console.log("otherNode.type");
						//console.log(otherNode.type);
						if (otherNode.type == 'easy setNode' && otherNode.widgets[0].value === name && name != '') {
							return true;
						}
						return false;
					})
				}

				// This node is purely frontend and does not impact the resulting prompt so should not be serialized
				this.isVirtualNode = true;
			}


			getInputLink(slot) {
                console.log("get.getInputLink(): " + slot);
				const setter = this.findSetter(this.graph);
                console.log("setter:");
                console.log(setter);

				
				// const setters = app.graph._nodes.filter((otherNode) => {
				// 	const name = this.widgets[0].value
				// 	if (otherNode.type == 'TunnelIn' && otherNode.widgets[0].value === name && name != '') {
				// 		return true;
				// 	}
				// 	return false;
				// });

				// if (setters.length > 1) {
				// 	throw new Error("Multiple setters found for " + this.widgets[0].value);
				// }

				// if (setters.length == 0) {
				// 	throw new Error("No setter found for " + this.widgets[0].value);
				// }
				

				if (setter) {
					const slot_info = setter.inputs[slot];
                    console.log("slot info");
                    console.log(slot_info);
                    console.log(this.graph.links);
                    const link = this.graph.links[ slot_info.link ];
                    console.log("link:");
                    console.log(link);
                    return link;
				} else {
                    console.log(this.widgets[0]);
                    console.log(this.widgets[0].value);
					throw new Error("No setter found for " + this.widgets[0].value + "(" + this.type + ")");
				}

			}
			onAdded(graph) {
				//this.setComboValues();
				//this.validateName(graph);
			}

		}


		LiteGraph.registerNodeType(
			"easy getNode",
			Object.assign(GetNode, {
				title: "Get",
			})
		);

		GetNode.category = "utils";
	},
});