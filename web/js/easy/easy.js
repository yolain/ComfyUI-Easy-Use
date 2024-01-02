import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";
import { $el } from "/scripts/ui.js";
import { api } from "/scripts/api.js";

const BETTER_COMBOS_NODES = ["easy a1111Loader"]
const CONVERTED_TYPE = "converted-widget";
const GET_CONFIG = Symbol();

function hideWidget(node, widget, suffix = "") {
	widget.origType = widget.type;
	widget.origComputeSize = widget.computeSize;
	widget.origSerializeValue = widget.serializeValue;
	widget.computeSize = () => [0, -4]; // -4 is due to the gap litegraph adds between widgets automatically
	widget.type = CONVERTED_TYPE + suffix;
	widget.serializeValue = () => {
		// Prevent serializing the widget if we have no input linked
		if (!node.inputs) {
			return undefined;
		}
		let node_input = node.inputs.find((i) => i.widget?.name === widget.name);

		if (!node_input || !node_input.link) {
			return undefined;
		}
		return widget.origSerializeValue ? widget.origSerializeValue() : widget.value;
	};

	// Hide any linked widgets, e.g. seed+seedControl
	if (widget.linkedWidgets) {
		for (const w of widget.linkedWidgets) {
			hideWidget(node, w, ":" + widget.name);
		}
	}
}

function convertToInput(node, widget, config) {
    console.log('config:', config)
	hideWidget(node, widget);

	const { type } = getWidgetType(config);

	// Add input and store widget config for creating on primitive node
	const sz = node.size;
	node.addInput(widget.name, type, {
		widget: { name: widget.name, [GET_CONFIG]: () => config },
	});

	for (const widget of node.widgets) {
		widget.last_y += LiteGraph.NODE_SLOT_HEIGHT;
	}

	// Restore original size but grow if needed
	node.setSize([Math.max(sz[0], node.size[0]), Math.max(sz[1], node.size[1])]);
}

function getWidgetType(config) {
	// Special handling for COMBO so we restrict links based on the entries
	let type = config[0];
	if (type instanceof Array) {
		type = "COMBO";
	}
	return { type };
}

app.registerExtension({
    name: "comfy.easyUse",
    init() {
        // 刷新节点
        const easyReloadNode = function (node) {
            const nodeType = node.constructor.type;
            const origVals = node.properties.origVals || {};

            const nodeTitle = origVals.title || node.title;
            const nodeColor = origVals.color || node.color;
            const bgColor = origVals.bgcolor || node.bgcolor;
            const oldNode = node
            const options = {
                'size': [...node.size],
                'color': nodeColor,
                'bgcolor': bgColor,
                'pos': [...node.pos]
            }

            let inputLinks = []
            let outputLinks = []
            if(node.inputs){
                for (const input of node.inputs) {
                    if (input.link) {
                        const input_name = input.name
                        const input_slot = node.findInputSlot(input_name)
                        const input_node = node.getInputNode(input_slot)
                        const input_link = node.getInputLink(input_slot)

                        inputLinks.push([input_link.origin_slot, input_node, input_name])
                    }
                }
            }
            if(node.outputs) {
                for (const output of node.outputs) {
                    if (output.links) {
                        const output_name = output.name

                        for (const linkID of output.links) {
                            const output_link = graph.links[linkID]
                            const output_node = graph._nodes_by_id[output_link.target_id]
                            outputLinks.push([output_name, output_node, output_link.target_slot])
                        }
                    }
                }
            }

            app.graph.remove(node)
            const newNode = app.graph.add(LiteGraph.createNode(nodeType, nodeTitle, options));

            function handleLinks() {
                // re-convert inputs
                for (let w of oldNode.widgets) {
                    if (w.type === 'converted-widget') {
                        const WidgetToConvert = newNode.widgets.find((nw) => nw.name === w.name);
                        for (let i of oldNode.inputs) {
                            if (i.name === w.name) {
                                convertToInput(newNode, WidgetToConvert, i.widget);
                            }
                        }
                    }
                }
                // replace input and output links
                for (let input of inputLinks) {
                    const [output_slot, output_node, input_name] = input;
                    output_node.connect(output_slot, newNode.id, input_name)
                }
                for (let output of outputLinks) {
                    const [output_name, input_node, input_slot] = output;
                    newNode.connect(output_name, input_node, input_slot)
                }
            }

            // fix widget values
            let values = oldNode.widgets_values;
            if (!values) {
                newNode.widgets.forEach((newWidget, index) => {
                    const oldWidget = oldNode.widgets[index];
                    if (newWidget.name === oldWidget.name && newWidget.type === oldWidget.type) {
                        newWidget.value = oldWidget.value;
                    }
                });
                handleLinks();
                return;
            }
            let pass = false
            const isIterateForwards = values.length <= newNode.widgets.length;
            let vi = isIterateForwards ? 0 : values.length - 1;
            function evalWidgetValues(testValue, newWidg) {
                if (testValue === true || testValue === false) {
                    if (newWidg.options?.on && newWidg.options?.off) {
                        return { value: testValue, pass: true };
                    }
                } else if (typeof testValue === "number") {
                    if (newWidg.options?.min <= testValue && testValue <= newWidg.options?.max) {
                        return { value: testValue, pass: true };
                    }
                } else if (newWidg.options?.values?.includes(testValue)) {
                    return { value: testValue, pass: true };
                } else if (newWidg.inputEl && typeof testValue === "string") {
                    return { value: testValue, pass: true };
                }
                return { value: newWidg.value, pass: false };
            }
            const updateValue = (wi) => {
                const oldWidget = oldNode.widgets[wi];
                let newWidget = newNode.widgets[wi];
                if (newWidget.name === oldWidget.name && newWidget.type === oldWidget.type) {
                    while ((isIterateForwards ? vi < values.length : vi >= 0) && !pass) {
                        let { value, pass } = evalWidgetValues(values[vi], newWidget);
                        if (pass && value !== null) {
                            newWidget.value = value;
                            break;
                        }
                        vi += isIterateForwards ? 1 : -1;
                    }
                    vi++
                    if (!isIterateForwards) {
                        vi = values.length - (newNode.widgets.length - 1 - wi);
                    }
                }
            };
            if (isIterateForwards) {
                for (let wi = 0; wi < newNode.widgets.length; wi++) {
                    updateValue(wi);
                }
            } else {
                for (let wi = newNode.widgets.length - 1; wi >= 0; wi--) {
                    updateValue(wi);
                }
            }
            handleLinks();
        };

        const getNodeMenuOptions = LGraphCanvas.prototype.getNodeMenuOptions;
        LGraphCanvas.prototype.getNodeMenuOptions = function (node) {
            const options = getNodeMenuOptions.apply(this, arguments);
            node.setDirtyCanvas(true, true);

            options.splice(options.length - 1, 0,
                {
                    content: "🔃 Reload Node (easyUse)",
                    callback: () => {
                        var graphcanvas = LGraphCanvas.active_canvas;
                        if (!graphcanvas.selected_nodes || Object.keys(graphcanvas.selected_nodes).length <= 1) {
                            easyReloadNode(node);
                        } else {
                            for (var i in graphcanvas.selected_nodes) {
                                easyReloadNode(graphcanvas.selected_nodes[i]);
                            }
                        }
                    }
                },
            );
            return options;
        };
    },
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name.startsWith("easy")) {
            const origOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const r = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
                return r;
            };
        }
    },
});


// easy Dropdown
var styleElement = document.createElement("style");
const cssCode = `
.easy-dropdown, .easy-nested-dropdown {
    position: relative;
    box-sizing: border-box;
    background-color: #171717;
    box-shadow: 0 4px 4px rgba(255, 255, 255, .25);
    padding: 0;
    margin: 0;
    list-style: none;
    z-index: 1000;
    overflow: visible;
    max-height: fit-content;
    max-width: fit-content;
}

.easy-dropdown {
    position: absolute;
    border-radius: 0;
}

/* Style for final items */
.easy-dropdown li.item, .easy-nested-dropdown li.item {
    font-weight: normal;
    min-width: max-content;
}

/* Style for folders (parent items) */
.easy-dropdown li.folder, .easy-nested-dropdown li.folder {
    cursor: default;
    position: relative;
    border-right: 3px solid cyan;
}

.easy-dropdown li.folder::after, .easy-nested-dropdown li.folder::after {
    content: ">"; 
    position: absolute; 
    right: 2px; 
    font-weight: normal;
}

.easy-dropdown li, .easy-nested-dropdown li {
    padding: 4px 10px;
    cursor: pointer;
    font-family: system-ui;
    font-size: 0.7rem;
    position: relative; 
}

/* Style for nested dropdowns */
.easy-nested-dropdown {
    position: absolute;
    top: 0;
    left: 100%;
    margin: 0;
    border: none;
    display: none;
}

.easy-dropdown li.selected > .easy-nested-dropdown,
.easy-nested-dropdown li.selected > .easy-nested-dropdown {
    display: block;
    border: none;
}
  
.easy-dropdown li.selected,
.easy-nested-dropdown li.selected {
    background-color: #e5e5e5;
    border: none;
}
`
styleElement.innerHTML = cssCode
document.head.appendChild(styleElement);

let activeDropdown = null;

export function easy_RemoveDropdown() {
    if (activeDropdown) {
        activeDropdown.removeEventListeners();
        activeDropdown.dropdown.remove();
        activeDropdown = null;
    }
}

class Dropdown {
    constructor(inputEl, suggestions, onSelect, isDict = false) {
        this.dropdown = document.createElement('ul');
        this.dropdown.setAttribute('role', 'listbox');
        this.dropdown.classList.add('easy-dropdown');
        this.selectedIndex = -1;
        this.inputEl = inputEl;
        this.suggestions = suggestions;
        this.onSelect = onSelect;
        this.isDict = isDict;

        this.focusedDropdown = this.dropdown;

        this.buildDropdown();

        this.onKeyDownBound = this.onKeyDown.bind(this);
        this.onWheelBound = this.onWheel.bind(this);
        this.onClickBound = this.onClick.bind(this);

        this.addEventListeners();
    }

    buildDropdown() {
        if (this.isDict) {
            this.buildNestedDropdown(this.suggestions, this.dropdown);
        } else {
            this.suggestions.forEach((suggestion, index) => {
                this.addListItem(suggestion, index, this.dropdown);
            });
        }

        const inputRect = this.inputEl.getBoundingClientRect();
        this.dropdown.style.top = (inputRect.top + inputRect.height - 10) + 'px';
        this.dropdown.style.left = inputRect.left + 'px';

        document.body.appendChild(this.dropdown);
        activeDropdown = this;
    }

    buildNestedDropdown(dictionary, parentElement) {
        let index = 0;
        Object.keys(dictionary).forEach((key) => {
            const item = dictionary[key];
            if (typeof item === "object" && item !== null) {
                const nestedDropdown = document.createElement('ul');
                nestedDropdown.setAttribute('role', 'listbox');
                nestedDropdown.classList.add('easy-nested-dropdown');
                const parentListItem = document.createElement('li');
                parentListItem.classList.add('folder');
                parentListItem.textContent = key;
                parentListItem.appendChild(nestedDropdown);
                parentListItem.addEventListener('mouseover', this.onMouseOver.bind(this, index, parentElement));
                parentElement.appendChild(parentListItem);
                this.buildNestedDropdown(item, nestedDropdown);
                index = index + 1;
            } else {
                const listItem = document.createElement('li');
                listItem.classList.add('item');
                listItem.setAttribute('role', 'option');
                listItem.textContent = key;
                listItem.addEventListener('mouseover', this.onMouseOver.bind(this, index, parentElement));
                listItem.addEventListener('mousedown', this.onMouseDown.bind(this, key));
                parentElement.appendChild(listItem);
                index = index + 1;
            }
        });
    }

    addListItem(item, index, parentElement) {
        const listItem = document.createElement('li');
        listItem.setAttribute('role', 'option');
        listItem.textContent = item;
        listItem.addEventListener('mouseover', this.onMouseOver.bind(this, index));
        listItem.addEventListener('mousedown', this.onMouseDown.bind(this, item));
        parentElement.appendChild(listItem);
    }

    addEventListeners() {
        document.addEventListener('keydown', this.onKeyDownBound);
        this.dropdown.addEventListener('wheel', this.onWheelBound);
        document.addEventListener('click', this.onClickBound);
    }

    removeEventListeners() {
        document.removeEventListener('keydown', this.onKeyDownBound);
        this.dropdown.removeEventListener('wheel', this.onWheelBound);
        document.removeEventListener('click', this.onClickBound);
    }

    onMouseOver(index, parentElement) {
        if (parentElement) {
            this.focusedDropdown = parentElement;
        }
        this.selectedIndex = index;
        this.updateSelection();
    }

    onMouseOut() {
        this.selectedIndex = -1;
        this.updateSelection();
    }

    onMouseDown(suggestion, event) {
        event.preventDefault();
        this.onSelect(suggestion);
        this.dropdown.remove();
        this.removeEventListeners();
    }

    onKeyDown(event) {
        const enterKeyCode = 13;
        const escKeyCode = 27;
        const arrowUpKeyCode = 38;
        const arrowDownKeyCode = 40;
        const arrowRightKeyCode = 39;
        const arrowLeftKeyCode = 37;
        const tabKeyCode = 9;

        const items = Array.from(this.focusedDropdown.children);
        const selectedItem = items[this.selectedIndex];

        if (activeDropdown) {
            if (event.keyCode === arrowUpKeyCode) {
                event.preventDefault();
                this.selectedIndex = Math.max(0, this.selectedIndex - 1);
                this.updateSelection();
            }

            else if (event.keyCode === arrowDownKeyCode) {
                event.preventDefault();
                this.selectedIndex = Math.min(items.length - 1, this.selectedIndex + 1);
                this.updateSelection();
            }

            else if (event.keyCode === arrowRightKeyCode) {
                event.preventDefault();
                if (selectedItem && selectedItem.classList.contains('folder')) {
                    const nestedDropdown = selectedItem.querySelector('.easy-nested-dropdown');
                    if (nestedDropdown) {
                        this.focusedDropdown = nestedDropdown;
                        this.selectedIndex = 0;
                        this.updateSelection();
                    }
                }
            }

            else if (event.keyCode === arrowLeftKeyCode && this.focusedDropdown !== this.dropdown) {
                const parentDropdown = this.focusedDropdown.closest('.easy-dropdown, .easy-nested-dropdown').parentNode.closest('.easy-dropdown, .easy-nested-dropdown');
                if (parentDropdown) {
                    this.focusedDropdown = parentDropdown;
                    this.selectedIndex = Array.from(parentDropdown.children).indexOf(this.focusedDropdown.parentNode);
                    this.updateSelection();
                }
            }

            else if ((event.keyCode === enterKeyCode || event.keyCode === tabKeyCode) && this.selectedIndex >= 0) {
                event.preventDefault();
                if (selectedItem.classList.contains('item')) {
                    this.onSelect(items[this.selectedIndex].textContent);
                    this.dropdown.remove();
                    this.removeEventListeners();
                }
                
                const nestedDropdown = selectedItem.querySelector('.easy-nested-dropdown');
                if (nestedDropdown) {
                    this.focusedDropdown = nestedDropdown;
                    this.selectedIndex = 0;
                    this.updateSelection();
                }
            }
            
            else if (event.keyCode === escKeyCode) {
                this.dropdown.remove();
                this.removeEventListeners();
            }
        } 
    }

    onWheel(event) {
        const top = parseInt(this.dropdown.style.top);
        if (localStorage.getItem("Comfy.Settings.Comfy.InvertMenuScrolling")) {
            this.dropdown.style.top = (top + (event.deltaY < 0 ? 10 : -10)) + "px";
        } else {
            this.dropdown.style.top = (top + (event.deltaY < 0 ? -10 : 10)) + "px";
        }
    }

    onClick(event) {
        if (!this.dropdown.contains(event.target) && event.target !== this.inputEl) {
            this.dropdown.remove();
            this.removeEventListeners();
        }
    }

    updateSelection() {
        Array.from(this.focusedDropdown.children).forEach((li, index) => {
            if (index === this.selectedIndex) {
                li.classList.add('selected');
            } else {
                li.classList.remove('selected');
            }
        });
    }
}

export function easy_CreateDropdown(inputEl, suggestions, onSelect, isDict = false) {
    easy_RemoveDropdown();
    new Dropdown(inputEl, suggestions, onSelect, isDict);
}