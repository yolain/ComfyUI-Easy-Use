// 1.0.2
import { app } from "/scripts/app.js";
import { GroupNodeConfig } from "/extensions/core/groupNode.js";

const nodeTemplateShortcutId = "Comfy.EasyUse.NodeTemplateShortcut"

let enableNodeTemplateShortcut = true
export function addNodeTemplateShortcutSetting(app) {
	app.ui.settings.addSetting({
		id: nodeTemplateShortcutId,
		name: "Enable ALT+1~9 to paste nodes from nodes template (ComfyUI-Easy-Use)",
		type: "boolean",
		defaultValue: enableNodeTemplateShortcut,
		onChange(value) {
			enableNodeTemplateShortcut = !!value;
		},
	});
}
const getEnableNodeTemplateShortcut = _ => app.ui.settings.getSettingValue(nodeTemplateShortcutId, true)

function loadTemplate(){
    return localStorage['Comfy.NodeTemplates'] ? JSON.parse(localStorage['Comfy.NodeTemplates']) : null
}
const clipboardAction = async (cb) => {
    const old = localStorage.getItem("litegrapheditor_clipboard");
    await cb();
    localStorage.setItem("litegrapheditor_clipboard", old);
};
async function addTemplateToCanvas(t){
    const data = JSON.parse(t.data);
    await GroupNodeConfig.registerFromWorkflow(data.groupNodes, {});
    localStorage.setItem("litegrapheditor_clipboard", t.data);
    app.canvas.pasteFromClipboard();
}

app.registerExtension({
	name: 'comfy.easyUse.quick',
	init() {
        const keybindListener = async function (event) {
			const modifierPressed = event.altKey;
            const isEnabled = getEnableNodeTemplateShortcut()
            if(isEnabled){
                const mac_alt_nums = ['¡','™','£','¢','∞','§','¶','•','ª']
                const nums = ['1','2','3','4','5','6','7','8','9']
                let key = event.key
                if(mac_alt_nums.includes(key)){
                    const idx = mac_alt_nums.findIndex(key)
                    key = nums[idx]
                }
                if(['1','2','3','4','5','6','7','8','9'].includes(key) && modifierPressed) {
                    const template = loadTemplate()
                    const idx = parseInt(key) - 1
                    if (template && template[idx]) {
                        let t = template[idx]
                        try{
                          let data = JSON.parse(t.data)
                          data.title = t.name
                          t.data = JSON.stringify(data)
                          clipboardAction(_ => {
                            addTemplateToCanvas(t)
                          })
                        }catch (e){
                            console.error(e)
                        }

                    }
                    if (event.ctrlKey || event.altKey || event.metaKey) {
                        return;
                    }
                }
            }

        }
        window.addEventListener("keydown", keybindListener, true);
    },

    setup(app) {
        addNodeTemplateShortcutSetting(app)
    }
});


// 修改粘贴指令
LGraphCanvas.prototype.pasteFromClipboard = function(isConnectUnselected = false) {
    // if ctrl + shift + v is off, return when isConnectUnselected is true (shift is pressed) to maintain old behavior
    if (!LiteGraph.ctrl_shift_v_paste_connect_unselected_outputs && isConnectUnselected) {
        return;
    }
    var data = localStorage.getItem("litegrapheditor_clipboard");
    if (!data) {
        return;
    }

    this.graph.beforeChange();

    //create nodes
    var clipboard_info = JSON.parse(data);
    // calculate top-left node, could work without this processing but using diff with last node pos :: clipboard_info.nodes[clipboard_info.nodes.length-1].pos
    var posMin = false;
    var posMinIndexes = false;
    for (var i = 0; i < clipboard_info.nodes.length; ++i) {
        if (posMin){
            if(posMin[0]>clipboard_info.nodes[i].pos[0]){
                posMin[0] = clipboard_info.nodes[i].pos[0];
                posMinIndexes[0] = i;
            }
            if(posMin[1]>clipboard_info.nodes[i].pos[1]){
                posMin[1] = clipboard_info.nodes[i].pos[1];
                posMinIndexes[1] = i;
            }
        }
        else{
            posMin = [clipboard_info.nodes[i].pos[0], clipboard_info.nodes[i].pos[1]];
            posMinIndexes = [i, i];
        }
    }
    var nodes = [];
    var left_arr = [], right_arr = [], top_arr =[], bottom_arr =[];

    for (var i = 0; i < clipboard_info.nodes.length; ++i) {
        var node_data = clipboard_info.nodes[i];
        var node = LiteGraph.createNode(node_data.type);
        if (node) {

            node.configure(node_data);
            //paste in last known mouse position
            node.pos[0] += this.graph_mouse[0] - posMin[0]; //+= 5;
            node.pos[1] += this.graph_mouse[1] - posMin[1]; //+= 5;

            left_arr.push(node.pos[0])
            right_arr.push(node.pos[0] + node.size[0])
            top_arr.push(node.pos[1])
            bottom_arr.push(node.pos[1] + node.size[1])

            this.graph.add(node,{doProcessChange:false});

            nodes.push(node);

        }
    }

    if(clipboard_info.title){
        var l = Math.min(...left_arr) - 15;
        var r = Math.max(...right_arr) - this.graph_mouse[0] + 30;
        var t = Math.min(...top_arr) - 100;
        var b = Math.max(...bottom_arr) - this.graph_mouse[1] + 130;

        // create group
        const groups = [
            {
              "title": clipboard_info.title,
              "bounding": [
                l,
                t,
                r,
                b
              ],
              "color": "#3f789e",
              "font_size": 24,
              "locked": false
            }
        ]

        for (var i = 0; i < groups.length; ++i) {
            var group = new LiteGraph.LGraphGroup();
            group.configure(groups[i]);
            this.graph.add(group);
        }
    }

    //create links
    for (var i = 0; i < clipboard_info.links.length; ++i) {
        var link_info = clipboard_info.links[i];
        var origin_node;
        var origin_node_relative_id = link_info[0];
        if (origin_node_relative_id != null) {
            origin_node = nodes[origin_node_relative_id];
        } else if (LiteGraph.ctrl_shift_v_paste_connect_unselected_outputs && isConnectUnselected) {
            var origin_node_id = link_info[4];
            if (origin_node_id) {
                origin_node = this.graph.getNodeById(origin_node_id);
            }
        }
        var target_node = nodes[link_info[2]];
        if( origin_node && target_node )
            origin_node.connect(link_info[1], target_node, link_info[3]);
        else
            console.warn("Warning, nodes missing on pasting");
    }

    this.selectNodes(nodes);
    this.graph.afterChange();
};