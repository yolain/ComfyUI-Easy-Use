import {app} from "../../../../scripts/app.js";
import {api} from "../../../../scripts/api.js";
import {$el} from "../../../../scripts/ui.js";

const propmts = ["easy wildcards", "easy positive", "easy negative", "easy stylesSelector", "easy promptConcat", "easy promptReplace"]
const loaders = ["easy a1111Loader", "easy comfyLoader", "easy fullLoader", "easy svdLoader", "easy cascadeLoader", "easy sv3dLoader"]
const preSamplingNodes = ["easy preSampling", "easy preSamplingAdvanced", "easy preSamplingNoiseIn", "easy preSamplingCustom", "easy preSamplingDynamicCFG","easy preSamplingSdTurbo", "easy preSamplingLayerDiffusion"]
const kSampler = ["easy kSampler", "easy kSamplerTiled","easy kSamplerInpainting", "easy kSamplerDownscaleUnet", "easy kSamplerSDTurbo"]
const controlNetNodes = ["easy controlnetLoader", "easy controlnetLoaderADV"]
const instantIDNodes = ["easy instantIDApply", "easy instantIDApplyADV"]
const ipadapterNodes = ["easy ipadapterApply", "easy ipadapterApplyADV" ,"easy ipadapterApplyFaceIDKolors", "easy ipadapterStyleComposition"]
const pipeNodes = ['easy pipeIn','easy pipeOut', 'easy pipeEdit']
const xyNodes = ['easy XYPlot', 'easy XYPlotAdvanced']
const extraNodes = ['easy setNode']
const modelNormalNodes = [...["Reroute"],...['RescaleCFG','LoraLoaderModelOnly','LoraLoader','FreeU','FreeU_v2'],...ipadapterNodes,...extraNodes]
const suggestions = {
    // prompt
    "easy seed":{
        "from":{
            "INT": [...["Reroute"],...preSamplingNodes,...['easy fullkSampler']]
        }
    },
    "easy positive":{
       "from":{
           "STRING": [...["Reroute"],...propmts]
       }
    },
    "easy negative":{
       "from":{
           "STRING": [...["Reroute"],...propmts]
       }
    },
    "easy wildcards":{
        "from":{
           "STRING": [...["Reroute","easy showAnything"],...propmts,]
       }
    },
    "easy stylesSelector":{
        "from":{
           "STRING": [...["Reroute","easy showAnything"],...propmts,]
       }
    },
    "easy promptConcat":{
        "from":{
           "STRING": [...["Reroute","easy showAnything"],...propmts,]
       }
    },
    "easy promptReplace":{
        "from":{
           "STRING": [...["Reroute","easy showAnything"],...propmts,]
       }
    },
    // sd相关
    "easy fullLoader": {
        "from":{
            "PIPE_LINE": [...["Reroute"],...preSamplingNodes,...['easy fullkSampler'],...pipeNodes,...extraNodes],
            "MODEL":modelNormalNodes
        },
        "to":{
            "STRING": [...["Reroute"],...propmts]
        }
    },
    "easy a1111Loader": {
        "from": {
            "PIPE_LINE": [...["Reroute"], ...preSamplingNodes, ...controlNetNodes, ...instantIDNodes, ...pipeNodes, ...extraNodes],
            "MODEL": modelNormalNodes
        },
        "to":{
            "STRING": [...["Reroute"],...propmts]
        }
    },
    "easy comfyLoader": {
         "from": {
             "PIPE_LINE": [...["Reroute"], ...preSamplingNodes, ...controlNetNodes, ...instantIDNodes, ...pipeNodes, ...extraNodes],
             "MODEL": modelNormalNodes
         },
        "to":{
            "STRING": [...["Reroute"],...propmts]
        }
    },
    "easy svdLoader":{
         "from": {
             "PIPE_LINE": [...["Reroute"], ...["easy preSampling", "easy preSamplingAdvanced", "easy preSamplingDynamicCFG"], ...pipeNodes, ...extraNodes],
             "MODEL": modelNormalNodes
         },
         "to":{
            "STRING": [...["Reroute"],...propmts]
         }
    },
    "easy zero123Loader":{
         "from": {
             "PIPE_LINE": [...["Reroute"], ...["easy preSampling", "easy preSamplingAdvanced", "easy preSamplingDynamicCFG"], ...pipeNodes, ...extraNodes],
             "MODEL": modelNormalNodes
         },
         "to":{
            "STRING": [...["Reroute"],...propmts]
         }
    },
    "easy sv3dLoader":{
         "from": {
             "PIPE_LINE": [...["Reroute"], ...["easy preSampling", "easy preSamplingAdvanced", "easy preSamplingDynamicCFG"], ...pipeNodes, ...extraNodes],
             "MODEL": modelNormalNodes
         },
         "to":{
            "STRING": [...["Reroute"],...propmts]
         }
    },
    "easy preSampling": {
         "from": {
             "PIPE_LINE": [...["Reroute"], ...kSampler, ...pipeNodes, ...controlNetNodes, ...xyNodes, ...extraNodes]
         },
    },
    "easy preSamplingAdvanced": {
         "from": {
             "PIPE_LINE": [...["Reroute"], ...kSampler, ...pipeNodes, ...controlNetNodes, ...xyNodes, ...extraNodes]
         }
    },
    "easy preSamplingDynamicCFG": {
         "from": {
             "PIPE_LINE": [...["Reroute"], ...kSampler, ...pipeNodes, ...controlNetNodes, ...xyNodes, ...extraNodes]
         }
    },
     "easy preSamplingCustom": {
         "from": {
             "PIPE_LINE": [...["Reroute"], ...kSampler, ...pipeNodes, ...controlNetNodes, ...xyNodes, ...extraNodes]
         }
    },
    "easy preSamplingLayerDiffusion": {
         "from": {
             "PIPE_LINE": [...["Reroute", "easy kSamplerLayerDiffusion"], ...kSampler, ...pipeNodes, ...controlNetNodes, ...xyNodes, ...extraNodes]
         }
    },
    "easy preSamplingNoiseIn": {
         "from": {
             "PIPE_LINE": [...["Reroute"], ...kSampler, ...pipeNodes, ...controlNetNodes, ...xyNodes, ...extraNodes]
         }
    },
    // ksampler
    "easy fullkSampler": {
         "from": {
             "PIPE_LINE": [...["Reroute"], ...pipeNodes.reverse(), ...['easy preDetailerFix', 'easy preMaskDetailerFix'], ...preSamplingNodes, ...extraNodes]
         }
    },
    "easy kSampler": {
         "from": {
             "PIPE_LINE": [...["Reroute"], ...pipeNodes.reverse(), ...['easy preDetailerFix', 'easy preMaskDetailerFix', 'easy hiresFix'], ...preSamplingNodes, ...extraNodes],
         }
    },
    // cn
    "easy controlnetLoader": {
         "from": {
             "PIPE_LINE": [...["Reroute"], ...preSamplingNodes, ...controlNetNodes, ...instantIDNodes, ...pipeNodes, ...extraNodes]
         }
    },
    "easy controlnetLoaderADV":{
         "from": {
             "PIPE_LINE": [...["Reroute"], ...preSamplingNodes, ...controlNetNodes, ...instantIDNodes, ...pipeNodes, ...extraNodes]
         }
    },
    // instant
    "easy instantIDApply": {
        "from": {
            "PIPE_LINE": [...["Reroute"], ...preSamplingNodes, ...controlNetNodes, ...instantIDNodes, ...pipeNodes, ...extraNodes],
             "MODEL": modelNormalNodes
        },
        "to":{
            "COMBO": [...["Reroute", "easy promptLine"]]
        }
    },
    "easy instantIDApplyADV":{
        "from": {
            "PIPE_LINE": [...["Reroute"], ...preSamplingNodes, ...controlNetNodes, ...instantIDNodes, ...pipeNodes, ...extraNodes],
            "MODEL": modelNormalNodes
        },
        "to":{
            "COMBO": [...["Reroute", "easy promptLine"]]
        }
    },
    "easy ipadapterApply":{
        "to":{
            "COMBO": [...["Reroute", "easy promptLine"]]
        }
    },
    "easy ipadapterApplyADV":{
        "to":{
          "STRING": [...["Reroute", "easy sliderControl"], ...propmts],
          "COMBO": [...["Reroute", "easy promptLine"]]
        }
    },
    "easy ipadapterStyleComposition":{
        "to":{
            "COMBO": [...["Reroute", "easy promptLine"]]
        }
    },
    // fix
    "easy preDetailerFix":{
        "from": {
            "PIPE_LINE": [...["Reroute", "easy detailerFix"], ...pipeNodes, ...extraNodes]
        },
        "to":{
            "PIPE_LINE": [...["Reroute", "easy ultralyticsDetectorPipe", "easy samLoaderPipe", "easy kSampler", "easy fullkSampler"]]
        }
    },
    "easy preMaskDetailerFix":{
        "from": {
            "PIPE_LINE": [...["Reroute", "easy detailerFix"], ...pipeNodes, ...extraNodes]
        }
    },
    "easy samLoaderPipe": {
        "from":{
            "PIPE_LINE": [...["Reroute", "easy preDetailerFix"], ...pipeNodes, ...extraNodes]
        }
    },
    "easy ultralyticsDetectorPipe": {
        "from":{
            "PIPE_LINE": [...["Reroute", "easy preDetailerFix"], ...pipeNodes, ...extraNodes]
        }
    },
    // cascade相关
    "easy cascadeLoader":{
         "from": {
             "PIPE_LINE": [...["Reroute"], ...["easy fullCascadeKSampler", 'easy preSamplingCascade'], ...controlNetNodes, ...pipeNodes, ...extraNodes],
             "MODEL": modelNormalNodes.filter(cate => !ipadapterNodes.includes(cate))
         }
    },
    "easy fullCascadeKSampler":{
         "from": {
             "PIPE_LINE": [...["Reroute"], ...["easy preSampling", "easy preSamplingAdvanced"], ...pipeNodes, ...extraNodes]
         }
    },
    "easy preSamplingCascade":{
         "from": {
             "PIPE_LINE": [...["Reroute"], ...["easy cascadeKSampler",], ...pipeNodes, ...extraNodes]
         }
    },
    "easy cascadeKSampler": {
         "from": {
             "PIPE_LINE": [...["Reroute"], ...["easy preSampling", "easy preSamplingAdvanced"], ...pipeNodes, ...extraNodes]
         }
    },
}


app.registerExtension({
    name: "comfy.easyuse.suggestions",
    async setup(app) {
        LGraphCanvas.prototype.createDefaultNodeForSlot = function(optPass) { // addNodeMenu for connection
            var optPass = optPass || {};
            var opts = Object.assign({   nodeFrom: null // input
                                        ,slotFrom: null // input
                                        ,nodeTo: null   // output
                                        ,slotTo: null   // output
                                        ,position: []	// pass the event coords
                                        ,nodeType: null	// choose a nodetype to add, AUTO to set at first good
                                        ,posAdd:[0,0]	// adjust x,y
                                        ,posSizeFix:[0,0] // alpha, adjust the position x,y based on the new node size w,h
                                    }
                                    ,optPass
                                );
            var that = this;

            var isFrom = opts.nodeFrom && opts.slotFrom!==null;
            var isTo = !isFrom && opts.nodeTo && opts.slotTo!==null;

            if (!isFrom && !isTo){
                console.warn("No data passed to createDefaultNodeForSlot "+opts.nodeFrom+" "+opts.slotFrom+" "+opts.nodeTo+" "+opts.slotTo);
                return false;
            }
            if (!opts.nodeType){
                console.warn("No type to createDefaultNodeForSlot");
                return false;
            }

            var nodeX = isFrom ? opts.nodeFrom : opts.nodeTo;
            var slotX = isFrom ? opts.slotFrom : opts.slotTo;
            var nodeType = nodeX.type

            var iSlotConn = false;
            switch (typeof slotX){
                case "string":
                    iSlotConn = isFrom ? nodeX.findOutputSlot(slotX,false) : nodeX.findInputSlot(slotX,false);
                    slotX = isFrom ? nodeX.outputs[slotX] : nodeX.inputs[slotX];
                break;
                case "object":
                    // ok slotX
                    iSlotConn = isFrom ? nodeX.findOutputSlot(slotX.name) : nodeX.findInputSlot(slotX.name);
                break;
                case "number":
                    iSlotConn = slotX;
                    slotX = isFrom ? nodeX.outputs[slotX] : nodeX.inputs[slotX];
                break;
                case "undefined":
                default:
                    // bad ?
                    //iSlotConn = 0;
                    console.warn("Cant get slot information "+slotX);
                    return false;
            }

            if (slotX===false || iSlotConn===false){
                console.warn("createDefaultNodeForSlot bad slotX "+slotX+" "+iSlotConn);
            }

            // check for defaults nodes for this slottype
            var fromSlotType = slotX.type==LiteGraph.EVENT?"_event_":slotX.type;
            var slotTypesDefault = isFrom ? LiteGraph.slot_types_default_out : LiteGraph.slot_types_default_in;
            if(slotTypesDefault && slotTypesDefault[fromSlotType]){
                if (slotX.link !== null) {
                    // is connected
                }else{
                    // is not not connected
                }
                let nodeNewType = false;
                const fromOrTo = isFrom ? 'from' : 'to'
                if(suggestions[nodeType] && suggestions[nodeType][fromOrTo] && suggestions[nodeType][fromOrTo][fromSlotType]?.length>0){
                    for(var typeX in suggestions[nodeType][fromOrTo][fromSlotType]){
                        if (opts.nodeType == suggestions[nodeType][fromOrTo][fromSlotType][typeX] || opts.nodeType == "AUTO") {
                            nodeNewType = suggestions[nodeType][fromOrTo][fromSlotType][typeX];
                            break
                        }
                    }
                }
                else if(typeof slotTypesDefault[fromSlotType] == "object" || typeof slotTypesDefault[fromSlotType] == "array"){
                    for(var typeX in slotTypesDefault[fromSlotType]){
                        if (opts.nodeType == slotTypesDefault[fromSlotType][typeX] || opts.nodeType == "AUTO"){
                            nodeNewType = slotTypesDefault[fromSlotType][typeX];
                            // console.log("opts.nodeType == slotTypesDefault[fromSlotType][typeX] :: "+opts.nodeType);
                            break; // --------
                        }
                    }
                }else{
                    if (opts.nodeType == slotTypesDefault[fromSlotType] || opts.nodeType == "AUTO") nodeNewType = slotTypesDefault[fromSlotType];
                }
                if (nodeNewType) {
                    var nodeNewOpts = false;
                    if (typeof nodeNewType == "object" && nodeNewType.node){
                        nodeNewOpts = nodeNewType;
                        nodeNewType = nodeNewType.node;
                    }

                    //that.graph.beforeChange();

                    var newNode = LiteGraph.createNode(nodeNewType);
                    if(newNode){
                        // if is object pass options
                        if (nodeNewOpts){
                            if (nodeNewOpts.properties) {
                                for (var i in nodeNewOpts.properties) {
                                    newNode.addProperty( i, nodeNewOpts.properties[i] );
                                }
                            }
                            if (nodeNewOpts.inputs) {
                                newNode.inputs = [];
                                for (var i in nodeNewOpts.inputs) {
                                    newNode.addOutput(
                                        nodeNewOpts.inputs[i][0],
                                        nodeNewOpts.inputs[i][1]
                                    );
                                }
                            }
                            if (nodeNewOpts.outputs) {
                                newNode.outputs = [];
                                for (var i in nodeNewOpts.outputs) {
                                    newNode.addOutput(
                                        nodeNewOpts.outputs[i][0],
                                        nodeNewOpts.outputs[i][1]
                                    );
                                }
                            }
                            if (nodeNewOpts.title) {
                                newNode.title = nodeNewOpts.title;
                            }
                            if (nodeNewOpts.json) {
                                newNode.configure(nodeNewOpts.json);
                            }

                        }

                        // add the node
                        that.graph.add(newNode);
                        newNode.pos = [	opts.position[0]+opts.posAdd[0]+(opts.posSizeFix[0]?opts.posSizeFix[0]*newNode.size[0]:0)
                                        ,opts.position[1]+opts.posAdd[1]+(opts.posSizeFix[1]?opts.posSizeFix[1]*newNode.size[1]:0)]; //that.last_click_position; //[e.canvasX+30, e.canvasX+5];*/

                        //that.graph.afterChange();

                        // connect the two!
                        if (isFrom){
                            opts.nodeFrom.connectByType( iSlotConn, newNode, fromSlotType );
                        }else{
                            opts.nodeTo.connectByTypeOutput( iSlotConn, newNode, fromSlotType );
                        }

                        // if connecting in between
                        if (isFrom && isTo){
                            // TODO
                        }

                        return true;

                    }else{
                        console.log("failed creating "+nodeNewType);
                    }
                }
            }
            return false;
        }

        LGraphCanvas.prototype.showConnectionMenu = function(optPass) { // addNodeMenu for connection
            var optPass = optPass || {};
            var opts = Object.assign({   nodeFrom: null  // input
                                        ,slotFrom: null // input
                                        ,nodeTo: null   // output
                                        ,slotTo: null   // output
                                        ,e: null
                                    }
                                    ,optPass
                                );
            var that = this;

            var isFrom = opts.nodeFrom && opts.slotFrom;
            var isTo = !isFrom && opts.nodeTo && opts.slotTo;

            if (!isFrom && !isTo){
                console.warn("No data passed to showConnectionMenu");
                return false;
            }

            var nodeX = isFrom ? opts.nodeFrom : opts.nodeTo;
            var slotX = isFrom ? opts.slotFrom : opts.slotTo;

            var iSlotConn = false;
            switch (typeof slotX){
                case "string":
                    iSlotConn = isFrom ? nodeX.findOutputSlot(slotX,false) : nodeX.findInputSlot(slotX,false);
                    slotX = isFrom ? nodeX.outputs[slotX] : nodeX.inputs[slotX];
                break;
                case "object":
                    // ok slotX
                    iSlotConn = isFrom ? nodeX.findOutputSlot(slotX.name) : nodeX.findInputSlot(slotX.name);
                break;
                case "number":
                    iSlotConn = slotX;
                    slotX = isFrom ? nodeX.outputs[slotX] : nodeX.inputs[slotX];
                break;
                default:
                    // bad ?
                    //iSlotConn = 0;
                    console.warn("Cant get slot information "+slotX);
                    return false;
            }

            var options = ["Add Node",null];

            if (that.allow_searchbox){
                options.push("Search");
                options.push(null);
            }

            // get defaults nodes for this slottype
            var fromSlotType = slotX.type==LiteGraph.EVENT?"_event_":slotX.type;
            var slotTypesDefault = isFrom ? LiteGraph.slot_types_default_out : LiteGraph.slot_types_default_in;
            var nodeType = nodeX.type
            if(slotTypesDefault && slotTypesDefault[fromSlotType]){
                const fromOrTo = isFrom ? 'from' : 'to'
                if(suggestions[nodeType] && suggestions[nodeType][fromOrTo] && suggestions[nodeType][fromOrTo][fromSlotType]?.length>0){
                    for(var typeX in suggestions[nodeType][fromOrTo][fromSlotType]){
                        options.push(suggestions[nodeType][fromOrTo][fromSlotType][typeX]);
                    }
                }
                else if(typeof slotTypesDefault[fromSlotType] == "object" || typeof slotTypesDefault[fromSlotType] == "array"){
                    for(var typeX in slotTypesDefault[fromSlotType]){
                        options.push(slotTypesDefault[fromSlotType][typeX]);
                    }
                }else{
                    options.push(slotTypesDefault[fromSlotType]);
                }
            }

            // build menu
            var menu = new LiteGraph.ContextMenu(options, {
                event: opts.e,
                title: (slotX && slotX.name!="" ? (slotX.name + (fromSlotType?" | ":"")) : "")+(slotX && fromSlotType ? fromSlotType : ""),
                callback: inner_clicked
            });

            // callback
            function inner_clicked(v,options,e) {
                //console.log("Process showConnectionMenu selection");
                switch (v) {
                    case "Add Node":
                        LGraphCanvas.onMenuAdd(null, null, e, menu, function(node){
                            if (isFrom){
                                opts.nodeFrom.connectByType( iSlotConn, node, fromSlotType );
                            }else{
                                opts.nodeTo.connectByTypeOutput( iSlotConn, node, fromSlotType );
                            }
                        });
                        break;
                    case "Search":
                        if(isFrom){
                            that.showSearchBox(e,{node_from: opts.nodeFrom, slot_from: slotX, type_filter_in: fromSlotType});
                        }else{
                            that.showSearchBox(e,{node_to: opts.nodeTo, slot_from: slotX, type_filter_out: fromSlotType});
                        }
                        break;
                    default:
                        // check for defaults nodes for this slottype
                        var nodeCreated = that.createDefaultNodeForSlot(Object.assign(opts,{ position: [opts.e.canvasX, opts.e.canvasY]
                                                                                            ,nodeType: v
                                                                                        }));
                        if (nodeCreated){
                            // new node created
                            //console.log("node "+v+" created")
                        }else{
                            // failed or v is not in defaults
                        }
                        break;
                }
            }

            return false;
        };
    }
})