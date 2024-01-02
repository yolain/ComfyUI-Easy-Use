import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { $el } from "/scripts/ui.js";

const customThemeColor = "#3f3eed"
const customThemeColorLight = "#006691"
// 增加Slot颜色
const customPipeLineLink = "#7737AA"
const customPipeLineSDXLLink = "#7737AA"
const customIntLink = "#29699C"
const customXYPlotLink = "#74DA5D"
const customXYLink = "#74DA5D"

var customLinkColors = JSON.parse(localStorage.getItem('Comfy.Settings.ttN.customLinkColors')) || {};
if (!customLinkColors["PIPE_LINE"] || !LGraphCanvas.link_type_colors["PIPE_LINE"]) {customLinkColors["PIPE_LINE"] = customPipeLineLink;}
if (!customLinkColors["PIPE_LINE_SDXL"] || !LGraphCanvas.link_type_colors["PIPE_LINE_SDXL"]) {customLinkColors["PIPE_LINE_SDXL"] = customPipeLineSDXLLink;}
if (!customLinkColors["INT"] || !LGraphCanvas.link_type_colors["INT"]) {customLinkColors["INT"] = customIntLink;}
if (!customLinkColors["XYPLOT"] || !LGraphCanvas.link_type_colors["XYPLOT"]) {customLinkColors["XYPLOT"] = customXYPlotLink;}
if (!customLinkColors["XY"] || !LGraphCanvas.link_type_colors["XY"]) {customLinkColors["XY"] = customXYLink;}

localStorage.setItem('Comfy.Settings.easyUse.customLinkColors', JSON.stringify(customLinkColors));

// 增加自定义主题
const ui = {
  "version": 100,
  "id": "obsidian",
  "name": "黑曜石 by乱乱呀",
  "colors": {
      "node_slot": {
          "CLIP": "#FFD500",
          "CLIP_VISION": "#A8DADC",
          "CLIP_VISION_OUTPUT": "#ad7452",
          "CONDITIONING": "#FFA931",
          "CONTROL_NET": "#6EE7B7",
          "IMAGE": "#64B5F6",
          "LATENT": "#FF9CF9",
          "MASK": "#81C784",
          "MODEL": "#B39DDB",
          "STYLE_MODEL": "#C2FFAE",
          "VAE": "#FF6E6E",
          "TAESD": "#DCC274"
      },
      "litegraph_base": {
          "BACKGROUND_IMAGE": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAIAAAD/gAIDAAAACXBIWXMAAAsTAAALEwEAmpwYAAAGlmlUWHRYTUw6Y29tLmFkb2JlLnhtcAAAAAAAPD94cGFja2V0IGJlZ2luPSLvu78iIGlkPSJXNU0wTXBDZWhpSHpyZVN6TlRjemtjOWQiPz4gPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyIgeDp4bXB0az0iQWRvYmUgWE1QIENvcmUgOS4xLWMwMDEgNzkuMTQ2Mjg5OSwgMjAyMy8wNi8yNS0yMDowMTo1NSAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvIiB4bWxuczpkYz0iaHR0cDovL3B1cmwub3JnL2RjL2VsZW1lbnRzLzEuMS8iIHhtbG5zOnBob3Rvc2hvcD0iaHR0cDovL25zLmFkb2JlLmNvbS9waG90b3Nob3AvMS4wLyIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0RXZ0PSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VFdmVudCMiIHhtcDpDcmVhdG9yVG9vbD0iQWRvYmUgUGhvdG9zaG9wIDI1LjEgKFdpbmRvd3MpIiB4bXA6Q3JlYXRlRGF0ZT0iMjAyMy0xMS0xM1QwMDoxODowMiswMTowMCIgeG1wOk1vZGlmeURhdGU9IjIwMjMtMTEtMTVUMDI6MDQ6NTkrMDE6MDAiIHhtcDpNZXRhZGF0YURhdGU9IjIwMjMtMTEtMTVUMDI6MDQ6NTkrMDE6MDAiIGRjOmZvcm1hdD0iaW1hZ2UvcG5nIiBwaG90b3Nob3A6Q29sb3JNb2RlPSIzIiB4bXBNTTpJbnN0YW5jZUlEPSJ4bXAuaWlkOmIyYzRhNjA5LWJmYTctYTg0MC1iOGFlLTk3MzE2ZjM1ZGIyNyIgeG1wTU06RG9jdW1lbnRJRD0iYWRvYmU6ZG9jaWQ6cGhvdG9zaG9wOjk0ZmNlZGU4LTE1MTctZmQ0MC04ZGU3LWYzOTgxM2E3ODk5ZiIgeG1wTU06T3JpZ2luYWxEb2N1bWVudElEPSJ4bXAuZGlkOjIzMWIxMGIwLWI0ZmItMDI0ZS1iMTJlLTMwNTMwM2NkMDdjOCI+IDx4bXBNTTpIaXN0b3J5PiA8cmRmOlNlcT4gPHJkZjpsaSBzdEV2dDphY3Rpb249ImNyZWF0ZWQiIHN0RXZ0Omluc3RhbmNlSUQ9InhtcC5paWQ6MjMxYjEwYjAtYjRmYi0wMjRlLWIxMmUtMzA1MzAzY2QwN2M4IiBzdEV2dDp3aGVuPSIyMDIzLTExLTEzVDAwOjE4OjAyKzAxOjAwIiBzdEV2dDpzb2Z0d2FyZUFnZW50PSJBZG9iZSBQaG90b3Nob3AgMjUuMSAoV2luZG93cykiLz4gPHJkZjpsaSBzdEV2dDphY3Rpb249InNhdmVkIiBzdEV2dDppbnN0YW5jZUlEPSJ4bXAuaWlkOjQ4OWY1NzlmLTJkNjUtZWQ0Zi04OTg0LTA4NGE2MGE1ZTMzNSIgc3RFdnQ6d2hlbj0iMjAyMy0xMS0xNVQwMjowNDo1OSswMTowMCIgc3RFdnQ6c29mdHdhcmVBZ2VudD0iQWRvYmUgUGhvdG9zaG9wIDI1LjEgKFdpbmRvd3MpIiBzdEV2dDpjaGFuZ2VkPSIvIi8+IDxyZGY6bGkgc3RFdnQ6YWN0aW9uPSJzYXZlZCIgc3RFdnQ6aW5zdGFuY2VJRD0ieG1wLmlpZDpiMmM0YTYwOS1iZmE3LWE4NDAtYjhhZS05NzMxNmYzNWRiMjciIHN0RXZ0OndoZW49IjIwMjMtMTEtMTVUMDI6MDQ6NTkrMDE6MDAiIHN0RXZ0OnNvZnR3YXJlQWdlbnQ9IkFkb2JlIFBob3Rvc2hvcCAyNS4xIChXaW5kb3dzKSIgc3RFdnQ6Y2hhbmdlZD0iLyIvPiA8L3JkZjpTZXE+IDwveG1wTU06SGlzdG9yeT4gPC9yZGY6RGVzY3JpcHRpb24+IDwvcmRmOlJERj4gPC94OnhtcG1ldGE+IDw/eHBhY2tldCBlbmQ9InIiPz4OTe6GAAAAx0lEQVR42u3WMQoAIQxFwRzJys77X8vSLiRgITif7bYbgrwYc/mKXyBoY4VVBgsWLFiwYFmOlTv+9jfDOjHmr8u6eVkGCxYsWLBgmc5S8ApewXvgYRksWLBgKXidpeBdloL3wMOCBctgwVLwCl7BuyyDBQsWLFiwTGcpeAWv4D3wsAwWLFiwFLzOUvAuS8F74GHBgmWwYCl4Ba/gXZbBggULFixYprMUvIJX8B54WAYLFixYCl5nKXiXpeA98LBgwTJYsGC9tg1o8f4TTtqzNQAAAABJRU5ErkJggg==",
          // "BACKGROUND_IMAGE": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABjCAAAAABIjPowAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAAmJLR0QA/4ePzL8AAAQjSURBVGje7VpdTxwxDPQE2lL1har9/7+QtyK1VZGmD5vEE2c/vHBCJyBC6G43sdfjj+TGix+g1QG2fz4+P36+sb0B+/V1vEADzWBN0N1DMZoZYGbLVU5CjgaAsIJmRvrqYmZm5IYsTErnQdLM0B6HumL5XLCoW5MFMyYsaeK2H6dUGxCQ6YqPLdnGsSupj8Bwj4ZZ80kFHYbiQlpcUeZlfCJP7cKooooLCc4nd3EW+YhCaxA4XPAvqyF2DNccmgzBVNQZPWWgIo6RIkxiuK1yBAvcAgL1ubrZLU93LbEaOhxV97t2q6E1ADtVmHNDdZY9FAaVOxMnz2HALipRWEOI7KqhqkJ0ZFAiVa16MlG7jBVaOgIVaoxKJMFbUtbClxob6bYEWZGvobzE1dkRdiUOcC0uwPD9WZWr5gB67oiSHsBwaDOlfowPzzIPzjIIgfuv23uIl05vBbWnt5mNZWW9hpxBDNTa3Y0qoeiMwpGrXWoUR7yaT9ijDj2SzWdnalf8GvaVotgRLZJTzkjo1WQETSNvA4xjuDa0FUMohacNoBmmbWhQX6QmP/PMYGZc4sPPifU00g4SIn2ou7rpnIAL1E2pHSQ8mjFIzKMmXl5q/oRImVN7FJ/eHmE01v/Bxg6XPoCmzUm4qDW2O/vLvoT733/TuG2O259HM+5vX67k38F9Pu1PAB+/7c/49FDsFcY7UvL8WnNCCS+g5diSlxyIs0ouMa5DyQXQupLousR4R3C9iiWXGNeh5COEr8+Sjzw5peTtwPV2LLnEeDtwvaEfQX86adgaQfXTYugne9oHDPZ4ZyunWb9w91ACDU4zjtxezinrtKKSBY2ghpL7W6vX5Dtvh0hNdSXtNju7dZJ/HLiAmU13NnV6fnbl+yP8jm+sCVx9meeN3FrGDGU5QRVvwgszrKp/OYYbYFiJ3iVB5IVb72Qy68iSidWv9AoZGW40v1AD+AyJvkGbcaTRXV9/skyerDSbBBizTj6HCKGTX8e/fimRMrByDUcr9HAL7YKsT1rf2Mnx2Eku5jlYWTcIZEmXQNRFKt1i14GVGu4BkmowK68sldGBLqpBcGokeg4ubbZ6ZaQogWvwktCnHnP1lNUTB9u5+sZvd7lVBYXnTbmmQRSXFNM4GBqSNUhObPJSFQeTlHyOC5wfP/KJQ0C9NjQEOOmBEYHk3UPJpsIwEP9FCnN/pNbhONECXFGseUIMb3vEcpXwCVY+mSZDWaSuV6hcFxsc3Yx4tyUj/LZITQfWfPTwN0x8PxHnDc1YpDRhvuARw8OzMFOeH9rSCMvizmi9NAjM2dc96hJ/vcdfuyleCuvRa3q/4NCO1loaWjyUMC3S4qVXyfkUeITW3hZadKLn5WqwJXStz5b9ZKM0pN6Paf884VTbdfwIuvl+kxK0N/4D5hWoRYiuxvAAAAAASUVORK5CYII=",
          "CLEAR_BACKGROUND_COLOR": "#101010",
          "NODE_TITLE_COLOR": "rgba(255,255,255,.75)",
          "NODE_SELECTED_TITLE_COLOR": "#FFF",
          "NODE_TEXT_SIZE": 14,
          "NODE_TEXT_COLOR": "#b8b8b8",
          "NODE_SUBTEXT_SIZE": 12,
          "NODE_DEFAULT_COLOR": "rgba(0,0,0,.8)",
          "NODE_DEFAULT_BGCOLOR": "rgba(22,22,22,.8)",
          "NODE_DEFAULT_BOXCOLOR": "rgba(255,255,255,.75)",
          "NODE_DEFAULT_SHAPE": "box",
          "NODE_BOX_OUTLINE_COLOR": customThemeColor,
          "DEFAULT_SHADOW_COLOR": "rgba(0,0,0,0)",
          "DEFAULT_GROUP_FONT": 24,

          "WIDGET_BGCOLOR": "#222",
          "WIDGET_OUTLINE_COLOR": "#333",
          "WIDGET_TEXT_COLOR": "#a3a3a8",
          "WIDGET_SECONDARY_TEXT_COLOR": "#97979c",

          "LINK_COLOR": "#9A9",
          "EVENT_LINK_COLOR": "#A86",
          "CONNECTING_LINK_COLOR": "#AFA"
      },
      "comfy_base": {
          "fg-color": "#fff",
          "bg-color": "#202020",
          "comfy-menu-bg": "rgba(19,19,19,.9)",
          "comfy-input-bg": "#222222",
          "input-text": "#ddd",
          "descrip-text": "#999",
          "drag-text": "#ccc",
          "error-text": "#ff4444",
          "border-color": "#29292c",
          "tr-even-bg-color": "rgba(22,22,22,.9)",
          "tr-odd-bg-color": "rgba(19,19,19,.9)"
      }
  }
}
try{
    // 修改自定义主题
    let custom_theme = localStorage.getItem('Comfy.Settings.Comfy.CustomColorPalettes') ? JSON.parse(localStorage.getItem('Comfy.Settings.Comfy.CustomColorPalettes')) : {};
    if(!custom_theme || !custom_theme.obsidian || !custom_theme.obsidian.version || custom_theme.obsidian.version<ui.version){
        custom_theme.obsidian = ui
        localStorage.setItem('Comfy.Settings.Comfy.CustomColorPalettes', JSON.stringify(custom_theme));
        localStorage.setItem('Comfy.Settings.Comfy.ColorPalette','"custom_obsidian"')
    }
    const theme_name = localStorage.getItem('Comfy.Settings.Comfy.ColorPalette')
    if(theme_name == '"custom_obsidian"'){
        // canvas
        const bgcolor = LGraphCanvas.node_colors.bgcolor;
        LGraphCanvas.node_colors = {
            red: { color: "#af3535", bgcolor, groupcolor: "#A88" },
            brown: { color: "#38291f", bgcolor, groupcolor: "#b06634" },
            green: { color: "#346434", bgcolor, groupcolor: "#8A8" },
            blue: { color: "#1f1f48", bgcolor, groupcolor: "#88A" },
            pale_blue: {color: "#006691", bgcolor, groupcolor: "#3f789e"},
            cyan: { color: "#008181", bgcolor, groupcolor: "#8AA" },
            purple: { color: "#422342", bgcolor, groupcolor: "#a1309b" },
            yellow: { color: "#c09430", bgcolor, groupcolor: "#b58b2a" },
            black: { color: "rgba(0,0,0,.8)", bgcolor, groupcolor: "#444" }
        };
        LiteGraph.NODE_TEXT_SIZE = 13
        LiteGraph.DEFAULT_BACKGROUND_IMAGE = ui.colors.litegraph_base.BACKGROUND_IMAGE
        LGraphCanvas.prototype.drawNodeShape = function(
            node,
            ctx,
            size,
            fgcolor,
            bgcolor,
            selected,
            mouse_over
        ) {
            //bg rect
            ctx.strokeStyle = fgcolor;
            ctx.fillStyle = bgcolor;

            var title_height = LiteGraph.NODE_TITLE_HEIGHT;
            var low_quality = this.ds.scale < 0.5;

            //render node area depending on shape
            var shape =
                node._shape || node.constructor.shape || LiteGraph.ROUND_SHAPE;

            var title_mode = node.constructor.title_mode;

            var render_title = true;
            if (title_mode == LiteGraph.TRANSPARENT_TITLE || title_mode == LiteGraph.NO_TITLE) {
                render_title = false;
            } else if (title_mode == LiteGraph.AUTOHIDE_TITLE && mouse_over) {
                render_title = true;
            }

            var area = new Float32Array(4);
            area[0] = 0; //x
            area[1] = render_title ? -title_height : 0; //y
            area[2] = size[0] + 1; //w
            area[3] = render_title ? size[1] + title_height : size[1]; //h

            var old_alpha = ctx.globalAlpha;

            //full node shape
            // if(node.flags.collapsed)
            {
                ctx.lineWidth = 1;
                ctx.beginPath();
                if (shape == LiteGraph.BOX_SHAPE || low_quality) {
                    ctx.fillRect(area[0], area[1], area[2], area[3]);
                } else if (
                    shape == LiteGraph.ROUND_SHAPE ||
                    shape == LiteGraph.CARD_SHAPE
                ) {
                    ctx.roundRect(
                        area[0],
                        area[1],
                        area[2],
                        area[3],
                        shape == LiteGraph.CARD_SHAPE ? [this.round_radius,this.round_radius,0,0] : [this.round_radius]
                    );
                } else if (shape == LiteGraph.CIRCLE_SHAPE) {
                    ctx.arc(
                        size[0] * 0.5,
                        size[1] * 0.5,
                        size[0] * 0.5,
                        0,
                        Math.PI * 2
                    );
                }
                ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
                ctx.stroke();
                ctx.strokeStyle = fgcolor;
                ctx.fill();

                //separator
                if(!node.flags.collapsed && render_title)
                {
                    ctx.shadowColor = "transparent";
                    ctx.fillStyle = "rgba(0,0,0,0.2)";
                    ctx.fillRect(0, -1, area[2], 2);
                }
            }
            ctx.shadowColor = "transparent";

            if (node.onDrawBackground) {
                node.onDrawBackground(ctx, this, this.canvas, this.graph_mouse );
            }

            //title bg (remember, it is rendered ABOVE the node)
            if (render_title || title_mode == LiteGraph.TRANSPARENT_TITLE) {
                //title bar
                if (node.onDrawTitleBar) {
                    node.onDrawTitleBar( ctx, title_height, size, this.ds.scale, fgcolor );
                } else if (
                    title_mode != LiteGraph.TRANSPARENT_TITLE &&
                    (node.constructor.title_color || this.render_title_colored)
                ) {
                    var title_color = node.constructor.title_color || fgcolor;

                    if (node.flags.collapsed) {
                        ctx.shadowColor = LiteGraph.DEFAULT_SHADOW_COLOR;
                    }

                    //* gradient test
                    if (this.use_gradients) {
                        var grad = LGraphCanvas.gradients[title_color];
                        if (!grad) {
                            grad = LGraphCanvas.gradients[ title_color ] = ctx.createLinearGradient(0, 0, 400, 0);
                            grad.addColorStop(0, title_color); // TODO refactor: validate color !! prevent DOMException
                            grad.addColorStop(1, "#000");
                        }
                        ctx.fillStyle = grad;
                    } else {
                        ctx.fillStyle = title_color;
                    }

                    //ctx.globalAlpha = 0.5 * old_alpha;
                    ctx.beginPath();
                    if (shape == LiteGraph.BOX_SHAPE || low_quality) {
                        ctx.rect(0, -title_height, size[0] + 1, title_height);
                    } else if (  shape == LiteGraph.ROUND_SHAPE || shape == LiteGraph.CARD_SHAPE ) {
                        ctx.roundRect(
                            0,
                            -title_height,
                            size[0] + 1,
                            title_height,
                            node.flags.collapsed ? [this.round_radius] : [this.round_radius,this.round_radius,0,0]
                        );
                    }
                    ctx.fill();
                    ctx.shadowColor = "transparent";
                }

                var colState = false;
                if (LiteGraph.node_box_coloured_by_mode){
                    if(LiteGraph.NODE_MODES_COLORS[node.mode]){
                        colState = LiteGraph.NODE_MODES_COLORS[node.mode];
                    }
                }
                if (LiteGraph.node_box_coloured_when_on){
                    colState = node.action_triggered ? "#FFF" : (node.execute_triggered ? "#AAA" : colState);
                }

                //title box
                var box_size = 10;
                if (node.onDrawTitleBox) {
                    node.onDrawTitleBox(ctx, title_height, size, this.ds.scale);
                } else if (
                    shape == LiteGraph.ROUND_SHAPE ||
                    shape == LiteGraph.CIRCLE_SHAPE ||
                    shape == LiteGraph.CARD_SHAPE
                ) {
                    if (low_quality) {
                        ctx.fillStyle = "black";
                        ctx.beginPath();
                        ctx.arc(
                            title_height * 0.5,
                            title_height * -0.5,
                            box_size * 0.5 + 1,
                            0,
                            Math.PI * 2
                        );
                        ctx.fill();
                    }

                    // BOX_TITLE_ICON
                    ctx.fillStyle = selected ? LiteGraph.NODE_SELECTED_TITLE_COLOR : (node.boxcolor || colState || LiteGraph.NODE_DEFAULT_BOXCOLOR);
                    ctx.beginPath();
                    ctx.fillRect(10,0-box_size-1,box_size * 1.15,box_size * 0.15);
                    ctx.fillRect(10,0-box_size*1.5-1,box_size * 1.15,box_size * 0.15);
                    ctx.fillRect(10,0-box_size*2-1,box_size * 1.15,box_size * 0.15);
                } else {
                    if (low_quality) {
                        ctx.fillStyle = "black";
                        ctx.fillRect(
                            (title_height - box_size) * 0.5 - 1,
                            (title_height + box_size) * -0.5 - 1,
                            box_size + 2,
                            box_size + 2
                        );
                    }
                    ctx.fillStyle = node.boxcolor || colState || LiteGraph.NODE_DEFAULT_BOXCOLOR;
                    ctx.fillRect(
                        (title_height - box_size) * 0.5,
                        (title_height + box_size) * -0.5,
                        box_size,
                        box_size
                    );
                }
                ctx.globalAlpha = old_alpha;

                //title text
                if (node.onDrawTitleText) {
                    node.onDrawTitleText(
                        ctx,
                        title_height,
                        size,
                        this.ds.scale,
                        this.title_text_font,
                        selected
                    );
                }
                if (!low_quality) {
                    ctx.font = this.title_text_font;
                    var title = String(node.getTitle());
                    if (title) {
                        if (selected) {
                            ctx.fillStyle = LiteGraph.NODE_SELECTED_TITLE_COLOR;
                        } else {
                            ctx.fillStyle =
                                node.constructor.title_text_color ||
                                this.node_title_color;
                        }
                        if (node.flags.collapsed) {
                            ctx.textAlign = "left";
                            var measure = ctx.measureText(title);
                            ctx.fillText(
                                title.substr(0,20), //avoid urls too long
                                title_height,// + measure.width * 0.5,
                                LiteGraph.NODE_TITLE_TEXT_Y - title_height
                            );
                            ctx.textAlign = "left";
                        } else {
                            ctx.textAlign = "left";
                            ctx.fillText(
                                title,
                                title_height,
                                LiteGraph.NODE_TITLE_TEXT_Y - title_height
                            );
                        }
                    }
                }

                //subgraph box
                if (!node.flags.collapsed && node.subgraph && !node.skip_subgraph_button) {
                    var w = LiteGraph.NODE_TITLE_HEIGHT;
                    var x = node.size[0] - w;
                    var over = LiteGraph.isInsideRectangle( this.graph_mouse[0] - node.pos[0], this.graph_mouse[1] - node.pos[1], x+2, -w+2, w-4, w-4 );
                    ctx.fillStyle = over ? "#888" : "#555";
                    if( shape == LiteGraph.BOX_SHAPE || low_quality)
                        ctx.fillRect(x+2, -w+2, w-4, w-4);
                    else
                    {
                        ctx.beginPath();
                        ctx.roundRect(x+2, -w+2, w-4, w-4,[4]);
                        ctx.fill();
                    }
                    ctx.fillStyle = "#333";
                    ctx.beginPath();
                    ctx.moveTo(x + w * 0.2, -w * 0.6);
                    ctx.lineTo(x + w * 0.8, -w * 0.6);
                    ctx.lineTo(x + w * 0.5, -w * 0.3);
                    ctx.fill();
                }

                //custom title render
                if (node.onDrawTitle) {
                    node.onDrawTitle(ctx);
                }
            }

            //render selection marker
            if (selected) {
                if (node.onBounding) {
                    node.onBounding(area);
                }

                if (title_mode == LiteGraph.TRANSPARENT_TITLE) {
                    area[1] -= title_height;
                    area[3] += title_height;
                }
                ctx.lineWidth = 2;
                ctx.globalAlpha = 0.8;
                ctx.beginPath();
                // var out_a = -6,out_b = 12,scale = 2
                var out_a = 0, out_b = 0, scale = 1
                if (shape == LiteGraph.BOX_SHAPE) {
                    ctx.rect(
                        out_a + area[0],
                        out_a + area[1],
                        out_b + area[2],
                        out_b + area[3]
                    );
                } else if (
                    shape == LiteGraph.ROUND_SHAPE ||
                    (shape == LiteGraph.CARD_SHAPE && node.flags.collapsed)
                ) {
                    ctx.roundRect(
                        out_a + area[0],
                        out_a + area[1],
                        out_b + area[2],
                        out_b + area[3],
                        [this.round_radius * scale]
                    );
                } else if (shape == LiteGraph.CARD_SHAPE) {
                    ctx.roundRect(
                        out_a + area[0],
                        out_a + area[1],
                        out_b + area[2],
                        out_b + area[3],
                        [this.round_radius * scale,scale,this.round_radius * scale,scale]
                    );
                } else if (shape == LiteGraph.CIRCLE_SHAPE) {
                    ctx.arc(
                        size[0] * 0.5,
                        size[1] * 0.5,
                        size[0] * 0.5 + 6,
                        0,
                        Math.PI * 2
                    );
                }
                ctx.strokeStyle = LiteGraph.NODE_BOX_OUTLINE_COLOR;
                ctx.stroke();
                ctx.strokeStyle = fgcolor;
                ctx.globalAlpha = 1;
            }

            // these counter helps in conditioning drawing based on if the node has been executed or an action occurred
            if (node.execute_triggered>0) node.execute_triggered--;
            if (node.action_triggered>0) node.action_triggered--;
        };
        LGraphCanvas.prototype.drawNodeWidgets = function(
            node,
            posY,
            ctx,
            active_widget
        ) {
            if (!node.widgets || !node.widgets.length) {
                return 0;
            }
            var width = node.size[0];
            var widgets = node.widgets;
            posY += 2;
            var H = LiteGraph.NODE_WIDGET_HEIGHT;
            var show_text = this.ds.scale > 0.5;
            ctx.save();
            ctx.globalAlpha = this.editor_alpha;
            var outline_color = LiteGraph.WIDGET_OUTLINE_COLOR;
            var background_color = LiteGraph.WIDGET_BGCOLOR;
            var text_color = LiteGraph.WIDGET_TEXT_COLOR;
            var secondary_text_color = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
            var margin = 12;

            for (var i = 0; i < widgets.length; ++i) {
                var w = widgets[i];
                var y = posY;
                if (w.y) {
                    y = w.y;
                }
                w.last_y = y;
                ctx.strokeStyle = outline_color;
                ctx.fillStyle = "#222";
                ctx.textAlign = "left";
                ctx.lineWidth = 1;
                if(w.disabled)
                    ctx.globalAlpha *= 0.5;
                var widget_width = w.width || width;

                switch (w.type) {
                    case "button":
                        ctx.font = "10px Inter"
                        ctx.fillStyle = background_color;
                        if (w.clicked) {
                            ctx.fillStyle = "#AAA";
                            w.clicked = false;
                            this.dirty_canvas = true;
                        }
                        ctx.beginPath();
                        ctx.roundRect(margin, y, widget_width - margin * 2, H, [H * 0.25]);
                        ctx.fill();
                        if(show_text && !w.disabled)
                            ctx.stroke();
                        if (show_text) {
                            ctx.textAlign = "center";
                            ctx.fillStyle = text_color;
                            ctx.fillText(w.label || w.name, widget_width * 0.5, y + H * 0.7);
                        }
                        break;
                    case "toggle":
                        ctx.font = "10px Inter"
                        ctx.textAlign = "left";
                        ctx.strokeStyle = outline_color;
                        ctx.fillStyle = background_color;
                        ctx.beginPath();
                        if (show_text)
                            ctx.roundRect(margin, y, widget_width - margin * 2, H, [H * 0.25]);
                        else
                            ctx.rect(margin, y, widget_width - margin * 2, H );
                        ctx.fill();
                        if(show_text && !w.disabled)
                            ctx.stroke();
                        ctx.fillStyle = w.value ? customThemeColorLight : "#333";
                        ctx.beginPath();
                        ctx.arc( widget_width - margin * 2, y + H * 0.5, H * 0.25, 0, Math.PI * 2 );
                        ctx.fill();
                        if (show_text) {
                            ctx.fillStyle = secondary_text_color;
                            const label = w.label || w.name;
                            if (label != null) {
                                ctx.fillText(label, margin * 1.6, y + H * 0.7);
                            }
                            ctx.font = "10px Inter"
                            ctx.fillStyle = w.value ? text_color : secondary_text_color;
                            ctx.textAlign = "right";
                            ctx.fillText(
                                w.value
                                    ? w.options.on || "true"
                                    : w.options.off || "false",
                                widget_width - 35,
                                y + H * 0.7
                            );
                        }
                        break;
                    case "slider":
                        ctx.font = "10px Inter"
                        ctx.fillStyle = background_color;
                        ctx.strokeStyle = outline_color;
                        ctx.beginPath();
                        ctx.roundRect(margin, y, widget_width - margin * 2, H, [H*0.25]);
                        ctx.fill();
                        ctx.stroke()
                        var range = w.options.max - w.options.min;
                        var nvalue = (w.value - w.options.min) / range;
                        if(nvalue < 0.0) nvalue = 0.0;
                        if(nvalue > 1.0) nvalue = 1.0;
                        ctx.fillStyle = w.options.hasOwnProperty("slider_color") ? w.options.slider_color : (active_widget == w ? "#333" : customThemeColorLight);
                        ctx.beginPath();
                        ctx.roundRect(margin, y, nvalue * (widget_width - margin * 2), H, [H*0.25]);
                        ctx.fill();
                        if (w.marker) {
                            var marker_nvalue = (w.marker - w.options.min) / range;
                            if(marker_nvalue < 0.0) marker_nvalue = 0.0;
                            if(marker_nvalue > 1.0) marker_nvalue = 1.0;
                            ctx.fillStyle = w.options.hasOwnProperty("marker_color") ? w.options.marker_color : "#AA9";
                            ctx.roundRect( margin + marker_nvalue * (widget_width - margin * 2), y, 2, H , [H * 0.25] );
                        }
                        if (show_text) {
                            ctx.textAlign = "center";
                            ctx.fillStyle = text_color;
                            var text = (w.label || w.name) + ": " + (Number(w.value).toFixed(w.options.precision != null ? w.options.precision : 3)).toString()
                            ctx.fillText(
                                text,
                                widget_width * 0.5,
                                y + H * 0.7
                            );

                        }
                        break;
                    case "number":
                    case "combo":
                        ctx.textAlign = "left";
                        ctx.strokeStyle = outline_color;
                        ctx.fillStyle = background_color;
                        ctx.beginPath();
                        if(show_text)
                            ctx.roundRect(margin, y, widget_width - margin * 2, H, [H * 0.25] );
                        else
                            ctx.rect(margin, y, widget_width - margin * 2, H );
                        ctx.fill();
                        if (show_text) {
                            if(!w.disabled)
                                ctx.stroke();
                            ctx.fillStyle = text_color;
                            if(!w.disabled)
                            {
                                ctx.beginPath();
                                ctx.moveTo(margin + 12, y + 6.5);
                                ctx.lineTo(margin + 6, y + H * 0.5);
                                ctx.lineTo(margin + 12, y + H - 6.5);
                                ctx.fill();
                                ctx.beginPath();
                                ctx.moveTo(widget_width - margin - 12, y + 6.5);
                                ctx.lineTo(widget_width - margin - 6, y + H * 0.5);
                                ctx.lineTo(widget_width - margin - 12, y + H - 6.5);
                                ctx.fill();
                            }
                            ctx.fillStyle = secondary_text_color;
                            ctx.font = "10px Inter"
                            ctx.fillText(w.label || w.name, margin * 2 + 5, y + H * 0.7);
                            ctx.fillStyle = text_color;
                            ctx.textAlign = "right";
                            var rightDistance = 6
                            if (w.type == "number") {
                                ctx.font = "10px Inter,JetBrains Mono,monospace"
                                ctx.fillText(
                                    Number(w.value).toFixed(
                                        w.options.precision !== undefined
                                            ? w.options.precision
                                            : 3
                                    ),
                                    widget_width - margin * 2 - rightDistance,
                                    y + H * 0.7
                                );
                            } else {
                                var v = w.value;
                                if( w.options.values )
                                {
                                    var values = w.options.values;
                                    if( values.constructor === Function )
                                        values = values();
                                    if(values && values.constructor !== Array)
                                        v = values[ w.value ];
                                }
                                ctx.fillText(
                                    v,
                                    widget_width - margin * 2 - rightDistance,
                                    y + H * 0.7
                                );
                            }
                        }
                        break;
                    case "string":
                    case "text":
                        ctx.textAlign = "left";
                        ctx.strokeStyle = outline_color;
                        ctx.fillStyle = background_color;
                        ctx.beginPath();
                        if (show_text)
                            ctx.roundRect(margin, y, widget_width - margin * 2, H, [H * 0.25]);
                        else
                            ctx.rect( margin, y, widget_width - margin * 2, H );
                        ctx.fill();
                        if (show_text) {
                            if(!w.disabled)
                                ctx.stroke();
                            ctx.save();
                            ctx.beginPath();
                            ctx.rect(margin, y, widget_width - margin * 2, H);
                            ctx.clip();

                            //ctx.stroke();
                            ctx.fillStyle = secondary_text_color;
                            const label = w.label || w.name;
                            ctx.font = "10px Inter"
                            if (label != null) {
                                ctx.fillText(label, margin * 2, y + H * 0.7);
                            }
                            ctx.fillStyle = text_color;
                            ctx.textAlign = "right";
                            ctx.fillText(String(w.value).substr(0,30), widget_width - margin * 2, y + H * 0.7); //30 chars max
                            ctx.restore();
                        }
                        break;
                    default:
                        if (w.draw) {
                            w.draw(ctx, node, widget_width, y, H);
                        }
                        break;
                }
                posY += (w.computeSize ? w.computeSize(widget_width)[1] : H) + 4;
                ctx.globalAlpha = this.editor_alpha;

            }
            ctx.restore();
            ctx.textAlign = "left";
        };
        // 字体文件
        const preconnect1 = document.createElement("link");
        preconnect1.rel = 'preconnect'
        preconnect1.href = 'https://fonts.googleapis.com'
        document.head.appendChild(preconnect1);
        const preconnect2 = document.createElement("link");
        preconnect2.rel = 'preconnect'
        preconnect2.href = 'https://fonts.googleapis.com'
        preconnect2.crossorigin = ''
        document.head.appendChild(preconnect2);
        const fontFile = document.createElement("link")
        fontFile.rel = "stylesheet"
        fontFile.href = "https://fonts.googleapis.com/css2?family=Inter:wght@100;200;300;400;500;600;700&amp;family=JetBrains+Mono&amp;display=swap"
        document.head.appendChild(fontFile);
        // 样式
        const styleElement = document.createElement("style");
        const fontFamily = 'Inter,-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Oxygen-Sans,Ubuntu,Cantarell,Helvetica Neue,sans-serif'
        const cssCode = `
            .pysssss-workflow-popup{
                min-width:200px!important;
                right:0px!important;
                left:auto!important;
            }
            body{
               font-family: ${fontFamily}!important;
               -webkit-font-smoothing: antialiased;
               -moz-osx-font-smoothing: grayscale;
            }
            textarea{
                font-family: ${fontFamily}!important;
            }
            
            .comfy-multiline-input{
                background-color: transparent;
                border:1px solid var(--border-color);
                border-radius:8px;
                padding: 8px;
                font-size: 12px;
            }
            .comfy-modal {
                border:1px solid var(--border-color);
                box-shadow:none;
                backdrop-filter: blur(8px) brightness(120%);
            }
            .comfy-menu{
                top:38%;
                border-radius:16px;
                border:1px solid var(--border-color);
                box-shadow:none;
                backdrop-filter: blur(8px) brightness(120%);
            }
            .comfy-menu button,.comfy-modal button {
                font-size: 16px;
                padding:6px 0;
                margin-bottom:8px;
            }
            .comfy-menu button.comfy-settings-btn{
                font-size: 12px;
            }
            .comfy-menu-btns {
                margin-bottom: 4px;
            }
            .comfy-menu-btns button,.comfy-list-actions button{
                font-size: 12px;
            }
            .comfy-menu > button,
            .comfy-menu-btns button,
            .comfy-menu .comfy-list button,
            .comfy-modal button {
                border-width:1px;
            }
            
            dialog{
                border:1px solid var(--border-color);
                background:transparent;
                backdrop-filter: blur(8px) brightness(120%);
                box-shadow:none;
            }
            .cm-title{
                background-color:transparent!important;
            }
            .cm-notice-board{
                border-radius:10px!important;
                border:1px solid var(--border-color)!important;
            }
            .cm-menu-container{
                margin-bottom:50px!important;
            }
            hr{
               border:1px solid var(--border-color);
            }
            #shareButton{
                background:linear-gradient(to left,${customThemeColor},${LGraphCanvas.node_colors.pale_blue.color})!important;
                color:white!important;
            }
            
            .litegraph .litemenu-entry.has_submenu {
                border-right: 2px solid #3f3eed;
            }
        `
        styleElement.innerHTML = cssCode
        document.head.appendChild(styleElement);
    }
}catch(e){
    console.error(e)
}


// 节点颜色
const COLOR_THEMES = LGraphCanvas.node_colors
const NODE_COLORS = {
    "easy positive":"green",
    "easy negative":"red",
}

function setNodeColors(node, theme) {
    if (!theme) {return;}
    if(theme.color) node.color = theme.color;
    if(theme.bgcolor) node.bgcolor = theme.bgcolor;
}

app.registerExtension({
    name: "comfy.easyUse.interface",
    setup() {
        Object.assign(app.canvas.default_connection_color_byType, customLinkColors);
        Object.assign(LGraphCanvas.link_type_colors, customLinkColors);
    },

    nodeCreated(node) {
        if (NODE_COLORS.hasOwnProperty(node.comfyClass)) {
            const colorKey = NODE_COLORS[node.comfyClass]
            const theme = COLOR_THEMES[colorKey];
            setNodeColors(node, theme);
        }

    }
})

// 不需要了，之前的解决方案（解决control_before_generate时从历史记录调起时seed值不一致）
// const element = document.getElementsByClassName('comfy-list')
// let new_element = null
// async function load(){
//     const items = await api.getItems('history');
//     new_element.replaceChildren(
//         ...Object.keys(items).flatMap((section) => [
//             $el("h4", {
//                 textContent: section,
//             }),
//             $el("div.comfy-list-items", [
//                 ...(items[section].reverse()).map((item) => {
//                     // Allow items to specify a custom remove action (e.g. for interrupt current prompt)
//                     const removeAction = item.remove || {
//                         name: "Delete",
//                         cb: () => api.deleteItem('history', item.prompt[1]),
//                     };
//                     return $el("div", {textContent: item.prompt[0] + ": "}, [
//                         $el("button", {
//                             textContent: "Load",
//                             onclick: async () => {
//                                 // 补充：解决control_before_generate时从历史记录调起时seed值错误
//                                 const outputs = item.outputs
//                                 let workflow = item.prompt[3].extra_pnginfo.workflow
//                                 const seed_widgets = workflow.seed_widgets
//                                 if(seed_widgets){
//                                     for(let id in seed_widgets){
//                                         id = parseInt(id)
//                                         if(outputs[id] && outputs[id]['value']){
//                                             const seed = outputs[id]['value'][0]
//                                             let nodeIndex = workflow.nodes.findIndex(cate=>cate.id == id)
//                                             if(nodeIndex!==-1 && workflow.nodes[nodeIndex]){
//                                                 workflow.nodes[nodeIndex]['widgets_values'][seed_widgets[id]] = seed
//                                             }
//                                         }
//                                     }
//                                 }
//                                 // end
//                                 await app.loadGraphData(workflow);
//                                 if (item.outputs) {
//                                     app.nodeOutputs = item.outputs;
//                                 }
//                             },
//                         }),
//                         $el("button", {
//                             textContent: removeAction.name,
//                             onclick: async () => {
//                                 await removeAction.cb();
//                                 if (new_element.style.display != 'none') {
//                                     await load();
//                                 }
//                             },
//                         }),
//                     ]);
//                 }),
//             ]),
//         ]),
//         $el("div.comfy-list-actions", [
//             $el("button", {
//                 textContent: "Clear History",
//                 onclick: async () => {
//                     await api.clearItems('history');
//                     await load();
//                 },
//             }),
//             $el("button", {textContent: "Refresh", onclick: () => load()}),
//         ])
//     );
// }
// if(element && element[1]){
//     const history_button = document.getElementById('comfy-view-history-button')
//     history_button.addEventListener('click',e=>{
//         if(new_element){
//             if(new_element.style.display == 'none') new_element.style.display = 'block'
//             else  new_element.style.display = 'none'
//         }
//         if(element[1].style.display != 'none'){
//             element[1].remove()
//             const parentNode = element[0].parentNode
//             new_element = document.createElement('div')
//             new_element.className = 'comfy-list easyuse'
//             new_element.style.display = 'block'
//             parentNode.insertBefore(new_element, element[0].nextSibling);
//             setTimeout(_=>load(),10)
//         }
//     })
// }
// api.addEventListener("status", () => {
//     if (new_element && new_element.style.display != 'none') {
//         load();
//     }
// });

