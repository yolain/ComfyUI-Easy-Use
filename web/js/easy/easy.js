import { api } from "/scripts/api.js";
import { app } from "/scripts/app.js";
import {deepEqual, addCss, isLocalNetwork} from "../common/utils.js";
import {$t} from '../common/i18n.js';
import {toast} from "../common/toast.js";

addCss('css/index.css')

api.addEventListener("easyuse-toast",event=>{
    const content = event.detail.content
    const type = event.detail.type
    const duration = event.detail.duration
    if(!type){
        toast.info(content, duration)
    }
    else{
      toast.showToast({
        id: `toast-${type}`,
        content: `${toast[type+"_icon"]} ${content}`,
        duration: duration || 3000,
      })
    }
})
app.registerExtension({
    name: "comfy.easyUse",
    init() {
        // Canvas Menu
        const getCanvasMenuOptions = LGraphCanvas.prototype.getCanvasMenuOptions;
        LGraphCanvas.prototype.getCanvasMenuOptions = function () {
            const options = getCanvasMenuOptions.apply(this, arguments);
            let draggerEl = null
            let isGroupMapcanMove = true
            let old_groups = []
            let emptyImg = new Image()
            emptyImg.src = "data:image/gif;base64,R0lGODlhAQABAIAAAAUEBAAAACwAAAAAAQABAAACAkQBADs=";

            options.push(null,
                // Groups Map
                {
                    content: '📜 '+ $t('Groups Map (EasyUse)'),
                    callback: async() => {
                        let groups = app.canvas.graph._groups
                        let nodes = app.canvas.graph._nodes
                        let old_nodes = groups.length
                        let div =
                            document.querySelector('#easyuse_groups_map') ||
                            document.createElement('div')
                        div.id = 'easyuse_groups_map'
                        div.innerHTML = ''
                        let btn = document.createElement('div')
                        btn.style = `display: flex;
                            width: calc(100% - 8px);
                            justify-content: space-between;
                            align-items: center;
                            padding: 0 6px;
                            height: 44px;`
                        let hideBtn = document.createElement('button')
                        let textB = document.createElement('p')
                        btn.appendChild(textB)
                        btn.appendChild(hideBtn)
                        textB.style.fontSize = '11px'
                        textB.innerHTML =  `<b>${$t('Groups Map (EasyUse)')}</b>`
                        hideBtn.style = `float: right;color: var(--input-text);border-radius:6px;font-size:9px;
                            background-color: var(--comfy-input-bg); border: 1px solid var(--border-color);cursor: pointer;padding: 5px;aspect-ratio: 1 / 1;`
                        hideBtn.addEventListener('click', () => {
                            div.style.display = 'none'
                        })
                        hideBtn.innerText = '❌'
                        div.appendChild(btn)

                        div.addEventListener('mousedown', function (e) {
                            var startX = e.clientX
                            var startY = e.clientY
                            var offsetX = div.offsetLeft
                            var offsetY = div.offsetTop

                            function moveBox (e) {
                              var newX = e.clientX
                              var newY = e.clientY
                              var deltaX = newX - startX
                              var deltaY = newY - startY
                              div.style.left = offsetX + deltaX + 'px'
                              div.style.top = offsetY + deltaY + 'px'
                            }

                            function stopMoving () {
                              document.removeEventListener('mousemove', moveBox)
                              document.removeEventListener('mouseup', stopMoving)
                            }

                            if(isGroupMapcanMove){
                                document.addEventListener('mousemove', moveBox)
                                document.addEventListener('mouseup', stopMoving)
                            }
                        })

                        function updateGroups(groups, groupsDiv, autoSortDiv){
                            if(groups.length>0){
                                autoSortDiv.style.display = 'block'
                            }else autoSortDiv.style.display = 'none'
                            for (let index in groups) {
                                const group = groups[index]
                                const title = group.title
                                const show_text = $t('Always')
                                const hide_text = $t('Bypass')
                                const mute_text = $t('Never')
                                let group_item = document.createElement('div')
                                let group_item_style = `justify-content: space-between;display:flex;background-color: var(--comfy-input-bg);border-radius: 5px;border:1px solid var(--border-color);margin-top:5px;`
                                group_item.addEventListener("mouseover",event=>{
                                    event.preventDefault()
                                    group_item.style = group_item_style + "filter:brightness(1.2);"
                                })
                                group_item.addEventListener("mouseleave",event=>{
                                    event.preventDefault()
                                    group_item.style = group_item_style + "filter:brightness(1);"
                                })
                                group_item.addEventListener("dragstart",e=>{
                                    draggerEl = e.currentTarget;
                                    e.currentTarget.style.opacity = "0.6";
									e.currentTarget.style.border = "1px dashed yellow";
									e.dataTransfer.effectAllowed = 'move';
									e.dataTransfer.setDragImage(emptyImg, 0, 0);
                                })
                                group_item.addEventListener("dragend",e=>{
                                    e.target.style.opacity = "1";
                                    e.currentTarget.style.border = "1px dashed transparent";
                                    e.currentTarget.removeAttribute("draggable");
									document.querySelectorAll('.easyuse-group-item').forEach((el,i) => {
										var prev_i = el.dataset.id;
										if (el == draggerEl && prev_i != i ) {
											groups.splice(i, 0, groups.splice(prev_i, 1)[0]);
										}
										el.dataset.id = i;
									});
                                     isGroupMapcanMove = true
                                })
                                group_item.addEventListener("dragover",e=>{
                                    e.preventDefault();
									if (e.currentTarget == draggerEl) return;
									let rect = e.currentTarget.getBoundingClientRect();
									if (e.clientY > rect.top + rect.height / 2) {
										e.currentTarget.parentNode.insertBefore(draggerEl, e.currentTarget.nextSibling);
									} else {
										e.currentTarget.parentNode.insertBefore(draggerEl, e.currentTarget);
									}
                                    isGroupMapcanMove = true
                                })


                                group_item.setAttribute('data-id',index)
                                group_item.className = 'easyuse-group-item'
                                group_item.style = group_item_style
                                // 标题
                                let text_group_title = document.createElement('div')
                                text_group_title.style = `flex:1;font-size:12px;color:var(--input-text);padding:4px;white-space: nowrap;overflow: hidden;text-overflow: ellipsis;cursor:pointer`
                                text_group_title.innerHTML = `${title}`
                                text_group_title.addEventListener('mousedown',e=>{
                                    isGroupMapcanMove = false
                                    e.currentTarget.parentNode.draggable = 'true';
                                })
                                text_group_title.addEventListener('mouseleave',e=>{
                                    setTimeout(_=>{
                                        isGroupMapcanMove = true
                                    },150)
                                })
                                group_item.append(text_group_title)
                                // 按钮组
                                let buttons = document.createElement('div')
                                group.recomputeInsideNodes();
                                const nodesInGroup = group._nodes;
                                let isGroupShow = nodesInGroup && nodesInGroup.length>0 && nodesInGroup[0].mode == 0
                                let isGroupMute = nodesInGroup && nodesInGroup.length>0 && nodesInGroup[0].mode == 2
                                let go_btn = document.createElement('button')
                                go_btn.style = "margin-right:6px;cursor:pointer;font-size:10px;padding:2px 4px;color:var(--input-text);background-color: var(--comfy-input-bg);border: 1px solid var(--border-color);border-radius:4px;"
                                go_btn.innerText = "Go"
                                go_btn.addEventListener('click', () => {
                                    app.canvas.ds.offset[0] =  -group.pos[0] - group.size[0] * 0.5 + (app.canvas.canvas.width * 0.5) / app.canvas.ds.scale;
                                    app.canvas.ds.offset[1] = -group.pos[1] - group.size[1] * 0.5 + (app.canvas.canvas.height * 0.5) / app.canvas.ds.scale;
                                    app.canvas.setDirty(true, true);
                                    app.canvas.setZoom(1)
                                })
                                buttons.append(go_btn)
                                let see_btn = document.createElement('button')
                                let defaultStyle = `cursor:pointer;font-size:10px;;padding:2px;border: 1px solid var(--border-color);border-radius:4px;width:36px;`
                                see_btn.style = isGroupMute ? `background-color:var(--error-text);color:var(--input-text);` + defaultStyle : (isGroupShow ? `background-color:#006691;color:var(--input-text);` + defaultStyle : `background-color: var(--comfy-input-bg);color:var(--descrip-text);` + defaultStyle)
                                see_btn.innerText = isGroupMute ? mute_text : (isGroupShow ? show_text : hide_text)
                                let pressTimer
                                let firstTime =0, lastTime =0
                                let isHolding = false
                                see_btn.addEventListener('click', () => {
                                    if(isHolding){
                                        isHolding = false
                                        return
                                    }
                                    for (const node of nodesInGroup) {
                                        node.mode = isGroupShow ? 4 : 0;
                                        node.graph.change();
                                    }
                                    isGroupShow = nodesInGroup[0].mode == 0 ? true : false
                                    isGroupMute = nodesInGroup[0].mode == 2 ? true : false
                                    see_btn.style = isGroupMute ? `background-color:var(--error-text);color:var(--input-text);` + defaultStyle : (isGroupShow ? `background-color:#006691;color:var(--input-text);` + defaultStyle : `background-color: var(--comfy-input-bg);color:var(--descrip-text);` + defaultStyle)
                                    see_btn.innerText = isGroupMute ? mute_text : (isGroupShow ? show_text : hide_text)
                                })
                                see_btn.addEventListener('mousedown', () => {
                                    firstTime = new Date().getTime();
                                    clearTimeout(pressTimer);
                                    pressTimer = setTimeout(_=>{
                                        for (const node of nodesInGroup) {
                                            node.mode = isGroupMute ? 0 : 2;
                                            node.graph.change();
                                        }
                                        isGroupShow = nodesInGroup[0].mode == 0 ? true : false
                                        isGroupMute = nodesInGroup[0].mode == 2 ? true : false
                                        see_btn.style = isGroupMute ? `background-color:var(--error-text);color:var(--input-text);` + defaultStyle : (isGroupShow ? `background-color:#006691;color:var(--input-text);` + defaultStyle : `background-color: var(--comfy-input-bg);color:var(--descrip-text);` + defaultStyle)
                                        see_btn.innerText = isGroupMute ? mute_text : (isGroupShow ? show_text : hide_text)
                                    },500)
                                })
                                see_btn.addEventListener('mouseup', () => {
                                    lastTime = new Date().getTime();
                                    if(lastTime - firstTime > 500) isHolding = true
                                    clearTimeout(pressTimer);
                                })
                                buttons.append(see_btn)
                                group_item.append(buttons)

                                groupsDiv.append(group_item)
                            }

                        }

                        let groupsDiv =  document.createElement('div')
                        groupsDiv.id = 'easyuse-groups-items'
                        groupsDiv.style = `overflow-y: auto;max-height: 400px;height:100%;width: 100%;`

                        let autoSortDiv = document.createElement('button')
                        autoSortDiv.style = `cursor:pointer;font-size:10px;padding:2px 4px;color:var(--input-text);background-color: var(--comfy-input-bg);border: 1px solid var(--border-color);border-radius:4px;`
                        autoSortDiv.innerText =  $t('Auto Sorting')
                        autoSortDiv.addEventListener('click',e=>{
                            e.preventDefault()
                            groupsDiv.innerHTML = ``
                            let new_groups = groups.sort((a,b)=> a['pos'][0] - b['pos'][0]).sort((a,b)=> a['pos'][1] - b['pos'][1])
                            updateGroups(new_groups, groupsDiv, autoSortDiv)
                        })

                        updateGroups(groups, groupsDiv, autoSortDiv)

                        div.appendChild(groupsDiv)

                        let remarkDiv =  document.createElement('p')
                        remarkDiv.style = `text-align:center; font-size:10px; padding:0 10px;color:var(--descrip-text)`
                        remarkDiv.innerText =  $t('Toggle `Show/Hide` can set mode of group, LongPress can set group nodes to never')
                        div.appendChild(groupsDiv)
                        div.appendChild(remarkDiv)
                        div.appendChild(autoSortDiv)

                        let graphDiv = document.getElementById("graph-canvas")
                        graphDiv.addEventListener('mouseover', async () => {
                          groupsDiv.innerHTML = ``
                          let new_groups = app.canvas.graph._groups
                          updateGroups(new_groups, groupsDiv, autoSortDiv)
                          old_nodes = nodes
                        })

                        if (!document.querySelector('#easyuse_groups_map')){
                            document.body.appendChild(div)
                        }else{
                            div.style.display = 'flex'
                        }

                    }
                }
            );
            // Only show the reboot option if the server is running on a local network 仅在本地或局域网环境可重启服务
            if(isLocalNetwork(window.location.host)){
                options.push(null,{
                    content: '🔴 '+ $t('Reboot ComfyUI (EasyUse)'),
                    callback: _ =>{
                        if (confirm($t("Are you sure you'd like to reboot the server?"))){
                            try {
                                api.fetchApi("/easyuse/reboot");
                            } catch (exception) {}
                        }
                    }
                })
            }
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