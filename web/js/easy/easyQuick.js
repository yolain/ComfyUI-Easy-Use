// 1.0.2版本的内容 增加node_template的快捷键
// ps: (测试中) 还差个设置管理开启，以免其他包或后续官方添加快捷键造成冲突，后续开放
// import { app } from "/scripts/app.js";
// import { GroupNodeConfig } from "extensions/core/groupNode.js";
//
// function loadTemplate(){
//     return localStorage['Comfy.NodeTemplates'] ? JSON.parse(localStorage['Comfy.NodeTemplates']) : null
// }
// const clipboardAction = async (cb) => {
//     const old = localStorage.getItem("litegrapheditor_clipboard");
//     await cb();
//     localStorage.setItem("litegrapheditor_clipboard", old);
// };
// async function addTemplateToCanvas(t){
//     const data = JSON.parse(t.data);
//     await GroupNodeConfig.registerFromWorkflow(data.groupNodes, {});
//     localStorage.setItem("litegrapheditor_clipboard", t.data);
//     app.canvas.pasteFromClipboard();
// }
//
// app.registerExtension({
// 	name: 'comfy.easyUse.quick',
// 	init() {
//         const keybindListener = async function (event) {
// 			const modifierPressed = event.altKey;
//             if(['1','2','3','4','5','6','7','8','9'].includes(event.key) && modifierPressed){
//                 const template = loadTemplate()
//                 const idx = parseInt(event.key) - 1
//                 if(template && template[idx]){
//                     const t = template[idx]
//                     clipboardAction(_=>{addTemplateToCanvas(t)})
//                 }
//             }
//             if (event.ctrlKey || event.altKey || event.metaKey) {
//                 return;
//             }
//         }
//         window.addEventListener("keydown", keybindListener, true);
//     }
// });

