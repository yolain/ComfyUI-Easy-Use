// import {app} from "/scripts/app.js";
// import {api} from "/scripts/api.js";
// import {$el} from "/scripts/ui.js";
//
// let script=document.createElement("script");
//     script.type="text/JavaScript";
//     script.innerHTML = `
//         function displayImage(e) {
//             console.log(e)
//         }
//     `
//     document.getElementsByTagName('head')[0].appendChild(script);
//
// const Loaders = ['easy fullLoader','easy a1111Loader','easy comfyLoader']
// app.registerExtension({
//     name:"comfy.easyUse.contextMenu",
//     async setup(){
//         const existingContextMenu = LiteGraph.ContextMenu;
//         LiteGraph.ContextMenu = function(values,options){
//             const threshold = 20;
//             const enabled = true;
//             if(!enabled || (values?.length || 0) <= threshold || !(options?.callback) || values.some(i => typeof i !== 'string')){
//                 if(enabled){
//                     // console.log('Skipping context menu auto nesting for incompatible menu.');
//                 }
//                 return existingContextMenu.apply(this,[...arguments]);
//             }
//             const compatValues = values;
//             const originalValues = [...compatValues];
//             const folders = {};
//             const specialOps = [];
//             const folderless = [];
//             for(const value of compatValues){
//                 const splitBy = value.indexOf('/') > -1 ? '/' : '\\';
//                 const valueSplit = value.split(splitBy);
//                 if(valueSplit.length > 1){
//                     const key = valueSplit.shift();
//                     folders[key] = folders[key] || [];
//                     folders[key].push(valueSplit.join(splitBy));
//                 }else if(value === 'CHOOSE' || value.startsWith('DISABLE ')){
//                     specialOps.push(value);
//                 }else{
//                     folderless.push(value);
//                 }
//             }
//             const foldersCount = Object.values(folders).length;
//             if(foldersCount > 0){
//                 const oldcallback = options.callback;
//                 options.callback = null;
//                 const newCallback = (item,options) => {
//                     if(['None','无','無','なし'].includes(item.content)) oldcallback('None',options)
//                     else oldcallback(originalValues.find(i => i.endsWith(item.content),options));
//                 };
//                 const addContent = (content, folderName='') => {
//                     const name = folderName ? `${folderName}/${content}` : content;
//                     // 获取图像
//                     // const imgRes = api.fetchApi(`/easyuse/model/thumbnail?name=${name}`)
//                     // if (imgRes.status === 200) {
//                     //     let data = await imgRes.json();
//                     //     console.log(data)
//                     // }
//
//                     const newContent = $el(
//                         "span.easyuse-model",
//                     {
//                         $: (el) => {
//                             el.onmousemove = (e) => {
//                                 console.log(1)
//                             };
//                             el.onmouseout = () => {
//                                 console.log(2)
//                                 // hiddenImage()
//                             };
//                             el.onmouseover = (e) => {
//                                 console.log(1)
//                                 // displayImage(el.dataset.imgName, styleName)
//                             };
//                         },
//                     },content)
//
//
//                     return {
//                         content,
//                         title:newContent.outerHTML,
//                         callback: newCallback
//                     }
//                 }
//                 const newValues = [];
//                 for(const [folderName,folder] of Object.entries(folders)){
//                     newValues.push({
//                         content:folderName,
//                         has_submenu:true,
//                         callback:() => {},
//                         submenu:{
//                             options:folder.map(f => addContent(f,folderName)),
//                         }
//                     });
//                 }
//                 newValues.push(...folderless.map(f => ({
//                     content:f,
//                     callback:newCallback
//                 })));
//                 if(specialOps.length > 0)
//                     newValues.push(...specialOps.map(f => ({
//                         content:f,
//                         callback:newCallback
//                     })));
//                 return existingContextMenu.call(this,newValues,options);
//             }
//             return existingContextMenu.apply(this,[...arguments]);
//         }
//         LiteGraph.ContextMenu.prototype = existingContextMenu.prototype;
//     },
//
// })
//
