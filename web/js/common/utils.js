export function addPreconnect(href, crossorigin=false){
    const preconnect = document.createElement("link");
    preconnect.rel = 'preconnect'
    preconnect.href = href
    if(crossorigin) preconnect.crossorigin = ''
    document.head.appendChild(preconnect);
}

export function addCss(href, base=true) {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.type = "text/css";
    link.href =  base ? "extensions/ComfyUI-Easy-Use/"+href : href;
    document.head.appendChild(link);
}

export function deepEqual(obj1, obj2) {
  if (typeof obj1 !== typeof obj2) {
    return false
  }
  if (typeof obj1 !== 'object' || obj1 === null || obj2 === null) {
    return obj1 === obj2
  }
  const keys1 = Object.keys(obj1)
  const keys2 = Object.keys(obj2)
  if (keys1.length !== keys2.length) {
    return false
  }
  for (let key of keys1) {
    if (!deepEqual(obj1[key], obj2[key])) {
      return false
    }
  }
  return true
}


export function getLocale(){
    const locale = localStorage['AGL.Locale'] || localStorage['Comfy.Settings.AGL.Locale'] || 'en-US'
    return locale
}

export function spliceExtension(fileName){
   return fileName.substring(0,fileName.lastIndexOf('.'))
}
export function getExtension(fileName){
   return fileName.substring(fileName.lastIndexOf('.') + 1)
}