export function sleep(ms = 100, value) {
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve(value);
        }, ms);
    });
}
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

export function formatTime(time, format) {
  time = typeof (time) === "number" ? time : (time instanceof Date ? time.getTime() : parseInt(time));
  if (isNaN(time)) return null;
  if (typeof (format) !== 'string' || !format) format = 'yyyy-MM-dd hh:mm:ss';
  let _time = new Date(time);
  time = _time.toString().split(/[\s\:]/g).slice(0, -2);
  time[1] = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'][_time.getMonth()];
  let _mapping = {
    MM: 1,
    dd: 2,
    yyyy: 3,
    hh: 4,
    mm: 5,
    ss: 6
  };
  return format.replace(/([Mmdhs]|y{2})\1/g, (key) => time[_mapping[key]]);
}
