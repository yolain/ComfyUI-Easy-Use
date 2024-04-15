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


let origProps = {};
export const findWidgetByName = (node, name) => node.widgets.find((w) => w.name === name);

export const doesInputWithNameExist = (node, name) => node.inputs ? node.inputs.some((input) => input.name === name) : false;

export function updateNodeHeight(node) {node.setSize([node.size[0], node.computeSize()[1]]);}

export function toggleWidget(node, widget, show = false, suffix = "") {
	if (!widget || doesInputWithNameExist(node, widget.name)) return;
	if (!origProps[widget.name]) {
		origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize };
	}
	const origSize = node.size;

	widget.type = show ? origProps[widget.name].origType : "easyHidden" + suffix;
	widget.computeSize = show ? origProps[widget.name].origComputeSize : () => [0, -4];

	widget.linkedWidgets?.forEach(w => toggleWidget(node, w, ":" + widget.name, show));

	const height = show ? Math.max(node.computeSize()[1], origSize[1]) : node.size[1];
	node.setSize([node.size[0], height]);
}

export function isLocalNetwork(ip) {
  const localNetworkRanges = [
    '192.168.',
    '10.',
    '127.',
    /^172\.((1[6-9]|2[0-9]|3[0-1])\.)/
  ];

  return localNetworkRanges.some(range => {
    if (typeof range === 'string') {
      return ip.startsWith(range);
    } else {
      return range.test(ip);
    }
  });
}