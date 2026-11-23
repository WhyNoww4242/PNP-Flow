// pnpflow icon set: hand-written SVG paths on a 24x24 grid, drawn in one style
// (1.75 stroke, round caps and joins, no fill, currentColor), in the manner of Lucide.
// Custom glyphs for electrochemistry: ion, electrode, mesh, boundary, double layer.

const P = {
  "home": "M4 11l8-7 8 7 M6 10v9a1 1 0 0 0 1 1h3v-5h4v5h3a1 1 0 0 0 1-1v-9",
  "zoom-in": "M11 18a7 7 0 1 0 0-14 7 7 0 0 0 0 14z M20 20l-4-4 M11 8v6 M8 11h6",
  "zoom-out": "M11 18a7 7 0 1 0 0-14 7 7 0 0 0 0 14z M20 20l-4-4 M8 11h6",
  "rescale": "M4 20V4 M4 20h16 M8 8l-3-3 M5 5l3 0 M5 5l0 3 M16 16l3 3 M19 19l-3 0 M19 19l0-3 M8 16h4 M16 8v4",
  // files and app
  // mesh tools
  "hand": "M8 13V5.5a1.5 1.5 0 0 1 3 0V11 M11 10.5v-6a1.5 1.5 0 0 1 3 0V11 M14 6.5a1.5 1.5 0 0 1 3 0V12 M17 9.5a1.5 1.5 0 0 1 3 0V15a6 6 0 0 1-6 6h-1.5a6 6 0 0 1-4.6-2.2L4.6 15.3a1.6 1.6 0 0 1 2.4-2.1L8 14.5",
  "box-region": "M4 4h4 M12 4h4 M20 4v4 M20 12v4 M20 20h-4 M12 20H8 M4 20v-4 M4 12V8 M9 9h6v6H9z",
  "disk-region": "M12 3a9 9 0 0 1 9 9 M21 12a9 9 0 0 1-9 9 M12 21a9 9 0 0 1-9-9 M3 12a9 9 0 0 1 9-9 M12 9.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5z",
  "spray": "M7 10h6v11H7z M8 10V7h4v3 M10 7V4 M14 4h2 M17 6h2 M15 8.5h2 M18 3.5h1 M19 9h1",
  "eraser": "M7 21h10 M5.6 15.4l8.5-8.5a2 2 0 0 1 2.8 0l2.2 2.2a2 2 0 0 1 0 2.8L12 19H9.2a2 2 0 0 1-1.4-.6l-2.2-2.2a.8.8 0 0 1 0-.8z M10 10l5 5",
  "corner": "M4 20V8a4 4 0 0 1 4-4h12 M4 4h.01 M8 8a1.5 1.5 0 1 0 0 .01",
  "file-plus": "M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z M14 3v5h5 M12 11v6 M9 14h6",
  "folder-open": "M3 19V6a2 2 0 0 1 2-2h4l2 2h6a2 2 0 0 1 2 2v2 M3 19l2.6-7.2A2 2 0 0 1 7.5 10.5H21l-2.8 7.7A2 2 0 0 1 16.3 19.5H3.5",
  "refresh": "M20 11a8 8 0 0 0-14.3-4.9L4 8 M4 4v4h4 M4 13a8 8 0 0 0 14.3 4.9L20 16 M20 20v-4h-4",
  "x": "M6 6l12 12 M18 6L6 18",
  "chevron-down": "M6 9l6 6 6-6",
  "chevron-right": "M9 6l6 6-6 6",
  "external": "M14 4h6v6 M20 4l-9 9 M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5",
  "reset": "M4 12a8 8 0 1 0 2.4-5.7L4 8.5 M4 4v4.5h4.5",
  "help": "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z M9.4 9.2a2.7 2.7 0 0 1 5.2.9c0 1.8-2.6 2.4-2.6 3.9 M12 17.2v.1",
  "keyboard": "M3 7a1 1 0 0 1 1-1h16a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1z M7 10h.01 M11 10h.01 M15 10h.01 M7 14h10",
  "sun": "M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8z M12 2v2 M12 20v2 M4.9 4.9l1.4 1.4 M17.7 17.7l1.4 1.4 M2 12h2 M20 12h2 M4.9 19.1l1.4-1.4 M17.7 6.3l1.4-1.4",
  "moon": "M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z",
  "inbox": "M3 13h5l1.5 3h5l1.5-3h5 M5.5 5h13L21 13v5a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1v-5z",
  "alert": "M12 3l9.5 16.5h-19z M12 10v4.5 M12 17.5v.1",
  "info": "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z M12 11v5.5 M12 7.6v.1",
  "check": "M5 12.5l4.5 4.5L19 7.5",
  "check-circle": "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z M8 12.3l2.8 2.8L16.2 9.6",
  "x-circle": "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z M9 9l6 6 M15 9l-6 6",
  "minus-circle": "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z M8 12h8",
  "lock": "M6 11h12v9H6z M8.5 11V8a3.5 3.5 0 0 1 7 0v3",

  // run control
  "play": "M7 4.5v15l12.5-7.5z",
  "stop": "M6.5 6.5h11v11h-11z",
  "repeat": "M17 3l3 3-3 3 M20 6H8a4 4 0 0 0-4 4v1 M7 21l-3-3 3-3 M4 18h12a4 4 0 0 0 4-4v-1",
  "sliders": "M4 6h9 M17 6h3 M15 4v4 M4 12h3 M11 12h9 M9 10v4 M4 18h11 M19 18h1 M17 16v4",
  "terminal": "M3 5a1 1 0 0 1 1-1h16a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1z M7 9l3 3-3 3 M12.5 15H17",
  "trend-down": "M3 6l6.5 6.5 4-4L21 16 M21 10.5V16h-5.5",

  // view
  "fit": "M4 9V5a1 1 0 0 1 1-1h4 M15 4h4a1 1 0 0 1 1 1v4 M20 15v4a1 1 0 0 1-1 1h-4 M9 20H5a1 1 0 0 1-1-1v-4",
  "focus": "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z M12 2.5v4 M12 17.5v4 M2.5 12h4 M17.5 12h4",
  "mirror": "M12 3v18 M9 6.5L3.5 18H9z M15 6.5L20.5 18H15z",
  "image-down": "M20 13.5V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1h8 M4 16l4.5-4.5 4 4 M14 10.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3z M18.5 16v5.5 M16 19l2.5 2.5L21 19",
  "contours": "M3.5 18c3-1 4-5 8.5-5s5.5 4 8.5 5 M5.5 12.5C8 11 9 8 12 8s4 3 6.5 4.5 M8.5 6.5C9.6 5.5 10.6 4 12 4s2.4 1.5 3.5 2.5",
  "palette": "M12 21a9 9 0 1 1 9-9c0 2.5-2 3.5-4 3.5h-2a2 2 0 0 0-1.5 3.3c.5.7.2 2.2-1.5 2.2z M7.5 11.5h.01 M10 7.5h.01 M14.5 7.5h.01",

  // workbenches / model tree
  "geometry": "M4 20V9l8-5 8 5v11z M4 9l8 5 8-5 M12 14v6",
  "mesh": "M3 20h18L12 4z M7.5 12h9 M12 20l-4.5-8 M12 20l4.5-8 M12 12l-2.25-4 M12 12l2.25-4",
  "materials": "M9 3h6 M10 3v6.2L4.6 18.5A1.6 1.6 0 0 0 6 21h12a1.6 1.6 0 0 0 1.4-2.5L14 9.2V3 M7 15h10",
  "boundary": "M7 5h13v14H7z M3.5 5v14",
  "equations": "M17 5H8l5.5 7L8 19h9",
  "study": "M4 6h9 M17 6h3 M15 4v4 M4 12h3 M11 12h9 M9 10v4 M4 18h6 M14 18l2 2 4-4",
  "results": "M4 4v16h16 M8 15l3-4 3 2 4.5-6",
  "field-map": "M4 5a1 1 0 0 1 1-1h14a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1z M12 16.5a4.5 4.5 0 1 0 0-9 4.5 4.5 0 0 0 0 9z M12 13.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3z",
  "profiles": "M4 4v16h16 M7 16c2.5-8 4.5-9 6-7.5s2.5 5.5 6.5 5.5",
  "checks": "M10 6h10 M10 12h10 M10 18h10 M3.5 6l1.5 1.5L7.5 5 M3.5 12l1.5 1.5L7.5 11 M3.5 18l1.5 1.5L7.5 17",
  "mesh-views": "M3 5h8v14H3z M13 5h8v14h-8z M4 17l3-9 3 9 M14.5 17l2.5-4 2.5 4z",
  "figures": "M7 7V4a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1h-3 M3 9a1 1 0 0 1 1-1h11a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1z M3 17l4-4 3.5 3.5 2-2L16 18",
  "diagnose": "M12 3l7.5 3v6c0 4.5-3.2 7.8-7.5 9-4.3-1.2-7.5-4.5-7.5-9V6z M8.5 12l2.5 2.5 4.5-5",
  "quality": "M4 20h16 M6 20v-5 M10 20V9 M14 20v-7 M18 20V5",
  "ion": "M12 18a6 6 0 1 0 0-12 6 6 0 0 0 0 12z M12 9.5v5 M9.5 12h5 M3 4.5l2 2 M21 19.5l-2-2",
  "electrode": "M5 3v18 M5 3h3v18H5 M12 7h.01 M15.5 10h.01 M12 13h.01 M15.5 16h.01 M19 12h.01",
  "units": "M3 8h18v8H3z M7 8v3 M11 8v4 M15 8v3 M19 8v4",
  "flow": "M3 8c3-2 6 2 9 0s6-2 9 0 M3 16c3-2 6 2 9 0s6-2 9 0 M17 5l3 3-3 3",
  "edl": "M4 3v18 M8 6h.01 M8 10h.01 M8 14h.01 M8 18h.01 M12 8h.01 M12 16h.01 M17 12h.01 M20 7h.01",
  "import": "M12 3v12 M7 10l5 5 5-5 M4 17v3h16v-3",
  "script": "M8 7l-5 5 5 5 M16 7l5 5-5 5 M13.5 4l-3 16",
  "list": "M8 6h12 M8 12h12 M8 18h12 M4 6h.01 M4 12h.01 M4 18h.01",
  "axis": "M12 3v18 M9 6l3-3 3 3 M4 12h4 M16 12h4",
  "edges": "M3 21L12 3l9 18z M7.5 12h9 M12 21l-4.5-9 M12 21l4.5-9",
};

// Glyphs drawn filled (play / stop read better solid at 16px).
const FILLED = new Set(["play", "stop"]);

/** Return an inline <svg> element for icon `name`. */
export function icon(name, size = 16, extraClass = "") {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("width", size);
  svg.setAttribute("height", size);
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("focusable", "false");
  svg.setAttribute("class", `icon icon-${name}${extraClass ? " " + extraClass : ""}`);
  const path = document.createElementNS(ns, "path");
  path.setAttribute("d", P[name] || P.inbox);
  if (FILLED.has(name)) {
    path.setAttribute("fill", "currentColor");
    path.setAttribute("stroke", "currentColor");
    path.setAttribute("stroke-linejoin", "round");
    path.setAttribute("stroke-width", "1.5");
  } else {
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", "currentColor");
    path.setAttribute("stroke-width", "1.75");
    path.setAttribute("stroke-linecap", "round");
    path.setAttribute("stroke-linejoin", "round");
  }
  svg.appendChild(path);
  return svg;
}

export const ICON_NAMES = Object.keys(P);
