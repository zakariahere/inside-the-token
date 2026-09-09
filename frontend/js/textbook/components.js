// Text-only DOM construction: lesson text and user input never become HTML.
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value == null) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on"))
      node.addEventListener(key.slice(2).toLowerCase(), value);
    else if (key === "value" || key === "checked" || key === "disabled")
      node[key] = value;
    else node.setAttribute(key, value);
  }
  for (const child of children.flat(Infinity))
    if (child != null && child !== false)
      node.append(
        child instanceof Node ? child : document.createTextNode(String(child)),
      );
  return node;
}
export const p = (text) => el("p", {}, text);
export const fmt = (n, digits = 4) =>
  n === "-inf"
    ? "−∞"
    : typeof n !== "number"
      ? String(n)
      : Number(n.toFixed(digits)).toString();
export const vectorText = (v) => "[" + v.map((x) => fmt(x)).join(", ") + "]";
export const button = (text, action, attrs = {}) =>
  el("button", { type: "button", onClick: action, ...attrs }, text);
export const formula = (text, arithmetic = false) =>
  el("div", { class: "formula" + (arithmetic ? " arithmetic" : "") }, text);
export function details(title, ...body) {
  return el(
    "details",
    {},
    el("summary", {}, title),
    el("div", { class: "details-body" }, body),
  );
}
export function tokens(names, selected, onSelect) {
  return el(
    "div",
    { class: "tokens", role: "group", "aria-label": "Select a token" },
    names.map((name, i) =>
      button(name, () => onSelect(i), {
        class: "token" + (i === selected ? " selected" : ""),
        "aria-pressed": String(i === selected),
        "aria-label": `Select token ${i}: ${name}`,
      }),
    ),
  );
}
export function vector(values, role = "q", onSelect = null, selected = -1) {
  return el(
    "div",
    { class: `vector ${role}` },
    values.map((v, i) =>
      onSelect
        ? button(fmt(v), () => onSelect(i), {
            class: "cell" + (i === selected ? " selected" : ""),
            "aria-label": `Component ${i}: ${fmt(v)}`,
          })
        : el("span", { class: "cell" }, fmt(v)),
    ),
  );
}
export function matrix(
  data,
  {
    rowNames = [],
    colNames = [],
    selectedRow = -1,
    selectedCol = -1,
    onSelect = null,
    label = "Matrix",
  } = {},
) {
  return el(
    "div",
    { class: "table-scroll", tabindex: "0", "aria-label": label },
    el(
      "table",
      {},
      el("caption", { class: "sr-only" }, label),
      el(
        "thead",
        {},
        el(
          "tr",
          {},
          el("th", {}, ""),
          data[0]?.map((_, j) =>
            el("th", { scope: "col" }, colNames[j] ?? `d${j}`),
          ),
        ),
      ),
      el(
        "tbody",
        {},
        data.map((row, i) =>
          el(
            "tr",
            {},
            el("th", { scope: "row" }, rowNames[i] ?? `row ${i}`),
            row.map((v, j) =>
              el(
                "td",
                {},
                onSelect
                  ? button(fmt(v), () => onSelect(i, j), {
                      class:
                        (i === selectedRow || j === selectedCol ? "hl " : "") +
                        (v === "-inf" ? "masked" : ""),
                      "aria-label": `${label}, ${rowNames[i] ?? i}, ${colNames[j] ?? j}: ${fmt(v)}`,
                    })
                  : el(
                      "span",
                      {
                        class:
                          "static-cell " +
                          (i === selectedRow || j === selectedCol
                            ? "hl "
                            : "") +
                          (v === "-inf" ? "masked" : ""),
                      },
                      fmt(v),
                    ),
              ),
            ),
          ),
        ),
      ),
    ),
  );
}
export function bars(
  names,
  values,
  { selected = -1, onSelect = () => {}, blocked = [], max = 1 } = {},
) {
  return el(
    "div",
    { class: "bars" },
    names.map((name, i) =>
      el(
        "div",
        {
          class:
            "bar-row" +
            (selected === i ? " selected" : "") +
            (blocked[i] ? " blocked" : ""),
        },
        button(name, () => onSelect(i), {
          "aria-label": `Inspect ${name} weight`,
        }),
        el(
          "div",
          { class: "bar-track" },
          el("div", {
            class: "bar-fill",
            style: `width:${Math.max(0, Math.min(100, (values[i] / max) * 100))}%`,
          }),
        ),
        el(
          "span",
          { class: "bar-number" },
          blocked[i] ? "blocked" : `${fmt(values[i] * 100, 1)}%`,
        ),
      ),
    ),
  );
}
export function field(label, value, onChange, opts = {}) {
  const {
    type = "number",
    options,
    min,
    max,
    step,
    wide = false,
    disabled = false,
  } = opts;
  let input;
  if (options)
    input = el(
      "select",
      { "aria-label": label, onChange: (e) => onChange(e.target.value) },
      options.map(([v, t]) => el("option", { value: v }, t)),
    );
  else
    input = el("input", {
      type,
      min,
      max,
      step,
      "aria-label": label,
      onChange: (e) =>
        onChange(
          type === "checkbox"
            ? e.target.checked
            : type === "number" || type === "range"
              ? +e.target.value
              : e.target.value,
        ),
    });
  if (type === "checkbox") input.checked = Boolean(value);
  else input.value = value;
  input.disabled = disabled;
  if (type === "text" || type === "number") {
    let timer;
    input.addEventListener("input", () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        if (input.isConnected)
          onChange(type === "number" ? +input.value : input.value);
      }, 350);
    });
  }
  return el(
    "label",
    {
      class:
        "field" + (wide ? " wide" : "") + (type === "checkbox" ? " check" : ""),
    },
    type === "checkbox" ? input : null,
    el("span", {}, label),
    type !== "checkbox" ? input : null,
  );
}
export const mini = (title, ...body) =>
  el("div", { class: "mini" }, el("h3", {}, title), body);
export function arithmetic(a, b, label = "Result") {
  const sign = [...a, ...b].every(Number.isInteger) ? "=" : "≈";
  return formula(
    `${a.map((v, i) => `(${fmt(v)} × ${fmt(b[i])})`).join(" + ")}\n${sign} ${fmt(a.reduce((s, v, i) => s + v * b[i], 0))}  ← ${label}`,
    true,
  );
}
