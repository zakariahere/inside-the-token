import { lessons, aliases } from "./content.js";
import { el, p, button, details } from "./components.js";
import { overview, loadLesson, lessonView, controls } from "./scenes.js";
import { post, get } from "../api.js";
const KEY = "inside-the-token-v1";
let saved = {};
try {
  saved = JSON.parse(localStorage.getItem(KEY) || "{}") || {};
} catch {}
const states = {};
let current,
  serial = 0,
  fullCode = null;
const main = document.querySelector("#main");
function state(id) {
  return (states[id] ??= {
    step: Math.max(
      0,
      Math.min(
        lessons.find((l) => l.id === id).steps.length - 1,
        Number(saved[id]?.step) || 0,
      ),
    ),
    mode: "learn",
    selected: 1,
    component: 0,
    key: 2,
    head: 0,
    window: 0,
    merge: 0,
    text: "The bank by the river is quiet. The river flows past the bank.",
    seed: 42,
    scaling: true,
    dropout: 0.5,
    training: true,
    batch: 2,
    length: 4,
    stride: 1,
    emb_dim: 4,
    d_out: 8,
    heads: 2,
    source: "text",
    weights: "random",
  });
}
function persist() {
  try {
    const previous = JSON.parse(localStorage.getItem(KEY) || "{}") || {};
    localStorage.setItem(
      KEY,
      JSON.stringify({
        ...previous,
        last: current.id,
        ...Object.fromEntries(
          Object.entries(states).map(([id, s]) => [id, { step: s.step }]),
        ),
      }),
    );
  } catch {}
}
function nav() {
  const target = document.querySelector("#lessons");
  target.replaceChildren();
  let ch = 0;
  lessons.forEach((l, i) => {
    if (l.chapter !== ch) {
      ch = l.chapter;
      target.append(
        el(
          "div",
          { class: "chapter-label" },
          [
            "",
            "01 / THE BIG PICTURE",
            "02 / WORKING WITH TEXT",
            "03 / ATTENTION",
          ][ch],
        ),
      );
    }
    target.append(
      el(
        "a",
        {
          href: "#" + l.id,
          class: "nav-item" + (current.id === l.id ? " active" : ""),
          "aria-current": current.id === l.id ? "page" : null,
        },
        el("span", { class: "num" }, String(i + 1).padStart(2, "0")),
        l.nav,
      ),
    );
  });
}
function update(patch) {
  Object.assign(state(current.id), patch);
  persist();
  render();
}
async function render() {
  const openDetails = new Set(
    [...main.querySelectorAll("details[open]")].map(
      (n) => n.querySelector("summary")?.textContent,
    ),
  );
  const focused = document.activeElement;
  const focusLabel = focused?.getAttribute("aria-label");
  const focusText = focused?.tagName === "BUTTON" ? focused.textContent : null;
  function restoreFocus() {
    for (const d of main.querySelectorAll("details"))
      if (openDetails.has(d.querySelector("summary")?.textContent))
        d.open = true;
    const candidates = [...main.querySelectorAll("button,input,select")];
    const target = focusLabel
      ? candidates.find((n) => n.getAttribute("aria-label") === focusLabel)
      : focusText
        ? candidates.find((n) => n.textContent === focusText)
        : null;
    target?.focus({ preventScroll: true });
  }
  const mine = ++serial,
    l = current,
    s = state(l.id),
    index = lessons.indexOf(l);
  persist();
  nav();
  document.title = `${l.title} · Inside the Token`;
  document.querySelector("#breadcrumb").textContent =
    `CHAPTER ${l.chapter} / ${l.nav.toUpperCase()}`;
  const heading = el(
    "section",
    { class: "lesson-heading" },
    el("span", { class: "lesson-count" }, String(index + 1).padStart(2, "0")),
    el("div", { class: "eyebrow" }, `CHAPTER ${l.chapter} · §${l.section}`),
    el("h1", {}, l.title),
    p(l.intro),
  );
  heading.lastChild.className = "intro";
  const modes = el(
    "div",
    { class: "modebar" },
    el(
      "div",
      { class: "tabs", role: "group", "aria-label": "Lesson mode" },
      button("Guided lesson", () => update({ mode: "learn" }), {
        class: s.mode === "learn" ? "active" : "",
        "aria-pressed": String(s.mode === "learn"),
      }),
      l.id !== "overview"
        ? button("Explore", () => update({ mode: "explore" }), {
            class: s.mode === "explore" ? "active" : "",
            "aria-pressed": String(s.mode === "explore"),
          })
        : null,
    ),
    el(
      "span",
      { class: "badge" },
      s.mode === "learn" ? l.kind : "Explore · change inputs & compare",
    ),
  );
  const card = el("section", {
    class: "card",
    "aria-label": "Interactive lesson",
  });
  const step = s.step;
  if (s.mode === "learn")
    card.append(
      el(
        "div",
        { class: "step-track" },
        l.steps.map((_, i) =>
          button("", () => update({ step: i }), {
            class: i <= step ? "active" : "",
            "aria-label": `Go to step ${i + 1}: ${l.steps[i][0]}`,
          }),
        ),
      ),
      el(
        "div",
        { class: "step-heading" },
        el("span", { class: "step-number" }, step + 1),
        el("h2", {}, l.steps[step][0]),
      ),
    );
  const surface = el("div", { id: "interactive" });
  card.append(controls(l.id, s, update), surface);
  if (s.mode === "learn")
    card.append(
      el("div", { class: "explain" }, p(l.steps[step][1])),
      el(
        "div",
        { class: "step-footer" },
        el(
          "span",
          { class: "counter" },
          `STEP ${step + 1} OF ${l.steps.length}`,
        ),
        el(
          "div",
          { class: "actions" },
          button("← Back", () => update({ step: step - 1 }), {
            disabled: step === 0,
          }),
          button(
            step === l.steps.length - 1 ? "Next lesson →" : "Next step →",
            () => {
              if (step === l.steps.length - 1)
                location.hash = lessons[(index + 1) % lessons.length].id;
              else update({ step: step + 1 });
            },
            { class: "primary" },
          ),
        ),
      ),
    );
  const math = details("See the math", el("div", { class: "formula" }, l.math));
  const code = details(
    "See the PyTorch",
    el("pre", {}, el("code", {}, l.code)),
  );
  if (l.full) {
    const full = details(
      "Full class · " + l.full,
      el("p", {}, "Open to load the source used by this app."),
    );
    full.addEventListener("toggle", async () => {
      if (!full.open) return;
      const body = full.querySelector(".details-body");
      try {
        fullCode ??= await get("/lessons/code");
        body.replaceChildren(el("pre", {}, el("code", {}, fullCode[l.full])));
      } catch {
        body.replaceChildren(
          p(
            "Could not load the source. Check the backend and reopen this section.",
          ),
        );
      }
    });
    code.querySelector(".details-body").append(full);
  }
  if (l.id === "dropout")
    code
      .querySelector(".details-body")
      .append(
        el(
          "a",
          {
            href: "https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout",
            target: "_blank",
            rel: "noreferrer",
          },
          "PyTorch dropout reference ↗",
        ),
      );
  const check = details(
    "One quick thought",
    p(l.question),
    button("Reveal explanation", (e) => {
      e.currentTarget.replaceWith(el("p", { class: "check-answer" }, l.answer));
    }),
  );
  const bottom = el(
    "nav",
    { class: "lesson-bottom", "aria-label": "Previous and next lesson" },
    index
      ? el(
          "a",
          { href: "#" + lessons[index - 1].id },
          "← " + lessons[index - 1].nav,
        )
      : el("span"),
    index < lessons.length - 1
      ? el(
          "a",
          { href: "#" + lessons[index + 1].id },
          lessons[index + 1].nav + " →",
        )
      : el("a", { href: "#overview" }, "Return to the big picture →"),
  );
  main.replaceChildren(heading, modes, card, math, code, check, bottom);
  if (l.id === "overview") {
    surface.append(overview(s, update));
    restoreFocus();
    return;
  }
  surface.append(
    el(
      "div",
      { class: "loading", role: "status" },
      "Calculating this example…",
    ),
  );
  try {
    const data = await loadLesson(l.id, s, post);
    if (mine !== serial) return;
    surface.replaceChildren(lessonView(l.id, s, data, update));
    restoreFocus();
  } catch (err) {
    if (mine !== serial) return;
    surface.replaceChildren(
      el(
        "div",
        { class: "error", role: "alert" },
        el("h3", {}, "Let’s fix the input"),
        p(
          err.message.includes("Failed to fetch")
            ? "The Python backend is unavailable. Start uvicorn and retry."
            : err.message,
        ),
        button("Retry", () => render()),
        button("Reset this lesson", () => {
          delete states[l.id];
          delete saved[l.id];
          render();
        }),
      ),
    );
  }
}
function navigate() {
  const hash = location.hash.slice(1);
  const id = aliases[hash] || hash || saved.last || "overview";
  current = lessons.find((l) => l.id === id) || lessons[0];
  document.querySelector("#sidebar").classList.remove("open");
  document.querySelector("#menu").setAttribute("aria-expanded", "false");
  render();
  window.scrollTo(0, 0);
}
window.addEventListener("hashchange", navigate);
document.querySelector(".skip").addEventListener("click", (e) => {
  e.preventDefault();
  main.focus();
  main.scrollIntoView();
});
document.querySelector("#menu").addEventListener("click", () => {
  const open = document.querySelector("#sidebar").classList.toggle("open");
  document.querySelector("#menu").setAttribute("aria-expanded", String(open));
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    document.querySelector("#sidebar").classList.remove("open");
    document.querySelector("#menu").setAttribute("aria-expanded", "false");
  }
});
navigate();
