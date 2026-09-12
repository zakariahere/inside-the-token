import {
  el,
  p,
  button,
  bars,
  formula,
  mini,
  details,
  tokens,
  vector,
  matrix,
  field,
  arithmetic,
  fmt,
  vectorText,
} from "./components.js";
export function overview(s) {
  const box = el("div");
  box.append(
    el(
      "div",
      { class: "tokens" },
      ["The", "cat", "sat", "on", "the"].map((t) =>
        el("span", { class: "token" }, t),
      ),
      el("span", { class: "token selected" }, "?"),
    ),
  );
  const labels = [
    ["Token IDs", "Addresses in a vocabulary"],
    ["Embeddings", "A vector at every position"],
    ["Attention", "Mix information from context"],
    ["Next token", "The full model predicts"],
  ];
  box.append(
    el(
      "div",
      { class: "pipeline" },
      labels.map(([title, caption], i) =>
        el(
          "div",
          { class: "pipeline-box" + (s.step === i ? " active" : "") },
          el("span", { class: "stage-no" }, `0${i + 1}`),
          el("h3", {}, title),
          p(caption),
        ),
      ),
    ),
  );
  if (s.step === 3)
    box.append(
      el(
        "div",
        { class: "two" },
        mini(
          "An illustrative prediction",
          bars(["mat", "floor", "sofa"], [0.55, 0.3, 0.15]),
          el(
            "small",
            {},
            "Hand-picked probabilities, not a running language model.",
          ),
        ),
        mini(
          "Two different moments",
          p(
            "Training: compare predictions with known next tokens and update parameters.",
          ),
          p(
            "Generation: use learned parameters, select a token, append it, and predict again.",
          ),
        ),
      ),
    );
  else
    box.append(
      formula(
        [
          '"The cat sat on the" → [token ID, token ID, …]',
          "token vector + position vector = input vector",
          "query → scores → proportions → mixture",
        ][s.step],
      ),
    );
  return box;
}
const cache = new Map();
async function request(post, path, body) {
  const key = path + JSON.stringify(body);
  if (cache.has(key)) return cache.get(key);
  const result = await post(path, body);
  if (cache.size > 60) cache.delete(cache.keys().next().value);
  cache.set(key, result);
  return result;
}
const ch3 = ["attention", "qkv", "scores", "causal", "dropout", "heads"];
const ch4 = [
  "architecture",
  "layernorm",
  "gelu",
  "shortcuts",
  "transformer",
  "gpt",
  "generation",
];
export async function loadLesson(id, s, post) {
  const run = (path, body) => request(post, path, body);
  if (s.mode === "explore" && ch3.includes(id)) {
    const endpoint = {
      attention: "simple",
      qkv: "self",
      scores: "self",
      causal: "causal",
      dropout: "causal",
      heads: "mha",
    }[id];
    if (!s.text.trim() && s.source === "text")
      throw Error("Enter some text, or select the book preset.");
    if (id === "heads" && s.weights !== "gpt2" && s.d_out % s.heads)
      throw Error(
        "Output dimensions must be divisible by the number of heads. Try 8 dimensions with 2 heads.",
      );
    const result = await run("/ch03/" + endpoint, {
      source: s.weights === "gpt2" ? "text" : s.source,
      text: s.text,
      seed: s.seed,
      weights: s.weights,
      emb_dim: s.emb_dim,
      d_out: s.d_out,
      num_heads: s.heads,
      dropout: s.dropout,
      train: s.training,
      show_dims: 16,
      query_index: 0,
    });
    return { explorer: true, result, endpoint };
  }
  if (id === "tokens") return run("/ch02/tokenize", { text: s.text });
  if (id === "windows")
    return run("/ch02/windows", {
      text: s.text,
      source: s.source,
      max_length: s.length,
      stride: s.stride,
      batch_size: s.batch,
      max_pairs: 64,
    });
  if (id === "embeddings")
    return run("/ch02/embed", {
      text: s.text,
      source: s.source,
      max_length: s.length,
      stride: s.stride,
      batch_size: s.batch,
      emb_dim: s.emb_dim,
      seed: s.seed,
      weights: s.weights,
      show_dims: 8,
    });
  if (id === "attention")
    return run("/ch03/simple", { query_index: 1, show_dims: 3 });
  if (["qkv", "scores", "causal", "dropout"].includes(id))
    return run("/lessons/bank-attention", {
      scaling: s.scaling,
      causal: id === "dropout" || (id === "causal" && s.step > 0),
      dropout: id === "dropout" ? s.dropout : 0,
      training:
        id === "dropout" && (s.step === 1 || s.step === 2) && s.training,
      seed: s.seed,
    });
  if (id === "heads")
    return run("/ch03/mha", {
      source: "book",
      seed: 123,
      d_out: 2,
      num_heads: 2,
      show_dims: 16,
    });
  if (["layernorm", "gelu", "shortcuts"].includes(id))
    return run(
      "/ch04/" +
        { layernorm: "layernorm", gelu: "feedforward", shortcuts: "shortcuts" }[
          id
        ],
      { seed: s.mode === "learn" ? 123 : s.seed },
    );
  if (id === "transformer")
    return run("/ch04/block", {
      seed: s.mode === "learn" ? 123 : s.seed,
      dropout: s.mode === "learn" ? 0 : s.dropout,
      train: s.mode === "explore" && s.training,
    });
  if (["architecture", "gpt"].includes(id))
    return run("/ch04/model", {
      text: s.mode === "learn" ? "Every effort moves you" : s.text,
      seed: s.mode === "learn" ? 123 : s.seed,
    });
  if (id === "generation")
    return run("/ch04/generate", {
      text: s.mode === "learn" ? "Every effort moves you" : s.text,
      seed: s.mode === "learn" ? 123 : s.seed,
      max_new_tokens: s.max_new_tokens,
    });
}
export function controls(id, s, update) {
  const c = el("div", { class: "controls" }),
    explore = s.mode === "explore",
    trained = s.weights === "gpt2" && (id === "heads" || id === "embeddings");
  const set = (key) => (v) => update({ [key]: v });
  if (
    ["tokens", "windows", "embeddings"].includes(id) ||
    (explore && ch3.includes(id)) ||
    (explore && ["architecture", "gpt", "generation"].includes(id))
  )
    c.append(
      field("Your text", s.text, set("text"), { type: "text", wide: true }),
    );
  if (["windows", "embeddings"].includes(id) || (explore && ch3.includes(id))) {
    c.append(
      field(
        "Input preset",
        trained && id === "heads" ? "text" : s.source,
        set("source"),
        {
          disabled: trained && id === "heads",
          options: ch3.includes(id)
            ? [
                ["text", "Your text"],
                ["book", "Book constants"],
              ]
            : [
                ["text", "Your text"],
                ["verdict", "The Verdict · book text"],
              ],
        },
      ),
    );
  }
  if (id === "windows" || id === "embeddings") {
    c.append(
      field("Window length", s.length, set("length"), { min: 1, max: 12 }),
      field("Stride", s.stride, set("stride"), { min: 1, max: 12 }),
    );
    if (s.step === 3 || explore || id === "embeddings")
      c.append(field("Batch size", s.batch, set("batch"), { min: 1, max: 8 }));
  }
  if (id === "embeddings" || (explore && ch3.includes(id))) {
    c.append(
      field("Random seed", s.seed, set("seed"), {
        disabled: trained,
        min: 0,
        max: 2147483647,
      }),
      field("Input dimensions", trained ? 768 : s.emb_dim, set("emb_dim"), {
        disabled: trained,
        options: trained
          ? [[768, "768 · GPT-2"]]
          : [
              [3, "3"],
              [4, "4"],
              [8, "8"],
              [16, "16"],
              [64, "64"],
            ],
      }),
    );
    if (id === "embeddings" || id === "heads")
      c.append(
        field("Weights", s.weights, set("weights"), {
          options: [
            ["random", "Random · not trained"],
            ["gpt2", "GPT-2 · trained"],
          ],
        }),
      );
  }
  if (explore && ch3.includes(id) && id !== "attention")
    c.append(
      field("Output dimensions", trained ? 768 : s.d_out, set("d_out"), {
        disabled: trained,
        options: trained
          ? [[768, "768 · GPT-2"]]
          : [
              [2, "2"],
              [4, "4"],
              [8, "8"],
              [12, "12"],
              [64, "64"],
            ],
      }),
    );
  if (explore && id === "heads")
    c.append(
      field("Heads", trained ? 12 : s.heads, set("heads"), {
        disabled: trained,
        options: [
          [1, "1"],
          [2, "2"],
          [3, "3"],
          [4, "4"],
          [8, "8"],
          [12, "12"],
        ],
      }),
    );
  if (
    (explore && ["causal", "dropout", "heads"].includes(id)) ||
    (id === "dropout" && [1, 2].includes(s.step))
  )
    c.append(
      field("Drop probability p", trained ? 0 : s.dropout, set("dropout"), {
        disabled: trained,
        min: 0,
        max: 0.9,
        step: 0.1,
      }),
      field("Training mode", trained ? false : s.training, set("training"), {
        disabled: trained,
        type: "checkbox",
      }),
      button("Resample dropout", () => update({ seed: s.seed + 1 }), {
        disabled: trained,
      }),
    );
  if (id === "scores" && s.mode === "learn" && s.step >= 1)
    c.append(
      field("Divide scores by √d_k", s.scaling, set("scaling"), {
        type: "checkbox",
      }),
    );
  if (explore && ch4.includes(id))
    c.append(
      field("Random seed", s.seed, set("seed"), { min: 0, max: 1000000 }),
    );
  if (explore && id === "transformer")
    c.append(
      field("Drop probability p", s.dropout, set("dropout"), {
        min: 0,
        max: 0.9,
        step: 0.1,
      }),
      field("Training mode", s.training, set("training"), {
        type: "checkbox",
      }),
      button("Resample dropout", () => update({ seed: s.seed + 1 })),
    );
  if (id === "generation")
    c.append(
      field("New tokens", s.max_new_tokens, set("max_new_tokens"), {
        min: 1,
        max: 8,
      }),
    );
  return c;
}
function focus(names, s, update) {
  s.selected = Math.max(0, Math.min(s.selected, names.length - 1));
  return el(
    "div",
    { class: "focus-banner" },
    el("span", { class: "selected-label" }, "FOLLOWING"),
    tokens(names, s.selected, (i) => update({ selected: i })),
  );
}
function tokenScene(s, r, update) {
  const box = el("div");
  if (!r.count)
    return el(
      "div",
      { class: "notice" },
      "No tokens yet. Enter a sentence above to begin.",
    );
  s.selected = Math.min(s.selected, r.count - 1);
  box.append(
    p(`${r.count} tokens · vocabulary of ${r.vocab_size.toLocaleString()} IDs`),
    tokens(
      r.tokens.map((t) => t.text.replaceAll(" ", "␣")),
      s.selected,
      (i) => update({ selected: i }),
    ),
  );
  const t = r.tokens[s.selected];
  if (s.step >= 1 || s.mode === "explore")
    box.append(
      el(
        "div",
        { class: "two" },
        mini(
          "Selected vocabulary row",
          formula(`ID ${t.id}`),
          p("This integer selects an embedding-table row."),
        ),
        mini(
          "Exact bytes",
          formula(vectorText(t.bytes)),
          p(
            "The text preview may show a replacement character if this token holds an incomplete UTF-8 sequence.",
          ),
        ),
      ),
    );
  if (s.step === 2 || s.mode === "explore") {
    const merges = [["l", "o", "w"], ["lo", "w"], ["low"]];
    box.append(
      details(
        "A tiny BPE merge illustration",
        p("Suppose learned merge rules first prefer l + o, then lo + w."),
        el(
          "div",
          { class: "bpe" },
          merges[s.merge].map((t) => el("span", {}, t)),
        ),
        button(s.merge === 2 ? "Start again" : "Apply next merge", () =>
          update({ merge: (s.merge + 1) % 3 }),
        ),
        p(
          "Illustration only. The tokenizer above uses GPT-2’s actual vocabulary and merge rules.",
        ),
      ),
    );
  }
  if (s.step === 3 || s.mode === "explore")
    box.append(
      mini(
        "Decoded complete sequence",
        p(r.decoded),
        el(
          "span",
          { class: "axis" },
          r.decoded === s.text ? "Exact round trip ✓" : "Check the input",
        ),
      ),
    );
  return box;
}
function windowsScene(s, r, update) {
  const box = el("div");
  if (!r.pairs.length)
    return el(
      "div",
      { class: "notice" },
      "This text is too short for a training pair. Add text or reduce the window length; a pair needs length + 1 tokens.",
    );
  s.window = Math.min(s.window, r.pairs.length - 1);
  const pair = r.pairs[s.window];
  box.append(
    el(
      "div",
      { class: "card-top" },
      el(
        "span",
        { class: "axis" },
        `window ${s.window + 1} / ${r.total_pairs} · start ${pair.start}`,
      ),
      el(
        "div",
        { class: "actions" },
        button("← Earlier", () => update({ window: s.window - 1 }), {
          disabled: s.window === 0,
        }),
        button("Later →", () => update({ window: s.window + 1 }), {
          disabled: s.window === r.pairs.length - 1,
        }),
      ),
    ),
    el("div", { class: "kicker" }, "INPUT"),
    el(
      "div",
      { class: "tokens" },
      pair.input_tokens.map((t, i) =>
        el("span", { class: "token" }, t, el("small", {}, pair.input_ids[i])),
      ),
    ),
  );
  if (s.step >= 1 || s.mode === "explore")
    box.append(
      el("div", { class: "kicker" }, "TARGET · ONE POSITION AHEAD"),
      el(
        "div",
        { class: "tokens" },
        pair.target_tokens.map((t, i) =>
          el(
            "span",
            { class: "token role o" },
            t,
            el("small", {}, pair.target_ids[i]),
          ),
        ),
      ),
      formula(
        `At input position 0: ${JSON.stringify(pair.input_tokens[0])} → predict ${JSON.stringify(pair.target_tokens[0])}`,
      ),
    );
  if (s.step >= 2 || s.mode === "explore")
    box.append(
      p(
        `Next window starts at ${pair.start} + ${s.stride} = ${pair.start + s.stride}. ${r.total_pairs > 64 ? "The first 64 windows are available here." : ""}`,
      ),
    );
  if (s.step === 3 || s.mode === "explore")
    box.append(
      details(
        "First DataLoader batch",
        p(
          "Rows are separate training examples; columns are sequence positions.",
        ),
        matrix(r.first_batch.inputs, { label: "Batch input IDs" }),
        el(
          "span",
          { class: "axis" },
          `shape [${r.first_batch.shape.join(", ")}]`,
        ),
      ),
    );
  return box;
}
function embeddingScene(s, r, update) {
  const box = el("div");
  s.batchIndex = Math.min(s.batchIndex || 0, r.inputs.length - 1);
  const b = s.batchIndex;
  if (r.inputs.length > 1)
    box.append(
      field("Inspect batch item", b, (v) => update({ batchIndex: +v }), {
        options: r.inputs.map((_, i) => [i, `Sequence ${i}`]),
      }),
    );
  box.append(focus(r.input_tokens[b], s, update));
  const i = s.selected,
    j = Math.min(s.component, r.token_embeddings.data[b][i].length - 1);
  const a = r.token_embeddings.data[b][i],
    pos = r.pos_embeddings.data[i],
    out = r.input_embeddings.data[b][i];
  box.append(
    el(
      "div",
      { class: "three" },
      mini(
        `Token row ${r.inputs[b][i]}`,
        vector(a, "q", (v) => update({ component: v }), j),
      ),
      s.step >= 1 || s.mode === "explore"
        ? mini(
            `Position row ${i}`,
            vector(pos, "k", (v) => update({ component: v }), j),
          )
        : null,
      s.step >= 2 || s.mode === "explore"
        ? mini(
            "Sum → input x",
            vector(out, "o", (v) => update({ component: v }), j),
          )
        : null,
    ),
  );
  if (s.step >= 2 || s.mode === "explore")
    box.append(
      formula(
        `component ${j}: ${fmt(a[j])} + ${fmt(pos[j])} ≈ ${fmt(out[j])}`,
        true,
      ),
    );
  box.append(
    p(
      `${s.weights === "gpt2" ? "Trained GPT-2 tables" : "Seeded random tables; not learned meanings"}. Showing ${a.length} of ${r.emb_dim} dimensions. Values are rounded for display.`,
    ),
  );
  if (s.step === 3 || s.mode === "explore")
    box.append(
      details(
        "All input rows in this sequence",
        matrix(r.input_embeddings.data[b], {
          rowNames: r.input_tokens[b],
          selectedRow: i,
          label: "Input embeddings",
        }),
        p(`Full batch shape: [${r.input_embeddings.shape.join(", ")}]`),
      ),
    );
  return box;
}
function simpleScene(s, r, update) {
  const box = el("div", {}, focus(r.tokens, s, update)),
    i = s.selected,
    j = Math.min(s.key, r.tokens.length - 1),
    x = r.inputs.data;
  if (s.step === 0)
    box.append(mini(`Input vector for ${r.tokens[i]}`, vector(x[i])));
  if (s.step === 1)
    box.append(
      el("div", { class: "kicker" }, "COMPARE WITH"),
      tokens(r.tokens, j, (key) => update({ key })),
      arithmetic(x[i], x[j], `score for ${r.tokens[j]}`),
    );
  if (s.step >= 2)
    box.append(
      bars(r.tokens, r.attn_weights.data[i], {
        selected: j,
        onSelect: (key) => update({ key }),
      }),
    );
  if (s.step === 3) {
    const c = Math.min(s.component, 2),
      w = r.attn_weights.data[i];
    box.append(
      mini(
        "Context vector",
        vector(
          r.context_vecs.data[i],
          "o",
          (component) => update({ component }),
          c,
        ),
      ),
      arithmetic(
        w,
        x.map((row) => row[c]),
        `output component ${c} (rounded inputs)`,
      ),
      details(
        "Every query, same operation",
        matrix(r.attn_weights.data, {
          rowNames: r.tokens,
          colNames: r.tokens,
          selectedRow: i,
          label: "Attention weights",
        }),
      ),
    );
  }
  return box;
}
function qkvScene(s, r, update) {
  const box = el("div", {}, focus(r.tokens, s, update)),
    i = s.selected;
  if (s.step === 0) {
    box.append(
      mini(`x for ${r.tokens[i]}`, vector(r.x[i])),
      details(
        "All rows of X",
        matrix(r.x, { rowNames: r.tokens, selectedRow: i, label: "Input X" }),
      ),
    );
    return box;
  }
  const role = s.step === 2 ? "key" : s.step === 3 ? "value" : "query",
    name = { query: "queries", key: "keys", value: "values" }[role],
    j = Math.min(s.component, 1),
    w = r["W_" + role];
  box.append(
    el(
      "div",
      { class: "projection-layout" },
      mini(
        "One input row",
        vector(r.x[i]),
        el("span", { class: "math-caption" }, "4 input dimensions"),
      ),
      mini(
        `W_${role} · 4 × 2`,
        matrix(w, {
          selectedCol: j,
          onSelect: (_, c) => update({ component: c }),
          label: `W_${role}`,
        }),
      ),
    ),
    el(
      "div",
      { class: "role-label " + role[0] },
      `${role.toUpperCase()} · ${r.tokens[i]}`,
    ),
    vector(r[name][i], role[0], (component) => update({ component }), j),
    arithmetic(
      r.x[i],
      w.map((row) => row[j]),
      `${role} component ${j}`,
    ),
  );
  if (s.step === 4)
    box.append(
      el(
        "div",
        { class: "three" },
        ["queries", "keys", "values"].map((n, k) =>
          mini(
            n,
            matrix(r[n], { rowNames: r.tokens, selectedRow: i, label: n }),
          ),
        ),
      ),
      p(
        "Every row uses the same weight matrix. Computing all rows together is efficient, but each row still follows the arithmetic above.",
      ),
    );
  return box;
}
function bankScene(id, s, r, update) {
  if (id === "qkv") return qkvScene(s, r, update);
  const box = el("div", {}, focus(r.tokens, s, update)),
    i = s.selected,
    j = Math.min(s.key, 2),
    c = Math.min(s.component, 1);
  if (id === "scores") {
    if (s.step === 0)
      box.append(
        mini("Query", vector(r.queries[i])),
        el("div", { class: "kicker" }, "COMPARE WITH A KEY"),
        tokens(r.tokens, j, (key) => update({ key })),
        vector(r.keys[j], "k"),
        arithmetic(r.queries[i], r.keys[j], `score for ${r.tokens[j]}`),
        formula("All scores: " + vectorText(r.scores[i])),
      );
    if (s.step === 1)
      box.append(
        formula(
          `${vectorText(r.scores[i])} ${s.scaling ? "÷ √2" : "÷ 1"} = ${vectorText(r.scaled_scores[i])}`,
        ),
        el(
          "div",
          { class: "two" },
          mini("Without scaling", bars(r.tokens, r.weights_unscaled[i])),
          mini("With scaling ÷ √2", bars(r.tokens, r.weights_scaled[i])),
        ),
      );
    if (s.step === 2)
      box.append(
        mini(
          "Subtract the largest score",
          formula(
            `${vectorText(r.scaled_scores[i])}\n− ${fmt(Math.max(...r.scaled_scores[i]))}\n= ${vectorText(r.shifted_scores[i])}`,
          ),
        ),
        mini("Exponentiate each result", vector(r.exponents[i])),
        formula(
          `Sum = ${fmt(r.exponents[i].reduce((a, b) => a + b, 0))}\nEach exponential ÷ sum → ${vectorText(r.weights[i])}`,
        ),
        bars(r.tokens, r.weights[i]),
      );
    if (s.step >= 3)
      box.append(
        bars(r.tokens, r.weights[i], {
          selected: j,
          onSelect: (key) => update({ key }),
        }),
        mini(
          "Output · select a component",
          vector(r.out[i], "o", (component) => update({ component }), c),
        ),
        arithmetic(
          r.weights[i],
          r.values.map((v) => v[c]),
          `output component ${c}`,
        ),
        mini(
          "Each weighted contribution",
          vector(
            r.contributions[i].map((v) => v[c]),
            "v",
          ),
        ),
        s.step === 4
          ? details(
              "All queries × all keys",
              matrix(r.weights, {
                rowNames: r.tokens,
                colNames: r.tokens,
                selectedRow: i,
                onSelect: (selected, key) => update({ selected, key }),
                label: "All attention weights",
              }),
            )
          : null,
      );
  }
  if (id === "causal") {
    box.append(
      el(
        "div",
        { class: "two" },
        mini(
          "Allowed context",
          el(
            "div",
            { class: "tokens" },
            r.tokens.map((t, k) =>
              el(
                "span",
                {
                  class: "token " + (r.causal_mask[i][k] ? "role k" : "role q"),
                },
                t,
                el(
                  "small",
                  {},
                  r.causal_mask[i][k] ? "future · blocked" : "available",
                ),
              ),
            ),
          ),
        ),
        mini(
          s.step === 0 ? "Full attention" : "Causal attention",
          bars(r.tokens, r.weights[i], { blocked: r.causal_mask[i] }),
        ),
      ),
    );
    if (s.step >= 1)
      box.append(
        matrix(r.masked_scores, {
          rowNames: r.tokens,
          colNames: r.tokens,
          selectedRow: i,
          label: "Scaled masked scores",
        }),
      );
    if (s.step >= 2)
      box.append(
        formula(
          `${vectorText(r.masked_scores[i])}\nsoftmax → ${vectorText(r.weights[i])}`,
        ),
      );
    if (s.step === 3)
      box.append(
        mini(
          "New context vector",
          vector(r.out[i], "o", (component) => update({ component }), c),
        ),
        arithmetic(
          r.weights[i],
          r.values.map((v) => v[c]),
          `output component ${c}`,
        ),
      );
  }
  if (id === "dropout") {
    box.append(
      el(
        "div",
        { class: "two" },
        mini(
          "Before dropout",
          bars(r.tokens, r.weights[i], { blocked: r.causal_mask[i] }),
          p("Row sum = 1"),
        ),
        mini(
          r.config.training
            ? "After dropout · training"
            : "Evaluation / dropout off",
          bars(r.tokens, r.dropped_weights[i], {
            blocked: r.causal_mask[i],
            max: Math.max(1, ...r.dropped_weights[i]),
          }),
          el("span", { class: "axis" }, `row sum = ${fmt(r.row_sums[i])}`),
        ),
      ),
    );
    if (s.step === 1 || s.step === 2)
      box.append(
        formula(
          `keep mask = ${vectorText(r.keep_mask[i].map(Number))}\nsurvivor multiplier = ${fmt(r.dropout_multiplier)}\n${vectorText(r.weights[i])} × mask × ${fmt(r.dropout_multiplier)}\n= ${vectorText(r.dropped_weights[i])}`,
        ),
        p(
          "The bar scale expands if a weight exceeds 100%. These sampled weights are not a probability distribution.",
        ),
      );
    if (s.step >= 2)
      box.append(
        mini(
          "Context after this mask",
          vector(r.out[i], "o", (component) => update({ component }), c),
        ),
        arithmetic(
          r.dropped_weights[i],
          r.values.map((v) => v[c]),
          `output component ${c}`,
        ),
      );
    if (s.step === 4)
      box.append(
        el(
          "div",
          { class: "three" },
          mini(
            "Q · before multiplication",
            el("span", { class: "axis" }, "[b, T, d]"),
            p("b separate sequences; T tokens in each; d query components."),
          ),
          mini(
            "K · transpose(1, 2)",
            el("span", { class: "axis" }, "[b, d, T]"),
            p("Swap token and component axes. Keep each sequence separate."),
          ),
          mini(
            "Q @ Kᵀ",
            el("span", { class: "axis" }, "[b, T, T]"),
            p("One score matrix per sequence. No cross-sequence comparisons."),
          ),
        ),
        p(
          "The registered mask buffer has the maximum context size. [:T, :T] selects the part needed by this input; masked_fill_ changes scores in place.",
        ),
      );
  }
  return box;
}
function singleHeadBridge() {
  const inputTokens = ["The", " river", " bank", " was"];
  const route = ["Project Q/K/V", "Q · Kᵀ", "÷ √dₖ", "Causal mask", "Softmax", "Dropout", "× V"];
  return el(
    "div",
    { class: "head-bridge" },
    el(
      "div",
      { class: "notice" },
      "Single-head checkpoint · the complete causal-attention route is already known.",
    ),
    el(
      "div",
      { class: "sequence-recap" },
      el("span", { class: "recap-label" }, "INPUT POSITIONS"),
      el(
        "div",
        { class: "tokens", "aria-label": "Single-head walkthrough input" },
        inputTokens.map((name, index) =>
          el(
            "span",
            { class: "token" + (index === 2 ? " selected" : "") },
            name,
          ),
        ),
      ),
      el("span", { class: "arrow" }, "→"),
      el("span", { class: "recap-label" }, "LAST TARGET"),
      el("span", { class: "token target-token" }, " muddy"),
    ),
    el(
      "div",
      { class: "flow-strip", "aria-label": "Operations inside one attention head" },
      route.map((label, index) => [
        el("span", { class: "flow-step" }, label),
        index < route.length - 1
          ? el("span", { class: "flow-arrow", "aria-hidden": "true" }, "→")
          : null,
      ]),
    ),
    el(
      "div",
      { class: "head-fork" },
      mini(
        "One completed head",
        el("span", { class: "axis" }, "[b, T, d_head]"),
        p("One score row, one softmax row, and one context slice per query token."),
      ),
      el(
        "div",
        { class: "fork-mark", "aria-label": "Repeat the same operation in parallel" },
        "same route × N →",
      ),
      el(
        "div",
        { class: "head-stack" },
        el("div", { class: "head-pill head-zero" }, "HEAD 0 · full route"),
        el("div", { class: "head-pill head-one" }, "HEAD 1 · full route"),
      ),
    ),
    p(
      "Multi-head attention changes how many learned feature slices run this route. Q and K still create each head’s routing weights; V still supplies each head’s payload.",
    ),
  );
}

function headsScene(s, r, update, all = false) {
  const box = el("div");
  if (s.step === 0 && !all) {
    box.append(singleHeadBridge());
    return box;
  }
  box.append(focus(r.tokens, s, update));
  s.head = Math.min(s.head, r.heads.length - 1);
  const h = r.heads[s.head],
    i = s.selected,
    c = Math.min(s.component, r.out.data[i].length - 1),
    cfg = r.config;
  box.append(
    el(
      "span",
      { class: "axis" },
      `${cfg.num_heads} heads × ${cfg.head_dim} dimensions = ${cfg.d_out}`,
    ),
  );
  if (s.step === 1 && !all)
    box.append(
      el(
        "div",
        { class: "notice" },
        "Fixture handoff · the running single-head example used d_out = 3. This guided example uses d_out = 2 because 2 ÷ 2 heads gives an integer head width of 1.",
      ),
      el(
        "div",
        { class: "two" },
        mini(`Book input · ${r.tokens[i]}`, vector(r.x.data[i])),
        mini(
          "Projected query · before splitting",
          vector(r.queries_flat.data[i], "q"),
          p("Component 0 goes to head 0; component 1 goes to head 1."),
        ),
      ),
    );
  if (s.step >= 2 || all) {
    if (!all && s.step === 2)
      box.append(
        el(
          "div",
          { class: "shape-ladder" },
          mini(
            "1 · Project",
            el("span", { class: "axis" }, `[${cfg.batch}, ${cfg.num_tokens}, ${cfg.d_out}]`),
            p("batch, tokens, total projected width"),
          ),
          el("span", { class: "shape-arrow", "aria-hidden": "true" }, "→"),
          mini(
            "2 · View",
            el(
              "span",
              { class: "axis" },
              `[${cfg.batch}, ${cfg.num_tokens}, ${cfg.num_heads}, ${cfg.head_dim}]`,
            ),
            p("add a head axis; values stay in the same order"),
          ),
          el("span", { class: "shape-arrow", "aria-hidden": "true" }, "→"),
          mini(
            "3 · Transpose",
            el(
              "span",
              { class: "axis" },
              `[${cfg.batch}, ${cfg.num_heads}, ${cfg.num_tokens}, ${cfg.head_dim}]`,
            ),
            p("put heads beside the batch so PyTorch runs them in parallel"),
          ),
        ),
      );
    box.append(
      field("Selected head", s.head, (v) => update({ head: +v }), {
        options: r.heads.map((_, j) => [j, `Head ${j}`]),
      }),
      el(
        "div",
        { class: "three" },
        mini("Query slice", vector(h.queries.data[i], "q")),
        mini("Key slice", vector(h.keys.data[i], "k")),
        mini("Value slice", vector(h.values.data[i], "v")),
      ),
    );
    if (!all && s.step === 2)
      box.append(
        p(
          "These slices come from different rows of the trainable projection matrices. They are separate learned coordinates, even though one Linear layer computes them together.",
        ),
      );
  }
  const shownWeights = cfg.training
    ? h.attn_weights_dropped.data[i]
    : h.attn_weights.data[i];
  if (s.step >= 3 || all) {
    box.append(
      mini(
        `Where head ${s.head} attends for ${r.tokens[i]}`,
        bars(r.tokens, shownWeights, {
          blocked: r.tokens.map((_, j) => j > i),
          selected: s.key,
          onSelect: (key) => update({ key }),
          max: Math.max(1, ...shownWeights),
        }),
        cfg.training
          ? p("After training dropout; the row need not sum to one.")
          : null,
      ),
    );
    if (!all && s.step === 3)
      box.append(
        mini(
          `Inspect one score · ${r.tokens[i]} → ${r.tokens[s.key]}`,
          formula(
            `${h.queries.data[i]
              .map(
                (value, component) =>
                  `(${fmt(value)} × ${fmt(h.keys.data[s.key][component])})`,
              )
              .join(" + ")}\n= ${fmt(h.attn_scores.data[i][s.key])} raw score${s.key > i ? "\n→ future position → replace with −∞ → softmax weight 0" : "\n→ allowed position → keep before scaling and softmax"}`,
            true,
          ),
        ),
        mini(
          `Collect Values in head ${s.head}`,
          formula(
            `${h.attn_weights.data[i]
              .map(
                (weight, key) =>
                  `(${fmt(weight)} × ${fmt(h.values.data[key][0])})`,
              )
              .join(" + ")}\n≈ ${fmt(h.context_heads.data[i][0])}  ← this head’s context slice`,
            true,
          ),
        ),
      );
    box.append(
      details(
        "Head attention matrix",
        matrix(
          cfg.training ? h.attn_weights_dropped.data : h.attn_weights.data,
          {
            rowNames: r.tokens,
            colNames: r.tokens,
            selectedRow: i,
            label: "Head attention weights",
          },
        ),
      ),
    );
  }
  if (s.step >= 4 || all)
    box.append(
      mini(
        `Each head’s context slice · ${r.tokens[i]}`,
        el(
          "div",
          { class: "tokens" },
          r.heads.map((hd, j) =>
            el(
              "div",
              {},
              el("small", {}, `head ${j}`),
              vector(hd.context_heads.data[i], "v"),
            ),
          ),
        ),
      ),
      mini("Concatenated context", vector(r.context_concat.data[i], "o")),
    );
  if (s.step >= 5 || all) {
    box.append(
      mini(
        "After output projection · select a component",
        vector(r.out.data[i], "o", (component) => update({ component }), c),
      ),
    );
    if (!r.context_concat.truncated && !r.out_proj_weight.truncated) {
      const a = r.context_concat.data[i],
        w = r.out_proj_weight.data[c],
        bias = r.out_proj_bias.data[c];
      box.append(
        formula(
          `${a.map((v, j) => `(${fmt(v)} × ${fmt(w[j])})`).join(" + ")} + bias ${fmt(bias)}\n≈ ${fmt(r.out.data[i][c])}`,
          true,
        ),
      );
    } else
      box.append(
        p(
          "Only the first 16 components are displayed. The backend computes the full projection; omitted components also contribute.",
        ),
      );
  }
  return box;
}
const shapeText = (tensor) => `[${tensor.shape.join(", ")}]`;
const tokenRow = (tensor, token = 0) =>
  tensor.data[0]?.[token] ?? tensor.data[token] ?? tensor.data;

function architectureScene(s, r) {
  const stages = [
    ["Token + position", r.combined_embeddings],
    ["2 transformer blocks", r.block_outputs.at(-1)],
    ["Final LayerNorm", r.final_norm],
    ["Vocabulary logits", r.logits],
  ];
  const [title, tensor] = stages[s.step];
  return el(
    "div",
    {},
    el(
      "div",
      { class: "pipeline" },
      stages.map(([name, value], i) =>
        el(
          "div",
          { class: "pipeline-box" + (i === s.step ? " active" : "") },
          el("span", { class: "stage-no" }, `0${i + 1}`),
          el("h3", {}, name),
          p(shapeText(value)),
        ),
      ),
    ),
    mini(
      title,
      s.step === 3
        ? p("One score for every GPT-2 vocabulary entry at every position.")
        : vector(tokenRow(tensor, Math.min(1, r.tokens.length - 1)), "o"),
    ),
    details(
      "Tiny ↔ GPT-2 124M map",
      matrix(
        r.architecture.shape_map.map((row) => [row.tiny, row.gpt2_124m]),
        {
          rowNames: r.architecture.shape_map.map((row) => row.name),
          colNames: ["tiny", "GPT-2 124M"],
          label: "Tiny and GPT-2 architecture dimensions",
        },
      ),
    ),
  );
}

function layerNormScene(s, r, update) {
  const selected = Math.max(0, Math.min(s.selected, 1));
  const choose = focus(["example 0", "example 1"], { ...s, selected }, update);
  const row = (name) => r[name].data[selected];
  const mean = r.mean.data[selected][0];
  const variance = r.variance.data[selected][0];
  const box = el("div", {}, choose);
  if (s.step === 0)
    box.append(
      mini("Linear + ReLU activations", matrix(r.linear_relu.data, {
        rowNames: ["example 0", "example 1"],
        label: "Activation rows before LayerNorm",
        selectedRow: selected,
      })),
    );
  if (s.step === 1)
    box.append(
      mini("Selected row", vector(row("linear_relu"), "q")),
      formula(`${row("linear_relu").map(fmt).join(" + ")}\n──────────── = ${fmt(mean)}\n      6`, true),
      mini("After subtracting the mean", vector(row("centered"), "k")),
    );
  if (s.step === 2)
    box.append(
      formula(`population variance = ${fmt(variance)}\nε = ${r.eps}\ndivisor = √(${fmt(variance)} + ${r.eps})`, true),
      mini("Normalized row", vector(row("normalized"), "o")),
    );
  if (s.step === 3)
    box.append(
      el("div", { class: "three" },
        mini("Trainable scale", vector(r.scale.data, "q")),
        mini("Trainable shift", vector(r.shift.data, "k")),
        mini("LayerNorm output", vector(row("output"), "o")),
      ),
      formula(`output[0] = ${fmt(r.scale.data[0])} × ${fmt(row("normalized")[0])} + ${fmt(r.shift.data[0])}\n= ${fmt(row("output")[0])}`, true),
    );
  if (s.step === 4)
    box.append(
      matrix([
        [r.output_mean.data[0][0], r.output_variance.data[0][0]],
        [r.output_mean.data[1][0], r.output_variance.data[1][0]],
      ], {
        rowNames: ["example 0", "example 1"],
        colNames: ["mean", "population variance"],
        label: "LayerNorm output checks",
      }),
      el("div", { class: "notice" }, "Variance is close to one rather than exactly one because ε is included in the divisor."),
    );
  return box;
}

function geluScene(s, r, update) {
  const selected = Math.max(0, Math.min(s.selected, 1));
  const component = Math.max(0, Math.min(s.component, 15));
  const box = el("div");
  if (s.step === 0)
    box.append(
      matrix(r.curve_x.data.map((x, i) => [r.relu_curve.data[i], r.gelu_curve.data[i]]), {
        rowNames: r.curve_x.data.map((x) => `x = ${fmt(x, 1)}`),
        colNames: ["ReLU(x)", "GELU(x)"],
        label: "ReLU and GELU comparison",
      }),
    );
  else {
    box.append(focus(["token 0", "token 1"], { ...s, selected }, update));
    if (s.step === 1)
      box.append(
        mini("Input · [1, 2, 4]", vector(r.input.data[0][selected], "q")),
        formula("Linear(4, 16)", true),
        mini("Expanded · [1, 2, 16]", vector(r.expanded.data[0][selected], "k")),
      );
    if (s.step === 2) {
      const before = r.expanded.data[0][selected][component];
      const after = r.activated.data[0][selected][component];
      box.append(
        mini("Select one expanded component", vector(r.expanded.data[0][selected], "k", (i) => update({ component: i }), component)),
        formula(`GELU(${fmt(before)}) ≈ ${fmt(after)}`, true),
        mini("All sixteen activated independently", vector(r.activated.data[0][selected], "o")),
      );
    }
    if (s.step === 3)
      box.append(
        mini("Activated · [1, 2, 16]", vector(r.activated.data[0][selected], "k")),
        formula("Linear(16, 4)", true),
        mini("Output · [1, 2, 4]", vector(r.output.data[0][selected], "o")),
      );
  }
  return box;
}

function shortcutsScene(s, r, update) {
  const layer = Math.max(0, Math.min(s.component, 3));
  const picker = el(
    "div",
    { class: "tokens", role: "group", "aria-label": "Select a shortcut layer" },
    [0, 1, 2, 3].map((i) =>
      button(`Layer ${i + 1}`, () => update({ component: i }), {
        class: "token" + (layer === i ? " selected" : ""),
        "aria-pressed": String(layer === i),
      }),
    ),
  );
  const box = el("div");
  if (s.step === 0)
    box.append(
      el("div", { class: "pipeline" }, r.plain_stages.map((stage, i) =>
        el("div", { class: "pipeline-box" }, el("span", { class: "stage-no" }, `0${i + 1}`), el("h3", {}, `Layer ${i + 1}`), p(vectorText(stage.output.data[0]))),
      )),
      el("div", { class: "notice" }, "Plain route: every output fully replaces the previous x."),
    );
  if (s.step === 1) {
    const stage = r.shortcut_stages[layer];
    const output = stage.output.data[0];
    const branch = stage.layer_output.data[0];
    const input = output.map((v, i) => v - branch[i]);
    box.append(
      picker,
      el("div", { class: "three" },
        mini("Saved x", vector(input, "q")),
        mini("layer(x)", vector(branch, "k")),
        mini("x + layer(x)", vector(output, "o")),
      ),
      formula(`${fmt(input[0])} + ${fmt(branch[0])} = ${fmt(output[0])}  ← component 0`, true),
    );
  }
  if (s.step === 2)
    box.append(
      el("div", { class: "two" },
        mini("Plain network", p(`output ${vectorText(r.plain_output.data[0])}`), p(`MSE loss ${fmt(r.plain_loss.data)}`)),
        mini("With shortcuts", p(`output ${vectorText(r.shortcut_output.data[0])}`), p(`MSE loss ${fmt(r.shortcut_loss.data)}`)),
      ),
      formula("loss.backward() computes ∂loss / ∂weight for every Linear layer", true),
    );
  if (s.step === 3)
    box.append(
      matrix(r.plain_gradients.data.map((v, i) => [v, r.shortcut_gradients.data[i]]), {
        rowNames: r.plain_gradients.data.map((_, i) => `Linear ${i + 1}`),
        colNames: ["plain |gradient| mean", "shortcut |gradient| mean"],
        label: "Actual gradient means with and without shortcuts",
      }),
    );
  return box;
}

function transformerScene(s, r, update) {
  const selected = Math.max(0, Math.min(s.selected, r.tokens.length - 1));
  const stages = r.stages;
  const box = el("div", {}, focus(r.tokens, { ...s, selected }, update));
  const card = (title, name, role = "o") =>
    mini(`${title} · ${shapeText(stages[name])}`, vector(tokenRow(stages[name], selected), role));
  if (s.step === 0)
    box.append(el("div", { class: "two" }, card("Input", "input", "q"), card("Saved shortcut", "shortcut1", "q")));
  if (s.step === 1)
    box.append(
      el("div", { class: "three" }, card("LayerNorm 1", "norm1", "q"), card("Attention branch", "attention_dropped", "k"), card("After addition", "after_attention", "o")),
      formula(`${fmt(tokenRow(stages.shortcut1, selected)[0])} + ${fmt(tokenRow(stages.attention_dropped, selected)[0])} = ${fmt(tokenRow(stages.after_attention, selected)[0])}`, true),
    );
  if (s.step === 2)
    box.append(el("div", { class: "two" }, card("Attention result", "after_attention", "o"), card("Second shortcut", "shortcut2", "q")));
  if (s.step === 3)
    box.append(
      el("div", { class: "three" }, card("LayerNorm 2", "norm2", "q"), card("Feed-forward branch", "feed_forward_dropped", "k"), card("Block output", "output", "o")),
      formula(`${fmt(tokenRow(stages.shortcut2, selected)[0])} + ${fmt(tokenRow(stages.feed_forward_dropped, selected)[0])} = ${fmt(tokenRow(stages.output, selected)[0])}`, true),
    );
  if (s.step === 4)
    box.append(
      matrix(Object.entries(stages).map(([_, tensor]) => tensor.shape), {
        rowNames: Object.keys(stages),
        colNames: ["batch", "tokens", "embedding"],
        label: "Shape preserved through transformer block",
      }),
    );
  return box;
}

function gptScene(s, r, update) {
  const selected = Math.max(0, Math.min(s.selected, r.tokens.length - 1));
  const box = el("div", {}, focus(r.tokens, { ...s, selected }, update));
  if (s.step === 0)
    box.append(
      el("div", { class: "three" },
        mini("Token embedding", vector(tokenRow(r.token_embeddings, selected), "q")),
        mini("Position embedding", vector(r.position_embeddings.data[selected], "k")),
        mini("Element-wise sum", vector(tokenRow(r.combined_embeddings, selected), "o")),
      ),
      formula(`${fmt(tokenRow(r.token_embeddings, selected)[0])} + ${fmt(r.position_embeddings.data[selected][0])} = ${fmt(tokenRow(r.combined_embeddings, selected)[0])}`, true),
    );
  if (s.step === 1)
    box.append(
      el("div", { class: "pipeline" }, r.block_outputs.map((tensor, i) =>
        el("div", { class: "pipeline-box" }, el("span", { class: "stage-no" }, `0${i + 1}`), el("h3", {}, `Transformer block ${i + 1}`), p(shapeText(tensor)), vector(tokenRow(tensor, selected), "o")),
      )),
    );
  if (s.step === 2)
    box.append(mini(`Final LayerNorm · ${shapeText(r.final_norm)}`, vector(tokenRow(r.final_norm, selected), "o")));
  if (s.step === 3)
    box.append(
      formula(`${shapeText(r.final_norm)} → Linear(4, 50257) → ${shapeText(r.logits)}`, true),
      mini("Largest logits at the final position", matrix(r.top_last_position.map((item) => [item.id, item.logit]), {
        rowNames: r.top_last_position.map((item) => item.token.replaceAll(" ", "␣")),
        colNames: ["token ID", "logit"],
        label: "Largest random-model logits",
      })),
      el("div", { class: "notice" }, "These rankings come from random weights. They demonstrate the output contract, not language ability."),
    );
  if (s.step === 4) {
    const counts = r.architecture.parameter_counts;
    box.append(
      matrix(Object.entries(r.parameter_groups).map(([_, count]) => [count]), {
        rowNames: Object.keys(r.parameter_groups),
        colNames: ["tiny parameters"],
        label: "Parameters by top-level module",
      }),
      el("div", { class: "two" },
        mini("Book implementation · separate output head", p(counts.gpt2_book_untied.toLocaleString())),
        mini("GPT-2 count · tied embedding/output weights", p(counts.gpt2_with_weight_tying.toLocaleString())),
      ),
    );
  }
  return box;
}

function generationScene(s, r, update) {
  const selected = Math.max(0, Math.min(s.generationStep, r.steps.length - 1));
  const current = r.steps[selected];
  const picker = el(
    "div",
    { class: "tokens", role: "group", "aria-label": "Select a generation iteration" },
    r.steps.map((step, i) => button(`Iteration ${step.step}`, () => update({ generationStep: i }), {
      class: "token" + (selected === i ? " selected" : ""),
      "aria-pressed": String(selected === i),
    })),
  );
  const box = el("div", {}, picker);
  if (s.step === 0)
    box.append(
      tokens(current.context_tokens.map((t) => t.replaceAll(" ", "␣")), current.context_tokens.length - 1, () => {}),
      formula(`context IDs = [${current.context_ids.join(", ")}]\nkept ${current.context_ids.length} of at most ${r.config.context_length} positions`, true),
    );
  if (s.step === 1)
    box.append(
      formula(`model(context) → [1, ${current.context_ids.length}, 50257]\nselect [:, -1, :] → [${current.logits_shape.join(", ")}]`, true),
    );
  if (s.step === 2)
    box.append(
      bars(current.top_candidates.map((c) => c.token.replaceAll(" ", "␣")), current.top_candidates.map((c) => c.probability), {
        max: current.top_candidates[0].probability,
      }),
      formula(`argmax → ID ${current.chosen_id} → ${JSON.stringify(current.chosen_token)}`, true),
    );
  if (s.step === 3)
    box.append(
      formula(`cat(ids, [[${current.chosen_id}]], dim=1)`, true),
      mini("Decoded sequence after all iterations", p(r.generated_text)),
      el("div", { class: "notice" }, r.warning),
    );
  return box;
}

function explorer(s, data, update) {
  const { result: r, endpoint } = data,
    box = el("div");
  box.append(
    el(
      "div",
      { class: "notice" },
      s.weights === "gpt2"
        ? "Trained GPT-2 embeddings and first attention block. This is one block, not a full text generator."
        : s.source === "book"
          ? "Book constants with seeded projection weights."
          : "Your text with random embeddings and seeded weights. These patterns do not demonstrate learned meanings.",
    ),
  );
  if (endpoint === "mha")
    return el("div", {}, box, headsScene(s, r, update, true));
  box.append(focus(r.tokens, s, update));
  const i = s.selected;
  if (r.queries)
    box.append(
      el(
        "div",
        { class: "three" },
        mini("Query", vector(r.queries.data[i], "q")),
        mini("Key", vector(r.keys.data[i], "k")),
        mini("Value", vector(r.values.data[i], "v")),
      ),
    );
  const weights = r.attn_weights_dropped || r.attn_weights;
  box.append(
    bars(r.tokens, weights.data[i], {
      max: Math.max(1, ...weights.data[i]),
      blocked: endpoint === "causal" ? r.tokens.map((_, j) => j > i) : [],
    }),
    mini(
      "Context vector",
      vector((r.context_vec || r.context_vecs).data[i], "o"),
    ),
    details(
      "All attention weights",
      matrix(weights.data, {
        rowNames: r.tokens,
        colNames: r.tokens,
        selectedRow: i,
        label: "Attention weights",
      }),
    ),
    p(
      "Values are rounded; large vectors show their first 16 components. Text inputs use at most 64 tokens.",
    ),
  );
  return box;
}
export function lessonView(id, s, data, update) {
  if (data?.explorer) return explorer(s, data, update);
  if (id === "architecture") return architectureScene(s, data);
  if (id === "layernorm") return layerNormScene(s, data, update);
  if (id === "gelu") return geluScene(s, data, update);
  if (id === "shortcuts") return shortcutsScene(s, data, update);
  if (id === "transformer") return transformerScene(s, data, update);
  if (id === "gpt") return gptScene(s, data, update);
  if (id === "generation") return generationScene(s, data, update);
  if (id === "tokens") return tokenScene(s, data, update);
  if (id === "windows") return windowsScene(s, data, update);
  if (id === "embeddings") return embeddingScene(s, data, update);
  if (id === "attention") return simpleScene(s, data, update);
  if (id === "heads") return headsScene(s, data, update);
  return bankScene(id, s, data, update);
}
