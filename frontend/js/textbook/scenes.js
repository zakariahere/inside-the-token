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
}
export function controls(id, s, update) {
  const c = el("div", { class: "controls" }),
    explore = s.mode === "explore",
    trained = s.weights === "gpt2" && (id === "heads" || id === "embeddings");
  const set = (key) => (v) => update({ [key]: v });
  if (
    ["tokens", "windows", "embeddings"].includes(id) ||
    (explore && ch3.includes(id))
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
function headsScene(s, r, update, all = false) {
  const box = el("div", {}, focus(r.tokens, s, update));
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
  if (s.step === 0 && !all)
    box.append(
      mini(`Input · ${r.tokens[i]}`, vector(r.x.data[i])),
      mini(
        "Projected queries · before splitting",
        vector(r.queries_flat.data[i], "q"),
      ),
    );
  if (s.step >= 1 || all)
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
  const shownWeights = cfg.training
    ? h.attn_weights_dropped.data[i]
    : h.attn_weights.data[i];
  if (s.step >= 2 || all)
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
  if (s.step >= 3 || all)
    box.append(
      mini(
        "Each head contributes a slice",
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
  if (s.step >= 4 || all) {
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
  if (id === "tokens") return tokenScene(s, data, update);
  if (id === "windows") return windowsScene(s, data, update);
  if (id === "embeddings") return embeddingScene(s, data, update);
  if (id === "attention") return simpleScene(s, data, update);
  if (id === "heads") return headsScene(s, data, update);
  return bankScene(id, s, data, update);
}
