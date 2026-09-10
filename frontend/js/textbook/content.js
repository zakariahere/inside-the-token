export const lessons = [
  {
    id: "overview",
    chapter: 1,
    title: "What an LLM does",
    nav: "The bigger picture",
    section: "1",
    intro:
      "A language model predicts what could come next. Let’s follow a piece of text into the model—and find the part we’re going to build.",
    kind: "Illustration · not model predictions",
    steps: [
      [
        "Start with a sequence",
        "A token is a piece of text: sometimes a word, sometimes just part of one. The model receives a sequence of token IDs.",
      ],
      [
        "Give the tokens a representation",
        "An embedding is a list of numbers for a token. Position information tells the model where that token occurs.",
      ],
      [
        "Let tokens use their context",
        "Attention builds a new vector for each position by mixing value vectors. In a causal model, only the current and earlier positions are available.",
      ],
      [
        "Predict, append, repeat",
        "A full model turns its final vector into scores for the vocabulary. A token is selected and appended; the process repeats. We stop at attention in this textbook.",
      ],
    ],
    math: "text → token IDs → embeddings → transformer blocks → vocabulary scores → next token",
    code: "token_ids = tokenizer.encode(text)\nx = token_embedding(token_ids) + position_embedding(positions)\n# Chapter 3: attention is one part of each transformer block.\ncontext = attention(x)\n# Later chapters add the rest of the model and generation.",
    question: "Does an attention layer directly output the next word?",
    answer:
      "No. It outputs vectors. Later parts of a complete model turn those vectors into vocabulary scores and select a token.",
  },
  {
    id: "tokens",
    chapter: 2,
    title: "Text becomes tokens",
    nav: "Text → tokens",
    section: "2.5",
    intro:
      "Before there are vectors, there are IDs. See where your text is split, and why an ID is simply an address in a vocabulary.",
    kind: "Actual GPT-2 tokenizer",
    steps: [
      [
        "Look at the pieces",
        "Edit the sentence below. Each chip is one token returned by GPT-2’s byte-level BPE tokenizer. Leading spaces belong to tokens too.",
      ],
      [
        "Inspect an ID and its bytes",
        "Select a token. Its ID is a lookup index, not a meaning or a ranking. Some tokens contain only part of a UTF-8 character; their individual text preview can show �.",
      ],
      [
        "Understand a merge",
        "BPE builds larger pieces by merging adjacent pieces according to learned merge rules. This small illustration explains merging; it is not a trace of GPT-2’s learned rules.",
      ],
      [
        "Put the bytes back together",
        "Decoding the complete token sequence recovers the text. Individual byte fragments do not need to be independently readable.",
      ],
    ],
    math: "text → UTF-8 bytes → byte-pair token pieces → vocabulary IDs\ndecode(encode(text)) = text",
    code: 'tokenizer = tiktoken.get_encoding("gpt2")\nids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})\noriginal = tokenizer.decode(ids)\nraw_bytes = tokenizer.decode_single_token_bytes(ids[0])',
    question: "If two tokens have IDs 10 and 11, are their meanings similar?",
    answer:
      "The numbers are vocabulary addresses. Numerical closeness of IDs says nothing about meaning.",
  },
  {
    id: "windows",
    chapter: 2,
    title: "Learning from sequences",
    nav: "Inputs & targets",
    section: "2.6",
    intro:
      "The training question repeats at every position: given this prefix, what came next? Build one training window before thinking about batches.",
    kind: "Actual tokenizer & DataLoader",
    steps: [
      [
        "Choose one window",
        "The input is a short run of token IDs. Move the window to see another training example.",
      ],
      [
        "Shift the target by one",
        "Each target is the token immediately after its corresponding input. During causal training, that input position cannot look ahead at later input positions.",
      ],
      [
        "Move by the stride",
        "Window length controls how much context one example contains. Stride controls how far we move before making the next example. They need not be equal.",
      ],
      [
        "Group windows into a batch",
        "A batch groups independent sequences. Its shape is [batch size, sequence length]. No attention is exchanged between batch items.",
      ],
    ],
    math: "input  = ids[start : start + length]\ntarget = ids[start + 1 : start + length + 1]\nnext start = start + stride",
    code: "loader = create_dataloader_v1(\n    text, batch_size=2, max_length=4, stride=1,\n    shuffle=False, drop_last=False\n)\ninputs, targets = next(iter(loader))",
    question:
      "If stride is 1 and window length is 4, do neighboring windows overlap?",
    answer:
      "Yes. They share three input positions. A stride of 4 would make these input windows non-overlapping.",
  },
  {
    id: "embeddings",
    chapter: 2,
    title: "An ID becomes a vector",
    nav: "Embeddings & position",
    section: "2.7–2.8",
    intro:
      "A token ID selects a row. A position selects another row. Add them component by component, and you have the input to attention.",
    kind: "Seeded random embeddings",
    steps: [
      [
        "Look up the token",
        "Each vocabulary ID selects one row of the token embedding table. The row is a vector: an ordered list of numbers.",
      ],
      [
        "Look up the position",
        "Position 0, position 1, and so on select rows from a separate table. A repeated token can have the same token vector at different positions.",
      ],
      [
        "Add the two vectors",
        "Select a component to see the addition. These are element-wise sums, not concatenation. The number of dimensions stays the same.",
      ],
      [
        "Now call the result x",
        "Stack one resulting vector per position to make X. We use lowercase x for one row and uppercase X for the collection when explaining the math.",
      ],
    ],
    math: "x[position, component] = token_table[token_id, component]\n                          + position_table[position, component]",
    code: "tok_emb = torch.nn.Embedding(50257, output_dim)\npos_emb = torch.nn.Embedding(context_length, output_dim)\nx = tok_emb(inputs) + pos_emb(torch.arange(inputs.shape[-1]))",
    question:
      "Does adding a 4-number token vector to a 4-number position vector produce 8 numbers?",
    answer:
      "No. Corresponding components are added, giving a 4-number vector. Concatenation would produce 8.",
  },
  {
    id: "attention",
    chapter: 3,
    title: "Give one token some context",
    nav: "Why attention?",
    section: "3.3",
    intro:
      "Start with one token. Compare it to the others, turn the scores into proportions, and use those proportions to mix their vectors.",
    kind: "Book constants · 6 tokens × 3 dimensions",
    steps: [
      [
        "Choose whose output to compute",
        "The highlighted token supplies the query in this simplified example. We have no learned Q/K/V projections yet: the input vectors do all three jobs.",
      ],
      [
        "Compare with every input",
        "A dot product multiplies matching components and adds the products. Select another token to inspect one score.",
      ],
      [
        "Turn scores into proportions",
        "Softmax produces non-negative weights summing to one across this query’s row. Bigger scores get larger weights.",
      ],
      [
        "Build a weighted mixture",
        "Multiply each input vector by its weight, then add. Repeat the same operation independently for every query to build all output rows.",
      ],
    ],
    math: "score[i, j] = x[i] · x[j]\nweights[i] = softmax(scores[i])\noutput[i] = Σ weights[i, j] × x[j]",
    code: "scores = inputs @ inputs.T\nweights = torch.softmax(scores, dim=-1)\ncontext_vectors = weights @ inputs",
    question: "Does attention simply copy the highest-scoring token?",
    answer:
      "Usually no. It makes a weighted sum of the vectors. Several tokens can contribute to one output.",
  },
  {
    id: "qkv",
    chapter: 3,
    title: "One token. Three different jobs.",
    nav: "Queries, keys & values",
    section: "3.4",
    intro:
      "Follow “bank” through three small projections. The input stays the same; the weights change which combinations of its numbers we take.",
    kind: "Hand-picked example · The / bank / river",
    steps: [
      [
        "Isolate bank’s input",
        "These four numbers are deliberately chosen for easy arithmetic. They are not trained word meanings. Each row in X belongs to one token.",
      ],
      [
        "Build its query",
        "A query participates in deciding where this token attends. Select an output component: one input row meets one column of W_query.",
      ],
      [
        "Build its key",
        "A key participates in how other queries score this token. The same input goes through a different weight matrix.",
      ],
      [
        "Build its value",
        "Values are the vectors that get mixed. Queries and keys determine the proportions; values supply the numbers being combined.",
      ],
      [
        "Do every row at once",
        "X @ W applies the same projection to each row independently. It does not mix tokens. Token mixing happens later, in attention.",
      ],
    ],
    math: "X: [3 tokens, 4 input dimensions]\nW_query: [4 input dimensions, 2 output dimensions]\nQ = X @ W_query: [3 tokens, 2 output dimensions]\nQ[i, j] = Σ X[i, c] × W_query[c, j]",
    code: "self.W_query = nn.Linear(4, 2, bias=False)\nself.W_key   = nn.Linear(4, 2, bias=False)\nself.W_value = nn.Linear(4, 2, bias=False)\nqueries = self.W_query(x)\nkeys = self.W_key(x)\nvalues = self.W_value(x)\n# nn.Linear stores W as [out, in]: it computes x @ W.T.",
    full: "SelfAttention_v2",
    question: "Does computing Q = X @ W_query mix bank with river?",
    answer:
      "No. Each input row is projected independently with the same weights. The later weighted sum is what mixes information across tokens.",
  },
  {
    id: "scores",
    chapter: 3,
    title: "From scores to a useful mixture",
    nav: "Scores → softmax",
    section: "3.4",
    intro:
      "A query meets every key. Those scores become proportions, and the proportions tell us how much of each value vector to use.",
    kind: "Hand-picked example · full-precision calculations",
    steps: [
      [
        "Compare one query to every key",
        "Select a key to reveal the dot product. For bank the raw scores are [6, 6, 8]. These scores are not percentages.",
      ],
      [
        "Scale before softmax",
        "d_k is the number of components in a query or key: here 2. Divide by √2. Toggle scaling and compare the resulting weight distribution.",
      ],
      [
        "Exponentiate and normalize",
        "We subtract the largest score first for numerical stability. That changes neither the weights nor their ordering. Divide each exponential by their sum.",
      ],
      [
        "Mix the values",
        "Select an output component. Each token contributes its attention weight times that component of its value vector. Add the contributions.",
      ],
      [
        "Expand to all queries",
        "Every output row follows the same recipe. Rows identify the token asking; columns identify the tokens being scored.",
      ],
    ],
    math: "scores = Q @ Kᵀ\na = softmax(scores / √d_k)\noutput = a @ V\n\nWhy √d_k? With independent zero-mean, unit-variance query/key components, the dot product has variance d_k. Dividing by √d_k gives variance 1 under those assumptions. This reduces a source of softmax saturation; it is not a guarantee about every trained layer.",
    code: "scores = queries @ keys.T\nscaled = scores / keys.shape[-1] ** 0.5\nweights = torch.softmax(scaled, dim=-1)\ncontext = weights @ values",
    full: "SelfAttention_v2",
    question: "Does scaling change which of [6, 6, 8] is largest?",
    answer:
      "No. Dividing all scores by the same positive value keeps their order. It changes the gaps, so softmax gives different proportions.",
  },
  {
    id: "causal",
    chapter: 3,
    title: "The future is off limits",
    nav: "Causal attention",
    section: "3.5",
    intro:
      "“bank” could use “river” in the previous lesson. A left-to-right language model must block that future position. Watch the same calculation change.",
    kind: "Same hand-picked example · causal mask",
    steps: [
      [
        "First, allow the whole sequence",
        "With the mask off, bank can use river’s value. This is our comparison point, not a valid causal training setup.",
      ],
      [
        "Mark the future positions",
        "For query position i, block key positions j > i. A token may still attend to itself. Select a different query to move the boundary.",
      ],
      [
        "Replace forbidden scores with −∞",
        "Mask before softmax. exp(−∞) = 0, so future weights become exactly zero. A score of zero would instead contribute exp(0) = 1.",
      ],
      [
        "Rebuild the output",
        "Bank now mixes only The and bank: its weights are [0.5, 0.5, 0]. The output changes because river is no longer available.",
      ],
    ],
    math: "mask[i, j] = (j > i)\nmasked_scores = scores.masked_fill(mask, −∞)\nweights = softmax(masked_scores / √d_k)\noutput = weights @ V",
    code: "mask = torch.triu(torch.ones(T, T), diagonal=1).bool()\nscores.masked_fill_(mask, -torch.inf)\nweights = torch.softmax(scores / keys.shape[-1] ** 0.5, dim=-1)",
    full: "CausalAttention",
    question: "Why not set future scores to zero?",
    answer:
      "Softmax exponentiates scores, and exp(0) is 1. A zero score can still get positive attention. A masked −∞ score contributes zero.",
  },
  {
    id: "dropout",
    chapter: 3,
    title: "Drop connections. Keep the expectation.",
    nav: "Dropout & batches",
    section: "3.5",
    intro:
      "Causal masking is a rule about time. Dropout is random regularization during training. They act for different reasons, at different points in the calculation.",
    kind: "Hand-picked example · seeded dropout",
    steps: [
      [
        "Begin with causal attention",
        "Softmax rows sum to one before dropout. Future positions are already zero because of the causal mask.",
      ],
      [
        "Drop each connection independently",
        "Each weight is independently kept with probability 1 − p. A small row need not lose exactly p of its entries. Resample to see different outcomes.",
      ],
      [
        "Scale the survivors",
        "During training, divide surviving weights by 1 − p. This preserves each weight’s expected value over many draws. A single row can sum to zero, one, or more than one.",
      ],
      [
        "Turn dropout off for evaluation",
        "Evaluation leaves the original attention weights untouched. No random removal or extra scaling happens.",
      ],
      [
        "Add a batch axis",
        "A batch is a stack of independent sequences. Transpose only the last two axes of K, leaving the batch axis in place.",
      ],
    ],
    math: "training: dropped = weights × keep_mask / (1 − p)\nevaluation: dropped = weights\ncontext = dropped @ V\n\nQ: [b, T, d]   Kᵀ: [b, d, T]\nQ @ Kᵀ: [b, T, T]\nb = batch size; T = sequence length; d = projection width.",
    code: 'self.dropout = nn.Dropout(p)\nself.register_buffer("mask", torch.triu(\n    torch.ones(context_length, context_length), diagonal=1))\n# Buffer: saved and moved with the model; not optimized.\nscores = queries @ keys.transpose(1, 2)\nscores.masked_fill_(self.mask.bool()[:T, :T], -torch.inf)\nweights = torch.softmax(scores / keys.shape[-1] ** 0.5, dim=-1)\nweights = self.dropout(weights)\n# module.train() activates dropout; module.eval() disables it.\ncontext = weights @ values',
    full: "CausalAttention",
    question: "Must the weights still sum to one after dropout?",
    answer:
      "No. Scaling preserves the expectation over many random masks, not the sum of every sampled row. Do not renormalize them.",
  },
  {
    id: "heads",
    chapter: 3,
    title: "One head becomes many.",
    nav: "Multiple heads",
    section: "3.6",
    intro:
      "Keep the causal-attention operation you already know. Run it in parallel over different learned Q/K/V slices, then join the resulting context slices into one token representation.",
    kind: "Single-head bridge · book-backed 2-head calculation",
    steps: [
      [
        "Keep the pipeline you know",
        "A head still performs the complete route: project Q/K/V, compare Q with K, scale, causally mask, softmax, optionally drop weights, then collect V. Multi-head attention repeats that route in parallel; it does not invent a new attention formula.",
      ],
      [
        "Make room for two equal heads",
        "The single-head walkthrough used a projection width of 3. Three cannot be divided evenly between two heads, so this book-backed fixture uses a total output width of 2: two heads, one component per head. The numbers change here, but the operation stays the same.",
      ],
      [
        "Split features, keep every token",
        "The view operation adds a head axis, then transpose moves that axis beside the batch. Tokens are not divided between heads: every head receives every token position, with its own Q/K/V feature slice.",
      ],
      [
        "Run the same attention in each head",
        "Select a head, query token, and key token. The app exposes that head’s dot product, causal decision, softmax row, and weighted Value sum. Each head has an independent softmax row, while every head obeys the same causal mask.",
      ],
      [
        "Concatenate the context slices",
        "Each head returns one context slice for the selected token. Put the slices side by side, in head order, to restore the total output width: number of heads × head dimension.",
      ],
      [
        "Mix with the output projection",
        "The final Linear layer can mix information across the concatenated head components. The guided numbers use seeded random weights, so they demonstrate mechanics rather than learned linguistic roles. Explore GPT-2’s 12 trained heads afterward.",
      ],
    ],
    math: "One head:\nContext_h(X) = Dropout(Softmax((Q_h K_hᵀ / √d_head) + M)) V_h\n\nhead_dim = d_out / num_heads\n[b, T, d_out] → [b, T, heads, head_dim]\n             → [b, heads, T, head_dim]\n\nAll heads run the same equation independently.\nConcatenate(Context_0, …, Context_h−1) → [b, T, d_out]\noutput = concatenated context @ W_outᵀ + bias",
    code: "# One Linear layer stores all head-specific projection rows.\nqueries = self.W_query(x)  # [b, T, d_out]\nkeys = self.W_key(x)\nvalues = self.W_value(x)\n\nqueries = queries.view(b, T, self.num_heads, self.head_dim)\nqueries = queries.transpose(1, 2)  # [b, heads, T, head_dim]\n# Repeat the same reshape for K and V.\n\nscores = queries @ keys.transpose(2, 3)\nscores.masked_fill_(mask_bool, -torch.inf)\nweights = torch.softmax(scores / self.head_dim**0.5, dim=-1)\ncontext = (self.dropout(weights) @ values).transpose(1, 2)\ncontext = context.contiguous().view(b, T, self.d_out)\noutput = self.out_proj(context)",
    full: "MultiHeadAttention",
    question:
      "When two heads process six tokens, does each head receive three tokens?",
    answer:
      "No. Both heads receive all six token positions. The feature width is divided: with d_out = 2 and two heads, each head receives one Q/K/V component per token. In GPT-2, 768 ÷ 12 gives 64 components per head.",
  },
];
export const aliases = {
  "ch01-intro": "overview",
  "ch02-tokenize": "tokens",
  "ch02-windows": "windows",
  "ch02-embed": "embeddings",
  "ch03-simple": "attention",
  "ch03-self": "qkv",
  "ch03-causal": "causal",
  "ch03-mha": "heads",
};
