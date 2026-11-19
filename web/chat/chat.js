"use strict";

(() => {
  const TARGETS = {
    selfhost: { label: "Self-hosted" },
    bedrock: { label: "Amazon Bedrock" },
  };
  const STORE_KEY = "costlab.conversations.v1";
  const TOKEN_KEY = "costlab.token";
  const $ = (id) => document.getElementById(id);

  let config = null;
  let conversations = loadConversations();
  let activeId = conversations[0]?.id ?? null;
  let busy = false;

  // ---------- storage (per-viewer convenience only; the page works without it) ----------
  function loadConversations() {
    try {
      const raw = localStorage.getItem(STORE_KEY);
      const parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }
  function saveConversations() {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(conversations.slice(0, 50)));
    } catch {
      /* storage unavailable: conversations live for this tab only */
    }
  }
  function getToken() {
    try { return localStorage.getItem(TOKEN_KEY) || ""; } catch { return ""; }
  }
  function setToken(value) {
    try { localStorage.setItem(TOKEN_KEY, value); } catch { /* ignore */ }
  }

  // ---------- formatting ----------
  const usd = (value, digits) => {
    if (value === undefined || value === null || Number.isNaN(value)) return "–";
    const d = digits ?? (Math.abs(value) >= 100 ? 0 : Math.abs(value) >= 1 ? 2 : 5);
    return "$" + value.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  };
  const compact = (n) =>
    n.toLocaleString("en-US", { notation: "compact", maximumFractionDigits: 1 });
  const pct = (x) => `${Math.round(x * 100)}%`;

  function escapeHtml(text) {
    return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  }

  // Small, safe markdown subset: input is escaped before any markup is added.
  function renderMarkdown(source) {
    const parts = source.split(/```/);
    return parts
      .map((part, i) => {
        if (i % 2 === 1) {
          const body = part.replace(/^[\w+-]*\n/, "");
          return `<pre><code>${escapeHtml(body)}</code></pre>`;
        }
        return renderBlocks(escapeHtml(part));
      })
      .join("");
  }
  function renderInline(text) {
    return text
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  }
  function renderBlocks(text) {
    return text
      .split(/\n{2,}/)
      .map((block) => block.trim())
      .filter(Boolean)
      .map((block) => {
        const lines = block.split("\n");
        if (lines.every((l) => /^\s*[-*] /.test(l))) {
          return `<ul>${lines.map((l) => `<li>${renderInline(l.replace(/^\s*[-*] /, ""))}</li>`).join("")}</ul>`;
        }
        if (lines.every((l) => /^\s*\d+[.)] /.test(l))) {
          return `<ol>${lines.map((l) => `<li>${renderInline(l.replace(/^\s*\d+[.)] /, ""))}</li>`).join("")}</ol>`;
        }
        const heading = block.match(/^#{1,4} (.*)$/);
        if (heading && lines.length === 1) return `<p><strong>${renderInline(heading[1])}</strong></p>`;
        return `<p>${renderInline(block).replace(/\n/g, "<br>")}</p>`;
      })
      .join("");
  }

  // ---------- API ----------
  async function api(path, options = {}) {
    const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
    const token = getToken();
    if (token) headers.Authorization = `Bearer ${token}`;
    const response = await fetch(path, { ...options, headers });
    if (response.status === 401) {
      askToken();
      throw new Error("The gateway needs an access token.");
    }
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const body = await response.json();
        detail = body.error || JSON.stringify(body.detail || body);
      } catch { /* not JSON */ }
      throw new Error(detail);
    }
    return response;
  }

  async function* readEvents(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const frame = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        const data = frame.split("\n").filter((l) => l.startsWith("data:")).map((l) => l.slice(5).trim()).join("");
        if (data) yield JSON.parse(data);
      }
    }
  }

  // ---------- conversations ----------
  function activeConversation() {
    return conversations.find((c) => c.id === activeId) || null;
  }
  function newConversation(extra = {}) {
    const conversation = { id: crypto.randomUUID(), title: "New chat", turns: [], ...extra };
    conversations = [conversation, ...conversations];
    activeId = conversation.id;
    $("sidebar").classList.remove("open");
    saveConversations();
    renderAll();
    $("prompt").focus();
  }
  // Both models must see the same user turns, or the comparison is unfair: a turn where a model
  // produced no visible text (an error, or a reasoning model that only thought) stays in its
  // history with a placeholder reply instead of being dropped.
  function historyFor(conversation, target, prompt) {
    const messages = [];
    for (const turn of conversation.turns) {
      const answer = turn.answers[target];
      const reply = answer && !answer.error && answer.text.trim() ? answer.text : "(no answer)";
      messages.push({ role: "user", content: turn.user }, { role: "assistant", content: reply });
    }
    messages.push({ role: "user", content: prompt });
    return messages;
  }

  function renderConversationList() {
    const list = $("conversation-list");
    list.replaceChildren(
      ...conversations.map((c) => {
        const li = document.createElement("li");
        li.className = c.id === activeId ? "active" : "";
        const open = document.createElement("button");
        open.type = "button";
        open.className = "open";
        open.textContent = c.title;
        open.addEventListener("click", () => {
          activeId = c.id;
          $("sidebar").classList.remove("open");
          renderAll();
        });
        const del = document.createElement("button");
        del.type = "button";
        del.className = "delete";
        del.setAttribute("aria-label", `Delete ${c.title}`);
        del.textContent = "×";
        del.addEventListener("click", () => {
          conversations = conversations.filter((x) => x.id !== c.id);
          if (activeId === c.id) activeId = conversations[0]?.id ?? null;
          saveConversations();
          renderAll();
        });
        li.append(open, del);
        return li;
      }),
    );
  }

  function metricChips(target, metrics) {
    const chips = [
      ["in", compact(metrics.input_tokens)],
      ["out", compact(metrics.output_tokens)],
    ];
    if (target === "selfhost") chips.push(["cached", compact(metrics.cached_tokens)]);
    chips.push(["TTFT", `${Math.round(metrics.ttft_ms)} ms`], ["speed", `${metrics.output_tokens_per_second} tok/s`]);
    const html = chips.map(([k, v]) => `<span class="metric">${k} <b>${escapeHtml(String(v))}</b></span>`);
    if (target === "selfhost") {
      html.push(
        `<span class="metric cost" title="GPU time this answer needs when the GPU is kept at the target utilization">at target load <b>${usd(metrics.capacity_cost_usd)}</b></span>`,
        `<span class="metric" title="Compute time this answer actually held, shared with ${metrics.avg_concurrency} concurrent request(s)">this machine now <b>${usd(metrics.cost_usd)}</b></span>`,
      );
    } else {
      html.push(`<span class="metric cost">cost <b>${usd(metrics.cost_usd)}</b></span>`);
    }
    return html.join("");
  }

  function reasoningBlock(text, open) {
    const details = document.createElement("details");
    details.className = "reasoning";
    details.open = Boolean(open);
    const summary = document.createElement("summary");
    summary.textContent = "Reasoning (billed as output tokens)";
    const pre = document.createElement("div");
    pre.className = "reasoning-text";
    pre.textContent = text;
    details.append(summary, pre);
    return details;
  }

  function answerCard(target, answer) {
    const card = document.createElement("article");
    card.className = `answer ${target}${answer.streaming ? " streaming" : ""}`;
    const head = document.createElement("div");
    head.className = "answer-head";
    const name = document.createElement("strong");
    name.textContent = TARGETS[target].label;
    const meta = document.createElement("span");
    if (target === "selfhost" && answer.metrics) {
      const m = answer.metrics;
      meta.textContent = `${m.replica} · prefix ${m.prefix_blocks_matched}/${m.prefix_blocks_total} blocks`;
    } else if (target === "bedrock" && config) {
      meta.textContent = config.bedrock.model_id;
    }
    head.append(name, meta);
    const body = document.createElement("div");
    body.className = "answer-body";
    if (answer.error) {
      body.innerHTML = `<p class="error">${escapeHtml(answer.error)}</p>`;
    } else if (answer.streaming) {
      body.textContent = answer.text;
    } else {
      body.innerHTML = renderMarkdown(answer.text || "");
    }
    card.append(head);
    if (answer.reasoning) card.append(reasoningBlock(answer.reasoning, answer.streaming));
    card.append(body);
    if (answer.metrics) {
      const metrics = document.createElement("div");
      metrics.className = "metrics";
      metrics.innerHTML = metricChips(target, answer.metrics);
      card.append(metrics);
    }
    return card;
  }

  function turnElement(turn) {
    const wrap = document.createElement("div");
    wrap.className = "turn";
    const user = document.createElement("div");
    user.className = "user-msg";
    user.textContent = turn.user;
    const answers = document.createElement("div");
    answers.className = "answers";
    for (const target of Object.keys(turn.answers)) answers.append(answerCard(target, turn.answers[target]));
    wrap.append(user, answers);
    return wrap;
  }

  // ---------- showcase ----------
  function showcaseSummary(conversation) {
    const self = conversation.turns.map((t) => t.answers.selfhost && t.answers.selfhost.metrics).filter(Boolean);
    const bed = conversation.turns.map((t) => t.answers.bedrock && t.answers.bedrock.metrics).filter(Boolean);
    if (self.length < 2) return null;
    const later = self.slice(1);
    const avg = (xs, f) => xs.reduce((sum, x) => sum + f(x), 0) / xs.length;
    const laterInput = later.reduce((sum, m) => sum + m.input_tokens, 0);
    return {
      coldTtft: self[0].ttft_ms,
      warmTtft: avg(later, (m) => m.ttft_ms),
      cachedShare: laterInput ? later.reduce((sum, m) => sum + m.cached_tokens, 0) / laterInput : 0,
      selfFirst: self[0].capacity_cost_usd,
      selfLater: avg(later, (m) => m.capacity_cost_usd),
      bedrockTurns: bed.length,
      bedrockInput: bed.reduce((sum, m) => sum + m.input_tokens, 0),
      bedrockAvg: bed.length ? avg(bed, (m) => m.cost_usd) : null,
      bedrockTtft: bed.length ? avg(bed, (m) => m.ttft_ms) : null,
    };
  }
  function summaryElement(s) {
    const box = document.createElement("section");
    box.className = "showcase-summary";
    const rows = [
      ["Self-hosted time to first token", `${Math.round(s.coldTtft)} ms on turn 1 (whole runbook processed) → ${Math.round(s.warmTtft)} ms on later turns`],
      ["Prompt served from the KV cache on later turns", pct(s.cachedShare)],
      ["Self-hosted cost per turn at target load", `${usd(s.selfFirst)} on turn 1 → ${usd(s.selfLater)} on later turns`],
    ];
    if (s.bedrockTurns) {
      rows.push(
        ["Bedrock cost per turn", `${usd(s.bedrockAvg)}: the whole runbook is billed every turn (${compact(s.bedrockInput)} input tokens over ${s.bedrockTurns} turns)`],
        ["Bedrock time to first token", `${Math.round(s.bedrockTtft)} ms average`],
      );
    }
    box.innerHTML = `<h3>Showcase result</h3><dl>${rows
      .map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd>`)
      .join("")}</dl><p class="note">The demo node is a small CPU machine, so absolute speeds are low; the effect to look at is the drop after turn 1. On the proposed GPU fleet the same cache hits apply at much higher throughput (see the projection presets).</p>`;
    return box;
  }
  async function runShowcase() {
    if (busy) return;
    const show = window.SHOWCASE;
    newConversation({ title: show.title, system: show.system, maxTokens: show.maxTokens, showcase: true });
    setStatus("Showcase: turn 1 processes the whole runbook on the CPU node, which takes a while…");
    for (const question of show.questions) {
      await send(question);
      if (activeConversation().turns.at(-1).answers.selfhost?.error) break;
    }
    const conversation = activeConversation();
    conversation.summary = showcaseSummary(conversation);
    saveConversations();
    renderAll();
  }

  function renderMessages() {
    const container = $("messages");
    const conversation = activeConversation();
    const turns = conversation ? conversation.turns : [];
    $("empty-state").hidden = turns.length > 0;
    container.querySelectorAll(".turn, .showcase-summary").forEach((n) => n.remove());
    for (const turn of turns) container.append(turnElement(turn));
    if (conversation && conversation.summary) container.append(summaryElement(conversation.summary));
    container.scrollTop = container.scrollHeight;
  }

  function renderSessionTotals() {
    const conversation = activeConversation();
    const totals = { selfhost: { cost: 0, capacity: 0, input: 0, cached: 0, n: 0, ttft: 0 }, bedrock: { cost: 0, input: 0, n: 0, ttft: 0 } };
    for (const turn of conversation ? conversation.turns : []) {
      for (const [target, answer] of Object.entries(turn.answers)) {
        const m = answer.metrics;
        if (!m) continue;
        const t = totals[target];
        t.cost += m.cost_usd;
        t.capacity = (t.capacity || 0) + (m.capacity_cost_usd || 0);
        t.input += m.input_tokens;
        t.cached = (t.cached || 0) + (m.cached_tokens || 0);
        t.ttft += m.ttft_ms;
        t.n += 1;
      }
    }
    const s = totals.selfhost;
    const b = totals.bedrock;
    $("total-selfhost").textContent = usd(s.capacity, 5);
    $("total-selfhost-sub").textContent = `at target load · ${usd(s.cost, 5)} on this machine now`;
    $("total-bedrock").textContent = usd(b.cost, 5);
    const verdict = $("session-verdict");
    if (s.n && b.n) {
      const ratio = b.cost / Math.max(s.capacity, 1e-12);
      verdict.textContent = ratio >= 1
        ? `At target load, self-hosting this conversation costs ${ratio.toFixed(1)}× less than Bedrock.`
        : `At target load, Bedrock costs ${(1 / ratio).toFixed(1)}× less than self-hosting for this conversation.`;
      if (config && !config.bedrock.same_model) {
        verdict.textContent += " The two sides run different models, so this compares price, not quality.";
      }
    } else if (s.n || b.n) {
      verdict.textContent = "Only one side has answered so far, so there is nothing to compare yet.";
    } else {
      verdict.textContent = "Send a message to start comparing.";
    }
    const facts = [
      ["Prompt tokens served from KV cache", s.input ? pct(s.cached / s.input) : "–"],
      ["Avg time to first token, self-hosted", s.n ? `${Math.round(s.ttft / s.n)} ms` : "–"],
      ["Avg time to first token, Bedrock", b.n ? `${Math.round(b.ttft / b.n)} ms` : "–"],
    ];
    if (config) {
      facts.push(
        ["Models", config.bedrock.same_model ? `${config.model.label} on both` : `${config.model.label} vs ${config.bedrock.label} (different)`],
        ["Instance", `${config.selfhost.instance_type} (${config.selfhost.kind.toUpperCase()}) · ${usd(config.selfhost.usd_per_hour, 3)}/h`],
        ["Bedrock price / 1M tokens", `${usd(config.bedrock.input_per_million, 2)} in · ${usd(config.bedrock.output_per_million, 2)} out`],
      );
    }
    $("session-facts").innerHTML = facts.map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd>`).join("");
  }

  function renderAll() {
    renderConversationList();
    renderMessages();
    renderSessionTotals();
  }

  // ---------- sending ----------
  function setStatus(text, isError = false) {
    $("status").textContent = text;
    $("status").classList.toggle("error", isError);
  }

  async function send(prompt) {
    if (busy || !prompt.trim()) return;
    const targets = Object.keys(TARGETS).filter((t) => $(`use-${t}`).checked);
    if (!targets.length) {
      setStatus("Choose at least one model under Options.", true);
      return;
    }
    if (!activeConversation()) newConversation();
    const conversation = activeConversation();
    const body = {
      system: conversation.system ?? $("system-prompt").value.trim(),
      max_tokens: conversation.maxTokens ?? (Number($("max-tokens").value) || 512),
      workload: $("workload").value,
      targets: Object.fromEntries(targets.map((t) => [t, historyFor(conversation, t, prompt)])),
    };
    const turn = { user: prompt, answers: Object.fromEntries(targets.map((t) => [t, { text: "", streaming: true }])) };
    conversation.turns.push(turn);
    if (conversation.title === "New chat") conversation.title = prompt.slice(0, 48);
    busy = true;
    $("send").disabled = true;
    setStatus("Both models are answering…");
    renderAll();
    const turnNode = $("messages").lastElementChild;
    const cards = Object.fromEntries(targets.map((t) => [t, turnNode.querySelector(`.answer.${t} .answer-body`)]));
    try {
      const response = await api("/v1/chat", { method: "POST", body: JSON.stringify(body) });
      for await (const event of readEvents(response)) {
        const answer = turn.answers[event.target];
        if (event.type === "delta" && answer) {
          answer.text += event.text;
          cards[event.target].textContent = answer.text;
          $("messages").scrollTop = $("messages").scrollHeight;
        } else if (event.type === "reasoning" && answer) {
          answer.reasoning = (answer.reasoning || "") + event.text;
          const body = cards[event.target];
          let block = body.parentElement.querySelector(".reasoning");
          if (!block) {
            block = reasoningBlock("", true);
            body.before(block);
          }
          block.querySelector(".reasoning-text").textContent = answer.reasoning;
        } else if (event.type === "done" && answer) {
          answer.metrics = event.metrics;
          answer.streaming = false;
        } else if (event.type === "error" && answer) {
          answer.error = event.message;
          answer.streaming = false;
        }
      }
      setStatus("");
    } catch (error) {
      for (const answer of Object.values(turn.answers)) {
        if (answer.streaming) Object.assign(answer, { streaming: false, error: error.message });
      }
      setStatus(error.message, true);
    } finally {
      for (const answer of Object.values(turn.answers)) answer.streaming = false;
      busy = false;
      $("send").disabled = false;
      saveConversations();
      renderAll();
    }
  }

  // ---------- projection ----------
  let projectionTimer = null;
  function readProjectionInputs() {
    return {
      monthly_input_tokens: 10 ** Number($("tokens").value),
      prefix_reuse: Number($("reuse").value),
      output_ratio: Number($("ratio").value),
      target_utilization: Number($("util").value),
      instance_type: $("instance").value || null,
      online_hours: Number($("hours").value),
      purchase: $("purchase").value,
    };
  }
  const PURCHASE_LABELS = { on_demand: "on-demand", spot: "Spot", savings_plan: "Savings Plan" };
  function applySetup(instanceType, purchase) {
    $("instance").value = instanceType;
    $("purchase").value = purchase;
    scheduleProjection();
  }
  function scheduleProjection() {
    const inputs = readProjectionInputs();
    $("tokens-out").textContent = compact(inputs.monthly_input_tokens);
    $("reuse-out").textContent = pct(inputs.prefix_reuse);
    $("ratio-out").textContent = inputs.output_ratio.toFixed(2);
    $("util-out").textContent = pct(inputs.target_utilization);
    $("hours-out").textContent = `${inputs.online_hours} h${inputs.online_hours >= 730 ? " (always on)" : ""}`;
    clearTimeout(projectionTimer);
    projectionTimer = setTimeout(() => updateProjection(inputs), 150);
  }
  async function updateProjection(inputs) {
    try {
      const response = await api("/v1/projection", { method: "POST", body: JSON.stringify(inputs) });
      const result = await response.json();
      const p = result.point;
      const cheaper = p.selfhost_usd < p.bedrock_usd ? "selfhost" : "bedrock";
      const ratio = Math.max(p.selfhost_usd, p.bedrock_usd) / Math.max(1e-9, Math.min(p.selfhost_usd, p.bedrock_usd));
      $("projection-result").innerHTML = `
        <div class="row ${cheaper === "selfhost" ? "winner" : ""}"><span>Self-hosted (${p.instances}× ${escapeHtml(result.instance_type)} ${PURCHASE_LABELS[result.purchase]}, ${pct(p.utilization)} busy)</span><span>${usd(p.selfhost_usd, 0)}/mo</span></div>
        <div class="row ${cheaper === "bedrock" ? "winner" : ""}"><span>Bedrock on-demand</span><span>${usd(p.bedrock_usd, 0)}/mo</span></div>
        <div class="row"><span>${cheaper === "selfhost" ? "Self-hosting" : "Bedrock"} is cheaper by</span><span>${ratio.toFixed(1)}×</span></div>
        <div class="row"><span>Self-hosting is cheaper from</span><span>${result.break_even_input_tokens ? `${compact(result.break_even_input_tokens)} tokens/mo` : "never, on this setup"}</span></div>
        <div class="cheapest"><strong>Cheapest setups for this workload</strong>${result.cheapest
          .map((o, i) => `<button type="button" class="option" data-i="${i}"><span>${escapeHtml(o.instance_type)} · ${PURCHASE_LABELS[o.purchase]} · ${o.instances}×</span><span class="${o.selfhost_usd < o.bedrock_usd ? "good" : ""}">${usd(o.selfhost_usd, 0)}/mo</span></button>`)
          .join("")}</div>`;
      $("projection-result").querySelectorAll(".option").forEach((button) => {
        const option = result.cheapest[Number(button.dataset.i)];
        button.addEventListener("click", () => applySetup(option.instance_type, option.purchase));
      });
      drawChart(result.curve, inputs.monthly_input_tokens);
    } catch (error) {
      $("projection-result").textContent = `Projection unavailable: ${error.message}`;
    }
  }
  function drawChart(curve, marker) {
    const svg = $("chart");
    const W = 320, H = 200, L = 40, R = 8, T = 8, B = 24;
    const xs = curve.map((p) => Math.log10(p.monthly_input_tokens));
    const ys = curve.flatMap((p) => [p.bedrock_usd, p.selfhost_usd]).map((v) => Math.log10(Math.max(v, 1)));
    const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
    const [y0, y1] = [Math.floor(Math.min(...ys)), Math.ceil(Math.max(...ys))];
    const sx = (v) => L + ((v - x0) / (x1 - x0)) * (W - L - R);
    const sy = (v) => H - B - ((Math.log10(Math.max(v, 1)) - y0) / Math.max(1, y1 - y0)) * (H - T - B);
    const line = (key) => curve.map((p, i) => `${i ? "L" : "M"}${sx(xs[i]).toFixed(1)},${sy(p[key]).toFixed(1)}`).join("");
    const grid = [];
    for (let y = y0; y <= y1; y += 1) {
      const py = H - B - ((y - y0) / Math.max(1, y1 - y0)) * (H - T - B);
      grid.push(`<line class="grid" x1="${L}" x2="${W - R}" y1="${py}" y2="${py}"/><text x="${L - 4}" y="${py + 3}" text-anchor="end">$${compact(10 ** y)}</text>`);
    }
    for (let x = Math.ceil(x0); x <= x1; x += 1) {
      grid.push(`<text x="${sx(x)}" y="${H - 8}" text-anchor="middle">${compact(10 ** x)}</text>`);
    }
    const mx = sx(Math.log10(marker));
    svg.innerHTML = `<title id="chart-title">Monthly cost against monthly input tokens</title>${grid.join("")}
      <path class="line-bedrock" d="${line("bedrock_usd")}"/>
      <path class="line-selfhost" d="${line("selfhost_usd")}"/>
      <line class="marker" x1="${mx}" x2="${mx}" y1="${T}" y2="${H - B}"/>`;
  }

  // ---------- token dialog ----------
  function askToken() {
    const dialog = $("token-dialog");
    if (dialog.open) return;
    $("token-input").value = getToken();
    dialog.showModal();
  }
  // Save is the only submit button, so pressing Enter in the field saves too.
  $("token-form").addEventListener("submit", (event) => {
    event.preventDefault();
    setToken($("token-input").value.trim());
    $("token-dialog").close();
    setStatus("");
    init();
  });
  $("token-cancel").addEventListener("click", () => $("token-dialog").close());

  // ---------- wiring ----------
  $("composer").addEventListener("submit", (event) => {
    event.preventDefault();
    const prompt = $("prompt").value;
    $("prompt").value = "";
    $("prompt").style.height = "";
    send(prompt);
  });
  $("prompt").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("composer").requestSubmit();
    }
  });
  $("prompt").addEventListener("input", () => {
    const box = $("prompt");
    box.style.height = "auto";
    box.style.height = `${Math.min(box.scrollHeight, 200)}px`;
  });
  document.querySelectorAll("[data-prompt]").forEach((chip) =>
    chip.addEventListener("click", () => send(chip.dataset.prompt)),
  );
  $("new-chat").addEventListener("click", () => newConversation());
  $("run-showcase").addEventListener("click", runShowcase);
  $("showcase-top").addEventListener("click", runShowcase);
  // Cheapest GPU machine the gateway lists for this model (it only lists ones the model fits on).
  function cheapestGpu(priceField) {
    const gpus = Object.entries(config.instances).filter(([, i]) => i.kind === "gpu" && i[priceField]);
    gpus.sort((a, b) => a[1][priceField] - b[1][priceField]);
    return gpus.length ? gpus[0][0] : config.selfhost.instance_type;
  }
  const PRESETS = {
    demo: {
      note: "What this demo machine would cost serving 1B tokens a month around the clock. CPU never breaks even.",
      values: { tokens: 9, reuse: 0.6, ratio: 0.15, util: 0.6, hours: 730 },
      instance: () => config.selfhost.instance_type,
      purchase: () => config.selfhost.purchase,
    },
    hours: {
      note: "The cheapest GPU machine the model fits on, on Spot, scaled to zero outside 220 hours a month. GPU speeds are assumptions until benchmarked.",
      values: { tokens: 9, reuse: 0.6, ratio: 0.15, util: 0.6, hours: 220 },
      instance: () => cheapestGpu("spot_usd_per_hour"),
      purchase: () => "spot",
    },
    scale: {
      note: "The proposed architecture's target: agent traffic with 90% of each prompt reused from the KV cache, 20B input tokens a month, GPUs on a 1-year Savings Plan. GPU speeds are assumptions until benchmarked.",
      values: { tokens: Math.log10(2e10), reuse: 0.9, ratio: 0.1, util: 0.6, hours: 730 },
      instance: () => cheapestGpu("savings_plan_usd_per_hour"),
      purchase: () => "savings_plan",
    },
  };
  document.querySelectorAll("[data-preset]").forEach((button) =>
    button.addEventListener("click", () => {
      if (!config) return;
      const preset = PRESETS[button.dataset.preset];
      for (const [id, value] of Object.entries(preset.values)) $(id).value = value;
      $("instance").value = preset.instance();
      $("purchase").value = preset.purchase();
      $("preset-note").textContent = preset.note;
      scheduleProjection();
    }),
  );
  $("token-button").addEventListener("click", askToken);
  $("toggle-sidebar").addEventListener("click", () => $("sidebar").classList.toggle("open"));
  $("toggle-costs").addEventListener("click", () => $("costs").classList.toggle("open"));
  $("close-costs").addEventListener("click", () => $("costs").classList.remove("open"));
  ["tokens", "reuse", "ratio", "util", "hours", "instance", "purchase"].forEach((id) => $(id).addEventListener("input", scheduleProjection));

  async function checkModelReady() {
    try {
      const health = await (await fetch("/healthz")).json();
      if (health.model_ready) {
        if ($("status").dataset.loading) setStatus("");
        delete $("status").dataset.loading;
        return;
      }
      $("status").dataset.loading = "1";
      setStatus("The self-hosted model is still starting; on first boot it downloads the model weights.");
      setTimeout(checkModelReady, 15000);
    } catch {
      /* health is advisory; chat errors explain themselves */
    }
  }

  async function init() {
    try {
      const response = await api("/v1/config");
      config = await response.json();
      TARGETS.selfhost.label = `Self-hosted ${config.model.engine} · ${config.selfhost.kind.toUpperCase()}`;
      TARGETS.bedrock.label = `Amazon Bedrock · ${config.bedrock.label}`;
      $("model-label").textContent = config.bedrock.same_model
        ? `${config.model.label} on both sides · self-hosted (${config.model.build}) on ${config.selfhost.instance_type} vs Bedrock ${config.bedrock.model_id}`
        : `Self-hosted ${config.model.label} on ${config.selfhost.instance_type} vs ${config.bedrock.label} on Bedrock (different models)`;
      $("max-tokens").max = config.max_output_tokens;
      $("util").value = config.selfhost.target_utilization;
      $("purchase").value = config.selfhost.purchase || "on_demand";
      $("instance").replaceChildren(
        ...Object.entries(config.instances).map(([name, info]) => {
          const option = document.createElement("option");
          option.value = name;
          option.textContent = `${name} · ${info.kind.toUpperCase()} ${info.hardware} · ${usd(info.usd_per_hour, 2)}/h`;
          option.selected = name === config.selfhost.instance_type;
          return option;
        }),
      );
      $("price-note").textContent = config.price_note;
      setStatus("");
      renderAll();
      checkModelReady();
      scheduleProjection();
    } catch (error) {
      $("model-label").textContent = "not connected";
      setStatus(error.message, true);
    }
  }

  renderAll();
  init();
})();
