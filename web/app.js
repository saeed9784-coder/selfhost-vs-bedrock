// Keepsake staff web app: Cognito Hosted UI login (authorization code + PKCE), document admin
// and usage dashboard. No framework and no build step. All API data is written with
// textContent / createElement only, never innerHTML, so document text cannot inject markup.
"use strict";

(function keepsake() {
  const CONFIG = window.KEEPSAKE_CONFIG;
  const SESSION = { idToken: "keepsake.idToken", expiresAt: "keepsake.expiresAt",
                    verifier: "keepsake.pkceVerifier", state: "keepsake.pkceState",
                    returnTo: "keepsake.returnTo", user: "keepsake.user" };
  const SVG_NS = "http://www.w3.org/2000/svg";

  // ---------- helpers ----------------------------------------------------------------------
  function storageGet(key) {
    try { return window.sessionStorage.getItem(key); } catch (_e) { return null; }
  }
  function storageSet(key, value) {
    try { window.sessionStorage.setItem(key, value); } catch (_e) { /* private mode: stay signed out */ }
  }
  function storageClear() {
    try { Object.values(SESSION).forEach((k) => window.sessionStorage.removeItem(k)); } catch (_e) { /* ignore */ }
  }

  function showStatus(message, kind) {
    const el = document.getElementById("status");
    if (!el) return;
    el.textContent = message;
    el.className = "status visible " + (kind || "info");
  }

  function base64Url(bytes) {
    let text = "";
    bytes.forEach((b) => { text += String.fromCharCode(b); });
    return btoa(text).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  }

  function randomString() {
    const bytes = new Uint8Array(32);
    crypto.getRandomValues(bytes);
    return base64Url(bytes);
  }

  async function sha256(text) {
    const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
    return base64Url(new Uint8Array(digest));
  }

  function decodeJwtPayload(token) {
    const part = token.split(".")[1] || "";
    const json = atob(part.replace(/-/g, "+").replace(/_/g, "/"));
    return JSON.parse(decodeURIComponent(escape(json)));
  }

  function cell(text, className) {
    const td = document.createElement("td");
    td.textContent = text === null || text === undefined ? "-" : String(text);
    if (className) td.className = className;
    return td;
  }

  function button(label, testId, onClick, className) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.dataset.testid = testId;
    if (className) b.className = className;
    b.addEventListener("click", onClick);
    return b;
  }

  // ---------- auth (Cognito Hosted UI, PKCE) -----------------------------------------------
  function isSignedIn() {
    const expires = Number(storageGet(SESSION.expiresAt) || 0);
    return Boolean(storageGet(SESSION.idToken)) && Date.now() < expires;
  }

  async function signIn() {
    const verifier = randomString();
    const state = randomString();
    storageSet(SESSION.verifier, verifier);
    storageSet(SESSION.state, state);
    storageSet(SESSION.returnTo, window.location.pathname);
    const params = new URLSearchParams({
      response_type: "code", client_id: CONFIG.clientId, redirect_uri: CONFIG.redirectUri,
      scope: "openid email profile", state: state,
      code_challenge_method: "S256", code_challenge: await sha256(verifier),
    });
    window.location.assign(CONFIG.cognitoDomain + "/oauth2/authorize?" + params.toString());
  }

  function signOut() {
    storageClear();
    const params = new URLSearchParams({ client_id: CONFIG.clientId, logout_uri: CONFIG.redirectUri });
    window.location.assign(CONFIG.cognitoDomain + "/logout?" + params.toString());
  }

  async function completeSignIn() {
    const query = new URLSearchParams(window.location.search);
    const code = query.get("code");
    if (!code) return;
    if (query.get("state") !== storageGet(SESSION.state)) {
      showStatus("Sign-in failed: the response did not match this browser session.", "error");
      return;
    }
    const body = new URLSearchParams({
      grant_type: "authorization_code", client_id: CONFIG.clientId, code: code,
      redirect_uri: CONFIG.redirectUri, code_verifier: storageGet(SESSION.verifier) || "",
    });
    const response = await fetch(CONFIG.cognitoDomain + "/oauth2/token", {
      method: "POST", headers: { "Content-Type": "application/x-www-form-urlencoded" }, body: body,
    });
    if (!response.ok) {
      showStatus("Sign-in failed. Please try again.", "error");
      return;
    }
    const tokens = await response.json();
    const claims = decodeJwtPayload(tokens.id_token);
    storageSet(SESSION.idToken, tokens.id_token);
    storageSet(SESSION.expiresAt, String(Date.now() + Number(tokens.expires_in || 3600) * 1000));
    storageSet(SESSION.user, claims.email || claims["cognito:username"] || "");
    const returnTo = storageGet(SESSION.returnTo) || "index.html";
    window.history.replaceState({}, "", window.location.pathname);
    if (!window.location.pathname.endsWith(returnTo.split("/").pop())) window.location.assign(returnTo);
  }

  function renderHeader() {
    const signedIn = isSignedIn();
    document.getElementById("login-button").hidden = signedIn;
    document.getElementById("logout-button").hidden = !signedIn;
    document.getElementById("user-name").textContent = signedIn ? (storageGet(SESSION.user) || "") : "";
    document.getElementById("login-button").addEventListener("click", () => {
      signIn().catch(() => showStatus("Could not start sign-in.", "error"));
    });
    document.getElementById("logout-button").addEventListener("click", signOut);
  }

  // ---------- API --------------------------------------------------------------------------
  async function api(method, path, body) {
    if (!isSignedIn()) throw new Error("Please sign in first.");
    const response = await fetch(CONFIG.apiUrl + path, {
      method: method,
      headers: { "Authorization": storageGet(SESSION.idToken), "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    let envelope;
    try { envelope = await response.json(); } catch (_e) { envelope = null; }
    if (!response.ok || !envelope || !envelope.success) {
      throw new Error((envelope && envelope.error) || ("Request failed (" + response.status + ")"));
    }
    return envelope.data;
  }

  function run(action, successMessage) {
    return action()
      .then(() => { if (successMessage) showStatus(successMessage, "info"); })
      .catch((error) => showStatus(error.message, "error"));
  }

  // ---------- admin page (FR2.6) -------------------------------------------------------------
  async function loadDocuments() {
    const data = await api("GET", "/documents");
    const tbody = document.getElementById("documents-body");
    tbody.replaceChildren();
    data.documents.forEach((doc) => {
      const row = document.createElement("tr");
      [doc.document_id, doc.title, doc.owner_team, doc.classification, doc.status,
       doc.current_version_id, doc.pending_version_id].forEach((v) => row.appendChild(cell(v)));
      const actions = document.createElement("td");
      actions.appendChild(button("History", "documents-row-history-button", () => run(() => loadHistory(doc.document_id)), "secondary"));
      actions.appendChild(button("Retire", "documents-row-retire-button", () => {
        if (window.confirm("Retire " + doc.document_id + "? Agents will no longer receive it.")) {
          run(async () => { await api("DELETE", "/documents/" + encodeURIComponent(doc.document_id)); await loadDocuments(); }, "Document retired.");
        }
      }, "danger"));
      if (doc.pending_version_id) {
        actions.appendChild(button("Approve " + doc.pending_version_id, "documents-row-approve-button", () => run(async () => {
          await api("POST", "/documents/" + encodeURIComponent(doc.document_id) + "/approve", { version_id: doc.pending_version_id });
          await loadDocuments();
        }, "Version approved.")));
      }
      row.appendChild(actions);
      tbody.appendChild(row);
    });
  }

  async function loadHistory(documentId) {
    const data = await api("GET", "/documents/" + encodeURIComponent(documentId) + "/versions");
    const tbody = document.getElementById("history-body");
    tbody.replaceChildren();
    data.versions.forEach((v) => {
      const row = document.createElement("tr");
      [v.version_id, v.status, v.author, v.created_at, v.chunk_hashes.length,
       v.injection_findings.join("; ") || "none"].forEach((value) => row.appendChild(cell(value)));
      tbody.appendChild(row);
    });
  }

  function formValues(form) {
    return Object.fromEntries(new FormData(form).entries());
  }

  function initAdmin() {
    document.getElementById("refresh-documents").addEventListener("click", () => run(loadDocuments));
    document.getElementById("upload-file").addEventListener("change", (event) => {
      const file = event.target.files[0];
      if (!file) return;
      file.text().then((text) => { document.querySelector("#upload-form textarea").value = text; })
        .catch(() => showStatus("Could not read that file.", "error"));
    });
    document.getElementById("upload-form").addEventListener("submit", (event) => {
      event.preventDefault();
      const values = formValues(event.target);
      run(async () => {
        const data = await api("POST", "/documents", values);
        event.target.reset();
        await loadDocuments();
        if (data.version.status === "quarantined") {
          throw new Error("Uploaded, but quarantined: " + data.version.injection_findings.join("; "));
        }
      }, "Document uploaded.");
    });
    document.getElementById("update-form").addEventListener("submit", (event) => {
      event.preventDefault();
      const values = formValues(event.target);
      run(async () => {
        await api("PUT", "/documents/" + encodeURIComponent(values.document_id), { text: values.text });
        await loadDocuments();
      }, "New version saved.");
    });
    document.getElementById("history-form").addEventListener("submit", (event) => {
      event.preventDefault();
      run(() => loadHistory(formValues(event.target).document_id));
    });
    if (isSignedIn()) run(loadDocuments);
  }

  // ---------- dashboard page (FR5.3) ---------------------------------------------------------
  const FORMATS = {
    tokens_sent: (v) => Number(v).toLocaleString(),
    tokens_from_cache: (v) => Number(v).toLocaleString(),
    full_price_tokens: (v) => Number(v).toLocaleString(),
    spend_usd: (v) => "$" + Number(v).toFixed(2),
    saved_usd: (v) => "$" + Number(v).toFixed(2),
    storage_cost_usd: (v) => "$" + Number(v).toFixed(4),
    hit_rate: (v) => (Number(v) * 100).toFixed(1) + "%",
    p50_latency_ms: (v) => Math.round(v) + " ms",
    p99_latency_ms: (v) => Math.round(v) + " ms",
  };

  function renderTotals(totals) {
    document.querySelectorAll("[data-testid=usage-totals] [data-field]").forEach((el) => {
      el.textContent = FORMATS[el.dataset.field](totals[el.dataset.field]);
    });
  }

  function renderGroups(groups) {
    const tbody = document.getElementById("usage-body");
    tbody.replaceChildren();
    groups.forEach((g) => {
      const row = document.createElement("tr");
      row.appendChild(cell(g.key));
      Object.keys(FORMATS).forEach((field) => row.appendChild(cell(FORMATS[field](g[field]), "num")));
      tbody.appendChild(row);
    });
  }

  function svg(tag, attrs, text) {
    const el = document.createElementNS(SVG_NS, tag);
    Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, String(v)));
    if (text !== undefined) el.textContent = text;
    return el;
  }

  function renderChart(points) {
    const chart = document.getElementById("hit-rate-chart");
    chart.replaceChildren();
    const fallback = document.getElementById("hit-rate-fallback");
    if (!points.length) { fallback.textContent = "No requests in this period."; return; }
    const width = 600, height = 220, pad = 30;
    const x = (i) => pad + (points.length === 1 ? (width - 2 * pad) / 2 : i * (width - 2 * pad) / (points.length - 1));
    const y = (rate) => height - pad - rate * (height - 2 * pad);
    chart.appendChild(svg("line", { class: "axis", x1: pad, y1: height - pad, x2: width - pad, y2: height - pad }));
    chart.appendChild(svg("line", { class: "axis", x1: pad, y1: pad, x2: pad, y2: height - pad }));
    chart.appendChild(svg("text", { x: 2, y: pad + 4 }, "100%"));
    chart.appendChild(svg("text", { x: 8, y: height - pad + 4 }, "0%"));
    const path = points.map((p, i) => (i ? "L" : "M") + x(i).toFixed(1) + " " + y(p.hit_rate).toFixed(1)).join(" ");
    chart.appendChild(svg("path", { class: "line", d: path }));
    points.forEach((p, i) => {
      const dot = svg("circle", { class: "point", cx: x(i), cy: y(p.hit_rate), r: 3 });
      dot.appendChild(svg("title", {}, p.bucket_start.slice(0, 10) + ": " + (p.hit_rate * 100).toFixed(1) + "% of " + p.requests + " requests"));
      chart.appendChild(dot);
    });
    chart.appendChild(svg("text", { x: pad, y: height - 8 }, points[0].bucket_start.slice(0, 10)));
    chart.appendChild(svg("text", { x: width - pad - 60, y: height - 8 }, points[points.length - 1].bucket_start.slice(0, 10)));
    fallback.textContent = points.map((p) => p.bucket_start.slice(0, 10) + " " + (p.hit_rate * 100).toFixed(0) + "%").join(", ");
  }

  async function loadUsage(form) {
    const values = formValues(form);
    const params = new URLSearchParams();
    Object.entries(values).forEach(([k, v]) => { if (v) params.set(k, v); });
    const data = await api("GET", "/usage?" + params.toString());
    renderTotals(data.totals);
    renderGroups(data.groups);
    renderChart(data.hit_rate_over_time);
  }

  function initDashboard() {
    const form = document.getElementById("usage-form");
    form.addEventListener("submit", (event) => { event.preventDefault(); run(() => loadUsage(form)); });
    if (isSignedIn()) run(() => loadUsage(form));
  }

  // ---------- boot -------------------------------------------------------------------------
  document.addEventListener("DOMContentLoaded", () => {
    if (!CONFIG) {
      showStatus("Configuration is missing (config.js).", "error");
      return;
    }
    completeSignIn()
      .catch(() => showStatus("Sign-in failed. Please try again.", "error"))
      .finally(() => {
        renderHeader();
        const page = document.body.dataset.page;
        if (page === "admin") initAdmin();
        if (page === "dashboard") initDashboard();
      });
  });
})();
