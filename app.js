// Claret: type what you sell, get the closest USPTO Trademark ID Manual entries.
// Everything runs in the browser: a small sentence-embedding model (all-MiniLM-L6-v2,
// via transformers.js) turns the query into a vector, which is compared with a
// pre-computed bit vector for every ID Manual entry, then blended with a word match.
const MODEL = "Xenova/all-MiniLM-L6-v2";
const TJS = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1";
const DIM = 384, BYTES = DIM / 8;
const SHOW = 8, MORE = 30;
const FEE = 350; // USPTO base application fee per class (since January 18, 2025)

const $ = (id) => document.getElementById(id);
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

let E = null, V = null, CLS = {}, LOW = null, embed = null, modelFailed = false;
let picks = [];
try { picks = JSON.parse(localStorage.getItem("claret-picks") || "[]"); } catch (e) { picks = []; }

const status = (t) => { $("status").textContent = t; };

// ---------- data ----------
const dataReady = (async () => {
  const [e, v, c] = await Promise.all([
    fetch("data/entries.json").then((r) => r.json()),
    fetch("data/vectors.bin").then((r) => r.arrayBuffer()),
    fetch("data/classes.json").then((r) => r.json()),
  ]);
  E = e; V = new Uint8Array(v); CLS = c;
  LOW = E.map((x) => " " + plain(x[2]).toLowerCase().replace(/[^a-z0-9]+/g, " ") + " ");
  $("ver").textContent = `${E.length.toLocaleString()} entries from the USPTO Trademark ID Manual.`;
})();

let modelPromise = null;
function loadModel() {
  if (!modelPromise) modelPromise = (async () => {
    try {
      const { pipeline, env } = await import(TJS);
      env.allowLocalModels = false;
      const ex = await pipeline("feature-extraction", MODEL, { dtype: "q8" });
      embed = async (t) => (await ex(t, { pooling: "mean", normalize: true })).data;
    } catch (err) {
      console.warn("model failed, word match only", err);
      modelFailed = true;
    }
  })();
  return modelPromise;
}

function plain(d) { return d.replace(/\[[^\]]*\]/g, " ").replace(/[{}]/g, " ").replace(/\s+/g, " ").trim(); }

// ---------- search ----------
const FILLER = new Set("i we my our me us a an the of for and or to in on with that which who is are be sell sells selling sold make makes making made offer offering provide providing do does company business brand products product stuff things kind kinds type types like etc".split(" "));

function stems(q) {
  return q.toLowerCase().replace(/[^a-z0-9]+/g, " ").split(" ")
    .filter((w) => w.length > 1 && !FILLER.has(w))
    .map((w) => w.length > 4 ? w.replace(/(ies)$/, "y").replace(/(es|s|ing|ed)$/, "") : w);
}

function cleanQuery(q) {
  return q.replace(/^\s*(i|we)\s+(sell|make|offer|provide|run|own|do)\s+/i, "").replace(/^\s*(an?|the|my|our)\s+/i, "").trim();
}

async function search(q) {
  const words = stems(q);
  const n = E.length, score = new Float32Array(n);
  if (embed) {
    const qv = await embed(cleanQuery(q));
    // per byte position, a 256-entry table of the dot product of q with that byte's 8 signs
    const T = new Float32Array(BYTES * 256);
    for (let k = 0; k < BYTES; k++) for (let b = 0; b < 256; b++) {
      let s = 0;
      for (let j = 0; j < 8; j++) s += (b & (128 >> j)) ? qv[k * 8 + j] : -qv[k * 8 + j];
      T[k * 256 + b] = s;
    }
    const norm = 1 / Math.sqrt(DIM);
    for (let i = 0, o = 0; i < n; i++) {
      let s = 0;
      for (let k = 0; k < BYTES; k++, o++) s += T[k * 256 + V[o]];
      score[i] = s * norm;
    }
  }
  if (words.length) {
    const w = embed ? 0.12 : 1, keys = words.map((x) => " " + x);
    for (let i = 0; i < n; i++) {
      const L = LOW[i];
      let hit = 0;
      for (const k of keys) if (L.includes(k)) hit++;
      if (hit) score[i] += w * hit / keys.length - (embed ? 0 : L.length / 1e4);
    }
  }
  const idx = [];
  for (let i = 0; i < n; i++) if (score[i] > 0) idx.push(i);
  idx.sort((a, b) => score[b] - score[a]);
  return idx.slice(0, MORE);
}

// ---------- fill-in templates ----------
// Apps, software and stores are best described by the ID Manual's fill-in entries,
// which a plain similarity search ranks low; offer them first with the blank filled.
const TEMPLATES = [
  [/\b(apps?|mobile app(lication)?s?|application)\b/i, ["009-4414", "042-2072"]],
  [/\b(software|saas|platform|web ?app|website|online tool)\b/i, ["009-5429", "042-2072", "042-3071"]],
  [/\b(store|shop|boutique|retail)\b/i, ["035-1089"]],
];
let byId = null;
function templates(q) {
  if (!byId) { byId = new Map(); E.forEach((e, i) => byId.set(e[0], i)); }
  const out = [];
  for (const [re, ids] of TEMPLATES) {
    const m = q.match(re);
    if (!m) continue;
    let fill;
    if (ids[0].startsWith("035")) {
      fill = q.replace(/\b(online|e-?commerce|retail|store|shop|boutique|an?|the|my|our|we|i|sell|selling|run|own)\b/gi, " ");
    } else {
      fill = q.slice(m.index + m[0].length).replace(/^\s*(that|which|to|for|used for|used to|where)\b/i, "");
      fill = fill.replace(/^\s*(helps|lets|allows|enables)\s+(\w+)\s+(to\s+)?/i, "use by $2 to ");
    }
    fill = fill.replace(/\s+/g, " ").trim();
    for (const id of ids) if (byId.has(id) && !out.some((o) => o[0] === byId.get(id))) out.push([byId.get(id), fill]);
  }
  return out;
}

// ---------- results ----------
function descHTML(d, i, pre) {
  let k = 0;
  return d.split(/(\{[^}]*\}|\[[^\]]*\])/).map((p) => {
    if (p.startsWith("{")) {
      const hint = p.slice(1, -1);
      const v = k === 0 && pre ? ` value="${esc(pre)}"` : "";
      return `<input class="fill"${v} data-k="${k++}" placeholder="${esc(hint)}" title="${esc(hint)}" size="${Math.min(40, Math.max(10, hint.length * 0.7 | 0))}">`;
    }
    if (p.startsWith("[")) return `<span class="meta">${esc(p)}</span>`;
    return esc(p);
  }).join("");
}

function resHTML(i, pre) {
  const [id, c, d, note] = E[i];
  const picked = picks.some((p) => p.id === id);
  return `<div class="res${picked ? " picked" : ""}" data-i="${i}">
    <div class="body"><div class="desc">${descHTML(d, i, pre)}</div>
    <div class="meta"><span class="cls">Class ${c}</span>${esc(CLS[c] || "")} · ${esc(id)}</div>
    ${note ? `<details class="note"><summary>USPTO note</summary> ${esc(note)}</details>` : ""}</div>
    <button class="btn sm ${picked ? "alt" : ""}" data-add="${i}">${picked ? "Added" : "Add"}</button></div>`;
}

async function run() {
  const raw = $("q").value.trim();
  if (!raw) { $("q").focus(); return; }
  $("go").disabled = true;
  status("Loading the ID Manual…");
  await dataReady;
  if (!embed && !modelFailed) { status("Loading the matching model (first time only, about 25 MB)…"); await loadModel(); }
  status(modelFailed ? "Word match only (the matching model did not load)." : "");
  const items = raw.split(/[\n;,]+/).map((s) => s.trim()).filter(Boolean).slice(0, 12);
  let html = "";
  for (const it of items) {
    const tpl = templates(it), seen = new Set(tpl.map((t) => t[0]));
    const ids = tpl.concat((await search(it)).filter((i) => !seen.has(i)).map((i) => [i, ""]));
    const one = ([i, pre]) => resHTML(i, pre);
    html += `<div class="group card"><h2>${esc(it)}</h2>` +
      (ids.length ? ids.slice(0, SHOW).map(one).join("") +
        (ids.length > SHOW ? `<div class="more" hidden>${ids.slice(SHOW).map(one).join("")}</div><button class="btn sm alt" data-more>More matches</button>` : "")
        : `<p class="empty">No close entries. Try other words.</p>`) + `</div>`;
  }
  $("results").innerHTML = html;
  $("go").disabled = false;
}

// ---------- picks ----------
function save() { try { localStorage.setItem("claret-picks", JSON.stringify(picks)); } catch (e) {} }

function textOf(p) {
  // fill the {blanks}; drop [bracketed] USPTO remarks, which are not part of the identification
  let k = 0;
  return p.d.split(/(\{[^}]*\}|\[[^\]]*\])/).map((s) => {
    if (s.startsWith("{")) { const v = (p.f[k++] || "").trim(); return v ? v : s; }
    if (s.startsWith("[")) return "";
    return s;
  }).join("").replace(/\s+/g, " ").replace(/\s+([,.;])/g, "$1").trim();
}

function byClass() {
  const m = new Map();
  for (const p of picks) { if (!m.has(p.c)) m.set(p.c, []); m.get(p.c).push(p); }
  return [...m.entries()].sort((a, b) => a[0] - b[0]);
}

function renderPicks() {
  const g = byClass();
  if (!g.length) {
    $("picks").innerHTML = `<p class="empty">Add entries to build the identification for each class.</p>`;
    $("pickfoot").hidden = true; return;
  }
  $("picks").innerHTML = g.map(([c, ps]) => `<div class="pc"><div class="h"><b>Class ${c}</b><span>${esc(CLS[c] || "")}</span>
      <button class="btn sm alt" data-copy="${c}">Copy</button></div>
      ${ps.map((p) => `<div class="t">${esc(textOf(p)).replace(/\{([^}]*)\}/g, "<i title=\"fill this in\">{$1}</i>")}<span class="id">${esc(p.id)}</span><button class="x" data-del="${esc(p.id)}" title="Remove">×</button></div>`).join("")}</div>`).join("");
  $("pickfoot").hidden = false;
  const blanks = picks.some((p) => /\{/.test(textOf(p)));
  $("tot").textContent = `${g.length} class${g.length > 1 ? "es" : ""} · USPTO filing fee $${(g.length * FEE).toLocaleString()} ($${FEE} per class)` + (blanks ? " · fill in the red blanks" : "");
}

function classText(c) { return picks.filter((p) => p.c === c).map(textOf).join("; "); }

async function copy(t, el) {
  try { await navigator.clipboard.writeText(t); } catch (e) {
    const ta = document.createElement("textarea"); ta.value = t; document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove();
  }
  if (el) { el.textContent = "Copied"; setTimeout(() => { el.textContent = ""; }, 1800); }
}

function fillsFor(row) {
  return [...row.querySelectorAll("input.fill")].map((x) => x.value);
}

// ---------- events ----------
$("go").addEventListener("click", run);
$("q").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); run(); } });
document.querySelectorAll("[data-ex]").forEach((b) => b.addEventListener("click", () => { $("q").value = b.dataset.ex; run(); }));
$("q").addEventListener("focus", () => { dataReady.then(loadModel); }, { once: true });

$("results").addEventListener("click", (e) => {
  const m = e.target.closest("[data-more]");
  if (m) { m.previousElementSibling.hidden = false; m.remove(); return; }
  const a = e.target.closest("[data-add]");
  if (!a) return;
  const i = +a.dataset.add, [id, c, d] = E[i];
  const at = picks.findIndex((p) => p.id === id);
  if (at >= 0) picks.splice(at, 1); else picks.push({ id, c, d, f: fillsFor(a.closest(".res")) });
  save(); renderPicks();
  const row = a.closest(".res"), on = at < 0;
  row.classList.toggle("picked", on); a.classList.toggle("alt", on); a.textContent = on ? "Added" : "Add";
});

$("results").addEventListener("input", (e) => {
  const f = e.target.closest("input.fill");
  if (!f) return;
  const row = f.closest(".res"), id = E[+row.dataset.i][0], p = picks.find((x) => x.id === id);
  if (p) { p.f = fillsFor(row); save(); renderPicks(); }
});

$("picks").addEventListener("click", (e) => {
  const d = e.target.closest("[data-del]");
  if (d) {
    picks = picks.filter((p) => p.id !== d.dataset.del); save(); renderPicks();
    document.querySelectorAll(".res").forEach((r) => {
      if (E[+r.dataset.i][0] === d.dataset.del) { r.classList.remove("picked"); const b = r.querySelector("[data-add]"); b.classList.remove("alt"); b.textContent = "Add"; }
    });
    return;
  }
  const c = e.target.closest("[data-copy]");
  if (c) { copy(classText(+c.dataset.copy)); c.textContent = "Copied"; setTimeout(() => { c.textContent = "Copy"; }, 1800); }
});

$("copyall").addEventListener("click", () => copy(byClass().map(([c]) => `Class ${c}: ${classText(c)}`).join("\n\n"), $("copiedall")));

renderPicks();
dataReady.then(renderPicks).catch(() => status("Could not load the ID Manual data."));
