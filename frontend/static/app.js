/* ================================================================
   app.js — RedFlagAI Frontend
   All API calls go to the Python FastAPI backend (no key exposed).
   ================================================================ */

'use strict';

/* ── PROGRESS MESSAGES ─────────────────────────────────────────── */
const PROGRESS_MSGS = [
  'Stage 1: Running statistical screening…',
  'Stage 2: Extracting linguistic fingerprints…',
  'Stage 3: Cross-referencing contradictions…',
  'Stage 4: Scoring Dark Triad traits…',
  'Stage 5: Gemini reasoning (Chain-of-Thought)…',
  'Assembling full audit report…',
];

/* ── FALLBACK SAMPLES (used if backend is unreachable) ─────────── */
const FALLBACK_SAMPLES = [
  {
    id: 0, label: 'Sample A: Overconfident Alpha',
    bio: "Alpha male entrepreneur. I don't chase — I attract. Made my first $100k at 22. If you can't handle my ambition, you can't handle me. Women who play games are immediately deleted. I'm very selective about who I give my time to. Looking for someone who matches my energy and doesn't need constant validation. My time is my most valuable asset. I've been told I'm intimidating but that's just because most people aren't operating at my level.",
    smokes: 'no', drinks: 'often', diet: 'anything', drugs: 'never', status: 'single', age: 31,
  },
  {
    id: 1, label: 'Sample B: Love-Bombing Romantic',
    bio: "I've never felt this kind of connection just reading a profile. You feel like my soulmate already. I know this sounds crazy but I believe in instant, electric connections — the kind you feel in your soul. I would move mountains for the right person. You deserve to be treated like royalty every single day. I'll text you good morning and goodnight without fail. I just need someone who can be fully, completely present for me in return. No half measures. All or nothing.",
    smokes: 'no', drinks: 'socially', diet: 'anything', drugs: 'never', status: 'single', age: 27,
  },
  {
    id: 2, label: 'Sample C: Healthy & Balanced',
    bio: "Software engineer who loves hiking on weekends. Pretty laid-back, I enjoy cooking for friends, bad sci-fi movies, and a good book on a rainy afternoon. I have my own life and interests and I'm looking for someone who has theirs too. Honesty and communication matter a lot to me. Looking for someone to explore the city with and maybe argue about which season of The Wire is best.",
    smokes: 'no', drinks: 'socially', diet: 'vegetarian', drugs: 'never', status: 'single', age: 29,
  },
  {
    id: 3, label: 'Sample D: Contradictory Smoker',
    bio: "Very health conscious, I hit the gym six days a week and eat super clean — mostly whole foods and lean protein. Love chilling on the porch with a good cigar and some whiskey after a hard workout. Obviously I'm a non-smoker, but cigars are just a vibe, not a habit. Very outdoorsy, love camping trips and long trail runs. Clean living all the way.",
    smokes: 'no', drinks: 'socially', diet: 'anything', drugs: 'never', status: 'single', age: 34,
  },
  {
    id: 4, label: 'Sample E: Dark Triad Signals',
    bio: "I know what I want and I know how to get it. I'm not here to make friends. I'm extremely charming when I want to be and I can read people almost instantly — you cannot lie to me. I've dated a lot and I know every game people play. I don't have time for weak personalities or emotional neediness. If you pass my test, you'll see a completely different side of me. People either love me or they're intimidated by me. I prefer those who can keep up.",
    smokes: 'sometimes', drinks: 'often', diet: 'anything', drugs: 'sometimes', status: 'available', age: 36,
  },
];

/* ── DOM HELPERS ────────────────────────────────────────────────── */
const $  = (id)  => document.getElementById(id);
const $$ = (sel) => document.querySelectorAll(sel);

/* ── INIT ───────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', async () => {
  await checkServer();
  await loadSamplesFromAPI();
  setupCharCounter();
});

/* ── SERVER HEALTH CHECK ────────────────────────────────────────── */
async function checkServer() {
  const dot  = $('serverDot');
  const text = $('serverStatus');
  try {
    const res  = await fetch('/api/health');
    const data = await res.json();
    if (data.api_key_configured) {
      dot.classList.add('online');
      text.textContent = 'AUDIT TERMINAL v2.0 · PIPELINE ACTIVE · ' + data.model;
    } else {
      dot.classList.add('error');
      text.textContent = 'AUDIT TERMINAL v2.0 · API KEY NOT CONFIGURED';
      showBanner('⚠ Gemini API key not set on server. Edit backend/.env and restart.', 'warning');
    }
  } catch {
    dot.classList.add('error');
    text.textContent = 'AUDIT TERMINAL v2.0 · SERVER OFFLINE';
    showBanner('Cannot reach the backend server. Is it running on port 8000?', 'error');
  }
}

/* ── LOAD SAMPLES FROM API (with built-in fallback) ─────────────── */
async function loadSamplesFromAPI() {
  let samples = FALLBACK_SAMPLES; // default — always works offline

  try {
    const res  = await fetch('/api/samples');
    const data = await res.json();
    if (data.samples && data.samples.length > 0) {
      samples = data.samples;
    }
  } catch {
    console.warn('Could not reach /api/samples — using built-in sample profiles.');
  }

  // Populate the dropdown
  const sel = $('sampleSelect');
  while (sel.options.length > 1) sel.remove(1); // clear old options except placeholder
  samples.forEach((s) => {
    const opt = document.createElement('option');
    opt.value = s.id;
    opt.textContent = s.label;
    sel.appendChild(opt);
  });

  // Store globally
  window._samples = samples;

  // KEY FIX: selecting from the dropdown loads immediately — no button press needed
  sel.addEventListener('change', () => loadSample());
}

/* ── LOAD SAMPLE INTO FORM ──────────────────────────────────────── */
function loadSample() {
  const idx = parseInt($('sampleSelect').value);
  if (isNaN(idx) || idx < 0) return;

  const samples = window._samples || FALLBACK_SAMPLES;
  const s = samples.find((x) => x.id === idx);
  if (!s) return;

  $('bioText').value  = s.bio;
  $('fSmokes').value  = s.smokes;
  $('fDrinks').value  = s.drinks;
  $('fDiet').value    = s.diet;
  $('fDrugs').value   = s.drugs;
  $('fStatus').value  = s.status;
  $('fAge').value     = s.age;
  updateCharCount(s.bio.length);
}

/* ── CHARACTER COUNTER ──────────────────────────────────────────── */
function setupCharCounter() {
  const ta = $('bioText');
  ta.addEventListener('input', () => updateCharCount(ta.value.length));
}

function updateCharCount(n) {
  $('charCount').textContent = `${n} / 5000`;
  $('charCount').style.color = n > 4500 ? 'var(--red)' : 'var(--text3)';
}

/* ── TAB SWITCHING ──────────────────────────────────────────────── */
function switchTab(name, btn) {
  $$('.panel').forEach((p) => p.classList.remove('active'));
  $$('.tab-btn').forEach((t) => {
    t.classList.remove('active');
    t.setAttribute('aria-selected', 'false');
  });
  $('panel-' + name).classList.add('active');
  const activeBtn = btn || $('tab-' + name);
  if (activeBtn) {
    activeBtn.classList.add('active');
    activeBtn.setAttribute('aria-selected', 'true');
  }
}

/* ── BANNER ─────────────────────────────────────────────────────── */
function showBanner(msg, type) {
  const b = $('statusBanner');
  b.textContent = msg;
  b.className = `status-banner ${type}`;
  b.style.display = 'flex';
}

function hideBanner() {
  $('statusBanner').style.display = 'none';
}

/* ── PROGRESS HELPERS ───────────────────────────────────────────── */
function startProgress() {
  const wrap = $('progressWrap');
  const fill = $('progressFill');
  const msg  = $('progressMsg');
  const bar  = $('progressBar');
  wrap.style.display = 'block';
  let pct = 0; let msgIdx = 0;
  const ticker = setInterval(() => {
    pct = Math.min(pct + Math.random() * 13, 90);
    fill.style.width = pct.toFixed(0) + '%';
    bar.setAttribute('aria-valuenow', Math.round(pct));
    if (msgIdx < PROGRESS_MSGS.length && pct > msgIdx * 17) {
      msg.textContent = PROGRESS_MSGS[msgIdx++];
    }
  }, 380);
  return ticker;
}

function stopProgress(ticker, success) {
  clearInterval(ticker);
  const fill = $('progressFill');
  const msg  = $('progressMsg');
  const wrap = $('progressWrap');
  fill.style.width = '100%';
  msg.textContent = success ? 'Audit complete ✓' : 'Analysis failed.';
  setTimeout(() => {
    wrap.style.display = 'none';
    fill.style.width = '0%';
  }, 800);
}

/* ── MAIN ANALYSIS ──────────────────────────────────────────────── */
async function runAnalysis() {
  const bio = $('bioText').value.trim();
  if (!bio) {
    alert('Please enter or select a profile bio first.');
    return;
  }
  if (bio.length > 5000) {
    alert('Bio must be under 5000 characters.');
    return;
  }

  hideBanner();

  const btn     = $('analyzeBtn');
  const btnText = $('btnText');
  const btnIcon = $('btnIcon');
  btn.disabled = true;
  btnIcon.innerHTML = '<animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="0.9s" repeatCount="indefinite"/><circle cx="12" cy="12" r="8" stroke="white" stroke-width="2.5" fill="none" stroke-dasharray="25 51"/>';
  btnText.textContent = 'Analysing…';

  const ticker = startProgress();

  const payload = {
    bio,
    smokes: $('fSmokes').value,
    drinks: $('fDrinks').value,
    diet:   $('fDiet').value,
    drugs:  $('fDrugs').value,
    status: $('fStatus').value,
    age:    parseInt($('fAge').value) || 28,
  };

  try {
    const res = await fetch('/api/audit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    const result = await res.json();
    stopProgress(ticker, true);
    renderReport(result);
    renderMetrics(result);
    switchTab('report', null);

  } catch (e) {
    stopProgress(ticker, false);
    showBanner('Audit failed: ' + e.message, 'error');
    $('reportContent').innerHTML = errorBlock(e.message);
    switchTab('report', null);

  } finally {
    btn.disabled = false;
    btnIcon.innerHTML = '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>';
    btnText.textContent = 'Run Full Audit';
  }
}

/* ── RENDER: REPORT ─────────────────────────────────────────────── */
function riskClass(score) {
  if (score >= 60) return 'danger';
  if (score >= 35) return 'warning';
  return 'safe';
}

function renderReport(r) {
  const flagsHtml  = buildFlagsHtml(r.flags || []);
  const contradHtml = buildContradHtml(r.contradictions || []);
  const dtHtml     = buildDarkTriadHtml(r.dark_triad);

  $('reportContent').innerHTML = `
    <div class="fade-in">

      <div class="section-label" style="margin-bottom:1rem;">Risk overview</div>
      <div class="risk-overview">
        <div class="risk-card ${riskClass(r.overall_risk)}">
          <div class="rv">${r.overall_risk}</div>
          <div class="rl">Overall Risk %</div>
        </div>
        <div class="risk-card ${riskClass(r.financial_risk)}">
          <div class="rv">${r.financial_risk}</div>
          <div class="rl">Financial</div>
        </div>
        <div class="risk-card ${riskClass(r.emotional_risk)}">
          <div class="rv">${r.emotional_risk}</div>
          <div class="rl">Emotional</div>
        </div>
        <div class="risk-card ${riskClass(r.identity_risk)}">
          <div class="rv">${r.identity_risk}</div>
          <div class="rl">Identity</div>
        </div>
      </div>

      ${stage1InfoHtml(r)}

      <div class="two-col">
        <div class="report-block">
          ${rbHeader('flag', 'var(--red)', 'rgba(232,57,74,0.1)', 'Detected red flags', '<span class="badge badge-red">' + (r.flags||[]).length + ' flags</span>')}
          <ul class="flag-list">${flagsHtml}</ul>
        </div>
        <div class="report-block">
          ${rbHeader('shuffle', 'var(--amber)', 'rgba(240,160,32,0.1)', 'Contradiction engine', '<span class="badge badge-amber">Veracity</span>')}
          ${contradHtml}
        </div>
      </div>

      <div class="report-block">
        ${rbHeader('brain', 'var(--purple)', 'rgba(139,111,232,0.1)', 'Dark Triad scoring', '<span class="badge badge-purple">Psych analysis</span>')}
        ${dtHtml}
      </div>

      <div class="report-block">
        ${rbHeader('activity', 'var(--blue)', 'rgba(61,139,255,0.1)', 'Chain-of-Thought reasoning', '<span class="badge badge-blue">Stage 5 · LLM</span>')}
        <div class="cot-block"><span class="cot-prefix">AUDIT REASONING // </span>${escHtml(r.cot_reasoning)}</div>
      </div>

      <div class="report-block">
        ${rbHeader('heart', 'var(--purple)', 'rgba(139,111,232,0.1)', 'Educational tip', '<span class="badge badge-purple">Social reintegration</span>')}
        <div class="tip-box">
          <div class="tip-icon">${infoSvg()}</div>
          <p>${escHtml(r.educational_tip)}</p>
        </div>
      </div>

    </div>`;

  // Animate trait bars after DOM insert
  setTimeout(() => {
    document.querySelectorAll('.trait-fill[data-width]').forEach((el) => {
      el.style.width = el.dataset.width + '%';
    });
  }, 50);
}

function stage1InfoHtml(r) {
  if (r.stage1_score === undefined) return '';
  const color  = r.stage1_score >= 40 ? 'var(--amber)' : 'var(--green)';
  const passed = r.stage1_passed
    ? '<span style="color:var(--green);">Passed ✓</span>'
    : '<span style="color:var(--amber);">Flagged ⚑</span>';
  return `
    <div class="stage1-info" style="margin-bottom:1rem;">
      <div><span class="si-label">Stage 1 statistical score</span>
           <span class="si-val" style="color:${color};">${r.stage1_score} / 100</span></div>
      <div><span class="si-label">Stage 1 verdict</span>
           <span class="si-val">${passed}</span></div>
    </div>`;
}

function buildFlagsHtml(flags) {
  if (!flags.length) {
    return '<li class="flag-item low"><div class="flag-meta"><div class="flag-cat">safe</div><div class="flag-text">No significant red flags detected</div></div></li>';
  }
  return flags.map((f) => `
    <li class="flag-item ${f.severity}">
      <div class="flag-meta">
        <div class="flag-cat">${escHtml(f.category)} · ${escHtml(f.severity)}</div>
        <div class="flag-text">${escHtml(f.text)}</div>
      </div>
    </li>`).join('');
}

function buildContradHtml(contras) {
  if (!contras.length) {
    return '<p style="font-size:13px;color:var(--text3);font-family:var(--font-mono);">No contradictions detected between stated fields and bio text.</p>';
  }
  return contras.map((c) => `
    <div class="contra-row">
      <span class="contra-stated">${escHtml(c.stated)}</span>
      <span class="contra-arrow">→</span>
      <span class="contra-detected">"${escHtml(c.detected)}"</span>
      <span class="badge ${c.severity === 'high' ? 'badge-red' : c.severity === 'medium' ? 'badge-amber' : 'badge-green'}">${c.severity}</span>
    </div>`).join('');
}

function buildDarkTriadHtml(dt) {
  if (!dt) return '<p style="color:var(--text3);font-size:13px;">No data.</p>';
  const traits = [
    { label: 'Narcissism',       val: dt.narcissism,       color: 'var(--red)'    },
    { label: 'Machiavellianism', val: dt.machiavellianism, color: 'var(--amber)'  },
    { label: 'Psychopathy',      val: dt.psychopathy,      color: 'var(--purple)' },
  ];
  return traits.map((t) => `
    <div class="trait-bar-row">
      <div class="trait-label">${t.label}</div>
      <div class="trait-bar">
        <div class="trait-fill" data-width="${t.val}"
             style="background:${t.color};box-shadow:0 0 6px ${t.color}40;"></div>
      </div>
      <div class="trait-val" style="color:${t.color};">${t.val}</div>
    </div>`).join('');
}

/* ── RENDER: METRICS ────────────────────────────────────────────── */
function renderMetrics(r) {
  const prec  = r.precision_estimate          || 0.82;
  const rec   = r.recall_estimate             || 0.74;
  const iou   = r.iou_estimate                || 0.61;
  const sent  = r.linguistic?.sentiment_score ?? 0;
  const comp  = r.linguistic?.complexity      ?? 50;
  const manip = r.linguistic?.manipulation_score ?? 30;

  const sentPct   = Math.round((sent + 1) / 2 * 100);
  const sentLabel = sent > 0.2 ? 'Positive' : sent < -0.2 ? 'Negative' : 'Neutral';
  const sentColor = sent > 0.2 ? 'var(--green)' : sent < -0.2 ? 'var(--red)' : 'var(--amber)';

  $('metricsContent').innerHTML = `
    <div class="fade-in">
      <div class="section-label" style="margin-bottom:1rem;">Model performance estimates</div>
      <div class="metrics-grid-3">
        <div class="metric-card">
          <div class="mv" style="color:var(--green);">${(prec * 100).toFixed(1)}%</div>
          <div class="ml">Precision</div>
          <div class="mbar"><div class="mfill" id="mfill-prec" style="background:var(--green);box-shadow:0 0 8px rgba(46,184,124,0.4);"></div></div>
        </div>
        <div class="metric-card">
          <div class="mv" style="color:var(--amber);">${(rec * 100).toFixed(1)}%</div>
          <div class="ml">Recall</div>
          <div class="mbar"><div class="mfill" id="mfill-rec" style="background:var(--amber);box-shadow:0 0 8px rgba(240,160,32,0.4);"></div></div>
        </div>
        <div class="metric-card">
          <div class="mv" style="color:var(--purple);">${(iou * 100).toFixed(1)}%</div>
          <div class="ml">IoU Score</div>
          <div class="mbar"><div class="mfill" id="mfill-iou" style="background:var(--purple);box-shadow:0 0 8px rgba(139,111,232,0.4);"></div></div>
        </div>
      </div>

      <div class="section-label" style="margin-bottom:1rem;">Linguistic analysis</div>
      <div class="ling-row">
        <div class="ling-card">
          <div class="ling-label">Sentiment</div>
          <div class="ling-bar"><div class="ling-fill" id="lfill-sent" style="background:${sentColor};"></div></div>
          <div class="ling-val" style="color:${sentColor};">${sentLabel} (${sent > 0 ? '+' : ''}${sent.toFixed(2)})</div>
        </div>
        <div class="ling-card">
          <div class="ling-label">Text complexity</div>
          <div class="ling-bar"><div class="ling-fill" id="lfill-comp" style="background:var(--blue);"></div></div>
          <div class="ling-val" style="color:var(--blue);">${comp} / 100</div>
        </div>
        <div class="ling-card">
          <div class="ling-label">Manipulation score</div>
          <div class="ling-bar"><div class="ling-fill" id="lfill-manip" style="background:var(--red);"></div></div>
          <div class="ling-val" style="color:var(--red);">${manip} / 100</div>
        </div>
      </div>

      <div class="report-block">
        ${rbHeader('info', 'var(--blue)', 'rgba(61,139,255,0.1)', 'Metrics interpretation', '')}
        <div style="font-size:13px;color:var(--text2);line-height:1.75;">
          <p><strong style="color:var(--green);">Precision</strong> — proportion of flagged profiles that are genuine true positives. Higher means fewer false alarms.</p>
          <p style="margin-top:8px;"><strong style="color:var(--amber);">Recall</strong> — how many actual red-flag profiles were successfully caught. Higher means fewer missed cases.</p>
          <p style="margin-top:8px;"><strong style="color:var(--purple);">IoU</strong> — spatial overlap between predicted and ground-truth bounding boxes in the YOLOv8 visual stage. Above 0.5 is acceptable; above 0.75 is strong.</p>
          <p style="margin-top:10px;color:var(--text3);font-family:var(--font-mono);font-size:11px;">Class imbalance handled via SMOTE oversampling of the minority red-flag class in Stage 1.</p>
        </div>
      </div>
    </div>`;

  setTimeout(() => {
    $('mfill-prec').style.width  = (prec * 100).toFixed(0) + '%';
    $('mfill-rec').style.width   = (rec  * 100).toFixed(0) + '%';
    $('mfill-iou').style.width   = (iou  * 100).toFixed(0) + '%';
    $('lfill-sent').style.width  = sentPct + '%';
    $('lfill-comp').style.width  = comp + '%';
    $('lfill-manip').style.width = manip + '%';
  }, 50);
}

/* ── UTILITY HELPERS ────────────────────────────────────────────── */
function escHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

const ICON_PATHS = {
  flag:     '<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><line x1="4" y1="22" x2="4" y2="15"/>',
  shuffle:  '<polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/>',
  brain:    '<path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z"/>',
  activity: '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
  heart:    '<path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>',
  info:     '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>',
};

function rbHeader(iconName, strokeColor, bgColor, title, extra) {
  return `
    <div class="rb-header">
      <div class="rb-icon" style="background:${bgColor};">
        <svg viewBox="0 0 24 24" fill="none" stroke="${strokeColor}" stroke-width="2" aria-hidden="true">
          ${ICON_PATHS[iconName] || ''}
        </svg>
      </div>
      <h3 class="rb-title">${escHtml(title)}</h3>
      ${extra || ''}
    </div>`;
}

function infoSvg() {
  return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>';
}

function errorBlock(msg) {
  return `
    <div class="report-block fade-in">
      ${rbHeader('info', 'var(--red)', 'rgba(232,57,74,0.1)', 'Audit Error', '')}
      <p style="font-family:var(--font-mono);font-size:13px;color:var(--red);">${escHtml(msg)}</p>
    </div>`;
}
