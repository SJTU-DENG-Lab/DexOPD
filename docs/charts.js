'use strict';

(function initDexOPDCharts() {
  const data = window.DEXOPD_RESULTS;
  if (!data) return;

  const SVG_NS = 'http://www.w3.org/2000/svg';
  const COLORS = {
    ours: '#0f7a50',
    oursBright: '#129260',
    oursSoft: '#dff3ea',
    clay: '#d8442b',
    blue: '#2c4296',
    slate: '#586170',
    muted: '#8a95a3',
    grid: 'var(--line)',
    axis: '#b8c2cc',
    ink: 'var(--ink)',
  };

  function el(tag, attrs = {}, text = '') {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v !== undefined && v !== null) node.setAttribute(k, v);
    }
    if (text) node.textContent = text;
    return node;
  }

  function svgEl(tag, attrs = {}, text = '') {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v !== undefined && v !== null) node.setAttribute(k, String(v));
    }
    if (text) node.textContent = text;
    return node;
  }

  const fmt4 = (v) => Number(v).toFixed(4);
  const fmtSigned4 = (v) => (v >= 0 ? '+' : '\u2212') + Math.abs(v).toFixed(4);

  // =========================================================================
  // 1. Primary 3-Task Small-Multiples Matrix (Results Overview)
  // =========================================================================
  function renderPrimaryMatrix() {
    const host = document.getElementById('chart-primary-matrix');
    if (!host) return;

    const groups = [
      { id: 'base', label: 'Base model' },
      { id: 'specialists', label: 'Single-task specialist teachers' },
      { id: 'rl', label: 'Multi-task RL' },
      { id: 'opd', label: 'Multi-teacher OPD' },
    ];
    const metrics = [
      { key: 'geneval', label: 'GenEval reward', min: 0.4, max: 1.0, oursBest: true },
      { key: 'ocr', label: 'OCR score', min: 0.5, max: 1.0, oursBest: true },
      { key: 'aesthetic', label: 'DrawBench Aesthetic', min: 5.2, max: 6.55, oursBest: true },
    ];

    const table = el('div', { class: 'matrix-chart', role: 'region', 'aria-label': 'Primary three-task comparison across 11 models' });

    // Header row
    const head = el('div', { class: 'matrix-row matrix-head' });
    head.append(el('div', { class: 'matrix-model' }, 'Model'));
    for (const m of metrics) {
      head.append(el('div', { class: 'matrix-col-head' }, m.label));
    }
    table.append(head);

    for (const grp of groups) {
      const rows = data.table1.filter((r) => r.group === grp.id);
      table.append(el('div', { class: 'matrix-group-label' }, grp.label));
      for (const row of rows) {
        const line = el('div', {
          class: `matrix-row${row.ours ? ' is-ours' : ''}${row.group === 'specialists' ? ' is-specialist' : ''}`,
          tabindex: '0',
        });
        line.append(el('div', { class: 'matrix-model' }, row.name));
        for (const m of metrics) {
          const val = row[m.key];
          const pct = Math.max(4, Math.min(100, ((val - m.min) / (m.max - m.min)) * 100));
          const isSpecialistTarget =
            (row.name === 'GenEval Teacher' && m.key === 'geneval') ||
            (row.name === 'OCR Teacher' && m.key === 'ocr') ||
            (row.name === 'Aesthetic Teacher' && m.key === 'aesthetic');
          const cell = el('div', { class: 'matrix-cell' });
          const track = el('span', { class: 'matrix-track', 'aria-hidden': 'true' });
          const barClass = row.ours
            ? 'matrix-bar bar-ours'
            : isSpecialistTarget
              ? 'matrix-bar bar-specialist'
              : row.group === 'opd'
                ? 'matrix-bar bar-opd'
                : 'matrix-bar bar-muted';
          const bar = el('span', { class: barClass, style: `width:${pct.toFixed(1)}%` });
          track.append(bar);
          const num = el('span', { class: 'matrix-val' }, fmt4(val));
          cell.append(track, num);
          line.append(cell);
        }
        table.append(line);
      }
    }

    host.replaceChildren(table);
  }

  // =========================================================================
  // 2. Learning Dynamics Panel A: GenEval OPD Curves (Full / Zoom 500-1000)
  // =========================================================================
  function renderGenEvalCurves() {
    const host = document.getElementById('chart-geneval-opd');
    const readout = document.getElementById('geneval-opd-readout');
    const fullBtn = document.getElementById('geneval-view-full');
    const zoomBtn = document.getElementById('geneval-view-zoom');
    if (!host || !readout) return;

    const ld = data.learningDynamics;
    const seriesMeta = {
      dexopd: { color: COLORS.ours, dash: '', width: 3.0 },
      dreopd: { color: COLORS.blue, dash: '', width: 2.2 },
      diffusionopd: { color: COLORS.clay, dash: '5 3', width: 2.2 },
    };

    let zoomed = false;
    let activeIdx = 10; // default 1,000 updates

    function updateReadout(idx) {
      activeIdx = idx;
      const step = ld.steps[idx];
      const items = ld.genevalOPD.series
        .map((s) => `<span class="readout-item"><i style="background:${seriesMeta[s.id].color}"></i>${s.label}: <strong>${fmt4(s.values[idx])}</strong></span>`)
        .join('');
      readout.innerHTML = `<span class="readout-step">Update ${step.toLocaleString()}</span>${items}`;
    }

    function draw() {
      const startIdx = zoomed ? 5 : 0;
      const steps = ld.steps.slice(startIdx);
      const yMin = zoomed ? 0.945 : 0.62;
      const yMax = zoomed ? 0.978 : 1.00;
      const yTicks = zoomed ? [0.95, 0.96, 0.97] : [0.65, 0.75, 0.85, 0.95, 1.00];

      const W = 540, H = 290;
      const pad = { l: 48, r: 22, t: 18, b: 36 };
      const pw = W - pad.l - pad.r;
      const ph = H - pad.t - pad.b;

      const xScale = (step) => pad.l + ((step - steps[0]) / (steps[steps.length - 1] - steps[0])) * pw;
      const yScale = (val) => pad.t + (1 - (val - yMin) / (yMax - yMin)) * ph;

      const svg = svgEl('svg', {
        viewBox: `0 0 ${W} ${H}`,
        class: 'chart-svg',
        role: 'img',
        'aria-label': 'GenEval reward progression over 1,000 updates for DexOPD, DreOPD, and DiffusionOPD',
      });

      // Grid + Y axis
      for (const tick of yTicks) {
        const y = yScale(tick);
        svg.append(svgEl('line', { x1: pad.l, x2: W - pad.r, y1: y, y2: y, stroke: COLORS.grid, 'stroke-width': 1 }));
        svg.append(svgEl('text', { x: pad.l - 8, y: y + 4, 'text-anchor': 'end', class: 'svg-tick' }, tick.toFixed(2)));
      }
      // X ticks
      const xTicks = zoomed ? [500, 600, 700, 800, 900, 1000] : [0, 200, 400, 600, 800, 1000];
      for (const tick of xTicks) {
        const x = xScale(tick);
        svg.append(svgEl('line', { x1: x, x2: x, y1: pad.t, y2: H - pad.b, stroke: '#eff3ef', 'stroke-width': 1 }));
        svg.append(svgEl('text', { x, y: H - pad.b + 18, 'text-anchor': 'middle', class: 'svg-tick' }, String(tick)));
      }
      svg.append(svgEl('text', { x: pad.l + pw / 2, y: H - 4, 'text-anchor': 'middle', class: 'svg-axis-label' }, 'Optimizer updates'));

      // Reference lines: RL teacher & Proxy teacher
      const refs = [
        {
          val: ld.genevalOPD.proxyTeacher,
          label: 'Proxy teacher (λ=1.2): 0.9652',
          color: '#2b3b37',
          dash: '6 3 2 3',
          x: pad.l + 6,
          anchor: 'start',
          dy: -6,
        },
        {
          val: ld.genevalOPD.rlTeacher,
          label: 'RL teacher: 0.9534',
          color: COLORS.muted,
          dash: '4 4',
          x: zoomed ? W - pad.r - 4 : pad.l + 6,
          anchor: zoomed ? 'end' : 'start',
          dy: 13,
        },
      ];
      for (const ref of refs) {
        const ry = yScale(ref.val);
        svg.append(svgEl('line', { x1: pad.l, x2: W - pad.r, y1: ry, y2: ry, stroke: ref.color, 'stroke-width': 1.4, 'stroke-dasharray': ref.dash }));
        svg.append(svgEl('text', {
          x: ref.x,
          y: ry + ref.dy,
          'text-anchor': ref.anchor,
          fill: ref.color,
          class: 'svg-ref-label svg-halo',
        }, ref.label));
      }

      // Vertical hover guide
      const clampedIdx = Math.max(startIdx, activeIdx);
      const guide = svgEl('line', {
        x1: xScale(ld.steps[clampedIdx]),
        x2: xScale(ld.steps[clampedIdx]),
        y1: pad.t,
        y2: H - pad.b,
        stroke: '#9eb2a7',
        'stroke-width': 1.2,
        'stroke-dasharray': '3 3',
      });
      svg.append(guide);

      // Draw series (baselines first, DexOPD on top)
      const order = ['diffusionopd', 'dreopd', 'dexopd'];
      const pointNodes = [];
      for (const id of order) {
        const s = ld.genevalOPD.series.find((x) => x.id === id);
        const meta = seriesMeta[id];
        const pts = steps.map((step, i) => `${xScale(step).toFixed(1)},${yScale(s.values[startIdx + i]).toFixed(1)}`).join(' ');
        svg.append(svgEl('polyline', {
          points: pts,
          fill: 'none',
          stroke: meta.color,
          'stroke-width': meta.width,
          'stroke-dasharray': meta.dash,
          'stroke-linejoin': 'round',
          'stroke-linecap': 'round',
        }));
        steps.forEach((step, i) => {
          const realIdx = startIdx + i;
          const cx = xScale(step);
          const cy = yScale(s.values[realIdx]);
          const circle = svgEl('circle', {
            cx,
            cy,
            r: realIdx === clampedIdx ? (id === 'dexopd' ? 5.2 : 4.4) : (id === 'dexopd' ? 3.5 : 2.8),
            fill: meta.color,
            stroke: '#ffffff',
            'stroke-width': 1.6,
          });
          pointNodes.push({ realIdx, id, circle });
          svg.append(circle);
        });
      }

      // Transparent interactive columns for hover/click
      const colW = pw / Math.max(1, steps.length - 1);
      steps.forEach((step, i) => {
        const realIdx = startIdx + i;
        const x = xScale(step);
        const rect = svgEl('rect', {
          x: Math.max(pad.l, x - colW / 2),
          y: pad.t,
          width: i === 0 || i === steps.length - 1 ? colW / 2 : colW,
          height: ph,
          fill: 'transparent',
          class: 'svg-hit',
        });
        const activate = () => {
          updateReadout(realIdx);
          guide.setAttribute('x1', String(x));
          guide.setAttribute('x2', String(x));
          for (const p of pointNodes) {
            p.circle.setAttribute('r', String(p.realIdx === realIdx ? (p.id === 'dexopd' ? 5.2 : 4.4) : (p.id === 'dexopd' ? 3.5 : 2.8)));
          }
        };
        rect.addEventListener('mouseenter', activate);
        rect.addEventListener('click', activate);
        svg.append(rect);
      });

      host.replaceChildren(svg);
      updateReadout(clampedIdx);
    }

    if (fullBtn && zoomBtn) {
      fullBtn.addEventListener('click', () => {
        zoomed = false;
        fullBtn.setAttribute('aria-pressed', 'true');
        zoomBtn.setAttribute('aria-pressed', 'false');
        draw();
      });
      zoomBtn.addEventListener('click', () => {
        zoomed = true;
        fullBtn.setAttribute('aria-pressed', 'false');
        zoomBtn.setAttribute('aria-pressed', 'true');
        draw();
      });
    }
    draw();
  }

  // =========================================================================
  // 3. Learning Dynamics Panel B: Three-Task Training Stability
  // =========================================================================
  function renderMultitaskStability() {
    const host = document.getElementById('chart-multitask-stability');
    const readout = document.getElementById('multitask-readout');
    const summaryEl = document.getElementById('multitask-summary');
    const tabsWrap = document.getElementById('multitask-tabs');
    if (!host || !readout || !tabsWrap) return;

    const ld = data.learningDynamics;
    let activeModel = 'dexopd';
    let activeIdx = 10;

    const metricColors = {
      geneval: COLORS.ours,
      ocr: COLORS.blue,
      aesthetic: COLORS.clay,
    };

    function updateReadout(idx) {
      activeIdx = idx;
      const m = ld.multitask[activeModel];
      readout.innerHTML =
        `<span class="readout-step">Update ${ld.steps[idx].toLocaleString()}</span>` +
        `<span class="readout-item"><i style="background:${metricColors.geneval}"></i>GenEval: <strong>${fmt4(m.geneval[idx])}</strong></span>` +
        `<span class="readout-item"><i style="background:${metricColors.ocr}"></i>OCR: <strong>${fmt4(m.ocr[idx])}</strong></span>` +
        `<span class="readout-item"><i style="background:${metricColors.aesthetic}"></i>Aesthetic: <strong>${fmt4(m.aesthetic[idx])}</strong></span>`;
    }

    function draw() {
      const m = ld.multitask[activeModel];
      if (summaryEl) summaryEl.textContent = m.summary;

      const W = 540, H = 290;
      const pad = { l: 48, r: 48, t: 18, b: 36 };
      const pw = W - pad.l - pad.r;
      const ph = H - pad.t - pad.b;

      const xScale = (step) => pad.l + (step / 1000) * pw;
      const yLeft = (v) => pad.t + (1 - (v - 0.35) / (1.02 - 0.35)) * ph;
      const yRight = (v) => pad.t + (1 - (v - 5.0) / (6.7 - 5.0)) * ph;

      const svg = svgEl('svg', {
        viewBox: `0 0 ${W} ${H}`,
        class: 'chart-svg',
        role: 'img',
        'aria-label': `Three-task learning dynamics for ${m.label}`,
      });

      // Horizontal grid tied to left axis
      for (const t of [0.4, 0.55, 0.7, 0.85, 1.0]) {
        const y = yLeft(t);
        svg.append(svgEl('line', { x1: pad.l, x2: W - pad.r, y1: y, y2: y, stroke: COLORS.grid, 'stroke-width': 1 }));
        svg.append(svgEl('text', { x: pad.l - 8, y: y + 4, 'text-anchor': 'end', class: 'svg-tick' }, t.toFixed(2)));
      }
      // Right axis ticks (Aesthetic)
      for (const t of [5.2, 5.6, 6.0, 6.4]) {
        const y = yRight(t);
        svg.append(svgEl('text', { x: W - pad.r + 8, y: y + 4, 'text-anchor': 'start', class: 'svg-tick svg-tick-right' }, t.toFixed(1)));
      }
      // Axis headers
      svg.append(svgEl('text', { x: pad.l - 4, y: 11, 'text-anchor': 'start', class: 'svg-mini-head', fill: COLORS.ours }, 'GenEval / OCR'));
      svg.append(svgEl('text', { x: W - pad.r + 4, y: 11, 'text-anchor': 'end', class: 'svg-mini-head', fill: COLORS.clay }, 'Aesthetic'));

      // X ticks
      for (const tick of [0, 200, 400, 600, 800, 1000]) {
        const x = xScale(tick);
        svg.append(svgEl('line', { x1: x, x2: x, y1: pad.t, y2: H - pad.b, stroke: '#eff3ef', 'stroke-width': 1 }));
        svg.append(svgEl('text', { x, y: H - pad.b + 18, 'text-anchor': 'middle', class: 'svg-tick' }, String(tick)));
      }
      svg.append(svgEl('text', { x: pad.l + pw / 2, y: H - 4, 'text-anchor': 'middle', class: 'svg-axis-label' }, 'Optimizer updates'));

      const guide = svgEl('line', {
        x1: xScale(ld.steps[activeIdx]),
        x2: xScale(ld.steps[activeIdx]),
        y1: pad.t,
        y2: H - pad.b,
        stroke: '#9eb2a7',
        'stroke-width': 1.2,
        'stroke-dasharray': '3 3',
      });
      svg.append(guide);

      const tracks = [
        { key: 'ocr', vals: m.ocr, yFn: yLeft, color: metricColors.ocr, width: 2.2, dash: '5 3' },
        { key: 'aesthetic', vals: m.aesthetic, yFn: yRight, color: metricColors.aesthetic, width: 2.5, dash: '' },
        { key: 'geneval', vals: m.geneval, yFn: yLeft, color: metricColors.geneval, width: 2.8, dash: '' },
      ];
      const dots = [];
      for (const tr of tracks) {
        const pts = ld.steps.map((s, i) => `${xScale(s).toFixed(1)},${tr.yFn(tr.vals[i]).toFixed(1)}`).join(' ');
        svg.append(svgEl('polyline', {
          points: pts,
          fill: 'none',
          stroke: tr.color,
          'stroke-width': tr.width,
          'stroke-dasharray': tr.dash,
          'stroke-linejoin': 'round',
          'stroke-linecap': 'round',
        }));
        ld.steps.forEach((s, i) => {
          const c = svgEl('circle', {
            cx: xScale(s),
            cy: tr.yFn(tr.vals[i]),
            r: i === activeIdx ? 4.8 : 3.0,
            fill: tr.color,
            stroke: '#ffffff',
            'stroke-width': 1.5,
          });
          dots.push({ i, c });
          svg.append(c);
        });
      }

      const colW = pw / 10;
      ld.steps.forEach((s, i) => {
        const x = xScale(s);
        const rect = svgEl('rect', {
          x: Math.max(pad.l, x - colW / 2),
          y: pad.t,
          width: i === 0 || i === 10 ? colW / 2 : colW,
          height: ph,
          fill: 'transparent',
          class: 'svg-hit',
        });
        const activate = () => {
          updateReadout(i);
          guide.setAttribute('x1', String(x));
          guide.setAttribute('x2', String(x));
          for (const d of dots) d.c.setAttribute('r', String(d.i === i ? 4.8 : 3.0));
        };
        rect.addEventListener('mouseenter', activate);
        rect.addEventListener('click', activate);
        svg.append(rect);
      });

      host.replaceChildren(svg);
      updateReadout(activeIdx);
    }

    const buttons = [...tabsWrap.querySelectorAll('button[data-model]')];
    for (const btn of buttons) {
      btn.addEventListener('click', () => {
        activeModel = btn.dataset.model;
        for (const b of buttons) b.setAttribute('aria-pressed', String(b === btn));
        draw();
      });
    }
    draw();
  }

  // =========================================================================
  // 4. Analysis 01: Component Ablation
  // =========================================================================
  function renderAblation() {
    const host = document.getElementById('chart-ablation');
    const primaryBtn = document.getElementById('ablation-primary');
    const secondaryBtn = document.getElementById('ablation-secondary');
    if (!host) return;

    const baseRow = data.ablation.find((r) => r.isDefault);
    let mode = 'primary';

    const metricSets = {
      primary: [
        { key: 'geneval', label: 'GenEval reward', min: 0.940, max: 0.980 },
        { key: 'ocr', label: 'OCR score', min: 0.928, max: 0.956 },
        { key: 'aesthetic', label: 'DrawBench Aesthetic', min: 6.20, max: 6.53 },
      ],
      secondary: [
        { key: 'pickscore', label: 'PickScore', min: 0.875, max: 0.928 },
        { key: 'imagereward', label: 'ImageReward', min: 1.38, max: 1.53 },
        { key: 'clipscore', label: 'CLIPScore', min: 0.280, max: 0.300 },
        { key: 'hps', label: 'HPSv2.1', min: 0.320, max: 0.358 },
      ],
    };

    function draw() {
      const cols = metricSets[mode];
      const grid = el('div', { class: `ablation-grid cols-${cols.length}` });
      for (const m of cols) {
        const panel = el('div', { class: 'ablation-panel' });
        panel.append(el('h4', {}, m.label));
        const list = el('div', { class: 'ablation-bars' });
        for (const row of data.ablation) {
          const val = row[m.key];
          const delta = val - baseRow[m.key];
          const pct = Math.max(8, Math.min(100, ((val - m.min) / (m.max - m.min)) * 100));
          const item = el('div', { class: `ablation-item${row.isDefault ? ' is-default' : ''}` });
          const top = el('div', { class: 'ablation-meta' });
          top.append(el('span', { class: 'ablation-name', title: row.desc }, row.label));
          const numWrap = el('span', { class: 'ablation-nums' });
          numWrap.append(el('strong', {}, fmt4(val)));
          if (!row.isDefault) {
            const cls = delta < -0.00005 ? 'delta-neg' : delta > 0.00005 ? 'delta-pos' : 'delta-flat';
            numWrap.append(el('em', { class: cls }, fmtSigned4(delta)));
          } else {
            numWrap.append(el('em', { class: 'delta-ref' }, 'default'));
          }
          top.append(numWrap);
          const track = el('div', { class: 'ablation-track', 'aria-hidden': 'true' });
          track.append(el('span', { style: `width:${pct.toFixed(1)}%` }));
          item.append(top, track);
          list.append(item);
        }
        panel.append(list);
        grid.append(panel);
      }
      host.replaceChildren(grid);
    }

    if (primaryBtn && secondaryBtn) {
      primaryBtn.addEventListener('click', () => {
        mode = 'primary';
        primaryBtn.setAttribute('aria-pressed', 'true');
        secondaryBtn.setAttribute('aria-pressed', 'false');
        draw();
      });
      secondaryBtn.addEventListener('click', () => {
        mode = 'secondary';
        primaryBtn.setAttribute('aria-pressed', 'false');
        secondaryBtn.setAttribute('aria-pressed', 'true');
        draw();
      });
    }
    draw();
  }

  // =========================================================================
  // 5. Analysis 02: Auxiliary-Weight Sweep (beta in [0.0, 2.0])
  // =========================================================================
  function renderBetaSweep() {
    const host = document.getElementById('chart-beta-sweep');
    const readout = document.getElementById('beta-sweep-readout');
    if (!host || !readout) return;

    const rows = data.betaSweep;
    let activeIdx = rows.findIndex((r) => Math.abs(r.beta - 1.8) < 1e-6);
    if (activeIdx < 0) activeIdx = rows.length - 1;

    const panels = [
      { key: 'geneval', label: 'GenEval reward', yMin: 0.948, yMax: 0.978, ticks: [0.95, 0.96, 0.97], color: COLORS.ours },
      { key: 'ocr', label: 'OCR score', yMin: 0.932, yMax: 0.956, ticks: [0.935, 0.945, 0.955], color: COLORS.blue },
      { key: 'aesthetic', label: 'DrawBench Aesthetic', yMin: 6.43, yMax: 6.50, ticks: [6.44, 6.46, 6.48, 6.50], color: COLORS.clay },
    ];

    const panelControllers = [];

    function updateSelection(idx) {
      activeIdx = idx;
      const r = rows[idx];
      for (const ctrl of panelControllers) ctrl.highlight(idx);
      const badge = Math.abs(r.beta - 1.8) < 1e-6 ? ' <span class="badge-default">Default</span>' : '';
      readout.innerHTML =
        `<div class="sweep-readout-head"><strong>&beta; = ${r.beta.toFixed(1)}</strong>${badge}</div>` +
        `<dl class="sweep-readout-grid">` +
        `<div><dt>GenEval</dt><dd>${fmt4(r.geneval)}</dd></div>` +
        `<div><dt>OCR</dt><dd>${fmt4(r.ocr)}</dd></div>` +
        `<div><dt>Aesthetic</dt><dd>${fmt4(r.aesthetic)}</dd></div>` +
        `<div><dt>PickScore</dt><dd>${fmt4(r.pickscore)}</dd></div>` +
        `<div><dt>ImageReward</dt><dd>${fmt4(r.imagereward)}</dd></div>` +
        `<div><dt>CLIPScore</dt><dd>${fmt4(r.clipscore)}</dd></div>` +
        `<div><dt>HPSv2.1</dt><dd>${fmt4(r.hps)}</dd></div>` +
        `</dl>`;
    }

    const wrap = el('div', { class: 'beta-panels' });
    for (const p of panels) {
      const card = el('div', { class: 'mini-chart-card' });
      const titleRow = el('div', { class: 'mini-chart-title' });
      titleRow.append(el('span', {}, p.label));
      const peakVal = Math.max(...rows.map((r) => r[p.key]));
      const peakBeta = rows.find((r) => r[p.key] === peakVal).beta;
      titleRow.append(el('small', {}, `Peak ${fmt4(peakVal)} at β=${peakBeta.toFixed(1)}`));
      card.append(titleRow);

      const W = 340, H = 200;
      const pad = { l: 44, r: 14, t: 14, b: 30 };
      const pw = W - pad.l - pad.r;
      const ph = H - pad.t - pad.b;
      const xScale = (b) => pad.l + (b / 2.0) * pw;
      const yScale = (v) => pad.t + (1 - (v - p.yMin) / (p.yMax - p.yMin)) * ph;

      const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart-svg', role: 'img', 'aria-label': `${p.label} across auxiliary weight beta` });

      // Grid
      for (const t of p.ticks) {
        const y = yScale(t);
        svg.append(svgEl('line', { x1: pad.l, x2: W - pad.r, y1: y, y2: y, stroke: COLORS.grid, 'stroke-width': 1 }));
        svg.append(svgEl('text', { x: pad.l - 6, y: y + 3.5, 'text-anchor': 'end', class: 'svg-tick' }, t.toFixed(p.key === 'aesthetic' ? 2 : 3)));
      }
      for (const b of [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]) {
        const x = xScale(b);
        svg.append(svgEl('text', { x, y: H - 10, 'text-anchor': 'middle', class: 'svg-tick' }, b.toFixed(1)));
      }

      // Baseline (beta=0) dashed line
      const y0 = yScale(rows[0][p.key]);
      svg.append(svgEl('line', { x1: pad.l, x2: W - pad.r, y1: y0, y2: y0, stroke: '#b2beb7', 'stroke-width': 1.2, 'stroke-dasharray': '3 3' }));

      // Default beta=1.8 vertical marker
      const xDef = xScale(1.8);
      svg.append(svgEl('line', { x1: xDef, x2: xDef, y1: pad.t, y2: H - pad.b, stroke: '#a9c7ba', 'stroke-width': 1.4, 'stroke-dasharray': '4 3' }));

      // Hover guide
      const guide = svgEl('line', { x1: xDef, x2: xDef, y1: pad.t, y2: H - pad.b, stroke: p.color, 'stroke-opacity': 0.45, 'stroke-width': 1.5 });
      svg.append(guide);

      // Curve
      const pts = rows.map((r) => `${xScale(r.beta).toFixed(1)},${yScale(r[p.key]).toFixed(1)}`).join(' ');
      svg.append(svgEl('polyline', {
        points: pts,
        fill: 'none',
        stroke: p.color,
        'stroke-width': 2.4,
        'stroke-linejoin': 'round',
        'stroke-linecap': 'round',
      }));

      const circles = rows.map((r, i) => {
        const c = svgEl('circle', {
          cx: xScale(r.beta),
          cy: yScale(r[p.key]),
          r: i === activeIdx ? 5.0 : 3.0,
          fill: Math.abs(r.beta - 1.8) < 1e-6 ? COLORS.ours : p.color,
          stroke: '#ffffff',
          'stroke-width': 1.5,
        });
        svg.append(c);
        return c;
      });

      const colW = pw / 10;
      rows.forEach((r, i) => {
        const x = xScale(r.beta);
        const rect = svgEl('rect', {
          x: Math.max(pad.l, x - colW / 2),
          y: pad.t,
          width: i === 0 || i === rows.length - 1 ? colW / 2 : colW,
          height: ph,
          fill: 'transparent',
          class: 'svg-hit',
        });
        rect.addEventListener('mouseenter', () => updateSelection(i));
        rect.addEventListener('click', () => updateSelection(i));
        svg.append(rect);
      });

      panelControllers.push({
        highlight(idx) {
          const x = xScale(rows[idx].beta);
          guide.setAttribute('x1', String(x));
          guide.setAttribute('x2', String(x));
          circles.forEach((c, i) => c.setAttribute('r', String(i === idx ? 5.2 : 3.0)));
        },
      });

      card.append(svg);
      wrap.append(card);
    }

    host.replaceChildren(wrap);
    updateSelection(activeIdx);
  }

  // =========================================================================
  // 6. Analysis 03: Teacher Inference Extrapolation (lambda trade-off)
  // =========================================================================
  function renderTeacherSweep() {
    const host = document.getElementById('chart-teacher-sweep');
    const readout = document.getElementById('teacher-sweep-readout');
    const tabsWrap = document.getElementById('teacher-tabs');
    if (!host || !readout || !tabsWrap) return;

    let activeTeacher = 'aesthetic';
    let activeIdx = 6; // lambda = 1.2

    function draw() {
      const spec = data.teacherSweeps[activeTeacher];
      const rows = spec.rows;

      const panels = [
        {
          role: 'On-task target',
          key: spec.onTaskKey,
          label: spec.onTaskLabel,
          color: COLORS.ours,
          refVal: spec.dexopdScore,
          refLabel: `Unified DexOPD: ${fmt4(spec.dexopdScore)}`,
          zeroLine: false,
        },
        {
          role: 'Off-task degradation',
          key: spec.offTaskKey,
          label: spec.offTaskLabel,
          color: COLORS.clay,
          refVal: rows[0][spec.offTaskKey],
          refLabel: `Native teacher (λ=0): ${fmt4(rows[0][spec.offTaskKey])}`,
          zeroLine: spec.offTaskKey === 'imagereward' && Math.min(...rows.map((r) => r[spec.offTaskKey])) < 0,
        },
      ];

      const controllers = [];

      function updateReadout(idx) {
        activeIdx = idx;
        const r = rows[idx];
        for (const c of controllers) c.highlight(idx);
        const badge = Math.abs(r.lambda - 1.2) < 1e-6 ? ' <span class="badge-default">Distillation default</span>' : '';
        readout.innerHTML =
          `<div class="sweep-readout-head"><strong>${spec.title} &middot; &lambda; = ${r.lambda.toFixed(1)}</strong>${badge}</div>` +
          `<dl class="sweep-readout-grid cols-4">` +
          `<div><dt>${spec.onTaskLabel} (on-task)</dt><dd><strong>${fmt4(r[spec.onTaskKey])}</strong></dd></div>` +
          `<div><dt>${spec.offTaskLabel} (off-task)</dt><dd>${fmt4(r[spec.offTaskKey])}</dd></div>` +
          `<div><dt>${spec.offTaskSecondaryLabel} (off-task)</dt><dd>${fmt4(r[spec.offTaskSecondaryKey])}</dd></div>` +
          `<div><dt>PickScore / HPSv2.1</dt><dd>${fmt4(r.pickscore)} / ${fmt4(r.hps)}</dd></div>` +
          `</dl>`;
      }

      const wrap = el('div', { class: 'teacher-panels' });
      for (const p of panels) {
        const card = el('div', { class: 'mini-chart-card' });
        const head = el('div', { class: 'mini-chart-title' });
        head.append(el('span', {}, `${p.role}: ${p.label}`));
        const nativeVal = rows[0][p.key];
        const endVal = rows[rows.length - 1][p.key];
        head.append(el('small', {}, `λ=0→2: ${fmtSigned4(endVal - nativeVal)}`));
        card.append(head);

        const vals = rows.map((r) => r[p.key]);
        const allVals = p.refVal !== undefined ? [...vals, p.refVal] : vals;
        const rawMin = Math.min(...allVals);
        const rawMax = Math.max(...allVals);
        const span = Math.max(0.015, rawMax - rawMin);
        const yMin = rawMin - span * 0.16;
        const yMax = rawMax + span * 0.18;
        const decimals = span < 0.08 ? 3 : 2;

        const W = 480, H = 236;
        const pad = { l: 56, r: 18, t: 28, b: 32 };
        const pw = W - pad.l - pad.r;
        const ph = H - pad.t - pad.b;
        const xScale = (l) => pad.l + (l / 2.0) * pw;
        const yScale = (v) => pad.t + (1 - (v - yMin) / (yMax - yMin)) * ph;

        const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart-svg', role: 'img', 'aria-label': `${spec.title} ${p.label} vs lambda` });

        // 4 evenly spaced y ticks with adaptive precision to avoid duplicate labels
        for (let t = 0; t <= 3; t++) {
          const val = rawMin + (rawMax - rawMin) * (t / 3);
          const y = yScale(val);
          svg.append(svgEl('line', { x1: pad.l, x2: W - pad.r, y1: y, y2: y, stroke: COLORS.grid, 'stroke-width': 1 }));
          svg.append(svgEl('text', { x: pad.l - 7, y: y + 4, 'text-anchor': 'end', class: 'svg-tick' }, val.toFixed(decimals)));
        }
        for (const l of [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]) {
          const x = xScale(l);
          svg.append(svgEl('text', { x, y: H - 10, 'text-anchor': 'middle', class: 'svg-tick' }, l.toFixed(1)));
        }

        // Training default lambda = 1.2 vertical marker (label sits cleanly in top header margin)
        const x12 = xScale(1.2);
        svg.append(svgEl('line', { x1: x12, x2: x12, y1: pad.t, y2: H - pad.b, stroke: '#9db8ac', 'stroke-width': 1.4, 'stroke-dasharray': '4 3' }));
        svg.append(svgEl('text', { x: x12, y: pad.t - 9, 'text-anchor': 'middle', fill: '#2c5a46', class: 'svg-ref-label' }, 'Distill default λ=1.2'));

        // Zero line for ImageReward collapse
        if (p.zeroLine) {
          const zy = yScale(0);
          svg.append(svgEl('line', { x1: pad.l, x2: W - pad.r, y1: zy, y2: zy, stroke: '#9f3b28', 'stroke-width': 1.2, 'stroke-dasharray': '2 3' }));
          svg.append(svgEl('text', { x: pad.l + 6, y: zy - 6, fill: '#9f3b28', class: 'svg-ref-label svg-halo-card' }, 'Zero ImageReward'));
        }

        // Reference line with smart left/right placement away from nearby curve endpoints
        if (p.refVal !== undefined) {
          const ry = yScale(p.refVal);
          const leftDist = Math.abs(vals[0] - p.refVal);
          const rightDist = Math.abs(vals[vals.length - 1] - p.refVal);
          const placeLeft = p.role === 'On-task target' && rightDist < leftDist;
          const curveMostlyAbove = p.role === 'On-task target' && vals.filter((v) => v > p.refVal).length >= 7;
          const placeBelow = ry < pad.t + 18 || curveMostlyAbove || (p.role === 'On-task target' && vals[vals.length - 1] > p.refVal && !placeLeft);
          const labelX = curveMostlyAbove ? pad.l + pw * 0.52 : (placeLeft ? pad.l + 6 : W - pad.r - 4);
          const labelAnchor = curveMostlyAbove ? 'middle' : (placeLeft ? 'start' : 'end');
          svg.append(svgEl('line', { x1: pad.l, x2: W - pad.r, y1: ry, y2: ry, stroke: p.role === 'On-task target' ? COLORS.ours : COLORS.muted, 'stroke-width': 1.3, 'stroke-dasharray': '5 4' }));
          svg.append(svgEl('text', {
            x: labelX,
            y: placeBelow ? ry + 13 : ry - 6,
            'text-anchor': labelAnchor,
            fill: p.role === 'On-task target' ? COLORS.ours : COLORS.slate,
            class: 'svg-ref-label svg-halo-card',
          }, p.refLabel));
        }

        // Hover guide
        const guide = svgEl('line', { x1: xScale(rows[activeIdx].lambda), x2: xScale(rows[activeIdx].lambda), y1: pad.t, y2: H - pad.b, stroke: p.color, 'stroke-opacity': 0.45, 'stroke-width': 1.5 });
        svg.append(guide);

        // Curve
        const pts = rows.map((r) => `${xScale(r.lambda).toFixed(1)},${yScale(r[p.key]).toFixed(1)}`).join(' ');
        svg.append(svgEl('polyline', {
          points: pts,
          fill: 'none',
          stroke: p.color,
          'stroke-width': 2.6,
          'stroke-linejoin': 'round',
          'stroke-linecap': 'round',
        }));

        const circles = rows.map((r, i) => {
          const isPeak = p.role === 'On-task target' && Math.abs(r.lambda - spec.peakLambda) < 1e-6;
          const c = svgEl('circle', {
            cx: xScale(r.lambda),
            cy: yScale(r[p.key]),
            r: i === activeIdx ? 5.2 : isPeak ? 4.4 : 3.1,
            fill: isPeak ? '#ffffff' : p.color,
            stroke: isPeak ? p.color : '#ffffff',
            'stroke-width': isPeak ? 2.4 : 1.5,
          });
          svg.append(c);
          return { c, isPeak };
        });

        const colW = pw / 10;
        rows.forEach((r, i) => {
          const x = xScale(r.lambda);
          const rect = svgEl('rect', {
            x: Math.max(pad.l, x - colW / 2),
            y: pad.t,
            width: i === 0 || i === rows.length - 1 ? colW / 2 : colW,
            height: ph,
            fill: 'transparent',
            class: 'svg-hit',
          });
          rect.addEventListener('mouseenter', () => updateReadout(i));
          rect.addEventListener('click', () => updateReadout(i));
          svg.append(rect);
        });

        controllers.push({
          highlight(idx) {
            const x = xScale(rows[idx].lambda);
            guide.setAttribute('x1', String(x));
            guide.setAttribute('x2', String(x));
            circles.forEach(({ c, isPeak }, i) => c.setAttribute('r', String(i === idx ? 5.4 : isPeak ? 4.4 : 3.1)));
          },
        });

        card.append(svg);
        wrap.append(card);
      }

      host.replaceChildren(wrap);
      updateReadout(activeIdx);
    }

    const buttons = [...tabsWrap.querySelectorAll('button[data-teacher]')];
    for (const btn of buttons) {
      btn.addEventListener('click', () => {
        activeTeacher = btn.dataset.teacher;
        for (const b of buttons) b.setAttribute('aria-pressed', String(b === btn));
        draw();
      });
    }
    draw();
  }

  // =========================================================================
  // 7. Table 1 Column Group Filter (All / GenEval / OCR + DrawBench)
  // =========================================================================
  function setupTableFilter() {
    const controls = document.getElementById('table-filter-controls');
    const table = document.getElementById('main-benchmark-table');
    if (!controls || !table) return;

    const buttons = [...controls.querySelectorAll('button[data-view]')];
    for (const btn of buttons) {
      btn.addEventListener('click', () => {
        const view = btn.dataset.view;
        table.dataset.view = view;
        for (const b of buttons) b.setAttribute('aria-pressed', String(b === btn));
      });
    }
    controls.hidden = false;
  }

  renderPrimaryMatrix();
  renderGenEvalCurves();
  renderMultitaskStability();
  renderAblation();
  renderBetaSweep();
  renderTeacherSweep();
  setupTableFilter();
})();
