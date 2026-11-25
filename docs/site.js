'use strict';

// Preserve legacy #demo and #gallery links by redirecting to the dedicated examples page.
function redirectLegacyHash() {
  if (window.location.hash === '#demo' || window.location.hash === '#gallery') {
    window.location.replace(`examples.html${window.location.hash}`);
  }
}
redirectLegacyHash();
window.addEventListener('hashchange', redirectLegacyHash);

const get = (id) => document.getElementById(id);

// Theme toggle (Light / Dark white-paper switch)
(function initTheme() {
  const root = document.documentElement;
  const saved = localStorage.getItem('dexopd-theme');
  if (saved === 'dark') {
    root.setAttribute('data-theme', 'dark');
  }
  const btn = get('theme-toggle');
  if (!btn) return;
  const syncLabel = () => {
    const isDark = root.getAttribute('data-theme') === 'dark';
    btn.textContent = isDark ? '\u2600' : '\u263E';
    btn.setAttribute('aria-label', isDark ? 'Switch to light mode' : 'Switch to dark mode');
  };
  syncLabel();
  btn.addEventListener('click', () => {
    const next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    if (next === 'dark') {
      root.setAttribute('data-theme', 'dark');
    } else {
      root.removeAttribute('data-theme');
    }
    localStorage.setItem('dexopd-theme', next);
    syncLabel();
  });
})();

// Top 2px scroll progress bar
const scrollbar = get('scrollbar');
if (scrollbar) {
  const updateProgress = () => {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    const pct = max > 0 ? Math.min(100, Math.max(0, (window.scrollY / max) * 100)) : 0;
    scrollbar.style.width = `${pct.toFixed(1)}%`;
  };
  window.addEventListener('scroll', updateProgress, { passive: true });
  updateProgress();
}

function setupTabs(list) {
  if (!list) return;
  const tabs = [...list.querySelectorAll('[role="tab"]')];
  function select(current) {
    for (const tab of tabs) {
      const selected = tab === current;
      tab.setAttribute('aria-selected', selected);
      tab.tabIndex = selected ? 0 : -1;
      const panel = get(tab.getAttribute('aria-controls'));
      if (panel) {
        panel.hidden = !selected;
        panel.setAttribute('aria-labelledby', tab.id);
      }
    }
  }
  for (const [index, tab] of tabs.entries()) {
    tab.addEventListener('click', () => select(tab));
    tab.addEventListener('keydown', (event) => {
      const count = tabs.length;
      const next = {
        ArrowLeft: (index + count - 1) % count,
        ArrowRight: (index + 1) % count,
        Home: 0,
        End: count - 1,
      }[event.key];
      if (next === undefined) return;
      event.preventDefault();
      select(tabs[next]);
      tabs[next].focus();
    });
  }
  select(tabs[0]);
  list.hidden = false;
}

setupTabs(get('method-tabs'));

// Interactive trajectory markers on the Student candidates manifold
const rolloutLayer = get('rollout-markers');
const progress = get('rollout-progress');
if (rolloutLayer && progress && window.DEXOPD_TRAJECTORIES) {
  const markers = window.DEXOPD_TRAJECTORIES.map(({ path, color }) => {
    const curve = document.createElementNS(rolloutLayer.namespaceURI, 'path');
    curve.setAttribute('d', path);
    curve.setAttribute('fill', 'none');
    const dot = document.createElementNS(rolloutLayer.namespaceURI, 'circle');
    dot.setAttribute('r', '5.1');
    dot.setAttribute('fill', color);
    dot.setAttribute('stroke', '#fff');
    dot.setAttribute('stroke-width', '2');
    rolloutLayer.append(curve, dot);
    return { curve, dot, length: curve.getTotalLength() };
  });

  function setRolloutProgress() {
    const fraction = progress.valueAsNumber / 100;
    for (const { curve, dot, length } of markers) {
      const point = curve.getPointAtLength(length * fraction);
      dot.setAttribute('cx', point.x);
      dot.setAttribute('cy', point.y);
    }
    progress.setAttribute('aria-valuetext', `${progress.value} percent`);
  }

  progress.addEventListener('input', setRolloutProgress);
  setRolloutProgress();
  const control = get('rollout-control');
  if (control) control.hidden = false;
}

// Reward feedback state switcher (Different vs Tied rewards)
const rewardDrawing = get('reward-drawing');
const rewardExplanation = get('reward-explanation');
const rewardEquation = get('reward-equation');
if (rewardDrawing && rewardExplanation && rewardEquation) {
  const rewardCaption = rewardDrawing.nextElementSibling;
  const unequalRewards = {
    src: rewardDrawing.getAttribute('src'),
    alt: rewardDrawing.alt,
    explanation: rewardExplanation.textContent,
    equation: rewardEquation.textContent,
    equationLabel: rewardEquation.getAttribute('aria-label'),
    caption: rewardCaption.textContent,
  };
  const tiedRewards = {
    src: 'assets/method-ties.svg',
    alt: 'The same four endpoints each score 0.5. Their endpoint\u2013reward covariance is zero.',
    explanation: 'All four candidates earn the same reward. Every pairwise reward difference is zero, so the auxiliary loss is inactive and the teacher supplies the training target.',
    equation: 'G = 0',
    equationLabel: 'G equals zero',
    caption: 'The endpoints stay fixed. Assigning 0.5 to all four cancels the auxiliary direction.',
  };

  function showRewardCase(example) {
    rewardDrawing.src = example.src;
    rewardDrawing.alt = example.alt;
    rewardExplanation.textContent = example.explanation;
    rewardCaption.textContent = example.caption;
    rewardEquation.textContent = example.equation;
    rewardEquation.setAttribute('aria-label', example.equationLabel);
    get('unequal-rewards').setAttribute('aria-pressed', example === unequalRewards);
    get('equal-rewards').setAttribute('aria-pressed', example === tiedRewards);
  }

  get('unequal-rewards').addEventListener('click', () => showRewardCase(unequalRewards));
  get('equal-rewards').addEventListener('click', () => showRewardCase(tiedRewards));
  get('reward-states').hidden = false;
}

// Reusable copy button handler
function wireCopyButton(buttonId, codeId, idleLabel, doneLabel) {
  const button = get(buttonId);
  const code = get(codeId);
  if (!button || !code) return;
  let timer;
  button.hidden = false;
  button.addEventListener('click', async () => {
    let message = doneLabel;
    try {
      await navigator.clipboard.writeText(code.textContent);
    } catch {
      const selection = window.getSelection();
      selection.selectAllChildren(code);
      message = 'Selected';
      try {
        if (document.execCommand('copy')) {
          selection.removeAllRanges();
          message = doneLabel;
        }
      } catch {
        // Keep selection available for manual copy.
      }
    }
    const status = get('copy-status');
    if (status) status.textContent = message;
    button.querySelector('span').textContent = message;
    clearTimeout(timer);
    timer = setTimeout(() => {
      if (status) status.textContent = '';
      button.querySelector('span').textContent = idleLabel;
    }, 2200);
  });
}

wireCopyButton('copy-citation', 'bibtex-code', 'Copy BibTeX', 'Copied');
wireCopyButton('copy-snippet', 'snippet-code', 'Copy', 'Copied');

// Highlight active section in sticky header
const navLinks = [...document.querySelectorAll('#primary-nav a[href^="#"]')];
if (navLinks.length && 'IntersectionObserver' in window) {
  const sections = navLinks
    .map((a) => document.querySelector(a.getAttribute('href')))
    .filter(Boolean);
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        const id = `#${entry.target.id}`;
        for (const a of navLinks) {
          a.classList.toggle('is-active', a.getAttribute('href') === id);
        }
      }
    },
    { rootMargin: '-25% 0px -60% 0px', threshold: 0 },
  );
  for (const sec of sections) observer.observe(sec);
}
