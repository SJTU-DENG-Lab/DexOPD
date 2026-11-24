'use strict';

const { comparison, gallery } = window.DEXOPD_EXAMPLES;
const categories = { composition: 'Composition', text: 'Text rendering', aesthetic: 'Aesthetics' };
const assets = 'assets/';
const get = (id) => document.getElementById(id);

const baseline = get('baseline');
const sideMode = get('side-mode');
const wipeMode = get('wipe-mode');
const wipePosition = get('wipe-position');
const lightbox = get('lightbox');
let activeCase = comparison.find((example) => example.id === 'ocr-089') || comparison[0];

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function imageSettings(example, cfg) {
  return `${example.resolution.join(' \u00d7 ')} \u00b7 ${example.steps} steps \u00b7 Seed ${example.seed} \u00b7 CFG ${cfg}`;
}

function openImage(example, image, title, cfg) {
  get('lightbox-title').textContent = title;
  get('lightbox-prompt').textContent = example.prompt;
  get('lightbox-settings').textContent = imageSettings(example, cfg);
  const preview = get('lightbox-image');
  preview.alt = title;
  preview.src = assets + image;
  lightbox.showModal();
}

function imageButton(example, image, title, cfg, lazy = false) {
  const button = element('button', 'zoom-image');
  button.type = 'button';
  button.title = 'View full image';
  button.setAttribute('aria-label', `View full image: ${title}`);
  const img = new Image(...example.resolution);
  img.alt = '';
  if (lazy) {
    img.loading = 'lazy';
    img.decoding = 'async';
  }
  img.src = assets + image;
  const icon = get('zoom-icon').content.cloneNode(true);
  button.append(img, icon);
  button.addEventListener('click', () => openImage(example, image, title, cfg));
  return button;
}

function renderComparison() {
  const model = baseline.selectedOptions[0].textContent;
  const example = activeCase;
  const other = example.samples[baseline.value];
  const dex = example.samples.dexopd;
  get('baseline-figure').replaceChildren(
    element('figcaption', '', model),
    imageButton(example, other.image, `${example.name} / ${model}`, other.cfg),
  );
  get('dex-figure').replaceChildren(
    element('figcaption', '', 'DexOPD'),
    imageButton(example, dex.image, `${example.name} / DexOPD`, dex.cfg),
  );
  get('wipe-baseline-name').textContent = model;
  get('wipe-baseline-image').src = assets + other.image;
  get('wipe-baseline-image').alt = `${example.name} from ${model}`;
  get('wipe-dex-image').src = assets + dex.image;
  get('wipe-dex-image').alt = `${example.name} from DexOPD`;
  const cfg = other.cfg === dex.cfg ? dex.cfg : `${other.cfg} (${model}) / ${dex.cfg} (DexOPD)`;
  get('compare-settings').textContent = imageSettings(example, cfg);
}

function selectPrompt(example) {
  activeCase = example;
  for (const button of get('prompt-list').children) {
    button.setAttribute('aria-pressed', button.dataset.case === example.id);
  }
  get('compare-category').textContent = categories[example.category];
  get('compare-prompt').textContent = example.prompt;
  renderComparison();
}

for (const example of comparison) {
  const button = element('button', 'prompt-option');
  const thumb = new Image(96, 96);
  thumb.alt = '';
  thumb.src = `${assets}thumb-${example.id}.webp`;
  button.type = 'button';
  button.dataset.case = example.id;
  button.append(thumb, element('span', '', example.name));
  button.addEventListener('click', () => selectPrompt(example));
  get('prompt-list').append(button);
}

function setView(wipe) {
  sideMode.setAttribute('aria-pressed', !wipe);
  wipeMode.setAttribute('aria-pressed', wipe);
  get('side-view').hidden = wipe;
  get('wipe-view').hidden = !wipe;
}

baseline.addEventListener('change', renderComparison);
sideMode.addEventListener('click', () => setView(false));
wipeMode.addEventListener('click', () => setView(true));
wipePosition.addEventListener('input', () => {
  wipePosition.parentElement.style.setProperty('--wipe', `${wipePosition.value}%`);
});
selectPrompt(activeCase);
get('comparison').hidden = false;

for (const example of gallery) {
  const tile = element('figure', 'gallery-tile');
  tile.append(
    imageButton(example, example.image, example.name, example.cfg, true),
    element('figcaption', '', example.name),
  );
  get(`gallery-${example.category}`).append(tile);
}

function setupTabs(list) {
  const tabs = [...list.querySelectorAll('[role="tab"]')];
  function select(current) {
    for (const tab of tabs) {
      const selected = tab === current;
      tab.setAttribute('aria-selected', selected);
      tab.tabIndex = selected ? 0 : -1;
      const panel = get(tab.getAttribute('aria-controls'));
      panel.hidden = !selected;
      panel.setAttribute('aria-labelledby', tab.id);
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

setupTabs(get('gallery-tabs'));

get('close-lightbox').addEventListener('click', () => lightbox.close());
lightbox.addEventListener('click', (event) => {
  if (event.target === lightbox) lightbox.close();
});
