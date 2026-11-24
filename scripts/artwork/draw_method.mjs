// Run with Three.js 0.180 installed alongside this file:
// node draw_method.mjs <output-directory>
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { OrthographicCamera, Vector3 } from 'three';

const output = process.argv[2];
if (!output) throw new Error('Provide the output directory for the SVG illustrations.');

const width = 560;
const height = 330;
const camera = new OrthographicCamera(-3.25, 3.25, 2.1, -2.1, 0.1, 30);
camera.position.set(2.1, 5.4, 9);
camera.lookAt(0, 0, 0);
camera.updateMatrixWorld();
const ink = '#193e38';
const clay = '#ae4e32';
const teal = '#127660';
const blue = '#415b90';
const grey = '#677675';
const format = (value) => Number(value.toFixed(1));
const coord = (point) => point.map(format).join(',');

// An invented surface provides depth cues. All three stages use the same view.
function surface(x, z) {
  return new Vector3(x, 0.55 * Math.sin(x * 1.29 - z * 0.56)
    + 0.23 * Math.cos(z * 1.8 + x * 0.44) + 0.095 * x * z, z);
}

function project(point) {
  const p = point.clone().project(camera);
  return [(p.x + 1) * width / 2, (1 - p.y) * height / 2 + 7];
}

function sampledPath(points) {
  return 'M' + points.map(coord).join(' L');
}

function sheet() {
  const grid = [];
  const nx = 36, nz = 20;
  for (let j = 0; j <= nz; j++) {
    grid.push(Array.from({ length: nx + 1 }, (_, i) =>
      surface(-2.76 + 5.52 * i / nx, -1.72 + 3.44 * j / nz)));
  }
  const light = new Vector3(-0.5, 1, -0.65).normalize();
  const facets = [];
  for (let j = 0; j < nz; j++) {
    for (let i = 0; i < nx; i++) {
      const a = grid[j][i], b = grid[j][i + 1], c = grid[j + 1][i];
      const normal = c.clone().sub(a).cross(b.clone().sub(a)).normalize();
      const t = Math.max(0, normal.dot(light));
      const diffuse = [148 + 85 * t, 183 + 59 * t, 171 + 59 * t].map(Math.round);
      const color = 'rgb(' + diffuse.join(',') + ')';
      const corners = [a, b, grid[j + 1][i + 1], c].map(project);
      facets.push('<path d="' + sampledPath(corners) + 'Z" fill="' + color
        + '" stroke="' + color + '" stroke-width="0.6"/>');
    }
  }
  const mesh = [];
  for (let j = 0; j <= nz; j += 2) mesh.push(sampledPath(grid[j].map(project)));
  for (let i = 0; i <= nx; i += 2) mesh.push(sampledPath(grid.map(row => project(row[i]))));
  return '<g>' + facets.join('') + '<path d="' + mesh.join(' ') +
    '" fill="none" stroke="#557d71" stroke-opacity=".39" stroke-width=".85"/></g>';
}

function line(d, color, options = {}) {
  const weight = options.width ?? 3.3;
  const dash = options.dash ? ' stroke-dasharray="' + options.dash + '"' : '';
  const end = options.arrow ? ' marker-end="url(#arrow-' + color.slice(1) + ')"' : '';
  return '<g fill="none" stroke-linejoin="round" stroke-linecap="round">' +
    '<path d="' + d + '" stroke="#fcfcf7" stroke-opacity=".86" stroke-width="' + (weight + 2.8) + '"/>' +
    '<path d="' + d + '" stroke="' + color + '" stroke-width="' + weight + '"' + dash + end + '/></g>';
}

function dot(p, color, radius = 5) {
  return '<circle cx="' + p[0] + '" cy="' + p[1] + '" r="' + radius + '" fill="' +
    color + '" stroke="white" stroke-width="2.3"/>';
}

function label(x, y, value, color = ink, anchor = 'start') {
  return '<text x="' + x + '" y="' + y + '" text-anchor="' + anchor + '" fill="' + color +
    '" stroke="#f8faf4" stroke-width="4" stroke-linejoin="round" paint-order="stroke">' + value + '</text>';
}

function teacher() {
  const state = [221, 183], base = [295, 225], specialist = [335, 176];
  const proxy = specialist.map((s, i) => s + 1.2 * (s - base[i]));
  return [
    line('M54,213 C90,236 118,243 156,224 S195,193 221,183', ink),
    line(sampledPath([base, proxy]), grey, {width: 1.8, dash: '3 5'}),
    line(sampledPath([state, base]), grey, {width: 2.6, dash: '6 5', arrow: true}),
    line(sampledPath([state, specialist]), clay, {arrow: true}),
    line(sampledPath([state, proxy]), teal, {arrow: true}),
    dot(state, ink, 6), label(48, 266, 'Student rollout'), label(212, 157, 'Visited state', ink, 'end'),
    label(303, 250, 'Base', grey), label(347, 184, 'Specialist', clay),
    label(387, 100, 'Proxy target', teal),
  ].join('');
}

const candidates = [
  {gain: '0', color: ink, y: 238, x: 370, bend: 224, dash: ''},
  {gain: '0.05', color: blue, y: 199, x: 416, bend: 185, dash: '2 5'},
  {gain: '0.25', color: teal, y: 148, x: 374, bend: 140, dash: '9 4'},
  {gain: '0.50', color: clay, y: 95, x: 416, bend: 91, dash: ''},
].map((entry, index) => ({
  ...entry,
  path: 'M79,205 C144,' + (251 - 8 * index) + ' 180,' + (235 - 12 * index) + ' 227,' +
    (210 - 15 * index) + ' S320,' + entry.bend + ' ' + entry.x + ',' + entry.y,
}));

function exploration() {
  const paths = candidates.map(({gain, color, y, x, path, dash}) =>
    line(path, color, {dash}) +
    dot([x, y], color) + label(x + 12, y + 6, 'g = ' + gain, color));
  return paths.join('') + dot([79,205], ink, 6) + label(30, 181, 'Shared noise');
}

function rewards(tied) {
  // Screen-space examples demonstrate the covariance direction. These are toy rewards.
  const points = candidates.map(candidate => [candidate.x, candidate.y]);
  const scores = tied ? [0.5, 0.5, 0.5, 0.5] : [0.2, 0.8, 0.6, 0.4];
  const mean = [0,1].map(k => points.reduce((sum,p) => sum + p[k], 0) / points.length);
  const meanScore = scores.reduce((a,b) => a+b, 0) / scores.length;
  const covariance = [0,1].map(k => points.reduce((sum,p,i) =>
    sum + (p[k] - mean[k]) * (scores[i] - meanScore), 0) / points.length);
  const norm = Math.hypot(...covariance);
  const content = '<g opacity=".32">' + candidates.map(({path}) =>
    '<path fill="none" stroke="#738d82" stroke-width="1.8" d="' + path + '"/>').join('') + '</g>' +
    points.map((point,i) => dot(point, candidates[i].color, 6) +
      label(point[0] + 12, point[1] + 6, 'r = ' + scores[i].toFixed(1), ink)).join('');
  // Translate the direction within the illustration to make it readable beside the cloud.
  const origin = [222,192];
  if (!norm) return content + dot(origin, grey, 6) + label(origin[0] - 10, origin[1] + 25, 'G = 0', ink, 'end');
  const tip = origin.map((v,i) => v + 76 * covariance[i] / norm);
  return content + line(sampledPath([origin, tip]), teal, {width: 4.8, arrow: true}) +
    dot(origin, teal) + label(tip[0] + 9, tip[1] - 4, 'G', teal);
}

const surfaceArt = sheet();
const markers = [ink, clay, teal, grey].map(color => '<marker id="arrow-' + color.slice(1) +
  '" viewBox="0 0 10 10" refX="8.7" refY="5" markerUnits="userSpaceOnUse" markerWidth="14" markerHeight="14" orient="auto">' +
  '<path d="M0,0 L10,5 L0,10 L2.5,5Z" fill="' + color + '"/></marker>').join('');
for (const [name,title,art] of [
  ['teacher','Teacher extrapolation at a visited state',teacher()],
  ['candidates','Four trajectories from shared initial noise',exploration()],
  ['rewards','Unequal candidate rewards give an auxiliary direction',rewards(false)],
  ['ties','Equal candidate rewards cancel the auxiliary direction',rewards(true)],
]) {
  const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="560" height="330" viewBox="0 0 560 330" role="img">' +
    '<title>' + title + '</title><defs>' + markers + '</defs>' + surfaceArt +
    '<g font-family="Arial, sans-serif" font-size="19" font-weight="600">' + art + '</g></svg>\n';
  writeFileSync(join(output, 'method-' + name + '.svg'), svg);
}
const motion = candidates.map(({color, path}) => ({color, path}));
writeFileSync(join(output, 'method-trajectories.js'),
  '// Generated from the same paths as method-candidates.svg.\nwindow.DEXOPD_TRAJECTORIES = ' +
  JSON.stringify(motion, null, 2) + ';\n');
