import "./style.css";
import { initWebGPU } from "./gpu/device";
import { loadImageToPixels, type LoadedImage } from "./image";
import { PaletteFinder } from "./palette-finder";

interface RGB { r: number; g: number; b: number; }

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

// UI elements
const fileInput = $<HTMLInputElement>("file");
const fileLabel = $<HTMLSpanElement>("fileLabel");
const kInput = $<HTMLInputElement>("k");
const kOut = $<HTMLOutputElement>("kOut");
const itersInput = $<HTMLInputElement>("iters");
const iterOut = $<HTMLOutputElement>("iterOut");
const runButton = $<HTMLButtonElement>("run");
const shuffleButton = $<HTMLButtonElement>("shuffle");
const paletteEl = $<HTMLElement>("palette");
const imagesSection = $<HTMLElement>("images");
const originalImg = $<HTMLImageElement>("original");
const outputCanvas = $<HTMLCanvasElement>("output");

// Compare slider + color info panel
const compare = $<HTMLDivElement>("compare");
const colorInfo = $<HTMLDivElement>("colorInfo");
const infoSwatch = $<HTMLDivElement>("infoSwatch");
const infoHex = $<HTMLSpanElement>("infoHex");
const infoRgb01 = $<HTMLSpanElement>("infoRgb01");
const infoRgb255 = $<HTMLSpanElement>("infoRgb255");
const infoHsl = $<HTMLSpanElement>("infoHsl");
const downloadButton = $<HTMLButtonElement>("download");

// Slider value read-outs + black fill on the left of the track.
kInput.addEventListener("input", () => {
  kOut.value = kInput.value;
  updateSliderFill(kInput);
});
itersInput.addEventListener("input", () => {
  iterOut.value = itersInput.value;
  updateSliderFill(itersInput);
});
updateSliderFill(kInput);
updateSliderFill(itersInput);

// App state
let device: GPUDevice | null = null;
let finder: PaletteFinder | null = null;
let currentImage: LoadedImage | null = null;
let running = false;
let seed = 0x9e3779b9;
let objectUrl: string | null = null;

// Palette selection, kept across re-renders so the info panel stays stable.
let paletteColors: RGB[] = [];
let selectedIndex = 0;

async function initPaletteFinder(): Promise<PaletteFinder> {
  if (!device) device = await initWebGPU();
  if (!finder) finder = new PaletteFinder(device, outputCanvas);
  return finder;
}

fileInput.addEventListener("change", async () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  try {
    currentImage = await loadImageToPixels(file);

    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    originalImg.src = objectUrl;

    outputCanvas.width = currentImage.width;
    outputCanvas.height = currentImage.height;
    fileLabel.textContent = file.name;

    // Drop the previous run's quantized frame so we don't show it next to the
    // new original. Only if the finder (and its GPU context) already exists.
    finder?.clearCanvas();

    imagesSection.hidden = false;
    setComparePos(50); // start centered
    paletteEl.innerHTML = "";
    paletteColors = [];
    selectedIndex = 0;
    colorInfo.hidden = true;
    downloadButton.hidden = true;
    runButton.disabled = false;
    shuffleButton.disabled = false;
  } catch (err) {
    reportError(err);
  }
});

runButton.addEventListener("click", () => runClusterization());

shuffleButton.addEventListener("click", () => {
  seed = (seed + 0x9e3779b9) >>> 0; // Advance the seed to explore an alternative
  runClusterization();
});

async function runClusterization() {
  if (!currentImage || running) return;

  running = true;
  runButton.disabled = true;
  shuffleButton.disabled = true;

  try {
    const finder = await initPaletteFinder();
    const k = Number(kInput.value);
    const maxIterations = Number(itersInput.value);

    // Perceptual k-means runs on the GPU: histogram + weighted Lloyd loop in CIELAB,
    // then the quantized preview is rendered against the final centroids.
    const result = await finder.run(currentImage, { k, maxIterations, seed });
    renderPaletteUI(result.centroids, k);
  } catch (err) {
    reportError(err);
  } finally {
    running = false;
    runButton.disabled = false;
    shuffleButton.disabled = false;
  }
}

// ---- Palette + color info ----

function renderPaletteUI(centroids: Float32Array, k: number) {
  const entries: (RGB & { count: number; })[] = [];
  for (let i = 0; i < k; i++) {
    entries.push({
      r: Math.round(centroids[i * 4]),
      g: Math.round(centroids[i * 4 + 1]),
      b: Math.round(centroids[i * 4 + 2]),
      count: centroids[i * 4 + 3],
    });
  }
  entries.sort((a, b) => b.count - a.count);

  paletteColors = entries.map(({ r, g, b }) => ({ r, g, b }));
  if (selectedIndex >= paletteColors.length) selectedIndex = 0;

  paletteEl.innerHTML = "";
  entries.forEach((e, i) => {
    const el = document.createElement("button");
    el.type = "button";
    el.className = i === selectedIndex ? "swatch selected" : "swatch";
    el.style.background = toHex(e.r, e.g, e.b);
    el.addEventListener("click", () => selectColor(i));
    paletteEl.appendChild(el);
  });

  if (paletteColors.length) {
    updateColorInfo(paletteColors[selectedIndex]);
    colorInfo.hidden = false;
    downloadButton.hidden = false;
  }
}

// ---- Download poster: image (padded) with the palette underneath ----

downloadButton.addEventListener("click", () => downloadPoster());

async function downloadPoster() {
  if (!paletteColors.length || !originalImg.complete) return;

  const iw = originalImg.naturalWidth || currentImage?.width || 0;
  const ih = originalImg.naturalHeight || currentImage?.height || 0;
  if (!iw || !ih) return;

  // Cap the longest side so the exported poster stays a sensible size.
  const scale = Math.min(1, 1600 / Math.max(iw, ih));
  const drawW = Math.max(1, Math.round(iw * scale));
  const drawH = Math.max(1, Math.round(ih * scale));

  // One uniform spacing value: the outer padding and the gaps between swatches.
  const gap = Math.round(drawW * 0.015);
  const swatchH = Math.round(drawW * 0.14);

  const canvas = document.createElement("canvas");
  canvas.width = drawW + gap * 2;
  canvas.height = gap + drawH + gap + swatchH + gap;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(originalImg, gap, gap, drawW, drawH);

  // Palette swatches span the image width, separated by `gap`-wide white strips.
  const paletteY = gap + drawH + gap;
  const n = paletteColors.length;
  const swatchW = (drawW - gap * (n - 1)) / n;
  for (let i = 0; i < n; i++) {
    const x = gap + i * (swatchW + gap);
    const { r, g, b } = paletteColors[i];
    ctx.fillStyle = toHex(r, g, b);
    ctx.fillRect(Math.round(x), paletteY, Math.round(swatchW), swatchH);
  }

  const blob = await new Promise<Blob | null>((resolve) =>
    canvas.toBlob(resolve, "image/png"),
  );
  if (!blob) return;

  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${(fileLabel.textContent || "palette").replace(/\.[^.]+$/, "")}-palette.png`;
  a.click();
  URL.revokeObjectURL(url);
}

function selectColor(i: number) {
  selectedIndex = i;
  Array.from(paletteEl.children).forEach((c, idx) =>
    c.classList.toggle("selected", idx === i),
  );
  updateColorInfo(paletteColors[i]);
}

function updateColorInfo({ r, g, b }: RGB) {
  const hex = toHex(r, g, b);
  infoSwatch.style.background = hex;
  infoHex.textContent = hex.toUpperCase();
  infoRgb01.textContent = `${(r / 255).toFixed(3)}, ${(g / 255).toFixed(3)}, ${(b / 255).toFixed(3)}`;
  infoRgb255.textContent = `${r}, ${g}, ${b}`;
  const [h, s, l] = rgbToHsl(r, g, b);
  infoHsl.textContent = `${h}, ${s}%, ${l}%`;
}

// ---- Before/after compare slider ----

let dragging = false;

compare.addEventListener("pointerdown", (e) => {
  dragging = true;
  compare.setPointerCapture(e.pointerId);
  moveCompare(e);
});
compare.addEventListener("pointermove", (e) => {
  if (dragging) moveCompare(e);
});
compare.addEventListener("pointerup", () => (dragging = false));
compare.addEventListener("pointercancel", () => (dragging = false));

function moveCompare(e: PointerEvent) {
  const rect = compare.getBoundingClientRect();
  setComparePos(((e.clientX - rect.left) / rect.width) * 100);
}

function setComparePos(pct: number) {
  const clamped = Math.max(0, Math.min(100, pct));
  compare.style.setProperty("--pos", `${clamped}%`);
}

// ---- Helpers ----

function toHex(r: number, g: number, b: number): string {
  const h = (v: number) => v.toString(16).padStart(2, "0");
  return `#${h(r)}${h(g)}${h(b)}`;
}

function rgbToHsl(r: number, g: number, b: number): [number, number, number] {
  const rn = r / 255, gn = g / 255, bn = b / 255;
  const max = Math.max(rn, gn, bn), min = Math.min(rn, gn, bn);
  const l = (max + min) / 2;
  let h = 0;
  let s = 0;
  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    switch (max) {
      case rn: h = (gn - bn) / d + (gn < bn ? 6 : 0); break;
      case gn: h = (bn - rn) / d + 2; break;
      default: h = (rn - gn) / d + 4; break;
    }
    h /= 6;
  }
  return [Math.round(h * 360), Math.round(s * 100), Math.round(l * 100)];
}

function updateSliderFill(input: HTMLInputElement) {
  const min = Number(input.min);
  const max = Number(input.max);
  const pct = ((Number(input.value) - min) / (max - min)) * 100;
  input.style.setProperty("--fill", `${pct}%`);
}

function reportError(err: unknown) {
  console.error(err);
}
