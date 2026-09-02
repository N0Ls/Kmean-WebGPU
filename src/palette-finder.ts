import type { LoadedImage } from "./image";
import { rgbToLab } from "./color";
import labWGSL from "./shaders/lab.wgsl?raw";
import histogramWGSL from "./shaders/histogram.wgsl?raw";
import binprepWGSL from "./shaders/binprep.wgsl?raw";
import bassignWGSL from "./shaders/bassign.wgsl?raw";
import bfinalizeWGSL from "./shaders/bfinalize.wgsl?raw";
import quantizeWGSL from "./shaders/quantize.wgsl?raw";
import blitWGSL from "./shaders/blit.wgsl?raw";

// 5 bits/channel -> 2^15 histogram bins. Bounds the clustering work regardless of
// image size and keeps the fixed-point atomics in bassign.wgsl inside a u32.
const NUM_BINS = 1 << 15;

// Shaders that cluster or render in Lab share the conversion code; WGSL has no
// #include, so we prepend it at compile time.
const withLab = (src: string) => `${labWGSL}\n${src}`;

export interface KMeansOptions {
  k: number;
  maxIterations: number;
  seed?: number;
}

export interface KMeansResult {
  centroids: Float32Array; // k*4: r,g,b,count
  iterations: number;
}

export class PaletteFinder {
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private canvasFormat: GPUTextureFormat;

  // Compute pipelines: GPU histogram -> bin prep -> Lloyd (assign/finalize) -> quantize.
  private histPipeline!: GPUComputePipeline;
  private binprepPipeline!: GPUComputePipeline;
  private assignPipeline!: GPUComputePipeline;
  private finalizePipeline!: GPUComputePipeline;
  private quantizePipeline!: GPUComputePipeline;
  private blitPipeline!: GPURenderPipeline;

  private sampler!: GPUSampler;
  private built = false;

  constructor(device: GPUDevice, canvas: HTMLCanvasElement) {
    this.device = device;
    const ctx = canvas.getContext("webgpu");
    if (!ctx) throw new Error("Could not get a WebGPU canvas context.");
    this.context = ctx;
    this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device, format: this.canvasFormat, alphaMode: "opaque" });
  }

  private build() {
    // Build the pipelines once, lazily on first run.
    if (this.built) return;
    const d = this.device;

    const compute = (code: string) =>
      d.createComputePipeline({
        layout: "auto",
        compute: { module: d.createShaderModule({ code }), entryPoint: "main" },
      });

    this.histPipeline = compute(histogramWGSL);
    this.binprepPipeline = compute(withLab(binprepWGSL));
    this.assignPipeline = compute(bassignWGSL);
    this.finalizePipeline = compute(withLab(bfinalizeWGSL));
    this.quantizePipeline = compute(withLab(quantizeWGSL));

    const blitModule = d.createShaderModule({ code: blitWGSL });
    this.blitPipeline = d.createRenderPipeline({
      layout: "auto",
      vertex: { module: blitModule, entryPoint: "vs" },
      fragment: {
        module: blitModule,
        entryPoint: "fs",
        targets: [{ format: this.canvasFormat }],
      },
      primitive: { topology: "triangle-list" },
    });

    this.sampler = d.createSampler({ magFilter: "nearest", minFilter: "nearest" });
    this.built = true;
  }

  /**
   * Clear the output canvas to black. Call when a new image is loaded so the
   * previously quantized frame doesn't linger on the WebGPU canvas (resizing the
   * canvas alone doesn't reliably clear a configured context).
   */
  clearCanvas() {
    const d = this.device;
    const enc = d.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          loadOp: "clear",
          storeOp: "store",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        },
      ],
    });
    pass.end();
    d.queue.submit([enc.finish()]);
  }

  /**
   * Perceptual palette extraction, fully on the GPU. The pipeline:
   *   1. histogram  — atomic-bin the pixels (GPU).
   *   2. binprep    — average colour per bin -> Lab + sqrt(count) weight (GPU).
   *   3. seed       — weighted k-means++ over the (small) histogram (CPU; sequential).
   *   4. Lloyd loop — assign bins + weighted-mean update in Lab (GPU), no per-iteration
   *                   readback: the whole loop is encoded and submitted once.
   *   5. quantize + blit — render the preview against the final palette (GPU).
   * Only one small histogram readback (for seeding) and one centroid readback (the
   * palette) touch the CPU; the iteration loop never leaves the GPU.
   */
  async run(image: LoadedImage, opts: KMeansOptions): Promise<KMeansResult> {
    this.build();
    const d = this.device;
    const { k, maxIterations } = opts;
    const seed = opts.seed ?? 0x9e3779b9;
    const { width, height, pixels } = image;
    const n = width * height;

    // ---- Buffers -------------------------------------------------------------
    const params = d.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(params, 0, new Uint32Array([k, n, width, height, NUM_BINS, 0, 0, 0]));

    const pixelBuf = d.createBuffer({
      size: pixels.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    d.queue.writeBuffer(pixelBuf, 0, pixels);

    // histogram: NUM_BINS * (sumR,sumG,sumB,count) u32
    const histBuf = d.createBuffer({
      size: NUM_BINS * 4 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    d.queue.writeBuffer(histBuf, 0, new Uint32Array(NUM_BINS * 4)); // clear to 0

    const binLabBuf = d.createBuffer({ size: NUM_BINS * 4 * 4, usage: GPUBufferUsage.STORAGE });
    const binCountBuf = d.createBuffer({ size: NUM_BINS * 4, usage: GPUBufferUsage.STORAGE });

    const centersLabBuf = d.createBuffer({
      size: k * 4 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const centersRgbBuf = d.createBuffer({
      size: k * 4 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const accumBuf = d.createBuffer({
      size: k * 8 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const histReadBuf = d.createBuffer({
      size: NUM_BINS * 4 * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const centersReadBuf = d.createBuffer({
      size: k * 4 * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const outputTexture = d.createTexture({
      size: [width, height],
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    // ---- Bind groups ---------------------------------------------------------
    const histBind = d.createBindGroup({
      layout: this.histPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: pixelBuf } },
        { binding: 1, resource: { buffer: histBuf } },
        { binding: 2, resource: { buffer: params } },
      ],
    });
    const binprepBind = d.createBindGroup({
      layout: this.binprepPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: histBuf } },
        { binding: 1, resource: { buffer: binLabBuf } },
        { binding: 2, resource: { buffer: binCountBuf } },
        { binding: 3, resource: { buffer: params } },
      ],
    });
    const assignBind = d.createBindGroup({
      layout: this.assignPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: binLabBuf } },
        { binding: 1, resource: { buffer: binCountBuf } },
        { binding: 2, resource: { buffer: centersLabBuf } },
        { binding: 3, resource: { buffer: accumBuf } },
        { binding: 4, resource: { buffer: params } },
      ],
    });
    const finalizeBind = d.createBindGroup({
      layout: this.finalizePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: centersLabBuf } },
        { binding: 1, resource: { buffer: accumBuf } },
        { binding: 2, resource: { buffer: centersRgbBuf } },
        { binding: 3, resource: { buffer: binLabBuf } },
        { binding: 4, resource: { buffer: params } },
      ],
    });
    const quantizeBind = d.createBindGroup({
      layout: this.quantizePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: pixelBuf } },
        { binding: 1, resource: { buffer: centersLabBuf } },
        { binding: 2, resource: { buffer: centersRgbBuf } },
        { binding: 3, resource: outputTexture.createView() },
        { binding: 4, resource: { buffer: params } },
      ],
    });
    const blitBind = d.createBindGroup({
      layout: this.blitPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: outputTexture.createView() },
        { binding: 1, resource: this.sampler },
      ],
    });

    // ---- Pass 1: histogram + bin prep, then read the histogram back for seeding.
    {
      const enc = d.createCommandEncoder();
      {
        const pass = enc.beginComputePass();
        pass.setPipeline(this.histPipeline);
        pass.setBindGroup(0, histBind);
        pass.dispatchWorkgroups(Math.ceil(n / 256));
        pass.end();
      }
      {
        const pass = enc.beginComputePass();
        pass.setPipeline(this.binprepPipeline);
        pass.setBindGroup(0, binprepBind);
        pass.dispatchWorkgroups(Math.ceil(NUM_BINS / 256));
        pass.end();
      }
      enc.copyBufferToBuffer(histBuf, 0, histReadBuf, 0, NUM_BINS * 4 * 4);
      d.queue.submit([enc.finish()]);
    }

    await histReadBuf.mapAsync(GPUMapMode.READ);
    const hist = new Uint32Array(histReadBuf.getMappedRange().slice(0));
    histReadBuf.unmap();

    // ---- Seed centroids on the CPU (weighted k-means++ in Lab). Sequential and
    // cheap over the small histogram; keeps small accents from being outvoted.
    const initLab = seedFromHistogram(hist, k, seed);
    d.queue.writeBuffer(centersLabBuf, 0, initLab);

    // ---- Pass 2: the Lloyd loop (GPU-resident, no per-iteration readback) then
    // the final quantize + blit. Everything is encoded once and submitted once.
    {
      const enc = d.createCommandEncoder();
      for (let iter = 0; iter < maxIterations; iter++) {
        enc.clearBuffer(accumBuf); // zero the accumulators (encoder-ordered)
        {
          const pass = enc.beginComputePass();
          pass.setPipeline(this.assignPipeline);
          pass.setBindGroup(0, assignBind);
          pass.dispatchWorkgroups(Math.ceil(NUM_BINS / 256));
          pass.end();
        }
        {
          const pass = enc.beginComputePass();
          pass.setPipeline(this.finalizePipeline);
          pass.setBindGroup(0, finalizeBind);
          pass.dispatchWorkgroups(Math.ceil(k / 64));
          pass.end();
        }
      }
      {
        const pass = enc.beginComputePass();
        pass.setPipeline(this.quantizePipeline);
        pass.setBindGroup(0, quantizeBind);
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        pass.end();
      }
      {
        const pass = enc.beginRenderPass({
          colorAttachments: [
            {
              view: this.context.getCurrentTexture().createView(),
              loadOp: "clear",
              storeOp: "store",
              clearValue: { r: 0, g: 0, b: 0, a: 1 },
            },
          ],
        });
        pass.setPipeline(this.blitPipeline);
        pass.setBindGroup(0, blitBind);
        pass.draw(3);
        pass.end();
      }
      enc.copyBufferToBuffer(centersRgbBuf, 0, centersReadBuf, 0, k * 4 * 4);
      d.queue.submit([enc.finish()]);
    }

    await centersReadBuf.mapAsync(GPUMapMode.READ);
    const centroids = new Float32Array(centersReadBuf.getMappedRange().slice(0));
    centersReadBuf.unmap();

    // Free per-run GPU resources.
    for (const b of [pixelBuf, histBuf, binLabBuf, binCountBuf, centersLabBuf,
      centersRgbBuf, accumBuf, histReadBuf, centersReadBuf, params]) {
      b.destroy();
    }
    outputTexture.destroy();

    return { centroids, iterations: maxIterations };
  }
}

// ---- CPU seeding ------------------------------------------------------------

// Seeded PRNG for deterministic k-means++ initialization. Returns a float in [0,1).
function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pickWeighted(weight: Float32Array, n: number, rng: () => number): number {
  let total = 0;
  for (let i = 0; i < n; i++) total += weight[i];
  let target = rng() * total;
  for (let i = 0; i < n - 1; i++) {
    target -= weight[i];
    if (target <= 0) return i;
  }
  return n - 1;
}

/**
 * Weighted k-means++ seeding in CIELAB over the GPU-built histogram. `hist` is
 * NUM_BINS * (sumR,sumG,sumB,count) u32. Returns k*4 floats (L,a,b,0) — the initial
 * centroids the GPU Lloyd loop refines. Weight = sqrt(count), matching binprep.wgsl,
 * so a small vivid bin can still be seeded instead of being drowned by area.
 */
function seedFromHistogram(hist: Uint32Array, k: number, seed: number): Float32Array<ArrayBuffer> {
  // Compact the non-empty bins into flat Lab + weight arrays.
  let m = 0;
  for (let i = 0; i < NUM_BINS; i++) if (hist[i * 4 + 3] > 0) m++;

  const lab = new Float32Array(m * 3);
  const weight = new Float32Array(m);
  let j = 0;
  for (let i = 0; i < NUM_BINS; i++) {
    const count = hist[i * 4 + 3];
    if (count === 0) continue;
    const inv = 1 / count;
    const [L, a, b] = rgbToLab(hist[i * 4] * inv, hist[i * 4 + 1] * inv, hist[i * 4 + 2] * inv);
    lab[j * 3] = L;
    lab[j * 3 + 1] = a;
    lab[j * 3 + 2] = b;
    weight[j] = Math.sqrt(count);
    j++;
  }

  const out = new Float32Array(k * 4);
  const rng = mulberry32(seed);

  // Fewer distinct colours than clusters: hand each bin back (rest reuse the last).
  if (m <= k) {
    for (let c = 0; c < k; c++) {
      const i = Math.min(c, m - 1);
      out[c * 4] = lab[i * 3];
      out[c * 4 + 1] = lab[i * 3 + 1];
      out[c * 4 + 2] = lab[i * 3 + 2];
    }
    return out;
  }

  const dist2 = new Float32Array(m).fill(Infinity);
  const first = pickWeighted(weight, m, rng);
  out[0] = lab[first * 3];
  out[1] = lab[first * 3 + 1];
  out[2] = lab[first * 3 + 2];

  for (let c = 1; c < k; c++) {
    const cx = out[(c - 1) * 4];
    const cy = out[(c - 1) * 4 + 1];
    const cz = out[(c - 1) * 4 + 2];
    let sum = 0;
    for (let i = 0; i < m; i++) {
      const dl = lab[i * 3] - cx;
      const da = lab[i * 3 + 1] - cy;
      const db = lab[i * 3 + 2] - cz;
      const dd = dl * dl + da * da + db * db;
      if (dd < dist2[i]) dist2[i] = dd;
      sum += weight[i] * dist2[i];
    }
    let target = rng() * sum;
    let idx = 0;
    for (; idx < m - 1; idx++) {
      target -= weight[idx] * dist2[idx];
      if (target <= 0) break;
    }
    out[c * 4] = lab[idx * 3];
    out[c * 4 + 1] = lab[idx * 3 + 1];
    out[c * 4 + 2] = lab[idx * 3 + 2];
  }

  return out;
}
