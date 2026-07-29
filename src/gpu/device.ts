export async function initWebGPU(): Promise<GPUDevice> {
  if (!("gpu" in navigator)) {
    throw new Error(
      "WebGPU is not available in this browser",
    );
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No GPU adapter found");
  }
  const device = await adapter.requestDevice();
  device.lost.then((info) => {
    console.error("WebGPU device lost :", info.message);
  });
  return device;
}
