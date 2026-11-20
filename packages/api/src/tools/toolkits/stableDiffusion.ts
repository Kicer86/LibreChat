import type { ImageInput } from './images';

/** Longest side of a generated image (txt2img renders a square of this size). */
const IMAGE_SIZE = 1024;
/** img2img dimensions are rounded to this multiple, which every common model accepts. */
const SIZE_MULTIPLE = 64;

const GENERATION_DEFAULTS = {
  cfg_scale: 4.5,
  steps: 22,
  width: IMAGE_SIZE,
  height: IMAGE_SIZE,
};

/**
 * The server behind `SD_WEBUI_URL`. stable-diffusion.cpp's `sd-server` fills omitted generation
 * parameters from the loaded model's own defaults, so it gets no hardcoded ones; other WebUI
 * servers (AUTOMATIC1111, Forge) fall back to SD 1.5-era values and keep getting ours.
 */
export type StableDiffusionBackend = 'sdcpp' | 'webui';

/** How long a detected backend is reused before the server is asked again. */
const BACKEND_CACHE_TTL_MS = 5 * 60 * 1000;
const BACKEND_CACHE_MAX_ENTRIES = 100;
const backendCache = new Map<string, { backend: StableDiffusionBackend; expiresAt: number }>();

/** Fetches a URL, resolving with any HTTP status; rejects only when no response arrived. */
export type StableDiffusionProbe = (url: string) => Promise<{ status: number; data: unknown }>;

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  value != null && typeof value === 'object' && !Array.isArray(value);

/**
 * Whether a `/sdcpp/v1/capabilities` response body is sd-server's: it names the current
 * generation mode and carries the loaded model's default generation parameters. A proxy or
 * another server answering the route with arbitrary JSON (`{}`, `[]`, an error object) is not.
 */
export function isStableDiffusionCppCapabilities(data: unknown): boolean {
  return (
    isPlainObject(data) &&
    typeof data.current_mode === 'string' &&
    data.current_mode.length > 0 &&
    isPlainObject(data.defaults)
  );
}

/**
 * Detects whether `url` is served by stable-diffusion.cpp, which answers
 * `/sdcpp/v1/capabilities`. A definite answer is cached per URL; when the server could not be
 * reached the result falls back to `webui` without caching, so the next call asks again.
 */
export async function detectStableDiffusionBackend(
  url: string,
  probe: StableDiffusionProbe,
  now: number = Date.now(),
): Promise<StableDiffusionBackend> {
  const cached = backendCache.get(url);
  if (cached && cached.expiresAt > now) {
    return cached.backend;
  }

  let backend: StableDiffusionBackend;
  try {
    const { status, data } = await probe(`${url}/sdcpp/v1/capabilities`);
    if (status >= 500) {
      return 'webui';
    }
    backend = status === 200 && isStableDiffusionCppCapabilities(data) ? 'sdcpp' : 'webui';
  } catch {
    return 'webui';
  }

  if (backendCache.size >= BACKEND_CACHE_MAX_ENTRIES) {
    backendCache.clear();
  }
  backendCache.set(url, { backend, expiresAt: now + BACKEND_CACHE_TTL_MS });
  return backend;
}

/** Forgets detected backends (for tests). */
export function clearStableDiffusionBackendCache(): void {
  backendCache.clear();
}

export interface StableDiffusionRequestParams {
  prompt: string;
  negative_prompt?: string;
  /** The starting image; switches the request from txt2img to img2img. */
  image?: ImageInput;
  /** How much img2img may change the image (0–1); omitted, the server default applies. */
  denoising_strength?: number;
  /** Defaults to `webui`, which sends this tool's fixed generation parameters. */
  backend?: StableDiffusionBackend;
}

export interface StableDiffusionRequest {
  /** Path on the Stable Diffusion WebUI-compatible API (`/sdapi/v1/...`). */
  path: '/sdapi/v1/txt2img' | '/sdapi/v1/img2img';
  payload: Record<string, unknown>;
}

/**
 * Scales an image's size so its longest side is `IMAGE_SIZE`, keeping the aspect ratio, with both
 * sides rounded to `SIZE_MULTIPLE`. img2img renders at this size, so the result keeps the shape of
 * the input instead of being squeezed into the txt2img square.
 */
export function fitStableDiffusionSize(
  width: number,
  height: number,
): { width: number; height: number } {
  const scale = IMAGE_SIZE / Math.max(width, height);
  const fit = (side: number) =>
    Math.max(SIZE_MULTIPLE, Math.round((side * scale) / SIZE_MULTIPLE) * SIZE_MULTIPLE);
  return { width: fit(width), height: fit(height) };
}

/**
 * Builds a Stable Diffusion WebUI API request (`txt2img`, or `img2img` when an image is given).
 * The same API is served by AUTOMATIC1111, Forge and stable-diffusion.cpp's `sd-server`; for the
 * latter, steps, CFG scale and the txt2img size are left to the loaded model's defaults.
 */
export function buildStableDiffusionRequest({
  prompt,
  negative_prompt,
  image,
  denoising_strength,
  backend = 'webui',
}: StableDiffusionRequestParams): StableDiffusionRequest {
  const payload: Record<string, unknown> = { prompt, negative_prompt };
  if (backend === 'webui') {
    Object.assign(payload, GENERATION_DEFAULTS);
  }
  if (!image) {
    return { path: '/sdapi/v1/txt2img', payload };
  }

  const { width, height } = image.file;
  if (width && height) {
    Object.assign(payload, fitStableDiffusionSize(width, height));
  }
  payload.init_images = [image.buffer.toString('base64')];
  if (denoising_strength != null) {
    payload.denoising_strength = denoising_strength;
  }
  return { path: '/sdapi/v1/img2img', payload };
}
