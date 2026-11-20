import axios from 'axios';
import type { TFile } from 'librechat-data-provider';
import type { AxiosRequestConfig } from 'axios';
import type { GeneratedImageRegistry, ImageInput, LoadImageInputsParams } from './images';
import type { ServerRequest } from '~/types';
import { configuredRequestBudget } from '~/utils/timeouts';
import { applySSRFSafeAgentIfDirect } from '~/auth/agent';
import { loadImageInputs } from './images';

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
 * `/sdcpp/v1/capabilities`. A definite answer (the contract, or any other non-5xx response) is
 * cached per URL. When the probe fails (no response, or 5xx), the last answer for the URL is
 * kept even if it expired, so a brief outage of the probe route does not change a known
 * sd-server's generation parameters; the next call asks again. A URL never answered before falls
 * back to `webui`: its fixed parameters work on every server, while omitting them on a WebUI
 * server would render with its SD 1.5-era defaults.
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

  let response: Awaited<ReturnType<StableDiffusionProbe>> | undefined;
  try {
    response = await probe(`${url}/sdcpp/v1/capabilities`);
  } catch {
    response = undefined;
  }
  if (!response || response.status >= 500) {
    return cached?.backend ?? 'webui';
  }

  const backend: StableDiffusionBackend =
    response.status === 200 && isStableDiffusionCppCapabilities(response.data) ? 'sdcpp' : 'webui';
  if (!cached && backendCache.size >= BACKEND_CACHE_MAX_ENTRIES) {
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

/** Shown to the model when `image_ids` names no image it may use. */
export const STABLE_DIFFUSION_IMAGE_NOT_LOADED =
  'The image to change could not be loaded. Use the ID of an image attached to this request or generated earlier.';

export interface StableDiffusionToolInput {
  prompt: string;
  negative_prompt?: string;
  image_ids?: string[];
  denoising_strength?: number;
}

/** What the `stable-diffusion` tool is created with, from the server's wiring. */
export interface StableDiffusionToolContext {
  /** Base URL of the WebUI-compatible API (`SD_WEBUI_URL`). */
  url: string;
  req?: ServerRequest;
  /** A URL the user entered is reached only through the SSRF-guarded agent. */
  isUserProvidedEndpoint?: boolean;
  /** Images attached to the request (the `image_edit` tool resource). */
  imageFiles?: TFile[];
  fileStrategy?: string;
  /** Owner-scoped image lookup and storage access; without them img2img is unavailable. */
  findImageFiles?: LoadImageInputsParams['findImageFiles'];
  getStrategyFunctions?: LoadImageInputsParams['getStrategyFunctions'];
  generatedImages?: GeneratedImageRegistry;
}

export type PreparedStableDiffusionRequest =
  | { error: string }
  | { url: string; payload: Record<string, unknown>; config: AxiosRequestConfig };

/** Axios config for a request to `requestUrl`, SSRF-guarded when the user set the URL. */
function stableDiffusionRequestConfig(
  context: StableDiffusionToolContext,
  requestUrl: string,
  config: AxiosRequestConfig,
): AxiosRequestConfig {
  return context.isUserProvidedEndpoint ? applySSRFSafeAgentIfDirect(config, requestUrl) : config;
}

async function loadInputImage(
  context: StableDiffusionToolContext,
  imageIds: string[],
): Promise<ImageInput | undefined> {
  const { findImageFiles, getStrategyFunctions } = context;
  if (!findImageFiles || !getStrategyFunctions) {
    return undefined;
  }
  const [image] = await loadImageInputs({
    imageIds: imageIds.slice(0, 1),
    requestFiles: context.imageFiles ?? [],
    req: context.req,
    fileStrategy: context.fileStrategy,
    findImageFiles,
    getStrategyFunctions,
    generatedImages: context.generatedImages,
    logPrefix: '[StableDiffusion]',
  });
  return image;
}

/**
 * Turns the tool's input into the generation request: loads the image to change (img2img),
 * detects the backend, and builds the payload and the HTTP config. The probe and the generation
 * get the `toolsMs` and `imageGenerationMs` budgets, and both go through the SSRF guard for a
 * user-provided URL, which throws for a blocked address.
 */
export async function prepareStableDiffusionRequest(
  input: StableDiffusionToolInput,
  context: StableDiffusionToolContext,
): Promise<PreparedStableDiffusionRequest> {
  let image: ImageInput | undefined;
  if (input.image_ids?.length) {
    image = await loadInputImage(context, input.image_ids);
    if (!image) {
      return { error: STABLE_DIFFUSION_IMAGE_NOT_LOADED };
    }
  }

  const appConfig = context.req?.config;
  const backend = await detectStableDiffusionBackend(context.url, (probeUrl) =>
    axios.get(
      probeUrl,
      stableDiffusionRequestConfig(context, probeUrl, {
        ...configuredRequestBudget(appConfig, 'toolsMs'),
        validateStatus: () => true,
      }),
    ),
  );
  const { path, payload } = buildStableDiffusionRequest({
    prompt: input.prompt,
    negative_prompt: input.negative_prompt,
    image,
    denoising_strength: input.denoising_strength,
    backend,
  });
  const url = `${context.url}${path}`;
  const config = {
    ...stableDiffusionRequestConfig(context, url, {}),
    ...configuredRequestBudget(appConfig, 'imageGenerationMs'),
  };
  return { url, payload, config };
}
