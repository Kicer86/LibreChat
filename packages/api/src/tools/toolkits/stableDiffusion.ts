import axios from 'axios';
import { logger } from '@librechat/data-schemas';
import type { AppConfig } from '@librechat/data-schemas';
import type { TFile } from 'librechat-data-provider';
import type { AxiosRequestConfig } from 'axios';
import type { SdcppModelCheck, SdcppModelStatus, SdcppProbe, SdcppModel, SdcppPost } from './sdcpp';
import type { GeneratedImageRegistry, ImageInput, LoadImageInputsParams } from './images';
import type { DistributedLock } from '~/cache/lock';
import type { ServerRequest } from '~/types';
import {
  loadSdcppModel,
  getSdcppCatalog,
  chooseSdcppModel,
  SdcppLockError,
  SdcppModelError,
  allowsSdcppModel,
  hasReadySdcppModel,
  restrictsSdcppModels,
  noteSdcppModelLoaded,
  readSdcppModelStatus,
  formatSdcppModels,
  withSdcppServerLock,
  verifySdcppModel,
} from './sdcpp';
import { configuredRequestBudget, getOutboundTimeoutMs } from '~/utils/timeouts';
import { applySSRFSafeAgentIfDirect } from '~/auth/agent';
import { buildImageToolContext } from './imageContext';
import { createDistributedLock } from '~/cache/lock';
import { loadImageInputs } from './images';

/** Longest side of a generated image (txt2img renders a square of this size). */
const IMAGE_SIZE = 1024;
/** img2img dimensions are rounded to this multiple, which every common model accepts. */
const SIZE_MULTIPLE = 64;

/**
 * Sent to every server. sd-server renders omitted steps and CFG scale with its general defaults
 * (20 steps, CFG 1), not the loaded recipe's, which suits few models.
 */
const GENERATION_DEFAULTS = {
  cfg_scale: 4.5,
  steps: 22,
};

/** The txt2img size for other WebUI servers, which fall back to SD 1.5-era 512² without it. */
const WEBUI_IMAGE_SIZE = {
  width: IMAGE_SIZE,
  height: IMAGE_SIZE,
};

/**
 * The server behind `SD_WEBUI_URL`. stable-diffusion.cpp's `sd-server` renders an omitted txt2img
 * size at the loaded recipe's own size, so it gets none; other WebUI servers (AUTOMATIC1111,
 * Forge) keep getting ours.
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

/** The size img2img renders at: the longest side, and the multiple both sides round to. */
export interface StableDiffusionSizeTarget {
  longestSide: number;
  multiple: number;
}

const DEFAULT_SIZE_TARGET: StableDiffusionSizeTarget = {
  longestSide: IMAGE_SIZE,
  multiple: SIZE_MULTIPLE,
};

/**
 * Scales an image's size so its longest side is `target.longestSide`, keeping the aspect ratio,
 * with both sides rounded to `target.multiple`. img2img renders at this size, so the result keeps
 * the shape of the input instead of being squeezed into the txt2img square.
 */
export function fitStableDiffusionSize(
  width: number,
  height: number,
  { longestSide, multiple }: StableDiffusionSizeTarget = DEFAULT_SIZE_TARGET,
): { width: number; height: number } {
  const scale = longestSide / Math.max(width, height);
  const fit = (side: number) =>
    Math.max(multiple, Math.round((side * scale) / multiple) * multiple);
  return { width: fit(width), height: fit(height) };
}

/**
 * The img2img size for a stable-diffusion.cpp model: its recipe's longest side (1024 when the
 * recipe gives no size) and its size multiple, so the request suits models with other
 * requirements than the WebUI defaults.
 */
export function sdcppSizeTarget(model: SdcppModel): StableDiffusionSizeTarget {
  const recipeSide = Math.max(model.width ?? 0, model.height ?? 0);
  return { longestSide: recipeSide > 0 ? recipeSide : IMAGE_SIZE, multiple: model.sizeMultiple };
}

/**
 * Builds a Stable Diffusion WebUI API request (`txt2img`, or `img2img` when an image is given).
 * The same API is served by AUTOMATIC1111, Forge and stable-diffusion.cpp's `sd-server`; for the
 * latter, the txt2img size is left to the loaded recipe.
 */
export function buildStableDiffusionRequest({
  prompt,
  negative_prompt,
  image,
  denoising_strength,
  backend = 'webui',
}: StableDiffusionRequestParams): StableDiffusionRequest {
  const payload: Record<string, unknown> = { prompt, negative_prompt, ...GENERATION_DEFAULTS };
  if (backend === 'webui') {
    Object.assign(payload, WEBUI_IMAGE_SIZE);
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
  /** stable-diffusion.cpp model (recipe ID) to generate with; omitted, the loaded one is kept. */
  model?: string;
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
  /**
   * Serializes model loads and generations on one stable-diffusion.cpp server. Pass the app's
   * shared lock (Redis-backed with several instances); without one, a lock local to this
   * process is used.
   */
  lock?: DistributedLock;
}

export type PreparedStableDiffusionRequest =
  | { error: string }
  | {
      url: string;
      payload: Record<string, unknown>;
      /**
       * The generation's HTTP config without its time budget, which `stableDiffusionGenerationConfig`
       * starts when the request is sent, so waiting for the server lock or a model load does not
       * use it up.
       */
      config: AxiosRequestConfig;
      backend: StableDiffusionBackend;
      /** The input image, for img2img. */
      image?: ImageInput;
    };

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
 * detects the backend, and builds the payload and the HTTP config. The probe gets the `toolsMs`
 * budget; both it and the generation go through the SSRF guard for a user-provided URL, which
 * throws for a blocked address.
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
  const config = stableDiffusionRequestConfig(context, url, {});
  return { url, payload, config, backend, ...(image && { image }) };
}

/**
 * The prepared generation's config with an `imageGenerationMs` budget starting now, which also
 * ends when `signal` fires.
 */
export function stableDiffusionGenerationConfig(
  config: AxiosRequestConfig,
  appConfig: Pick<AppConfig, 'timeouts'> | null | undefined,
  signal?: AbortSignal,
): AxiosRequestConfig {
  return { ...config, ...configuredRequestBudget(appConfig, 'imageGenerationMs', signal) };
}

/** A Stable Diffusion WebUI API generation response (`images` base64, `info` JSON text). */
export interface StableDiffusionResponse {
  images: string[];
  info?: string;
  [key: string]: unknown;
}

export type StableDiffusionGeneration =
  | { error: string }
  | {
      data: StableDiffusionResponse;
      /** The stable-diffusion.cpp model (recipe ID) that generated the image, when known. */
      model?: string;
    };

/** Shown to the model when it names a model on a server that has no model list. */
export const STABLE_DIFFUSION_MODEL_UNSUPPORTED =
  'This image server does not offer a choice of models. Omit "model".';
/** Shown to the model when sd.cpp's model list cannot be read and models are restricted. */
export const STABLE_DIFFUSION_MODELS_UNAVAILABLE =
  "The image server's model list cannot be read right now, so no allowed model can be chosen. Try again later.";
/** Shown to the model when the chosen model could not be loaded or kept loaded. */
export const stableDiffusionModelFailed = (model: string) =>
  `The image model "${model}" could not be loaded. Try again later, or choose another model.`;
/** Shown to the model when other requests kept the image server busy for the whole budget. */
export const STABLE_DIFFUSION_SERVER_BUSY =
  'The image server is busy with other images. Try again later.';
/** Shown to the model when the generation stopped because its hold on the server was lost. */
export const STABLE_DIFFUSION_LOCK_LOST =
  'The image generation was stopped because another request may be using the image server. Try again.';

/**
 * A generation named by `model` only when its files confirm it. A result that names no files is
 * kept, since the lock makes another model unlikely, but it is not attributed to a model.
 */
function withCheckedModel(
  data: StableDiffusionResponse,
  model: string,
  check: SdcppModelCheck,
): StableDiffusionGeneration {
  if (check.result === 'match') {
    return { data, model };
  }
  logger.warn(
    `[StableDiffusion] Could not tell which model generated the image (${model} expected)`,
  );
  return { data };
}

/** Used when the caller passes no lock; it coordinates only this process's generations. */
const processLock = createDistributedLock();

/** sd.cpp model requests for `context`'s server; each one also ends when `signal` fires. */
function sdcppClient(
  context: StableDiffusionToolContext,
  signal?: AbortSignal,
): { probe: SdcppProbe; post: SdcppPost } {
  const appConfig = context.req?.config;
  return {
    probe: (requestUrl) =>
      axios.get(
        requestUrl,
        stableDiffusionRequestConfig(context, requestUrl, {
          ...configuredRequestBudget(appConfig, 'toolsMs', signal),
          validateStatus: () => true,
        }),
      ),
    post: (requestUrl, body) =>
      axios.post(
        requestUrl,
        body,
        stableDiffusionRequestConfig(context, requestUrl, {
          ...configuredRequestBudget(appConfig, 'imageGenerationMs', signal),
          validateStatus: () => true,
        }),
      ),
  };
}

/**
 * Generates an image for the tool's input. On a stable-diffusion.cpp server that lists its
 * models, the model is chosen (`chooseSdcppModel`), loaded when it is not loaded yet, and checked
 * against the files the result names; a model switched by another client during the generation
 * is retried once. Every sd.cpp generation holds the server lock, also when the model list
 * cannot be read; `librechat.yaml`'s model limits then still apply. Other servers get the
 * request as it is.
 */
export async function generateStableDiffusionImage(
  input: StableDiffusionToolInput,
  context: StableDiffusionToolContext,
): Promise<StableDiffusionGeneration> {
  const request = await prepareStableDiffusionRequest(input, context);
  if ('error' in request) {
    return request;
  }
  const appConfig = context.req?.config;
  /** Each generation, a retry included, gets its own budget from the moment it is sent. */
  const generate = async (payload = request.payload, signal?: AbortSignal) =>
    (
      await axios.post<StableDiffusionResponse>(
        request.url,
        payload,
        stableDiffusionGenerationConfig(request.config, appConfig, signal),
      )
    ).data;
  /** img2img sized for the model that renders it. */
  const payloadFor = (model: SdcppModel) => {
    const file = request.image?.file;
    if (!file?.width || !file.height) {
      return request.payload;
    }
    const size = fitStableDiffusionSize(file.width, file.height, sdcppSizeTarget(model));
    return { ...request.payload, ...size };
  };

  if (request.backend !== 'sdcpp') {
    if (input.model) {
      return { error: STABLE_DIFFUSION_MODEL_UNSUPPORTED };
    }
    return { data: await generate() };
  }

  const modelsConfig = appConfig?.stableDiffusion?.models;
  const lookup = await getSdcppCatalog(context.url, sdcppClient(context).probe);
  const lockParams = {
    lock: context.lock ?? processLock,
    url: context.url,
    waitMs: getOutboundTimeoutMs(appConfig, 'imageGenerationMs'),
  };
  /**
   * Without a catalog no model can be chosen or loaded. The loaded one is used as it is, but when
   * `librechat.yaml` restricts models only after the server confirms it is a ready, allowed one,
   * and only if the result was made by its files.
   */
  const generateUnlisted = async (
    status: 'unsupported' | 'unavailable',
    signal: AbortSignal,
  ): Promise<StableDiffusionGeneration> => {
    const unavailable = { error: STABLE_DIFFUSION_MODELS_UNAVAILABLE };
    if (input.model) {
      return status === 'unsupported' ? { error: STABLE_DIFFUSION_MODEL_UNSUPPORTED } : unavailable;
    }
    if (!restrictsSdcppModels(modelsConfig)) {
      return { data: await generate(request.payload, signal) };
    }
    let loaded: SdcppModelStatus;
    try {
      loaded = await readSdcppModelStatus(context.url, sdcppClient(context, signal).probe);
    } catch (error) {
      if (!(error instanceof SdcppModelError)) {
        throw error;
      }
      return unavailable;
    }
    const { recipe } = loaded;
    if (!recipe || !hasReadySdcppModel(loaded) || !allowsSdcppModel(recipe, modelsConfig)) {
      return unavailable;
    }
    signal.throwIfAborted();
    const data = await generate(request.payload, signal);
    const check = verifySdcppModel(loaded, data.info);
    if (check.result === 'mismatch') {
      logger.warn(
        `[StableDiffusion] The model changed while ${recipe} generated (${check.detail})`,
      );
      return { error: stableDiffusionModelFailed(recipe) };
    }
    return withCheckedModel(data, recipe, check);
  };

  type Attempt = StableDiffusionGeneration | { switchedAway: string };
  /** One choose, load and generate pass; every request ends when the server lock is lost. */
  const attempt = async (
    models: SdcppModel[],
    requested: string | undefined,
    signal: AbortSignal,
  ): Promise<Attempt> => {
    const locked = sdcppClient(context, signal);
    let status: SdcppModelStatus = await readSdcppModelStatus(context.url, locked.probe);
    noteSdcppModelLoaded(context.url, status.recipe);
    const choice = chooseSdcppModel({
      catalog: { models, ...(status.recipe && { loaded: status.recipe }) },
      config: modelsConfig,
      requested,
      withImage: request.image != null,
      readyOutsideCatalog: !status.recipe && hasReadySdcppModel(status),
    });
    if (!choice.ok) {
      return { error: choice.error.message };
    }
    signal.throwIfAborted();
    if (!choice.model) {
      const data = await generate(request.payload, signal);
      return { data, ...(status.recipe && { model: status.recipe }) };
    }

    const model = choice.model.id;
    try {
      status = await loadSdcppModel(context.url, model, status, locked.post);
    } catch (error) {
      if (!(error instanceof SdcppModelError)) {
        throw error;
      }
      logger.warn(`[StableDiffusion] Could not load ${model}: ${error.message}`);
      return { error: stableDiffusionModelFailed(model) };
    }
    noteSdcppModelLoaded(context.url, status.recipe);

    signal.throwIfAborted();
    const data = await generate(payloadFor(choice.model), signal);
    const check = verifySdcppModel(status, data.info);
    if (check.result === 'mismatch') {
      logger.warn(`[StableDiffusion] The model changed while ${model} generated (${check.detail})`);
      return { switchedAway: model };
    }
    return withCheckedModel(data, model, check);
  };

  try {
    return await withSdcppServerLock(lockParams, async (signal) => {
      if (lookup.status !== 'listed') {
        return generateUnlisted(lookup.status, signal);
      }
      const { models } = lookup.catalog;
      const first = await attempt(models, input.model, signal);
      if (!('switchedAway' in first)) {
        return first;
      }
      signal.throwIfAborted();
      const second = await attempt(models, first.switchedAway, signal);
      return 'switchedAway' in second
        ? { error: stableDiffusionModelFailed(second.switchedAway) }
        : second;
    });
  } catch (error) {
    if (!(error instanceof SdcppLockError)) {
      throw error;
    }
    return {
      error: error.code === 'busy' ? STABLE_DIFFUSION_SERVER_BUSY : STABLE_DIFFUSION_LOCK_LOST,
    };
  }
}

export interface StableDiffusionContextParams {
  /** Base URL of the WebUI-compatible API (`SD_WEBUI_URL`); without it no models are listed. */
  url?: string;
  /** A URL the user entered is reached only through the SSRF-guarded agent. */
  isUserProvidedEndpoint?: boolean;
  /** Images attached to the request (the `image_edit` tool resource). */
  imageFiles?: Array<{ file_id: string }>;
  appConfig?: AppConfig;
}

/**
 * Tool context for `stable-diffusion` at the start of a turn: the image models it may use, when
 * the server is stable-diffusion.cpp, and the IDs of the attached images. The model list is read
 * within the `imageModelsMs` budget and cached; a server that cannot be reached lists none.
 */
export async function buildStableDiffusionToolContext({
  url,
  isUserProvidedEndpoint,
  imageFiles = [],
  appConfig,
}: StableDiffusionContextParams): Promise<string> {
  const images = buildImageToolContext({
    imageFiles,
    toolName: 'stable-diffusion',
    contextDescription: 'image-to-image (img2img)',
  });
  if (!url) {
    return images;
  }
  const context: StableDiffusionToolContext = { url, isUserProvidedEndpoint };
  const catalog = await getSdcppCatalog(url, (probeUrl) =>
    axios.get(
      probeUrl,
      stableDiffusionRequestConfig(context, probeUrl, {
        ...configuredRequestBudget(appConfig, 'imageModelsMs'),
        validateStatus: () => true,
      }),
    ),
  );
  const models =
    catalog.status === 'listed'
      ? formatSdcppModels(catalog.catalog, appConfig?.stableDiffusion?.models)
      : '';
  return [models, images].filter(Boolean).join('\n\n');
}
