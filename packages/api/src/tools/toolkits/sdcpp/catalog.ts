/** Fetches a URL, resolving with any HTTP status; rejects only when no response arrived. */
export type SdcppProbe = (url: string) => Promise<{ status: number; data: unknown }>;

/** What a model produces: still images, or video clips. */
export type SdcppModelMode = 'image' | 'video';

/** A model the server can load: one of its recipes that names model files. */
export interface SdcppModel {
  /** Recipe ID, the name `POST /sdcpp/v1/models/load` takes. */
  id: string;
  name: string;
  description?: string;
  mode: SdcppModelMode;
  /** Whether the model can start from an input image (img2img, or image-to-video). */
  acceptsImage: boolean;
  /** Generated widths and heights must be multiples of this. */
  sizeMultiple: number;
  /** The recipe's own output size, which the server uses when a request gives none. */
  width?: number;
  height?: number;
  /**
   * The recipe's sampling steps, CFG scale, sampler and scheduler. sd-server does not apply them
   * itself (it renders with its general defaults), so they are sent when a request gives none.
   */
  steps?: number;
  cfgScale?: number;
  sampler?: string;
  scheduler?: string;
  /** Video models only: the clip length and frame rate the recipe renders by default. */
  frames?: number;
  fps?: number;
  /** Video models only: which frame counts the model can render, and its fixed frame rate. */
  frameRule?: SdcppFrameRule;
}

/**
 * The frame counts a video model renders: `offset + step·n` frames (LTX takes 8n+1, Wan 4n+1),
 * at least `min` and at most `max`. sd-server rounds another count to the rule (`rounding`).
 * Some models play at one frame rate only (`fixedFps`).
 */
export interface SdcppFrameRule {
  offset: number;
  step: number;
  rounding: 'floor' | 'ceil';
  min: number;
  max?: number;
  fixedFps?: number;
}

export interface SdcppCatalog {
  models: SdcppModel[];
  /** Recipe loaded on the server when the catalog was read; absent when none is loaded. */
  loaded?: string;
}

/**
 * What a server says about its models: a catalog (`listed`); no catalog, because it answered
 * without one (`unsupported`: not sd.cpp, or a build without recipes); or nothing yet, because
 * it has not answered and no earlier catalog is known (`unavailable`).
 */
export type SdcppCatalogLookup =
  | { status: 'listed'; catalog: SdcppCatalog }
  | { status: 'unsupported' }
  | { status: 'unavailable' };

/** What `/sdcpp/v1/models/status` reports about the loaded model. */
export interface SdcppModelStatus {
  recipe?: string;
  state?: string;
  error?: string;
  /** Model files by slot (`diffusion_model`, `vae`, ...), as paths relative to the model root. */
  models: Record<string, string>;
}

/** Size multiple for recipes that do not state one; every common model accepts it. */
const DEFAULT_SIZE_MULTIPLE = 64;
const MAX_DESCRIPTION_LENGTH = 300;

/** How long a catalog is reused before the server is asked again. */
const CATALOG_CACHE_TTL_MS = 5 * 60 * 1000;
const CATALOG_CACHE_MAX_ENTRIES = 100;

/** `null` records a server that answered but has no catalog (not sd.cpp). */
const catalogCache = new Map<string, { catalog: SdcppCatalog | null; expiresAt: number }>();

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  value != null && typeof value === 'object' && !Array.isArray(value);

const positiveInteger = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isInteger(value) && value > 0 ? value : undefined;

const wholeNumber = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isInteger(value) && value >= 0 ? value : undefined;

const finiteNumber = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined;

const nonEmptyString = (value: unknown): string | undefined =>
  typeof value === 'string' && value.trim().length > 0 ? value.trim() : undefined;

/** Any frame count, from one frame up, for a video recipe that states no constraints. */
const ANY_FRAME_COUNT: SdcppFrameRule = { offset: 0, step: 1, rounding: 'floor', min: 1 };

/** A recipe's `video_constraints`; a malformed one is treated as stating none. */
function parseFrameRule(constraints: unknown): SdcppFrameRule {
  if (!isPlainObject(constraints)) {
    return ANY_FRAME_COUNT;
  }
  const offset = wholeNumber(constraints.frame_offset) ?? 0;
  const step = positiveInteger(constraints.frame_step) ?? 1;
  const min = positiveInteger(constraints.min_frames) ?? 1;
  const max = positiveInteger(constraints.max_frames);
  const fixedFps = positiveInteger(constraints.fixed_fps);
  return {
    offset,
    step,
    rounding: constraints.frame_rounding === 'ceil' ? 'ceil' : 'floor',
    min,
    ...(max != null && max >= min && { max }),
    ...(fixedFps && { fixedFps }),
  };
}

function parseRecipe(recipe: unknown): SdcppModel | undefined {
  if (!isPlainObject(recipe) || !isPlainObject(recipe.models)) {
    return undefined;
  }
  const id = nonEmptyString(recipe.id);
  if (!id) {
    return undefined;
  }
  const defaults = isPlainObject(recipe.defaults) ? recipe.defaults : {};
  const features = isPlainObject(recipe.effective_features) ? recipe.effective_features : {};
  const mode: SdcppModelMode =
    defaults.video_frames != null || isPlainObject(recipe.video_constraints) ? 'video' : 'image';
  const description = nonEmptyString(recipe.description);
  const steps = positiveInteger(defaults.steps);
  const cfgScale = finiteNumber(defaults.cfg_scale);
  const sampler = nonEmptyString(defaults.sampling_method);
  const scheduler = nonEmptyString(defaults.scheduler);
  const frames = positiveInteger(defaults.video_frames);
  const fps = positiveInteger(defaults.fps);
  return {
    id,
    name: nonEmptyString(recipe.name) ?? id,
    ...(description && { description: description.slice(0, MAX_DESCRIPTION_LENGTH) }),
    mode,
    acceptsImage: (mode === 'image' ? features.img2img : features.init_image) === true,
    sizeMultiple: positiveInteger(recipe.size_multiple) ?? DEFAULT_SIZE_MULTIPLE,
    width: positiveInteger(defaults.width),
    height: positiveInteger(defaults.height),
    ...(steps && { steps }),
    ...(cfgScale != null && { cfgScale }),
    ...(sampler && { sampler }),
    ...(scheduler && { scheduler }),
    ...(mode === 'video' && {
      ...(frames && { frames }),
      ...(fps && { fps }),
      frameRule: parseFrameRule(recipe.video_constraints),
    }),
  };
}

/**
 * The loadable models in a `/sdcpp/v1/recipes` response, or `undefined` when the body is not one.
 * Built-in recipes that name no model files cannot be loaded and are left out.
 */
export function parseSdcppRecipes(data: unknown): SdcppModel[] | undefined {
  if (!isPlainObject(data) || !Array.isArray(data.recipes)) {
    return undefined;
  }
  const models: SdcppModel[] = [];
  for (const recipe of data.recipes) {
    const model = parseRecipe(recipe);
    if (model) {
      models.push(model);
    }
  }
  return models;
}

/** A `/sdcpp/v1/models/status` (or `models/load`) response, or `undefined` when it is not one. */
export function parseSdcppModelStatus(data: unknown): SdcppModelStatus | undefined {
  if (!isPlainObject(data)) {
    return undefined;
  }
  const models: Record<string, string> = {};
  if (isPlainObject(data.models)) {
    for (const [slot, file] of Object.entries(data.models)) {
      if (typeof file === 'string' && file.length > 0) {
        models[slot] = file;
      }
    }
  }
  const error = data.error == null ? undefined : String(data.error);
  return {
    recipe: nonEmptyString(data.recipe),
    state: nonEmptyString(data.state),
    ...(error && { error }),
    models,
  };
}

async function probeQuietly(probe: SdcppProbe, url: string) {
  try {
    return await probe(url);
  } catch {
    return undefined;
  }
}

const lookupOf = (catalog: SdcppCatalog | null): SdcppCatalogLookup =>
  catalog ? { status: 'listed', catalog } : { status: 'unsupported' };

/**
 * The models the stable-diffusion.cpp server at `url` can load, and which one is loaded.
 * The answer is cached per URL; when the server cannot be reached (no response, or 5xx) the last
 * answer for the URL is kept even if it expired, and without one the lookup is `unavailable`.
 */
export async function getSdcppCatalog(
  url: string,
  probe: SdcppProbe,
  now: number = Date.now(),
): Promise<SdcppCatalogLookup> {
  const cached = catalogCache.get(url);
  if (cached && cached.expiresAt > now) {
    return lookupOf(cached.catalog);
  }

  const [recipes, status] = await Promise.all([
    probeQuietly(probe, `${url}/sdcpp/v1/recipes`),
    probeQuietly(probe, `${url}/sdcpp/v1/models/status`),
  ]);
  if (!recipes || recipes.status >= 500) {
    return cached ? lookupOf(cached.catalog) : { status: 'unavailable' };
  }

  const models = recipes.status === 200 ? parseSdcppRecipes(recipes.data) : undefined;
  let catalog: SdcppCatalog | null = null;
  if (models) {
    const loaded =
      status?.status === 200 ? parseSdcppModelStatus(status.data)?.recipe : cached?.catalog?.loaded;
    catalog = { models, ...(loaded && { loaded }) };
  }
  if (!cached && catalogCache.size >= CATALOG_CACHE_MAX_ENTRIES) {
    catalogCache.clear();
  }
  catalogCache.set(url, { catalog, expiresAt: now + CATALOG_CACHE_TTL_MS });
  return lookupOf(catalog);
}

/** Records the model loaded on the server at `url`, so the cached catalog names it. */
export function noteSdcppModelLoaded(url: string, recipe: string | undefined): void {
  const cached = catalogCache.get(url);
  if (!cached?.catalog) {
    return;
  }
  const { loaded: _previous, ...rest } = cached.catalog;
  cached.catalog = recipe ? { ...rest, loaded: recipe } : rest;
}

/** Forgets cached catalogs (for tests). */
export function clearSdcppCatalogCache(): void {
  catalogCache.clear();
}
