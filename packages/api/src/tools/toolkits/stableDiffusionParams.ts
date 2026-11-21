import type { TStableDiffusionConfig } from 'librechat-data-provider';
import type { SdcppCapabilities } from './sdcpp';

/** What one generation may ask for when `stableDiffusion.limits` in `librechat.yaml` is unset. */
export const STABLE_DIFFUSION_LIMITS = { maxImages: 4, maxSteps: 50, maxSize: 2048 } as const;

/** Highest CFG scale accepted; WebUI servers offer the same range. */
const MAX_CFG_SCALE = 30;
/** Highest seed accepted; every server takes an unsigned 32-bit seed. */
const MAX_SEED = 2 ** 32 - 1;
/** Smallest side of a generated image when the server states no minimum. */
const MIN_SIZE = 64;
/** Longest side of a generated image when the model states no size. */
export const IMAGE_SIZE = 1024;
/** Sizes round to this multiple when the model's is unknown; every common model accepts it. */
export const SIZE_MULTIPLE = 64;

/**
 * What other WebUI servers (AUTOMATIC1111, Forge) get unless the calling model sets it: their own
 * defaults are SD 1.5-era values. stable-diffusion.cpp gets none, since it fills omitted settings
 * from the loaded recipe.
 */
export const WEBUI_GENERATION_DEFAULTS: {
  cfg_scale: number;
  steps: number;
  width: number;
  height: number;
} = {
  cfg_scale: 4.5,
  steps: 22,
  width: IMAGE_SIZE,
  height: IMAGE_SIZE,
};

/** Generation settings as the calling model passes them to the tool. */
export interface StableDiffusionParamsInput {
  width?: number;
  height?: number;
  steps?: number;
  cfg_scale?: number;
  /** `-1` asks for a random seed, as an omitted one does. */
  seed?: number;
  sampler?: string;
  scheduler?: string;
  /** How many images to make in one generation. */
  count?: number;
}

/** Checked settings, named as the WebUI-compatible API takes them. */
export interface StableDiffusionParams {
  /** Requested size, before it is rounded to the model's size multiple. */
  width?: number;
  height?: number;
  steps?: number;
  cfg_scale?: number;
  seed?: number;
  sampler_name?: string;
  scheduler?: string;
  batch_size?: number;
}

/** The sizes and steps the server and `librechat.yaml` allow. */
export interface StableDiffusionBounds {
  width: { min: number; max: number };
  height: { min: number; max: number };
  maxSteps: number;
}

export type StableDiffusionParamsCheck =
  | { ok: true; value: StableDiffusionParams; bounds: StableDiffusionBounds }
  | { ok: false; error: { code: StableDiffusionParamsErrorCode; message: string } };

export type StableDiffusionParamsErrorCode =
  | 'invalid_size'
  | 'invalid_steps'
  | 'invalid_cfg_scale'
  | 'invalid_seed'
  | 'invalid_count'
  | 'unknown_sampler'
  | 'unknown_scheduler'
  | 'size_out_of_range'
  | 'defaults_unknown';

export interface StableDiffusionParamsContext {
  limits?: TStableDiffusionConfig['limits'];
  /** A stable-diffusion.cpp server's samplers, schedulers and limits; absent for other servers. */
  capabilities?: SdcppCapabilities;
}

const isWholeNumber = (value: number, min: number, max: number) =>
  Number.isInteger(value) && value >= min && value <= max;

const failure = (
  code: StableDiffusionParamsErrorCode,
  message: string,
): { ok: false; error: { code: StableDiffusionParamsErrorCode; message: string } } => ({
  ok: false,
  error: { code, message },
});

/** The listed name matching `name` regardless of case, or `undefined` when none does. */
function matchName(name: string, listed: string[]): string | undefined {
  const wanted = name.trim().toLowerCase();
  return listed.find((candidate) => candidate.toLowerCase() === wanted);
}

/**
 * Checks the calling model's generation settings against `librechat.yaml`'s limits and, on a
 * stable-diffusion.cpp server, against the server's own limits and sampler and scheduler lists.
 * The server ignores an unknown sampler or scheduler, quietly changes a step count of 0 and makes
 * no more images than its batch limit, so those are refused here with a message that says what
 * is accepted. A sampler or scheduler named in another case is given in the server's spelling,
 * since sd.cpp matches schedulers exactly. Settings left out are checked once the model's
 * defaults are known (`resolveStableDiffusionSettings`).
 */
export function checkStableDiffusionParams(
  input: StableDiffusionParamsInput,
  { limits, capabilities }: StableDiffusionParamsContext = {},
): StableDiffusionParamsCheck {
  const server = capabilities?.limits ?? {};
  const maxSize = limits?.maxSize ?? STABLE_DIFFUSION_LIMITS.maxSize;
  const maxSteps = limits?.maxSteps ?? STABLE_DIFFUSION_LIMITS.maxSteps;
  const maxImagesConfigured = limits?.maxImages ?? STABLE_DIFFUSION_LIMITS.maxImages;
  const maxImages = Math.min(maxImagesConfigured, server.maxBatchCount ?? maxImagesConfigured);
  const max = {
    width: Math.min(maxSize, server.maxWidth ?? maxSize),
    height: Math.min(maxSize, server.maxHeight ?? maxSize),
  };
  const min = { width: server.minWidth ?? MIN_SIZE, height: server.minHeight ?? MIN_SIZE };
  const bounds: StableDiffusionBounds = {
    width: { min: min.width, max: max.width },
    height: { min: min.height, max: max.height },
    maxSteps,
  };
  const value: StableDiffusionParams = {};

  for (const side of ['width', 'height'] as const) {
    const size = input[side];
    if (size == null) {
      continue;
    }
    if (!isWholeNumber(size, min[side], max[side])) {
      return failure(
        'invalid_size',
        `"${side}" must be a whole number of pixels from ${min[side]} to ${max[side]}.`,
      );
    }
    value[side] = size;
  }

  if (input.steps != null) {
    if (!isWholeNumber(input.steps, 1, maxSteps)) {
      return failure('invalid_steps', `"steps" must be a whole number from 1 to ${maxSteps}.`);
    }
    value.steps = input.steps;
  }

  if (input.cfg_scale != null) {
    const { cfg_scale } = input;
    if (!Number.isFinite(cfg_scale) || cfg_scale < 0 || cfg_scale > MAX_CFG_SCALE) {
      return failure('invalid_cfg_scale', `"cfg_scale" must be from 0 to ${MAX_CFG_SCALE}.`);
    }
    value.cfg_scale = cfg_scale;
  }

  if (input.seed != null && input.seed !== -1) {
    if (!isWholeNumber(input.seed, 0, MAX_SEED)) {
      return failure(
        'invalid_seed',
        `"seed" must be a whole number from 0 to ${MAX_SEED}, or -1 for a random seed.`,
      );
    }
    value.seed = input.seed;
  }

  if (input.count != null) {
    if (!isWholeNumber(input.count, 1, maxImages)) {
      return failure('invalid_count', `"count" must be a whole number from 1 to ${maxImages}.`);
    }
    value.batch_size = input.count;
  }

  const lists = [
    ['sampler', 'sampler_name', capabilities?.samplers, 'unknown_sampler'],
    ['scheduler', 'scheduler', capabilities?.schedulers, 'unknown_scheduler'],
  ] as const;
  for (const [field, key, listed, code] of lists) {
    const name = input[field]?.trim();
    if (!name) {
      continue;
    }
    if (!listed?.length) {
      value[key] = name;
      continue;
    }
    const match = matchName(name, listed);
    if (!match) {
      return failure(
        code,
        `Unknown ${field} "${name}". Omit "${field}" for the model's default, or use one of: ${listed.join(', ')}.`,
      );
    }
    value[key] = match;
  }

  return { ok: true, value, bounds };
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

/** What a model renders with when a request leaves a setting out. */
export interface StableDiffusionDefaults {
  width?: number;
  height?: number;
  steps?: number;
}

/** What `resolveStableDiffusionSettings` needs besides the model's defaults and size target. */
export interface StableDiffusionSettingsRequest {
  /** A server not identified yet (`unknown`) is sent the WebUI settings. */
  backend: 'sdcpp' | 'webui' | 'unknown';
  /** The img2img input's size, whose aspect ratio is kept. */
  image?: { width?: number | null; height?: number | null };
  /** Settings the calling model asked for, already checked against `bounds`. */
  width?: number;
  height?: number;
  steps?: number;
  bounds: StableDiffusionBounds;
}

export interface StableDiffusionSettingsParams extends StableDiffusionSettingsRequest {
  /**
   * The model's defaults on stable-diffusion.cpp; fields left out are unknown. WebUI servers
   * always use `WEBUI_GENERATION_DEFAULTS`.
   */
  defaults?: StableDiffusionDefaults;
  /** The model's size target; without one, sizes round to 64 and img2img fits into 1024. */
  target?: StableDiffusionSizeTarget;
}

/** The size and steps a generation is sent with; a field left out keeps the model's default. */
export interface StableDiffusionSettings {
  width?: number;
  height?: number;
  steps?: number;
}

export type StableDiffusionSettingsResult =
  | { ok: true; value: StableDiffusionSettings }
  | { ok: false; error: { code: StableDiffusionParamsErrorCode; message: string } };

/** Shown to the model when a default that may exceed a limit cannot be read. */
export const STABLE_DIFFUSION_DEFAULTS_UNKNOWN =
  'The image model\'s default size and steps cannot be read right now, so they cannot be checked against the limits. Pass "width", "height" and "steps", or try again later.';

const sideNames = ['width', 'height'] as const;
type Side = (typeof sideNames)[number];
type Size = Record<Side, number>;

/** The scale that brings `size` within `range` on both sides, nearest to 1; none when no scale fits. */
function scaleWithin(size: Size, range: Record<Side, { lo: number; hi: number }>) {
  const lowest = Math.max(range.width.lo / size.width, range.height.lo / size.height);
  const highest = Math.min(range.width.hi / size.width, range.height.hi / size.height);
  return lowest > highest ? undefined : Math.min(highest, Math.max(lowest, 1));
}

/**
 * The size and steps a generation actually renders with, kept within `bounds` also for the
 * settings the calling model left out, so a default above `librechat.yaml`'s limits cannot slip
 * through:
 * - steps left out are the model's default, lowered to `maxSteps` when it is above;
 * - sizes are multiples of the model's size multiple between the smallest and largest allowed,
 *   so the server does not round them out of range; no such multiple is refused;
 * - img2img keeps the input's aspect ratio: fitted to the target's longest side and scaled as a
 *   whole into the allowed range, or derived from the one side asked for, which is refused when
 *   the derived side falls out of range;
 * - txt2img renders the sides asked for; the others are the model's defaults, scaled as a whole
 *   (or the one side clamped) when they are out of range.
 *
 * On stable-diffusion.cpp a setting equal to the recipe's default is left out of the request, so
 * the recipe applies as before; a default that is needed and unknown is refused with
 * `defaults_unknown`. WebUI servers get every setting, from `WEBUI_GENERATION_DEFAULTS`.
 */
export function resolveStableDiffusionSettings(
  params: StableDiffusionSettingsParams,
): StableDiffusionSettingsResult {
  const { backend, image, bounds, target } = params;
  const webui = backend !== 'sdcpp';
  const defaults: StableDiffusionDefaults = webui
    ? WEBUI_GENERATION_DEFAULTS
    : (params.defaults ?? {});
  const unknown = failure('defaults_unknown', STABLE_DIFFUSION_DEFAULTS_UNKNOWN);
  const value: StableDiffusionSettings = {};

  if (params.steps != null) {
    value.steps = params.steps;
  } else if (defaults.steps == null) {
    return unknown;
  } else {
    /** sd-server renders with its general steps, not the recipe's, unless they are sent. */
    value.steps = Math.min(defaults.steps, bounds.maxSteps);
  }

  const multiple = target?.multiple ?? SIZE_MULTIPLE;
  const range = {} as Record<Side, { lo: number; hi: number }>;
  for (const side of sideNames) {
    const lo = Math.ceil(bounds[side].min / multiple) * multiple;
    const hi = Math.floor(bounds[side].max / multiple) * multiple;
    if (lo > hi) {
      return failure(
        'size_out_of_range',
        `No image ${side} fits the limits: the model needs a multiple of ${multiple} pixels from ${bounds[side].min} to ${bounds[side].max}.`,
      );
    }
    range[side] = { lo, hi };
  }
  const round = (side: Side, size: number) =>
    Math.min(range[side].hi, Math.max(range[side].lo, Math.round(size / multiple) * multiple));
  const asked = { width: params.width, height: params.height };
  const outOfRange = (side: Side, size: number) =>
    size < bounds[side].min || size > bounds[side].max;

  const inputWidth = image?.width;
  const inputHeight = image?.height;
  if (inputWidth && inputHeight) {
    const input: Size = { width: inputWidth, height: inputHeight };
    if (asked.width != null && asked.height != null) {
      value.width = round('width', asked.width);
      value.height = round('height', asked.height);
      return { ok: true, value };
    }
    const given = sideNames.find((side) => asked[side] != null);
    if (given) {
      const other: Side = given === 'width' ? 'height' : 'width';
      const size = asked[given] as number;
      const derived = (size * input[other]) / input[given];
      if (outOfRange(other, derived)) {
        const ratio = input[given] / input[other];
        const from = Math.ceil(Math.max(bounds[given].min, bounds[other].min * ratio));
        const to = Math.floor(Math.min(bounds[given].max, bounds[other].max * ratio));
        const advice =
          from <= to
            ? `"${given}" from ${from} to ${to} keeps it in range`
            : 'give both "width" and "height"';
        return failure(
          'size_out_of_range',
          `With this image's aspect ratio, "${given}" ${size} makes the ${other} ${Math.round(derived)}, outside ${bounds[other].min} to ${bounds[other].max}; ${advice}.`,
        );
      }
      value[given] = round(given, size);
      value[other] = round(other, derived);
      return { ok: true, value };
    }
    const fitted = fitStableDiffusionSize(inputWidth, inputHeight, {
      longestSide: target?.longestSide ?? IMAGE_SIZE,
      multiple: 1,
    });
    const scale = scaleWithin(fitted, range);
    if (scale == null) {
      return failure(
        'size_out_of_range',
        `This image's aspect ratio cannot fit the size limits (${bounds.width.min} to ${bounds.width.max} wide, ${bounds.height.min} to ${bounds.height.max} high). Give both "width" and "height".`,
      );
    }
    return {
      ok: true,
      value: {
        ...value,
        width: round('width', fitted.width * scale),
        height: round('height', fitted.height * scale),
      },
    };
  }

  /** txt2img, or img2img whose input size is unknown: the server would use the input's size. */
  const sendAll = webui || image != null;
  const size = {} as Partial<Size>;
  for (const side of sideNames) {
    const sideSize = asked[side] ?? defaults[side];
    if (sideSize == null) {
      return unknown;
    }
    size[side] = sideSize;
  }
  const resolved = size as Size;
  if (asked.width == null && asked.height == null) {
    const fits = sideNames.every((side) => !outOfRange(side, resolved[side]));
    if (fits && !sendAll) {
      return { ok: true, value };
    }
    const scale = fits ? 1 : scaleWithin(resolved, range);
    if (scale == null) {
      return failure(
        'size_out_of_range',
        `The model's default size cannot fit the size limits. Give both "width" and "height".`,
      );
    }
    return {
      ok: true,
      value: {
        ...value,
        width: round('width', resolved.width * scale),
        height: round('height', resolved.height * scale),
      },
    };
  }
  for (const side of sideNames) {
    if (asked[side] != null || sendAll || outOfRange(side, resolved[side])) {
      value[side] = round(side, resolved[side]);
    }
  }
  return { ok: true, value };
}
