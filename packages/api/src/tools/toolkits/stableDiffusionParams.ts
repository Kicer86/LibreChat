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
}

/** The largest image the server and `librechat.yaml` allow. */
export interface StableDiffusionMaxSize {
  width: number;
  height: number;
}

export type StableDiffusionParamsCheck =
  | { ok: true; value: StableDiffusionParams; maxSize: StableDiffusionMaxSize }
  | { ok: false; error: { code: StableDiffusionParamsErrorCode; message: string } };

export type StableDiffusionParamsErrorCode =
  | 'invalid_size'
  | 'invalid_steps'
  | 'invalid_cfg_scale'
  | 'invalid_seed'
  | 'unknown_sampler'
  | 'unknown_scheduler';

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
): StableDiffusionParamsCheck => ({ ok: false, error: { code, message } });

/** The listed name matching `name` regardless of case, or `undefined` when none does. */
function matchName(name: string, listed: string[]): string | undefined {
  const wanted = name.trim().toLowerCase();
  return listed.find((candidate) => candidate.toLowerCase() === wanted);
}

/**
 * Checks the calling model's generation settings against `librechat.yaml`'s limits and, on a
 * stable-diffusion.cpp server, against the server's own limits and sampler and scheduler lists.
 * The server ignores an unknown sampler or scheduler, and quietly changes a step count of 0, so
 * those are refused here with a message that says what is accepted. A sampler or scheduler named
 * in another case is given in the server's spelling, since sd.cpp matches schedulers exactly.
 */
export function checkStableDiffusionParams(
  input: StableDiffusionParamsInput,
  { limits, capabilities }: StableDiffusionParamsContext = {},
): StableDiffusionParamsCheck {
  const server = capabilities?.limits ?? {};
  const maxSize = limits?.maxSize ?? STABLE_DIFFUSION_LIMITS.maxSize;
  const maxSteps = limits?.maxSteps ?? STABLE_DIFFUSION_LIMITS.maxSteps;
  const max: StableDiffusionMaxSize = {
    width: Math.min(maxSize, server.maxWidth ?? maxSize),
    height: Math.min(maxSize, server.maxHeight ?? maxSize),
  };
  const min = { width: server.minWidth ?? MIN_SIZE, height: server.minHeight ?? MIN_SIZE };
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

  return { ok: true, value, maxSize: max };
}
