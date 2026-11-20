/** Generation limits a stable-diffusion.cpp server enforces, from `/sdcpp/v1/capabilities`. */
export interface SdcppLimits {
  minWidth?: number;
  maxWidth?: number;
  minHeight?: number;
  maxHeight?: number;
  /** Most images one request makes; the server silently makes no more. */
  maxBatchCount?: number;
}

/**
 * What a stable-diffusion.cpp server accepts in a generation. The server ignores an unknown
 * sampler or scheduler without an error, so a request is checked against these lists first.
 */
export interface SdcppCapabilities {
  /** Sampler IDs (`euler_a`, `dpm++2m`, ...); empty when the server lists none. */
  samplers: string[];
  /** Scheduler IDs (`karras`, `simple`, ...), matched only in this spelling. */
  schedulers: string[];
  limits: SdcppLimits;
}

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  value != null && typeof value === 'object' && !Array.isArray(value);

const positiveInteger = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isInteger(value) && value > 0 ? value : undefined;

const names = (value: unknown): string[] =>
  Array.isArray(value)
    ? value.filter((name): name is string => typeof name === 'string' && name.length > 0)
    : [];

/** The samplers, schedulers and limits in a `/sdcpp/v1/capabilities` response body. */
export function parseSdcppCapabilities(data: Record<string, unknown>): SdcppCapabilities {
  const limits = isPlainObject(data.limits) ? data.limits : {};
  const minWidth = positiveInteger(limits.min_width);
  const maxWidth = positiveInteger(limits.max_width);
  const minHeight = positiveInteger(limits.min_height);
  const maxHeight = positiveInteger(limits.max_height);
  const maxBatchCount = positiveInteger(limits.max_batch_count);
  return {
    samplers: names(data.samplers),
    schedulers: names(data.schedulers),
    limits: {
      ...(minWidth && { minWidth }),
      ...(maxWidth && { maxWidth }),
      ...(minHeight && { minHeight }),
      ...(maxHeight && { maxHeight }),
      ...(maxBatchCount && { maxBatchCount }),
    },
  };
}
