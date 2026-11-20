import { logger } from '@librechat/data-schemas';
import type { SdcppProbe } from './catalog';
import type { SdcppPost } from './load';
import { sleepUnlessAborted } from './load';

/** A finished clip: an animated WebP (or AVI), with what sd-server says it rendered. */
export interface SdcppVideo {
  /** The clip, base64-encoded. */
  base64: string;
  mimeType: string;
  fps?: number;
  frames?: number;
  /** `generation_info`: size, frames, seed, sampling settings and the model files used. */
  info: Record<string, unknown>;
}

/**
 * How a video job ended: with a clip (`completed`); refused before it started (`rejected`, an
 * HTTP 4xx answer to the request); failed or cancelled on the server (`failed`, with sd-server's
 * error code); or still running when the budget ran out (`timeout`), in which case it was
 * cancelled.
 */
export type SdcppVideoJobResult =
  | { status: 'completed'; video: SdcppVideo }
  | { status: 'rejected'; httpStatus: number }
  | { status: 'failed'; code: string }
  | { status: 'timeout' };

/** sd-server answered a job request with something other than its job contract. */
export class SdcppVideoJobError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SdcppVideoJobError';
  }
}

export interface SdcppJobClient {
  probe: SdcppProbe;
  post: SdcppPost;
}

export interface RunSdcppVideoJobParams {
  url: string;
  /** The `vid_gen` body, in sd-server's native schema. */
  body: Record<string, unknown>;
  /** Requests made while the generation runs; each ends when `signal` fires. */
  client: SdcppJobClient;
  /**
   * Requests made after the generation stopped, to cancel the job and wait for it to end; they
   * must not end with `signal`, which has already fired.
   */
  cleanup: SdcppJobClient;
  /** Fires when the caller stops the generation, or the server lock is lost. */
  signal: AbortSignal;
  /** How long the job may run, from the moment it is sent. */
  budgetMs: number;
  now?: () => number;
  sleep?: (ms: number, signal?: AbortSignal) => Promise<void>;
}

/** How often a running job is asked for its state. */
const POLL_MS = 2_000;
/**
 * How long a cancelled job is waited for. sd-server stops a job at the end of its current
 * sampling step, which takes tens of seconds for large video models, and cannot load another
 * model until then, so the server lock is kept until the job has ended or this has passed.
 */
const CANCEL_SETTLE_MS = 120_000;

const TERMINAL = new Set(['completed', 'failed', 'cancelled']);

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  value != null && typeof value === 'object' && !Array.isArray(value);

const positiveInteger = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isInteger(value) && value > 0 ? value : undefined;

interface JobState {
  status: string;
  data: Record<string, unknown>;
}

async function readJob(probe: SdcppProbe, jobUrl: string): Promise<JobState> {
  const response = await probe(jobUrl);
  if (response.status !== 200 || !isPlainObject(response.data)) {
    throw new SdcppVideoJobError(`video job answered HTTP ${response.status}`);
  }
  const status = typeof response.data.status === 'string' ? response.data.status : '';
  return { status, data: response.data };
}

function parseClip(job: Record<string, unknown>): SdcppVideo {
  const result = isPlainObject(job.result) ? job.result : undefined;
  if (!result || typeof result.b64_json !== 'string' || result.b64_json.length === 0) {
    throw new SdcppVideoJobError('completed video job has no clip');
  }
  const fps = positiveInteger(result.fps);
  const frames = positiveInteger(result.frame_count);
  return {
    base64: result.b64_json,
    mimeType: typeof result.mime_type === 'string' ? result.mime_type : 'image/webp',
    ...(fps && { fps }),
    ...(frames && { frames }),
    info: isPlainObject(result.generation_info) ? result.generation_info : {},
  };
}

function failureCode(job: Record<string, unknown>, status: string): string {
  const error = isPlainObject(job.error) ? job.error : undefined;
  return typeof error?.code === 'string' && error.code.length > 0 ? error.code : status;
}

/**
 * Cancels a job and waits until sd-server reports it ended, within `CANCEL_SETTLE_MS`. Failures
 * are logged, not thrown: the caller is already ending with another outcome.
 */
async function cancelJob(
  cleanup: SdcppJobClient,
  jobUrl: string,
  now: () => number,
  sleep: (ms: number) => Promise<void>,
): Promise<void> {
  try {
    await cleanup.post(`${jobUrl}/cancel`, {});
    const deadline = now() + CANCEL_SETTLE_MS;
    while (now() < deadline) {
      const { status } = await readJob(cleanup.probe, jobUrl);
      if (TERMINAL.has(status)) {
        return;
      }
      await sleep(POLL_MS);
    }
    logger.warn('[StableDiffusion] A cancelled video job is still running');
  } catch (error) {
    logger.warn('[StableDiffusion] Could not cancel a video job', error);
  }
}

/**
 * Runs a stable-diffusion.cpp video job (`POST /sdcpp/v1/vid_gen`) and waits for its clip,
 * asking `GET /sdcpp/v1/jobs/{id}` for its state. A job still running when the budget ends, when
 * `signal` fires, or when its state cannot be read is cancelled (`POST .../cancel`), and the
 * call waits for it to end, so the server is free for the next one. A fired `signal` then
 * rejects with its reason. sd-server keeps a finished job only for a while, so its clip is read
 * as soon as it is done.
 */
export async function runSdcppVideoJob({
  url,
  body,
  client,
  cleanup,
  signal,
  budgetMs,
  now = Date.now,
  sleep = sleepUnlessAborted,
}: RunSdcppVideoJobParams): Promise<SdcppVideoJobResult> {
  const deadline = now() + budgetMs;
  const submitted = await client.post(`${url}/sdcpp/v1/vid_gen`, body);
  if (submitted.status >= 400 && submitted.status < 500) {
    return { status: 'rejected', httpStatus: submitted.status };
  }
  const id = isPlainObject(submitted.data) ? submitted.data.id : undefined;
  if (submitted.status >= 300 || typeof id !== 'string' || id.length === 0) {
    throw new SdcppVideoJobError(`video job request answered HTTP ${submitted.status}`);
  }

  const jobUrl = `${url}/sdcpp/v1/jobs/${encodeURIComponent(id)}`;
  const settle = () => cancelJob(cleanup, jobUrl, now, (ms) => sleep(ms));
  try {
    for (;;) {
      const job = await readJob(client.probe, jobUrl);
      if (job.status === 'completed') {
        return { status: 'completed', video: parseClip(job.data) };
      }
      if (job.status === 'failed' || job.status === 'cancelled') {
        return { status: 'failed', code: failureCode(job.data, job.status) };
      }
      if (now() >= deadline) {
        await settle();
        return { status: 'timeout' };
      }
      await sleep(Math.min(POLL_MS, Math.max(0, deadline - now())), signal);
    }
  } catch (error) {
    await settle();
    signal.throwIfAborted();
    throw error;
  }
}

/** What `buildSdcppVideoBody` reads from a WebUI-style generation payload. */
export interface SdcppVideoSettings {
  prompt: string;
  negative_prompt?: string;
  width?: number;
  height?: number;
  steps?: number;
  cfg_scale?: number;
  seed?: number;
  sampler_name?: string;
  scheduler?: string;
}

/**
 * The `vid_gen` body for a clip: the prompt, size and seed as top-level fields, the sampling
 * settings nested the way sd-server's native schema takes them (flat `steps` or `cfg_scale` are
 * ignored), and an animated WebP as the output, which browsers play in an `<img>`. Settings left
 * out keep the loaded recipe's defaults.
 */
export function buildSdcppVideoBody(
  settings: SdcppVideoSettings,
  clip: { frames: number; fps?: number },
): Record<string, unknown> {
  const { prompt, negative_prompt, width, height, steps, cfg_scale, seed } = settings;
  const sampleParams: Record<string, unknown> = {
    ...(steps != null && { sample_steps: steps }),
    ...(settings.sampler_name && { sample_method: settings.sampler_name }),
    ...(settings.scheduler && { scheduler: settings.scheduler }),
    ...(cfg_scale != null && { guidance: { txt_cfg: cfg_scale } }),
  };
  return {
    prompt,
    ...(negative_prompt && { negative_prompt }),
    ...(width != null && { width }),
    ...(height != null && { height }),
    video_frames: clip.frames,
    ...(clip.fps != null && { fps: clip.fps }),
    ...(seed != null && { seed }),
    output_format: 'webp',
    ...(Object.keys(sampleParams).length > 0 && { sample_params: sampleParams }),
  };
}
