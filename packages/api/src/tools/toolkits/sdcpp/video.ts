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
  /**
   * Requests made while the generation runs, each ending when the given signal fires: the
   * caller's `signal`, or the end of `budgetMs`.
   */
  connect: (signal: AbortSignal) => SdcppJobClient;
  /**
   * Requests made after the generation stopped, to cancel the job and wait for it to end, each
   * ending when the given signal fires: the end of the time a cancelled job is waited for. They
   * must not end with `signal`, which has already fired.
   */
  cleanup: (signal: AbortSignal) => SdcppJobClient;
  /** Fires when the caller stops the generation, or the server lock is lost. */
  signal: AbortSignal;
  /** How long the job may run, from the moment it is sent. */
  budgetMs: number;
  /** Told where the job is each time that changes; it must not throw. */
  onProgress?: (progress: SdcppJobProgress) => void;
  now?: () => number;
  sleep?: (ms: number, signal?: AbortSignal) => Promise<void>;
}

/**
 * Where a running job is, as sd-server reports it. `stage` is sd-server's name for the step it is
 * on (`conditioning`, `sampling`, `vae_decode`, ...), or `queued` while other jobs run first.
 * `pass` counts the images of a batch while they are sampled, from 1; `completed` of `total`
 * counts what the stage works through (sampling steps).
 */
export interface SdcppJobProgress {
  stage: string;
  pass?: number;
  completed?: number;
  total?: number;
  /** Jobs ahead of this one, while it is queued. */
  queuePosition?: number;
}

/** How often a running job is asked for its state. */
const POLL_MS = 1_000;
/** How often a cancelled job is asked whether it has ended. */
const CANCEL_POLL_MS = 2_000;
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

const count = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isInteger(value) && value >= 0 ? value : undefined;

/** The progress in a running job's state; none when sd-server reports nothing readable. */
export function readSdcppJobProgress(job: Record<string, unknown>): SdcppJobProgress | undefined {
  if (job.status === 'queued') {
    const queuePosition = count(job.queue_position);
    return { stage: 'queued', ...(queuePosition != null && { queuePosition }) };
  }
  const progress = isPlainObject(job.progress) ? job.progress : undefined;
  const stage = typeof progress?.stage === 'string' ? progress.stage : '';
  if (!progress || stage.length === 0) {
    return undefined;
  }
  const pass = positiveInteger(progress.pass);
  const completed = count(progress.completed);
  const total = positiveInteger(progress.total);
  return {
    stage,
    ...(pass != null && { pass }),
    ...(completed != null && { completed }),
    ...(total != null && { total }),
  };
}

/** Whether two reports say the same, so an unchanged state is not reported again. */
export const sameSdcppJobProgress = (
  a: SdcppJobProgress | undefined,
  b: SdcppJobProgress,
): boolean =>
  a != null &&
  a.stage === b.stage &&
  a.pass === b.pass &&
  a.completed === b.completed &&
  a.total === b.total &&
  a.queuePosition === b.queuePosition;

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

/** Whether a cancelled job was seen to end, or still ran when waiting for it stopped. */
export type SdcppCancelOutcome = 'ended' | 'unconfirmed';

/** Whether a job answer says the job is over: a final state, or no such job any more. */
function hasEnded(response: { status: number; data: unknown }): boolean {
  if (response.status === 404 || response.status === 410) {
    return true;
  }
  const status = isPlainObject(response.data) ? response.data.status : undefined;
  return response.status === 200 && typeof status === 'string' && TERMINAL.has(status);
}

/**
 * Cancels a job and waits until sd-server reports it ended, within `CANCEL_SETTLE_MS`. A failed
 * cancel request or state read is retried until then, so a passing network error does not free
 * the server while the job still runs. The time limit also ends a request still waiting for an
 * answer, so the call returns once it has passed. Nothing is thrown: the caller is already
 * ending with another outcome.
 */
async function cancelJob(
  cleanup: (signal: AbortSignal) => SdcppJobClient,
  jobUrl: string,
  now: () => number,
  sleep: (ms: number, signal?: AbortSignal) => Promise<void>,
): Promise<SdcppCancelOutcome> {
  const deadline = now() + CANCEL_SETTLE_MS;
  const waiting = new AbortController();
  const timer = setTimeout(() => waiting.abort(), CANCEL_SETTLE_MS);
  timer.unref?.();
  const client = cleanup(waiting.signal);
  const expired = new Promise<never>((_resolve, reject) =>
    waiting.signal.addEventListener('abort', () => reject(new Error('cancel wait ended')), {
      once: true,
    }),
  );
  expired.catch(() => undefined);
  /** An answer arriving after the wait ended is not waited for, even if its request goes on. */
  const bounded = <T>(request: Promise<T>) => Promise.race([request, expired]);
  let requested = false;
  try {
    while (now() < deadline && !waiting.signal.aborted) {
      try {
        if (!requested) {
          const response = await bounded(client.post(`${jobUrl}/cancel`, {}));
          if (response.status === 404) {
            return 'ended';
          }
          requested = response.status >= 200 && response.status < 300;
        }
        if (requested && hasEnded(await bounded(client.probe(jobUrl)))) {
          return 'ended';
        }
      } catch (error) {
        logger.debug('[StableDiffusion] Cancelling a video job failed; retrying', error);
      }
      await sleep(Math.min(CANCEL_POLL_MS, Math.max(0, deadline - now())), waiting.signal).catch(
        () => undefined,
      );
    }
  } finally {
    clearTimeout(timer);
  }
  logger.warn(
    `[StableDiffusion] A cancelled video job did not end within ${CANCEL_SETTLE_MS} ms; releasing the server`,
  );
  return 'unconfirmed';
}

/** The budget of a video job ran out while one of its requests was waiting for an answer. */
class SdcppVideoBudgetError extends Error {
  constructor() {
    super('video job budget ran out');
    this.name = 'SdcppVideoBudgetError';
  }
}

/**
 * Runs a stable-diffusion.cpp video job (`POST /sdcpp/v1/vid_gen`) and waits for its clip,
 * asking `GET /sdcpp/v1/jobs/{id}` for its state and passing each new stage or step to
 * `onProgress`. The budget starts when the job is sent and also
 * ends a request still waiting for an answer. A job still running when the budget ends, when
 * `signal` fires, or when its state cannot be read is cancelled (`POST .../cancel`), and the
 * call waits for it to end, so the server is free for the next one. A fired `signal` then
 * rejects with its reason. sd-server keeps a finished job only for a while, so its clip is read
 * as soon as it is done.
 */
export async function runSdcppVideoJob({
  url,
  body,
  connect,
  cleanup,
  signal,
  budgetMs,
  onProgress,
  now = Date.now,
  sleep = sleepUnlessAborted,
}: RunSdcppVideoJobParams): Promise<SdcppVideoJobResult> {
  const deadline = now() + budgetMs;
  const budget = new AbortController();
  const timer = setTimeout(() => budget.abort(new SdcppVideoBudgetError()), budgetMs);
  timer.unref?.();
  const jobSignal = AbortSignal.any([signal, budget.signal]);
  const client = connect(jobSignal);
  let jobUrl: string | undefined;
  let reported: SdcppJobProgress | undefined;
  const settle = async () => {
    if (jobUrl) {
      await cancelJob(cleanup, jobUrl, now, sleep);
    }
  };
  try {
    const submitted = await client.post(`${url}/sdcpp/v1/vid_gen`, body);
    if (submitted.status >= 400 && submitted.status < 500) {
      return { status: 'rejected', httpStatus: submitted.status };
    }
    const id = isPlainObject(submitted.data) ? submitted.data.id : undefined;
    if (submitted.status >= 300 || typeof id !== 'string' || id.length === 0) {
      throw new SdcppVideoJobError(`video job request answered HTTP ${submitted.status}`);
    }
    jobUrl = `${url}/sdcpp/v1/jobs/${encodeURIComponent(id)}`;

    for (;;) {
      const job = await readJob(client.probe, jobUrl);
      if (job.status === 'completed') {
        return { status: 'completed', video: parseClip(job.data) };
      }
      if (job.status === 'failed' || job.status === 'cancelled') {
        return { status: 'failed', code: failureCode(job.data, job.status) };
      }
      const progress = readSdcppJobProgress(job.data);
      if (progress && onProgress && !sameSdcppJobProgress(reported, progress)) {
        reported = progress;
        onProgress(progress);
      }
      if (now() >= deadline || budget.signal.aborted) {
        await settle();
        return { status: 'timeout' };
      }
      await sleep(Math.min(POLL_MS, Math.max(0, deadline - now())), jobSignal);
    }
  } catch (error) {
    await settle();
    signal.throwIfAborted();
    if (budget.signal.aborted) {
      return { status: 'timeout' };
    }
    throw error;
  } finally {
    clearTimeout(timer);
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

/** Still images a clip starts from and ends on (PNG, JPEG or still WebP bytes). */
export interface SdcppClipImages {
  start?: Buffer;
  end?: Buffer;
}

/**
 * Seed sent when the calling model gave none. sd-server's native API takes an omitted seed from
 * the recipe (42), so every call would render the same noise; `-1` asks it for a random one, as
 * the WebUI API does for an omitted seed.
 */
export const SDCPP_RANDOM_SEED = -1;

/**
 * The `vid_gen` body for a clip: the prompt, size and seed as top-level fields, the sampling
 * settings nested the way sd-server's native schema takes them (flat `steps` or `cfg_scale` are
 * ignored), and an animated WebP as the output, which browsers play in an `<img>`. Other settings
 * left out get sd-server's own defaults (for steps, CFG scale and the sampler not the recipe's, so
 * the caller fills those in); an omitted seed is random. A start or end image goes in `init_image`
 * or `end_image` as base64; sd-server crops it to the clip's aspect ratio and scales it to the
 * clip's size.
 */
export function buildSdcppVideoBody(
  settings: SdcppVideoSettings,
  clip: { frames: number; fps?: number },
  images: SdcppClipImages = {},
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
    seed: seed ?? SDCPP_RANDOM_SEED,
    output_format: 'webp',
    ...(Object.keys(sampleParams).length > 0 && { sample_params: sampleParams }),
    ...(images.start && { init_image: images.start.toString('base64') }),
    ...(images.end && { end_image: images.end.toString('base64') }),
  };
}
