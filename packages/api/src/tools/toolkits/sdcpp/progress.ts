import { logger } from '@librechat/data-schemas';
import type { ToolProgress } from 'librechat-data-provider';
import type { SdcppJobProgress } from './video';
import type { SdcppProbe } from './catalog';
import { sameSdcppJobProgress } from './video';
import { sleepUnlessAborted } from './load';

/** sd-server stages before sampling, shown together as `preparing`. */
const PREPARING = new Set(['preparing', 'vae_encode', 'conditioning']);

/** Share of the call each stage starts at; sampling fills the span up to `decoding`. */
const SAMPLING_START = 0.05;
const SAMPLING_END = 0.9;
const DECODING = 0.92;
const ENCODING = 0.97;

/**
 * A job's progress as the tool reports it to the call's card: sd-server's stage as a stable code
 * (`queued`, `preparing`, `sampling`, `decoding`, `encoding`, or `working` for one it does not
 * know), and the share of the whole call done. Sampling, which takes most of the time, fills
 * most of it, image by image for a batch of `items`.
 */
export function toToolProgress(job: SdcppJobProgress, items = 1): ToolProgress {
  if (job.stage === 'queued') {
    return {
      stage: 'queued',
      progress: 0,
      ...(job.queuePosition != null && { queuePosition: job.queuePosition }),
    };
  }
  if (PREPARING.has(job.stage)) {
    return { stage: 'preparing', progress: SAMPLING_START / 2 };
  }
  const item = job.pass ?? 1;
  const count = Math.max(items, item);
  const batch = count > 1 ? { item, items: count } : {};
  if (job.stage === 'sampling' && job.total != null) {
    const completed = Math.min(job.completed ?? 0, job.total);
    const done = ((item - 1) * job.total + completed) / (count * job.total);
    return {
      stage: 'sampling',
      progress: SAMPLING_START + (SAMPLING_END - SAMPLING_START) * done,
      completed,
      total: job.total,
      ...batch,
    };
  }
  if (job.stage === 'vae_decode') {
    return { stage: 'decoding', progress: DECODING, ...batch };
  }
  if (job.stage === 'encoding') {
    return { stage: 'encoding', progress: ENCODING };
  }
  return { stage: job.stage === 'sampling' ? 'sampling' : 'working' };
}

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  value != null && typeof value === 'object' && !Array.isArray(value);

const count = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isInteger(value) && value >= 0 ? value : undefined;

/**
 * The progress in a WebUI `/sdapi/v1/progress` answer, in a job's terms: sampling with the step
 * and the image of a batch (`job_no`, from 0) once sampling has started, `preparing` before.
 * None when the server reports no generation.
 */
export function readWebuiProgress(data: unknown): SdcppJobProgress | undefined {
  const state = isPlainObject(data) && isPlainObject(data.state) ? data.state : undefined;
  if (!state) {
    return undefined;
  }
  const total = count(state.sampling_steps);
  const completed = count(state.sampling_step);
  const jobNo = count(state.job_no);
  if (total != null && total > 0) {
    return {
      stage: 'sampling',
      ...(jobNo != null && { pass: jobNo + 1 }),
      completed: Math.min(completed ?? 0, total),
      total,
    };
  }
  return typeof state.job === 'string' && state.job.length > 0 ? { stage: 'preparing' } : undefined;
}

/** How often a running WebUI generation is asked how far it is. */
const WEBUI_POLL_MS = 1_000;

export interface WatchWebuiProgressParams {
  url: string;
  probe: SdcppProbe;
  /** Told each new stage or step; it must not throw. */
  onProgress: (progress: SdcppJobProgress) => void;
  /** Ends the watch, when the generation it follows has answered. */
  signal: AbortSignal;
  sleep?: (ms: number, signal?: AbortSignal) => Promise<void>;
}

/**
 * Asks `GET /sdapi/v1/progress` how far the running generation is, every second until `signal`
 * fires, and reports each change. The answer describes the server's current generation, not a
 * given request, so it is ours only while the server lock is held. A failed read is skipped;
 * the watch never fails the generation it follows.
 */
export async function watchWebuiProgress({
  url,
  probe,
  onProgress,
  signal,
  sleep = sleepUnlessAborted,
}: WatchWebuiProgressParams): Promise<void> {
  let reported: SdcppJobProgress | undefined;
  while (!signal.aborted) {
    try {
      const response = await probe(`${url}/sdapi/v1/progress?skip_current_image=true`);
      const progress = response.status === 200 ? readWebuiProgress(response.data) : undefined;
      if (progress && !signal.aborted && !sameSdcppJobProgress(reported, progress)) {
        reported = progress;
        onProgress(progress);
      }
      await sleep(WEBUI_POLL_MS, signal);
    } catch (error) {
      if (signal.aborted) {
        return;
      }
      logger.debug('[StableDiffusion] Reading the generation progress failed', error);
      await sleep(WEBUI_POLL_MS, signal).catch(() => undefined);
    }
  }
}
