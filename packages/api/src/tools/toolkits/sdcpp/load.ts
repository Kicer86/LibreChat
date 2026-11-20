import { logger } from '@librechat/data-schemas';
import type { SdcppModelStatus, SdcppProbe } from './catalog';
import type { DistributedLock } from '~/cache/lock';
import { parseSdcppModelStatus } from './catalog';

/** Posts a JSON body, resolving with any HTTP status; rejects only when no response arrived. */
export type SdcppPost = (url: string, body: unknown) => Promise<{ status: number; data: unknown }>;

/** A model load or status read that failed; the message is safe to log, not to show. */
export class SdcppModelError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SdcppModelError';
  }
}

/** The model loaded on the server at `url`. */
export async function readSdcppModelStatus(
  url: string,
  probe: SdcppProbe,
): Promise<SdcppModelStatus> {
  const response = await probe(`${url}/sdcpp/v1/models/status`);
  const status = response.status === 200 ? parseSdcppModelStatus(response.data) : undefined;
  if (!status) {
    throw new SdcppModelError(`model status answered HTTP ${response.status}`);
  }
  return status;
}

/** Whether the server has a usable model loaded: ready, without a load error, naming its files. */
export function hasReadySdcppModel(status: SdcppModelStatus): boolean {
  return (
    status.error == null &&
    (status.state ?? 'ready') === 'ready' &&
    Object.keys(status.models).length > 0
  );
}

const isReady = (status: SdcppModelStatus, recipe: string) =>
  status.recipe === recipe && status.error == null && (status.state ?? 'ready') === 'ready';

/**
 * Makes `recipe` the server's loaded model and returns the resulting status. `models/load` is
 * synchronous and answers with the final status; the server validates the recipe before
 * unloading the current model, so a refused load leaves that model in place.
 */
export async function loadSdcppModel(
  url: string,
  recipe: string,
  current: SdcppModelStatus,
  post: SdcppPost,
): Promise<SdcppModelStatus> {
  if (isReady(current, recipe)) {
    return current;
  }
  const response = await post(`${url}/sdcpp/v1/models/load`, { recipe });
  const status = parseSdcppModelStatus(response.data);
  if (response.status !== 200 || !status || !isReady(status, recipe)) {
    const detail = status?.error ?? (status?.state && `state ${status.state}`);
    throw new SdcppModelError(
      `loading ${recipe} answered HTTP ${response.status}${detail ? ` (${detail})` : ''}`,
    );
  }
  return status;
}

const basename = (file: string) => file.split(/[\\/]/).pop() ?? file;

/** The model files a WebUI-style response says it generated with (`info.generation_info`). */
function generatedModels(info: unknown): Record<string, unknown> | undefined {
  let parsed = info;
  if (typeof info === 'string') {
    try {
      parsed = JSON.parse(info);
    } catch {
      return undefined;
    }
  }
  if (parsed == null || typeof parsed !== 'object') {
    return undefined;
  }
  const generationInfo = (parsed as { generation_info?: unknown }).generation_info;
  const first = Array.isArray(generationInfo) ? generationInfo[0] : generationInfo;
  const models = (first as { models?: unknown } | undefined)?.models;
  return models != null && typeof models === 'object'
    ? (models as Record<string, unknown>)
    : undefined;
}

/**
 * Whether a result was made by the model loaded as `status`: its files (`match`), other files
 * (`mismatch`, with the differing slots), or cannot tell (`unknown`: the result names no files,
 * or none of the slots the status names).
 */
export type SdcppModelCheck =
  | { result: 'match' }
  | { result: 'mismatch'; detail: string }
  | { result: 'unknown' };

/**
 * Checks which model produced a generation. Generation requests cannot name a model, so another
 * client of the server may switch models between the load and the generation; the result's
 * `generation_info.models` names the files it used. Files are compared by name per slot (the
 * status gives paths relative to the model root, the result bare names).
 *
 * This detects another set of weights, not another recipe: two recipes using the same files
 * with other settings, or same-named files in other folders, look alike. Telling recipes apart
 * would need an identifier from sd-server, so LibreChat assumes it is the server's only client.
 */
export function verifySdcppModel(status: SdcppModelStatus, info: unknown): SdcppModelCheck {
  const actual = generatedModels(info);
  if (!actual) {
    return { result: 'unknown' };
  }
  const differences: string[] = [];
  let compared = 0;
  for (const [slot, expected] of Object.entries(status.models)) {
    const used = actual[slot];
    if (typeof used !== 'string') {
      continue;
    }
    compared++;
    if (basename(used) !== basename(expected)) {
      differences.push(`${slot}: expected ${basename(expected)}, got ${basename(used)}`);
    }
  }
  if (differences.length > 0) {
    return { result: 'mismatch', detail: differences.join('; ') };
  }
  return compared > 0 ? { result: 'match' } : { result: 'unknown' };
}

/** How long a holder keeps the server lock without renewing it (it renews while working). */
const LOCK_TTL_MS = 60_000;
const LOCK_RENEW_MS = 20_000;
/** A holder whose lease is not confirmed stops this long before it could expire. */
const LOCK_MARGIN_MS = 10_000;
const LOCK_RETRY_MS = 500;

export interface WithSdcppServerLockParams {
  lock: DistributedLock;
  url: string;
  /** How long to wait for another holder before giving up. */
  waitMs: number;
  /** Fires when the caller gives up (the user stopped the response); waiting ends with its reason. */
  signal?: AbortSignal;
  now?: () => number;
  sleep?: (ms: number, signal?: AbortSignal) => Promise<void>;
}

/** Waits `ms`, or rejects with `signal`'s reason as soon as it fires. */
export function sleepUnlessAborted(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason);
      return;
    }
    const onAbort = () => {
      clearTimeout(timer);
      reject(signal?.reason);
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

/** Why the server lock could not protect a task: still held by another, or lost while held. */
export class SdcppLockError extends Error {
  constructor(
    readonly code: 'busy' | 'lost',
    message: string,
  ) {
    super(message);
    this.name = 'SdcppLockError';
  }
}

/**
 * Runs `task` while holding the lock for the sd.cpp server at `url`, so a model load and the
 * generation after it are not interleaved with another LibreChat request's load. With a
 * Redis-backed lock this holds across LibreChat instances. It does not stop clients outside
 * LibreChat from switching the model; `verifySdcppModel` detects that afterwards.
 *
 * The lease is renewed while the task runs, one renewal at a time. When it is lost (renewal
 * refused, or not confirmed shortly before the lease could expire, also while a renewal hangs)
 * `task`'s signal is aborted, so it must pass the signal to every request and start none after
 * it fired; the call then throws `SdcppLockError` (`lost`) even if the task finished, since its
 * result was not protected. A renewal answered after that does not bring the lease back.
 * Aborting a request does not stop work sd-server already started. Throws `SdcppLockError`
 * (`busy`) when the lock is not free within `waitMs`. When `signal` fires while waiting, no
 * further attempt is made and the call rejects with its reason; a lock acquired as it fires is
 * released again.
 */
export async function withSdcppServerLock<T>(
  {
    lock,
    url,
    waitMs,
    signal,
    now = Date.now,
    sleep = sleepUnlessAborted,
  }: WithSdcppServerLockParams,
  task: (signal: AbortSignal) => Promise<T>,
): Promise<T> {
  const key = `stable-diffusion:sdcpp:${url}`;
  const deadline = now() + waitMs;
  signal?.throwIfAborted();
  /** When the last acquire or renewal was sent: the lease runs at least that long from there. */
  let sentAt = now();
  let token = await lock.acquire(key, LOCK_TTL_MS);
  while (!token) {
    signal?.throwIfAborted();
    if (now() >= deadline) {
      throw new SdcppLockError('busy', `server lock still held after ${waitMs} ms`);
    }
    await sleep(LOCK_RETRY_MS, signal);
    signal?.throwIfAborted();
    sentAt = now();
    token = await lock.acquire(key, LOCK_TTL_MS);
  }
  if (signal?.aborted) {
    await lock.release(key, token).catch(() => undefined);
    signal.throwIfAborted();
  }

  const held = token;
  const controller = new AbortController();
  let done = false;
  let renewal: ReturnType<typeof setTimeout> | undefined;
  let expiry: ReturnType<typeof setTimeout> | undefined;
  const lose = (reason: string) => {
    if (!done && !controller.signal.aborted) {
      logger.warn(`[StableDiffusion] ${reason}; stopping the generation`);
      controller.abort(new SdcppLockError('lost', reason));
    }
  };
  /** Stops the task before the lease confirmed at `since` could expire, unless renewed first. */
  const guard = (since: number) => {
    clearTimeout(expiry);
    expiry = setTimeout(
      () => lose('Could not renew the sd.cpp server lock before it expires'),
      Math.max(0, since + LOCK_TTL_MS - LOCK_MARGIN_MS - now()),
    );
    expiry.unref?.();
  };
  /** Sends the next renewal when the last one settled, so they never overlap. */
  const schedule = () => {
    renewal = setTimeout(() => {
      const sent = now();
      lock.extend(key, held, LOCK_TTL_MS).then(
        (extended) => {
          if (done || controller.signal.aborted) {
            return;
          }
          if (!extended) {
            lose('Lost the sd.cpp server lock');
            return;
          }
          guard(sent);
          schedule();
        },
        () => {
          if (!done && !controller.signal.aborted) {
            schedule();
          }
        },
      );
    }, LOCK_RENEW_MS);
    renewal.unref?.();
  };
  guard(sentAt);
  schedule();
  try {
    const result = await task(controller.signal);
    controller.signal.throwIfAborted();
    return result;
  } catch (error) {
    controller.signal.throwIfAborted();
    throw error;
  } finally {
    done = true;
    clearTimeout(renewal);
    clearTimeout(expiry);
    await lock.release(key, held).catch(() => undefined);
  }
}
