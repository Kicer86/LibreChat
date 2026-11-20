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
 * Whether the model loaded as `status` produced a generation. Generation requests cannot name a
 * model, so another client of the server may switch models between the load and the generation;
 * the result's `generation_info.models` names the files it used. Files are compared by name per
 * slot (the status gives paths relative to the model root, the result bare names). Returns the
 * differing slots, or `undefined` when they match or the result names no files.
 */
export function findSdcppModelMismatch(
  status: SdcppModelStatus,
  info: unknown,
): string | undefined {
  const actual = generatedModels(info);
  if (!actual) {
    return undefined;
  }
  const differences: string[] = [];
  for (const [slot, expected] of Object.entries(status.models)) {
    const used = actual[slot];
    if (typeof used === 'string' && basename(used) !== basename(expected)) {
      differences.push(`${slot}: expected ${basename(expected)}, got ${basename(used)}`);
    }
  }
  return differences.length > 0 ? differences.join('; ') : undefined;
}

/** How long a holder keeps the server lock without renewing it (it renews while working). */
const LOCK_TTL_MS = 60_000;
const LOCK_RENEW_MS = 20_000;
const LOCK_RETRY_MS = 500;

export interface WithSdcppServerLockParams {
  lock: DistributedLock;
  url: string;
  /** How long to wait for another holder before giving up. */
  waitMs: number;
  now?: () => number;
  sleep?: (ms: number) => Promise<void>;
}

const defaultSleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

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
 * LibreChat from switching the model; `findSdcppModelMismatch` detects that afterwards.
 *
 * The lease is renewed while the task runs. When it is lost (renewal refused, or failing until
 * the lease would expire) `task`'s signal is aborted, so it must pass the signal to every
 * request and start none after it fired; the call then throws `SdcppLockError` (`lost`) even if
 * the task finished, since its result was not protected. Aborting a request does not stop work
 * sd-server already started. Throws `SdcppLockError` (`busy`) when the lock is not free within
 * `waitMs`.
 */
export async function withSdcppServerLock<T>(
  { lock, url, waitMs, now = Date.now, sleep = defaultSleep }: WithSdcppServerLockParams,
  task: (signal: AbortSignal) => Promise<T>,
): Promise<T> {
  const key = `stable-diffusion:sdcpp:${url}`;
  const deadline = now() + waitMs;
  let token = await lock.acquire(key, LOCK_TTL_MS);
  while (!token) {
    if (now() >= deadline) {
      throw new SdcppLockError('busy', `server lock still held after ${waitMs} ms`);
    }
    await sleep(LOCK_RETRY_MS);
    token = await lock.acquire(key, LOCK_TTL_MS);
  }

  const held = token;
  const controller = new AbortController();
  const lose = (reason: string) => {
    if (!controller.signal.aborted) {
      logger.warn(`[StableDiffusion] ${reason}; stopping the generation`);
      controller.abort(new SdcppLockError('lost', reason));
    }
  };
  let renewedAt = now();
  const renew = setInterval(() => {
    lock.extend(key, held, LOCK_TTL_MS).then(
      (extended) => {
        if (extended) {
          renewedAt = now();
        } else {
          lose('Lost the sd.cpp server lock');
        }
      },
      () => {
        if (now() - renewedAt + LOCK_RENEW_MS >= LOCK_TTL_MS) {
          lose('Could not renew the sd.cpp server lock before it expires');
        }
      },
    );
  }, LOCK_RENEW_MS);
  renew.unref?.();
  try {
    const result = await task(controller.signal);
    controller.signal.throwIfAborted();
    return result;
  } catch (error) {
    controller.signal.throwIfAborted();
    throw error;
  } finally {
    clearInterval(renew);
    await lock.release(key, held).catch(() => undefined);
  }
}
