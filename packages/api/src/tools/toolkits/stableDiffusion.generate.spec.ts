import axios from 'axios';
import type { TFile } from 'librechat-data-provider';
import type { ServerRequest } from '~/types';
import {
  STABLE_DIFFUSION_LOCK_LOST,
  generateStableDiffusionImage,
  STABLE_DIFFUSION_SERVER_BUSY,
  STABLE_DIFFUSION_MODELS_UNAVAILABLE,
  clearStableDiffusionBackendCache,
  STABLE_DIFFUSION_MODEL_UNSUPPORTED,
} from './stableDiffusion';
import { createGeneratedImageRegistry } from './images';
import { clearSdcppCatalogCache } from './sdcpp';

jest.mock('axios', () => ({ get: jest.fn(), post: jest.fn() }));

const mockGet = axios.get as jest.Mock;
const mockPost = axios.post as jest.Mock;
const url = 'http://sd.local:7860';

const recipe = (id: string, file: string, extra: Record<string, unknown> = {}) => ({
  id,
  name: id,
  models: { diffusion_model: `diffusion_models/${file}` },
  size_multiple: 32,
  defaults: { width: 1024, height: 1024 },
  effective_features: { img2img: true },
  ...extra,
});
const recipes = [
  recipe('qwen', 'qwen.gguf'),
  recipe('flux', 'flux.gguf', { defaults: { width: 768, height: 768 } }),
  recipe('ltx', 'ltx.gguf', { defaults: { video_frames: 33 } }),
];
const status = (id: string, file: string) => ({
  recipe: id,
  state: 'ready',
  error: null,
  models: { diffusion_model: `diffusion_models/${file}` },
});
const generated = (file: string) => ({
  images: ['aW1hZ2U='],
  info: JSON.stringify({ generation_info: [{ models: { diffusion_model: file } }] }),
});

/** Answers after `ms`, failing like axios when the request's signal fired meanwhile. */
async function after<T>(ms: number, config: { signal?: AbortSignal } | undefined, value: T) {
  if (ms > 0) {
    await new Promise((resolve) => setTimeout(resolve, ms));
  }
  if (config?.signal?.aborted) {
    throw new Error('canceled');
  }
  return value;
}

/**
 * A fake sd-server; `loaded` is what models/status reports and models/load changes. An empty
 * `id` is a server with no recipe loaded: nothing at all without a `file`, else a model loaded
 * from the command line.
 */
function serve(
  initial: { id: string; file: string; error?: string },
  generatedFiles?: string[],
  delays: { loadMs?: number; generateMs?: number; recipesStatus?: number } = {},
) {
  let loaded = initial;
  const files = [...(generatedFiles ?? [])];
  mockGet.mockImplementation(async (requestUrl: string) => {
    if (requestUrl.endsWith('/sdcpp/v1/capabilities')) {
      return {
        status: 200,
        data: {
          current_mode: 'img_gen',
          defaults: {},
          samplers: ['euler', 'euler_a'],
          schedulers: ['karras', 'simple'],
          limits: { min_width: 64, max_width: 4096, min_height: 64, max_height: 4096 },
        },
      };
    }
    if (requestUrl.endsWith('/sdcpp/v1/recipes')) {
      const recipesStatus = delays.recipesStatus ?? 200;
      return { status: recipesStatus, data: recipesStatus === 200 ? { recipes } : {} };
    }
    if (requestUrl.endsWith('/sdcpp/v1/models/status')) {
      if (loaded.id) {
        return { status: 200, data: status(loaded.id, loaded.file) };
      }
      const models = loaded.file ? { diffusion_model: loaded.file } : {};
      const error = 'error' in loaded ? loaded.error : null;
      return { status: 200, data: { recipe: null, state: 'ready', error, models } };
    }
    return { status: 404, data: 'Not Found' };
  });
  mockPost.mockImplementation(
    async (requestUrl: string, body: { recipe?: string }, config?: { signal?: AbortSignal }) => {
      if (requestUrl.endsWith('/sdcpp/v1/models/load')) {
        const next = recipes.find((candidate) => candidate.id === body.recipe);
        loaded = { id: next!.id, file: next!.models.diffusion_model.split('/').pop()! };
        return after(delays.loadMs ?? 0, config, {
          status: 200,
          data: status(loaded.id, loaded.file),
        });
      }
      const data = generated(files.shift() ?? loaded.file);
      return after(delays.generateMs ?? 0, config, { status: 200, data });
    },
  );
}

const req = (models?: Record<string, unknown>, timeouts?: Record<string, number>) =>
  ({
    user: { id: 'user-1' },
    config: { stableDiffusion: models && { models }, timeouts },
  }) as unknown as ServerRequest;

const loadCalls = () =>
  mockPost.mock.calls.filter(([requestUrl]) => String(requestUrl).endsWith('/models/load'));

describe('generateStableDiffusionImage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    clearStableDiffusionBackendCache();
    clearSdcppCatalogCache();
  });

  it('generates with the loaded model without loading anything', async () => {
    serve({ id: 'qwen', file: 'qwen.gguf' });
    const result = await generateStableDiffusionImage({ prompt: 'a cat' }, { url, req: req() });
    expect(result).toEqual({ data: generated('qwen.gguf'), model: 'qwen' });
    expect(loadCalls()).toHaveLength(0);
  });

  it('loads a requested model before generating', async () => {
    serve({ id: 'qwen', file: 'qwen.gguf' });
    const result = await generateStableDiffusionImage(
      { prompt: 'a cat', model: 'flux' },
      { url, req: req() },
    );
    expect(result).toEqual({ data: generated('flux.gguf'), model: 'flux' });
    expect(loadCalls()).toEqual([
      [`${url}/sdcpp/v1/models/load`, { recipe: 'flux' }, expect.any(Object)],
    ]);
  });

  it('sizes img2img for the chosen model', async () => {
    serve({ id: 'qwen', file: 'qwen.gguf' });
    const findImageFiles = jest.fn(async () => [] as TFile[]);
    const generatedImages = createGeneratedImageRegistry();
    const request = req();
    generatedImages.remember(request, {
      file: { file_id: 'gen-1', width: 1600, height: 900 },
      buffer: Buffer.from('generated'),
      mimeType: 'image/png',
    });
    const context = {
      url,
      req: request,
      generatedImages,
      findImageFiles,
      getStrategyFunctions: jest.fn(() => ({})),
    };
    const img2img = () =>
      mockPost.mock.calls.find(([requestUrl]) => String(requestUrl).endsWith('/img2img'))?.[1];

    await generateStableDiffusionImage({ prompt: 'red', image_ids: ['gen-1'] }, context);
    expect(img2img()).toMatchObject({ width: 1024, height: 576 });

    mockPost.mockClear();
    await generateStableDiffusionImage(
      { prompt: 'red', image_ids: ['gen-1'], model: 'flux' },
      context,
    );
    expect(img2img()).toMatchObject({ width: 768, height: 448 });
  });

  it("sends the settings asked for, sized for the loaded model's multiple", async () => {
    serve({ id: 'qwen', file: 'qwen.gguf' });
    const result = await generateStableDiffusionImage(
      { prompt: 'a cat', width: 500, steps: 8, seed: 7, sampler: 'Euler_A', scheduler: 'Karras' },
      { url, req: req() },
    );
    expect(result).toEqual({ data: generated('qwen.gguf'), model: 'qwen' });
    const txt2img = mockPost.mock.calls.find(([requestUrl]) =>
      String(requestUrl).endsWith('/txt2img'),
    );
    expect(txt2img?.[1]).toEqual({
      prompt: 'a cat',
      negative_prompt: undefined,
      width: 512,
      steps: 8,
      cfg_scale: 4.5,
      seed: 7,
      sampler_name: 'euler_a',
      scheduler: 'karras',
    });
  });

  it('refuses settings the server would ignore before locking or loading anything', async () => {
    serve({ id: 'qwen', file: 'qwen.gguf' });
    const result = await generateStableDiffusionImage(
      { prompt: 'a cat', model: 'flux', scheduler: 'exponential' },
      { url, req: req() },
    );
    expect(result).toEqual({
      error:
        'Unknown scheduler "exponential". Omit "scheduler" for the model\'s default, or use one of: karras, simple.',
    });
    expect(mockPost).not.toHaveBeenCalled();
  });

  it('gives the generation its own budget after a slow model load', async () => {
    serve({ id: 'qwen', file: 'qwen.gguf' }, undefined, { loadMs: 70, generateMs: 70 });
    const result = await generateStableDiffusionImage(
      { prompt: 'a cat', model: 'flux' },
      { url, req: req(undefined, { imageGenerationMs: 110 }) },
    );
    expect(result).toEqual({ data: generated('flux.gguf'), model: 'flux' });
  });

  it('gives a retried generation its own budget', async () => {
    serve({ id: 'qwen', file: 'qwen.gguf' }, ['other.gguf'], { generateMs: 70 });
    const result = await generateStableDiffusionImage(
      { prompt: 'a cat' },
      { url, req: req(undefined, { imageGenerationMs: 110 }) },
    );
    expect(result).toEqual({ data: generated('qwen.gguf'), model: 'qwen' });
  });

  it.each([
    ['no model loaded', { id: '', file: '' }],
    ['a failed load', { id: '', file: 'qwen.gguf', error: 'out of memory' }],
  ])('loads the default image model after %s', async (_case, initial) => {
    serve(initial);
    const result = await generateStableDiffusionImage(
      { prompt: 'a cat' },
      { url, req: req({ defaultImage: 'flux' }) },
    );
    expect(result).toEqual({ data: generated('flux.gguf'), model: 'flux' });
    expect(loadCalls()).toHaveLength(1);
  });

  it('keeps a ready model loaded from the command line', async () => {
    serve({ id: '', file: 'custom.gguf' });
    const result = await generateStableDiffusionImage(
      { prompt: 'a cat' },
      { url, req: req({ defaultImage: 'flux' }) },
    );
    expect(result).toEqual({ data: generated('custom.gguf') });
    expect(loadCalls()).toHaveLength(0);
  });

  it('switches from a loaded video model to the default image model', async () => {
    serve({ id: 'ltx', file: 'ltx.gguf' });
    const result = await generateStableDiffusionImage(
      { prompt: 'a cat' },
      { url, req: req({ defaultImage: 'flux' }) },
    );
    expect(result).toMatchObject({ model: 'flux' });
  });

  it('refuses an excluded model without loading or generating', async () => {
    serve({ id: 'qwen', file: 'qwen.gguf' });
    const result = await generateStableDiffusionImage(
      { prompt: 'a cat', model: 'flux' },
      { url, req: req({ exclude: ['flux'] }) },
    );
    expect(result).toEqual({ error: expect.stringContaining('Unknown model "flux"') });
    expect(mockPost).not.toHaveBeenCalled();
  });

  it('retries once when another client switched the model during the generation', async () => {
    serve({ id: 'qwen', file: 'qwen.gguf' }, ['other.gguf']);
    const result = await generateStableDiffusionImage({ prompt: 'a cat' }, { url, req: req() });
    expect(result).toEqual({ data: generated('qwen.gguf'), model: 'qwen' });
  });

  it('gives up when the model is switched again', async () => {
    serve({ id: 'qwen', file: 'qwen.gguf' }, ['other.gguf', 'other.gguf']);
    const result = await generateStableDiffusionImage({ prompt: 'a cat' }, { url, req: req() });
    expect(result).toEqual({ error: expect.stringContaining('"qwen" could not be loaded') });
  });

  it('reports a model that fails to load', async () => {
    serve({ id: 'qwen', file: 'qwen.gguf' });
    mockPost.mockResolvedValueOnce({ status: 500, data: { error: 'out of memory', models: {} } });
    const result = await generateStableDiffusionImage(
      { prompt: 'a cat', model: 'flux' },
      { url, req: req() },
    );
    expect(result).toEqual({ error: expect.stringContaining('"flux" could not be loaded') });
  });

  it('holds the server lock around the load and the generation', async () => {
    serve({ id: 'qwen', file: 'qwen.gguf' });
    const lock = {
      acquire: jest.fn(async () => 'token'),
      extend: jest.fn(async () => true),
      release: jest.fn(async () => true),
    };
    await generateStableDiffusionImage({ prompt: 'a cat' }, { url, req: req(), lock });
    expect(lock.acquire).toHaveBeenCalledWith(`stable-diffusion:sdcpp:${url}`, expect.any(Number));
    expect(lock.release).toHaveBeenCalledWith(`stable-diffusion:sdcpp:${url}`, 'token');
  });

  describe('with the server lock', () => {
    beforeEach(() => jest.useFakeTimers());
    afterEach(() => jest.useRealTimers());

    it('stops when the lock is lost during the generation', async () => {
      serve({ id: 'qwen', file: 'qwen.gguf' }, undefined, { generateMs: 30_000 });
      const lock = {
        acquire: jest.fn(async () => 'token'),
        extend: jest.fn(async () => false),
        release: jest.fn(async () => true),
      };
      const result = generateStableDiffusionImage({ prompt: 'a cat' }, { url, req: req(), lock });
      await jest.advanceTimersByTimeAsync(30_000);
      await expect(result).resolves.toEqual({ error: STABLE_DIFFUSION_LOCK_LOST });
    });

    it('reports a server kept busy by other requests', async () => {
      serve({ id: 'qwen', file: 'qwen.gguf' });
      const lock = { acquire: jest.fn(async () => null), extend: jest.fn(), release: jest.fn() };
      const result = generateStableDiffusionImage(
        { prompt: 'a cat' },
        { url, req: req(undefined, { imageGenerationMs: 1000 }), lock },
      );
      await jest.advanceTimersByTimeAsync(2000);
      await expect(result).resolves.toEqual({ error: STABLE_DIFFUSION_SERVER_BUSY });
      expect(mockPost).not.toHaveBeenCalled();
    });
  });

  it('keeps a result that names no model files without attributing it to a model', async () => {
    serve({ id: 'qwen', file: 'qwen.gguf' });
    const plain = { images: ['aW1hZ2U='], info: JSON.stringify({ seed: 1 }) };
    mockPost.mockImplementation(async (requestUrl: string) =>
      String(requestUrl).endsWith('/models/load')
        ? { status: 200, data: status('flux', 'flux.gguf') }
        : { status: 200, data: plain },
    );
    await expect(
      generateStableDiffusionImage({ prompt: 'a cat', model: 'flux' }, { url, req: req() }),
    ).resolves.toEqual({ data: plain });
  });

  describe('when the model list cannot be read', () => {
    const unreadable = (initial = { id: 'qwen', file: 'qwen.gguf' }) =>
      serve(initial, undefined, { recipesStatus: 503 });
    const spyLock = () => ({
      acquire: jest.fn(async () => 'token'),
      extend: jest.fn(async () => true),
      release: jest.fn(async () => true),
    });
    const generations = () =>
      mockPost.mock.calls.filter(([requestUrl]) => String(requestUrl).includes('/sdapi/'));

    it.each([
      ['no model is allowed', { include: [] }],
      ['the loaded model is excluded', { exclude: ['qwen'] }],
    ])('refuses to generate when %s', async (_case, models) => {
      unreadable();
      const result = await generateStableDiffusionImage(
        { prompt: 'a cat' },
        { url, req: req(models) },
      );
      expect(result).toEqual({ error: STABLE_DIFFUSION_MODELS_UNAVAILABLE });
      expect(generations()).toHaveLength(0);
    });

    it('refuses a model loaded outside the catalog when models are restricted', async () => {
      unreadable({ id: '', file: 'custom.gguf' });
      const result = await generateStableDiffusionImage(
        { prompt: 'a cat' },
        { url, req: req({ include: ['qwen'] }) },
      );
      expect(result).toEqual({ error: STABLE_DIFFUSION_MODELS_UNAVAILABLE });
      expect(generations()).toHaveLength(0);
    });

    it('generates with a loaded model the configuration allows, under the lock', async () => {
      unreadable();
      const lock = spyLock();
      const result = await generateStableDiffusionImage(
        { prompt: 'a cat' },
        { url, req: req({ include: ['qwen'] }), lock },
      );
      expect(result).toEqual({ data: generated('qwen.gguf'), model: 'qwen' });
      expect(lock.acquire).toHaveBeenCalled();
    });

    it('refuses an allowed result made by other files', async () => {
      serve({ id: 'qwen', file: 'qwen.gguf' }, ['other.gguf'], { recipesStatus: 503 });
      const result = await generateStableDiffusionImage(
        { prompt: 'a cat' },
        { url, req: req({ include: ['qwen'] }) },
      );
      expect(result).toEqual({ error: expect.stringContaining('"qwen" could not be loaded') });
    });

    it('generates with the loaded model under the lock without restrictions', async () => {
      unreadable();
      const lock = spyLock();
      const result = await generateStableDiffusionImage(
        { prompt: 'a cat' },
        { url, req: req(), lock },
      );
      expect(result).toEqual({ data: generated('qwen.gguf') });
      expect(lock.acquire).toHaveBeenCalled();
    });

    it('refuses a requested model', async () => {
      unreadable();
      await expect(
        generateStableDiffusionImage({ prompt: 'a cat', model: 'qwen' }, { url, req: req() }),
      ).resolves.toEqual({ error: STABLE_DIFFUSION_MODELS_UNAVAILABLE });
    });

    it('tells an sd.cpp server without a model list from one that cannot be read', async () => {
      serve({ id: 'qwen', file: 'qwen.gguf' }, undefined, { recipesStatus: 404 });
      await expect(
        generateStableDiffusionImage({ prompt: 'a cat', model: 'qwen' }, { url, req: req() }),
      ).resolves.toEqual({ error: STABLE_DIFFUSION_MODEL_UNSUPPORTED });
    });
  });

  describe('when the server has not answered the probe yet', () => {
    /** sd-server whose capabilities route fails on first contact, others answering as usual. */
    const probeFails = (
      failure: 'timeout' | 'error',
      initial = { id: 'qwen', file: 'qwen.gguf' },
      statusCode?: number,
    ) => {
      serve(initial);
      const served = mockGet.getMockImplementation()!;
      mockGet.mockImplementation(async (requestUrl: string, config?: unknown) => {
        if (requestUrl.endsWith('/sdcpp/v1/capabilities')) {
          if (failure === 'timeout') {
            throw new Error('timeout of 60000ms exceeded');
          }
          return { status: 503, data: '' };
        }
        if (statusCode && requestUrl.endsWith('/sdcpp/v1/models/status')) {
          return { status: statusCode, data: '' };
        }
        return served(requestUrl, config);
      });
    };
    const spyLock = () => ({
      acquire: jest.fn(async () => 'token'),
      extend: jest.fn(async () => true),
      release: jest.fn(async () => true),
    });
    const generations = () =>
      mockPost.mock.calls.filter(([requestUrl]) => String(requestUrl).includes('/sdapi/'));

    it.each([
      ['times out', 'timeout', { include: [] }],
      ['answers 503', 'error', { include: [] }],
      ['times out with the loaded model excluded', 'timeout', { exclude: ['qwen'] }],
      ['answers 503 with the loaded model excluded', 'error', { exclude: ['qwen'] }],
    ] as const)(
      'refuses a model the configuration forbids when the probe %s',
      async (_case, failure, models) => {
        probeFails(failure);
        const lock = spyLock();
        const result = await generateStableDiffusionImage(
          { prompt: 'a cat' },
          { url, req: req(models), lock },
        );
        expect(result).toEqual({ error: STABLE_DIFFUSION_MODELS_UNAVAILABLE });
        expect(generations()).toHaveLength(0);
        expect(lock.acquire).toHaveBeenCalled();
      },
    );

    it('refuses when models are restricted and the model status cannot be read either', async () => {
      probeFails('error', undefined, 503);
      const result = await generateStableDiffusionImage(
        { prompt: 'a cat' },
        { url, req: req({ include: ['qwen'] }) },
      );
      expect(result).toEqual({ error: STABLE_DIFFUSION_MODELS_UNAVAILABLE });
      expect(generations()).toHaveLength(0);
    });

    it('generates with an allowed loaded model under the lock', async () => {
      probeFails('error');
      const lock = spyLock();
      const result = await generateStableDiffusionImage(
        { prompt: 'a cat' },
        { url, req: req({ include: ['qwen'] }), lock },
      );
      expect(result).toEqual({ data: generated('qwen.gguf'), model: 'qwen' });
      expect(lock.acquire).toHaveBeenCalled();
    });

    it('sends the WebUI request under the lock without restrictions', async () => {
      probeFails('timeout');
      const lock = spyLock();
      const result = await generateStableDiffusionImage(
        { prompt: 'a cat' },
        { url, req: req(), lock },
      );
      expect(result).toEqual({ data: generated('qwen.gguf') });
      expect(lock.acquire).toHaveBeenCalled();
      expect(generations()).toEqual([
        [
          `${url}/sdapi/v1/txt2img`,
          expect.objectContaining({ width: 1024, height: 1024 }),
          expect.any(Object),
        ],
      ]);
    });

    it('asks to try again for a requested model', async () => {
      probeFails('error');
      await expect(
        generateStableDiffusionImage({ prompt: 'a cat', model: 'qwen' }, { url, req: req() }),
      ).resolves.toEqual({ error: STABLE_DIFFUSION_MODELS_UNAVAILABLE });
      expect(generations()).toHaveLength(0);
    });
  });

  it('sends the request as it is to a server without a model list', async () => {
    mockGet.mockResolvedValue({ status: 404, data: 'Not Found' });
    mockPost.mockResolvedValue({ status: 200, data: { images: ['x'] } });
    const lock = { acquire: jest.fn(), extend: jest.fn(), release: jest.fn() };
    await expect(
      generateStableDiffusionImage({ prompt: 'a cat' }, { url, req: req({ include: [] }), lock }),
    ).resolves.toEqual({ data: { images: ['x'] } });
    expect(mockPost).toHaveBeenCalledTimes(1);
    expect(lock.acquire).not.toHaveBeenCalled();

    await expect(
      generateStableDiffusionImage({ prompt: 'a cat', model: 'flux' }, { url, req: req() }),
    ).resolves.toEqual({ error: STABLE_DIFFUSION_MODEL_UNSUPPORTED });
  });
});
