import axios from 'axios';
import type { TFile } from 'librechat-data-provider';
import type { ServerRequest } from '~/types';
import {
  generateStableDiffusionImage,
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

/** A fake sd-server; `loaded` is what models/status reports and models/load changes. */
function serve(initial: { id: string; file: string }, generatedFiles?: string[]) {
  let loaded = initial;
  const files = [...(generatedFiles ?? [])];
  mockGet.mockImplementation(async (requestUrl: string) => {
    if (requestUrl.endsWith('/sdcpp/v1/capabilities')) {
      return { status: 200, data: { current_mode: 'img_gen', defaults: {} } };
    }
    if (requestUrl.endsWith('/sdcpp/v1/recipes')) {
      return { status: 200, data: { recipes } };
    }
    if (requestUrl.endsWith('/sdcpp/v1/models/status')) {
      return { status: 200, data: status(loaded.id, loaded.file) };
    }
    return { status: 404, data: 'Not Found' };
  });
  mockPost.mockImplementation(async (requestUrl: string, body: { recipe?: string }) => {
    if (requestUrl.endsWith('/sdcpp/v1/models/load')) {
      const next = recipes.find((candidate) => candidate.id === body.recipe);
      loaded = { id: next!.id, file: next!.models.diffusion_model.split('/').pop()! };
      return { status: 200, data: status(loaded.id, loaded.file) };
    }
    return { status: 200, data: generated(files.shift() ?? loaded.file) };
  });
}

const req = (models?: Record<string, unknown>) =>
  ({
    user: { id: 'user-1' },
    config: { stableDiffusion: models && { models } },
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

  it('sends the request as it is to a server without a model list', async () => {
    mockGet.mockResolvedValue({ status: 404, data: 'Not Found' });
    mockPost.mockResolvedValue({ status: 200, data: { images: ['x'] } });
    await expect(
      generateStableDiffusionImage({ prompt: 'a cat' }, { url, req: req() }),
    ).resolves.toEqual({ data: { images: ['x'] } });
    expect(mockPost).toHaveBeenCalledTimes(1);

    await expect(
      generateStableDiffusionImage({ prompt: 'a cat', model: 'flux' }, { url, req: req() }),
    ).resolves.toEqual({ error: STABLE_DIFFUSION_MODEL_UNSUPPORTED });
  });
});
