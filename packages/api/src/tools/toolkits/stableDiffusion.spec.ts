import type { TFile } from 'librechat-data-provider';
import type { ImageInput } from './images';
import {
  sdcppSizeTarget,
  fitStableDiffusionSize,
  resolveStableDiffusionSize,
  buildStableDiffusionRequest,
  detectStableDiffusionServer,
  clearStableDiffusionBackendCache,
} from './stableDiffusion';

const capabilities = { current_mode: 'img_gen', defaults: { width: 1024, height: 1024 } };

const input = (width?: number, height?: number): ImageInput => ({
  file: { file_id: 'img-1', width, height } as TFile,
  buffer: Buffer.from('png-bytes'),
  mimeType: 'image/png',
});

describe('fitStableDiffusionSize', () => {
  it.each([
    [1600, 900, { width: 1024, height: 576 }],
    [900, 1600, { width: 576, height: 1024 }],
    [512, 512, { width: 1024, height: 1024 }],
    [3000, 100, { width: 1024, height: 64 }],
  ])('fits %ix%i into 1024 keeping the aspect ratio', (width, height, expected) => {
    expect(fitStableDiffusionSize(width, height)).toEqual(expected);
  });
});

describe('sdcppSizeTarget', () => {
  const model = { id: 'm', name: 'm', mode: 'image' as const, acceptsImage: true };

  it("fits img2img into the recipe's size and multiple", () => {
    const target = sdcppSizeTarget({ ...model, sizeMultiple: 32, width: 768, height: 512 });
    expect(target).toEqual({ longestSide: 768, multiple: 32 });
    expect(fitStableDiffusionSize(1600, 900, target)).toEqual({ width: 768, height: 448 });
  });

  it('falls back to 1024 for a recipe without a size', () => {
    expect(sdcppSizeTarget({ ...model, sizeMultiple: 16 })).toEqual({
      longestSide: 1024,
      multiple: 16,
    });
  });
});

describe('buildStableDiffusionRequest', () => {
  it('builds the unchanged txt2img request without an image', () => {
    expect(buildStableDiffusionRequest({ prompt: 'a cat', negative_prompt: 'blur' })).toEqual({
      path: '/sdapi/v1/txt2img',
      payload: {
        prompt: 'a cat',
        negative_prompt: 'blur',
        cfg_scale: 4.5,
        steps: 22,
        width: 1024,
        height: 1024,
      },
    });
  });

  it('builds an img2img request sized like the input image', () => {
    const { path, payload } = buildStableDiffusionRequest({
      prompt: 'as a watercolor',
      negative_prompt: '',
      image: input(1600, 900),
      denoising_strength: 0.4,
    });
    expect(path).toBe('/sdapi/v1/img2img');
    expect(payload).toMatchObject({
      prompt: 'as a watercolor',
      width: 1024,
      height: 576,
      denoising_strength: 0.4,
      init_images: [Buffer.from('png-bytes').toString('base64')],
    });
  });

  it('leaves the denoising strength and unknown sizes to the server defaults', () => {
    const { payload } = buildStableDiffusionRequest({ prompt: 'x', image: input() });
    expect(payload).not.toHaveProperty('denoising_strength');
    expect(payload).toMatchObject({ width: 1024, height: 1024 });
  });

  it('leaves the txt2img size to the stable-diffusion.cpp recipe, with fixed steps and CFG scale', () => {
    expect(buildStableDiffusionRequest({ prompt: 'a cat', backend: 'sdcpp' })).toEqual({
      path: '/sdapi/v1/txt2img',
      payload: { prompt: 'a cat', cfg_scale: 4.5, steps: 22 },
    });
  });

  it('sends the WebUI txt2img size to a server not identified yet', () => {
    expect(buildStableDiffusionRequest({ prompt: 'a cat', backend: 'unknown' })).toEqual({
      path: '/sdapi/v1/txt2img',
      payload: { prompt: 'a cat', cfg_scale: 4.5, steps: 22, width: 1024, height: 1024 },
    });
  });

  it('sends the settings asked for in place of the defaults', () => {
    const params = { steps: 8, cfg_scale: 1, seed: 42, sampler_name: 'euler_a', width: 768 };
    expect(buildStableDiffusionRequest({ prompt: 'a cat', params }).payload).toEqual({
      prompt: 'a cat',
      negative_prompt: undefined,
      cfg_scale: 1,
      steps: 8,
      seed: 42,
      sampler_name: 'euler_a',
      width: 768,
      height: 1024,
    });
    expect(
      buildStableDiffusionRequest({ prompt: 'a cat', backend: 'sdcpp', params }).payload,
    ).toEqual({
      prompt: 'a cat',
      negative_prompt: undefined,
      cfg_scale: 1,
      steps: 8,
      seed: 42,
      sampler_name: 'euler_a',
      width: 768,
    });
  });

  it('still fits img2img to the input image on stable-diffusion.cpp', () => {
    const { payload } = buildStableDiffusionRequest({
      prompt: 'x',
      image: input(900, 1600),
      backend: 'sdcpp',
    });
    expect(payload).toMatchObject({ width: 576, height: 1024, cfg_scale: 4.5, steps: 22 });
  });
});

describe('resolveStableDiffusionSize', () => {
  const max = { width: 2048, height: 2048 };

  it('rounds the sides asked for to the target multiple, within the largest size', () => {
    const target = { longestSide: 1024, multiple: 32 };
    expect(
      resolveStableDiffusionSize({ backend: 'sdcpp', width: 500, height: 300, target }),
    ).toEqual({ width: 512, height: 288 });
    expect(
      resolveStableDiffusionSize({
        backend: 'sdcpp',
        width: 1000,
        target,
        maxSize: { width: 1000, height: 1000 },
      }),
    ).toEqual({ width: 992 });
  });

  it('leaves the sides not asked for to the stable-diffusion.cpp recipe, unrounded without a model', () => {
    expect(resolveStableDiffusionSize({ backend: 'sdcpp', height: 500 })).toEqual({ height: 500 });
    expect(resolveStableDiffusionSize({ backend: 'sdcpp' })).toEqual({});
  });

  it('fills the other side with the WebUI default and rounds to 64', () => {
    expect(resolveStableDiffusionSize({ backend: 'webui', width: 500 })).toEqual({
      width: 512,
      height: 1024,
    });
  });

  it.each([
    [{ width: 800 }, { width: 800, height: 448 }],
    [{ height: 900 }, { width: 1600, height: 896 }],
    [
      { width: 640, height: 640 },
      { width: 640, height: 640 },
    ],
    [{}, { width: 1024, height: 576 }],
  ])('keeps the img2img input shape for %j', (asked, expected) => {
    expect(
      resolveStableDiffusionSize({
        backend: 'sdcpp',
        image: { width: 1600, height: 900 },
        target: { longestSide: 1024, multiple: 32 },
        maxSize: max,
        ...asked,
      }),
    ).toEqual(expected);
  });
});

describe('detectStableDiffusionServer', () => {
  const url = 'http://sd.local:7860';
  const probe = jest.fn();
  const detectStableDiffusionBackend = async (...args: [string, typeof probe, number]) =>
    (await detectStableDiffusionServer(...args)).backend;

  beforeEach(() => {
    clearStableDiffusionBackendCache();
    probe.mockReset();
  });

  it('detects stable-diffusion.cpp by its capabilities endpoint and caches the answer', async () => {
    probe.mockResolvedValue({ status: 200, data: capabilities });
    await expect(detectStableDiffusionBackend(url, probe, 0)).resolves.toBe('sdcpp');
    await expect(detectStableDiffusionBackend(url, probe, 1000)).resolves.toBe('sdcpp');
    expect(probe).toHaveBeenCalledTimes(1);
    expect(probe).toHaveBeenCalledWith(`${url}/sdcpp/v1/capabilities`);
  });

  it("keeps the server's samplers, schedulers and limits with the detected backend", async () => {
    probe.mockResolvedValue({
      status: 200,
      data: {
        ...capabilities,
        samplers: ['euler', 'euler_a'],
        schedulers: ['karras'],
        limits: { min_width: 64, max_width: 4096, max_batch_count: 8 },
      },
    });
    await expect(detectStableDiffusionServer(url, probe, 0)).resolves.toEqual({
      backend: 'sdcpp',
      capabilities: {
        samplers: ['euler', 'euler_a'],
        schedulers: ['karras'],
        limits: { minWidth: 64, maxWidth: 4096, maxBatchCount: 8 },
      },
    });
  });

  it('treats a missing endpoint as another WebUI server and asks again after the TTL', async () => {
    probe.mockResolvedValue({ status: 404, data: 'Not Found' });
    await expect(detectStableDiffusionBackend(url, probe, 0)).resolves.toBe('webui');
    await expect(detectStableDiffusionBackend(url, probe, 1000)).resolves.toBe('webui');
    expect(probe).toHaveBeenCalledTimes(1);
    await detectStableDiffusionBackend(url, probe, 10 * 60 * 1000);
    expect(probe).toHaveBeenCalledTimes(2);
  });

  it.each([
    ['an empty object', {}],
    ['an array', []],
    ['an error object', { error: 'unknown route' }],
    ['a mode without defaults', { current_mode: 'img_gen' }],
    ['an empty mode', { current_mode: '', defaults: {} }],
  ])('does not take %s answering with HTTP 200 for stable-diffusion.cpp', async (_case, data) => {
    probe.mockResolvedValue({ status: 200, data });
    await expect(detectStableDiffusionBackend(url, probe, 0)).resolves.toBe('webui');
  });

  it.each([
    ['an unreachable server', () => probe.mockRejectedValueOnce(new Error('ECONNREFUSED'))],
    ['a server error', () => probe.mockResolvedValueOnce({ status: 503, data: '' })],
  ])('reports an unknown backend without caching for %s', async (_case, fail) => {
    fail();
    await expect(detectStableDiffusionBackend(url, probe, 0)).resolves.toBe('unknown');
    probe.mockResolvedValueOnce({ status: 200, data: capabilities });
    await expect(detectStableDiffusionBackend(url, probe, 1)).resolves.toBe('sdcpp');
  });

  it.each([
    ['times out', () => probe.mockRejectedValueOnce(new Error('timeout of 5000ms exceeded'))],
    ['answers 503', () => probe.mockResolvedValueOnce({ status: 503, data: '' })],
  ])('keeps a confirmed stable-diffusion.cpp when a later probe %s', async (_case, fail) => {
    const expired = 10 * 60 * 1000;
    probe.mockResolvedValueOnce({ status: 200, data: capabilities });
    await detectStableDiffusionBackend(url, probe, 0);
    fail();
    await expect(detectStableDiffusionBackend(url, probe, expired)).resolves.toBe('sdcpp');
    probe.mockResolvedValueOnce({ status: 404, data: 'Not Found' });
    await expect(detectStableDiffusionBackend(url, probe, expired + 1)).resolves.toBe('webui');
    expect(probe).toHaveBeenCalledTimes(3);
  });
});
