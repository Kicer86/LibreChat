import type { TFile } from 'librechat-data-provider';
import type { ImageInput } from './images';
import {
  fitStableDiffusionSize,
  buildStableDiffusionRequest,
  detectStableDiffusionBackend,
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

  it('leaves steps, CFG scale and the size to stable-diffusion.cpp model defaults', () => {
    expect(buildStableDiffusionRequest({ prompt: 'a cat', backend: 'sdcpp' })).toEqual({
      path: '/sdapi/v1/txt2img',
      payload: { prompt: 'a cat' },
    });
  });

  it('still fits img2img to the input image on stable-diffusion.cpp', () => {
    const { payload } = buildStableDiffusionRequest({
      prompt: 'x',
      image: input(900, 1600),
      backend: 'sdcpp',
    });
    expect(payload).toMatchObject({ width: 576, height: 1024 });
    expect(payload).not.toHaveProperty('steps');
    expect(payload).not.toHaveProperty('cfg_scale');
  });
});

describe('detectStableDiffusionBackend', () => {
  const url = 'http://sd.local:7860';
  const probe = jest.fn();

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
  ])('falls back to webui without caching for %s', async (_case, fail) => {
    fail();
    await expect(detectStableDiffusionBackend(url, probe, 0)).resolves.toBe('webui');
    probe.mockResolvedValueOnce({ status: 200, data: capabilities });
    await expect(detectStableDiffusionBackend(url, probe, 1)).resolves.toBe('sdcpp');
  });
});
