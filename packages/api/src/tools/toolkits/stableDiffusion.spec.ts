import type { TFile } from 'librechat-data-provider';
import type { ImageInput } from './images';
import { buildStableDiffusionRequest, fitStableDiffusionSize } from './stableDiffusion';

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
});
