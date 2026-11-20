import type { SdcppCatalog } from './catalog';
import { formatSdcppModels } from './context';

const catalog: SdcppCatalog = {
  loaded: 'qwen',
  models: [
    {
      id: 'qwen',
      name: 'Qwen Image 2.1',
      description: 'Qwen Image 2.1 with Q8 diffusion weights.',
      mode: 'image',
      acceptsImage: true,
      sizeMultiple: 32,
      width: 1024,
      height: 1024,
      steps: 20,
      cfgScale: 1,
    },
    { id: 'sd15', name: 'SD 1.5', mode: 'image', acceptsImage: false, sizeMultiple: 64 },
    {
      id: 'ltx',
      name: 'LTX 2.5',
      mode: 'video',
      acceptsImage: true,
      sizeMultiple: 32,
      width: 768,
      height: 512,
      steps: 8,
      frames: 121,
      fps: 24,
      frameRule: { offset: 1, step: 8, rounding: 'floor', min: 1 },
    },
    {
      id: 'hyperflow',
      name: 'Hyperflow',
      mode: 'video',
      acceptsImage: false,
      sizeMultiple: 32,
      frames: 124,
      frameRule: { offset: 5, step: 17, rounding: 'ceil', min: 5, max: 260, fixedFps: 24 },
    },
    {
      id: 'animatediff',
      name: 'AnimateDiff',
      mode: 'video',
      acceptsImage: false,
      sizeMultiple: 64,
      frames: 8,
      frameRule: { offset: 0, step: 1, rounding: 'floor', min: 1, max: 32 },
    },
  ],
};

describe('formatSdcppModels', () => {
  it('lists the image models, then the video models, marking the loaded one', () => {
    expect(formatSdcppModels(catalog)).toBe(
      [
        'Image models available to `stable-diffusion` (pass the ID in "model"; omit "model" to keep the loaded one, since loading another model takes extra time):',
        '\t- qwen (loaded): Qwen Image 2.1 (text to image, img2img; defaults: 1024x1024, 20 steps, cfg_scale 1). Qwen Image 2.1 with Q8 diffusion weights.',
        '\t- sd15: SD 1.5 (text to image only).',
        '',
        'Video models available to `stable-diffusion` (set "video": true to make a video clip; pass the ID in "model" to choose one, or omit it to keep the loaded video model or use the default one):',
        '\t- ltx: LTX 2.5 (text to video; defaults: 768x512, 121 frames at 24 fps (5 s), 8 steps; frames: 8n+1).',
        '\t- hyperflow: Hyperflow (text to video; defaults: 124 frames at 24 fps (5.2 s); frames: 17n+5, at least 5, up to 260, fixed 24 fps).',
        '\t- animatediff: AnimateDiff (text to video; defaults: 8 frames; frames: any count, up to 32).',
      ].join('\n'),
    );
  });

  it('lists only video models when no image model is allowed', () => {
    const text = formatSdcppModels(catalog, { include: ['ltx'] });
    expect(text).toMatch(/^Video models/);
    expect(text).not.toContain('qwen');
  });

  it('leaves out models librechat.yaml does not allow', () => {
    const text = formatSdcppModels(catalog, { exclude: ['qwen'] });
    expect(text).not.toContain('qwen');
    expect(text).toContain('sd15');
  });

  it('is empty without an allowed model', () => {
    expect(formatSdcppModels(catalog, { include: ['unknown'] })).toBe('');
  });
});
