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
    },
    { id: 'sd15', name: 'SD 1.5', mode: 'image', acceptsImage: false, sizeMultiple: 64 },
    { id: 'ltx', name: 'LTX 2.5', mode: 'video', acceptsImage: true, sizeMultiple: 32 },
  ],
};

describe('formatSdcppModels', () => {
  it('lists the image models with what they can do, marking the loaded one', () => {
    expect(formatSdcppModels(catalog)).toBe(
      [
        'Image models available to `stable-diffusion` (pass the ID in "model"; omit "model" to keep the loaded one, since loading another model takes extra time):',
        '\t- qwen (loaded): Qwen Image 2.1 (text to image, img2img). Qwen Image 2.1 with Q8 diffusion weights.',
        '\t- sd15: SD 1.5 (text to image only).',
      ].join('\n'),
    );
  });

  it('leaves out models librechat.yaml does not allow', () => {
    const text = formatSdcppModels(catalog, { exclude: ['qwen'] });
    expect(text).not.toContain('qwen');
    expect(text).toContain('sd15');
  });

  it('is empty without an allowed image model', () => {
    expect(formatSdcppModels(catalog, { include: ['ltx'] })).toBe('');
  });
});
