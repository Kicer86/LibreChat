import type { StableDiffusionBounds } from './stableDiffusionParams';
import type { SdcppCapabilities } from './sdcpp';
import {
  STABLE_DIFFUSION_LIMITS,
  checkStableDiffusionParams,
  resolveStableDiffusionSettings,
  STABLE_DIFFUSION_DEFAULTS_UNKNOWN,
} from './stableDiffusionParams';

const capabilities: SdcppCapabilities = {
  samplers: ['euler', 'euler_a', 'dpm++2m'],
  schedulers: ['karras', 'simple'],
  limits: { minWidth: 64, maxWidth: 4096, minHeight: 64, maxHeight: 4096, maxBatchCount: 8 },
};

describe('checkStableDiffusionParams', () => {
  it('passes the settings on under the API names', () => {
    expect(
      checkStableDiffusionParams(
        {
          width: 768,
          height: 512,
          steps: 8,
          cfg_scale: 1.5,
          seed: 123,
          sampler: 'euler_a',
          scheduler: 'karras',
          count: 2,
        },
        { capabilities },
      ),
    ).toEqual({
      ok: true,
      value: {
        width: 768,
        height: 512,
        steps: 8,
        cfg_scale: 1.5,
        seed: 123,
        sampler_name: 'euler_a',
        scheduler: 'karras',
        batch_size: 2,
      },
      bounds: {
        width: { min: 64, max: 2048 },
        height: { min: 64, max: 2048 },
        maxSteps: 50,
      },
    });
  });

  it('gives nothing when no setting is asked for, and treats seed -1 as random', () => {
    expect(checkStableDiffusionParams({ seed: -1 })).toEqual({
      ok: true,
      value: {},
      bounds: {
        width: { min: 64, max: STABLE_DIFFUSION_LIMITS.maxSize },
        height: { min: 64, max: STABLE_DIFFUSION_LIMITS.maxSize },
        maxSteps: STABLE_DIFFUSION_LIMITS.maxSteps,
      },
    });
  });

  it("uses the server's spelling of a sampler or scheduler named in another case", () => {
    const check = checkStableDiffusionParams(
      { sampler: 'DPM++2M', scheduler: ' Karras ' },
      { capabilities },
    );
    expect(check).toMatchObject({
      ok: true,
      value: { sampler_name: 'dpm++2m', scheduler: 'karras' },
    });
  });

  it.each([
    ['sampler', { sampler: 'Euler a' }, 'unknown_sampler', 'euler, euler_a, dpm++2m'],
    ['scheduler', { scheduler: 'exponential' }, 'unknown_scheduler', 'karras, simple'],
  ])(
    'refuses a %s the server does not list, naming the ones it does',
    (_field, input, code, list) => {
      const check = checkStableDiffusionParams(input, { capabilities });
      expect(check).toMatchObject({ ok: false, error: { code } });
      expect(!check.ok && check.error.message).toContain(list);
    },
  );

  it('passes any sampler and scheduler to a server that lists none', () => {
    expect(checkStableDiffusionParams({ sampler: 'Euler a', scheduler: 'Karras' })).toMatchObject({
      ok: true,
      value: { sampler_name: 'Euler a', scheduler: 'Karras' },
    });
  });

  it.each([
    [{ width: 32 }, 'invalid_size', '"width" must be a whole number of pixels from 64 to 2048.'],
    [
      { height: 4096 },
      'invalid_size',
      '"height" must be a whole number of pixels from 64 to 2048.',
    ],
    [{ width: 512.5 }, 'invalid_size', '"width"'],
    [{ steps: 0 }, 'invalid_steps', '"steps" must be a whole number from 1 to 50.'],
    [{ steps: 51 }, 'invalid_steps', '"steps"'],
    [{ cfg_scale: -1 }, 'invalid_cfg_scale', '"cfg_scale" must be from 0 to 30.'],
    [{ cfg_scale: Number.NaN }, 'invalid_cfg_scale', '"cfg_scale"'],
    [{ seed: -2 }, 'invalid_seed', '"seed"'],
    [{ seed: 2 ** 32 }, 'invalid_seed', '"seed"'],
    [{ seed: 1.5 }, 'invalid_seed', '"seed"'],
  ])('refuses %j', (input, code, message) => {
    const check = checkStableDiffusionParams(input, { capabilities });
    expect(check).toMatchObject({ ok: false, error: { code } });
    expect(!check.ok && check.error.message).toContain(message);
  });

  it("limits the image count to librechat.yaml's and the server's batch limit", () => {
    expect(checkStableDiffusionParams({ count: 4 })).toMatchObject({ ok: true });
    expect(checkStableDiffusionParams({ count: 5 })).toMatchObject({
      ok: false,
      error: { code: 'invalid_count', message: '"count" must be a whole number from 1 to 4.' },
    });
    expect(checkStableDiffusionParams({ count: 0 })).toMatchObject({ ok: false });
    const server = { ...capabilities, limits: { maxBatchCount: 2 } };
    expect(
      checkStableDiffusionParams({ count: 3 }, { limits: { maxImages: 8 }, capabilities: server }),
    ).toMatchObject({
      ok: false,
      error: { message: '"count" must be a whole number from 1 to 2.' },
    });
  });

  it("applies librechat.yaml's limits and the server's lower ones", () => {
    const limits = { maxSize: 8192, maxSteps: 10 };
    const narrow = { ...capabilities, limits: { minWidth: 128, maxWidth: 1536, maxHeight: 1024 } };
    expect(checkStableDiffusionParams({ width: 2048 }, { limits, capabilities })).toMatchObject({
      ok: true,
    });
    expect(checkStableDiffusionParams({ steps: 11 }, { limits })).toMatchObject({
      ok: false,
      error: { code: 'invalid_steps', message: '"steps" must be a whole number from 1 to 10.' },
    });
    expect(checkStableDiffusionParams({}, { limits, capabilities: narrow })).toMatchObject({
      ok: true,
      bounds: { width: { min: 128, max: 1536 }, height: { min: 64, max: 1024 } },
    });
    expect(checkStableDiffusionParams({ width: 100 }, { capabilities: narrow })).toMatchObject({
      ok: false,
      error: { message: '"width" must be a whole number of pixels from 128 to 1536.' },
    });
  });
});

describe('resolveStableDiffusionSettings', () => {
  const bounds = (max = 2048, maxSteps = 50, min = 64, maxHeight = max): StableDiffusionBounds => ({
    width: { min, max },
    height: { min, max: maxHeight },
    maxSteps,
  });
  const recipe = { width: 1024, height: 1024, steps: 20 };
  const target = { longestSide: 1024, multiple: 32 };

  describe('WebUI', () => {
    it('sends the WebUI defaults when nothing is asked for', () => {
      expect(resolveStableDiffusionSettings({ backend: 'webui', bounds: bounds() })).toEqual({
        ok: true,
        value: { steps: 22, width: 1024, height: 1024 },
      });
    });

    it('keeps the WebUI defaults within lowered limits', () => {
      expect(resolveStableDiffusionSettings({ backend: 'webui', bounds: bounds(512, 10) })).toEqual(
        { ok: true, value: { steps: 10, width: 512, height: 512 } },
      );
    });

    it('rounds a side asked for to 64 and fills the other with the default', () => {
      expect(
        resolveStableDiffusionSettings({ backend: 'webui', width: 500, bounds: bounds() }),
      ).toEqual({ ok: true, value: { steps: 22, width: 512, height: 1024 } });
    });
  });

  describe('stable-diffusion.cpp txt2img', () => {
    const sdcpp = { backend: 'sdcpp' as const, defaults: recipe, target };

    it("sends the recipe's steps, which sd-server would not apply, and leaves it the size", () => {
      expect(resolveStableDiffusionSettings({ ...sdcpp, bounds: bounds() })).toEqual({
        ok: true,
        value: { steps: 20 },
      });
    });

    it('sends the recipe defaults lowered to the limits', () => {
      expect(resolveStableDiffusionSettings({ ...sdcpp, bounds: bounds(512, 10) })).toEqual({
        ok: true,
        value: { steps: 10, width: 512, height: 512 },
      });
    });

    it.each([
      ['a square', { width: 1000, height: 1000 }, { width: 960, height: 960 }],
      ['a wide size', { width: 1000, height: 600 }, { width: 960, height: 576 }],
    ])(
      'sends %s default that sd-server would round up past the limit, scaled within it',
      (_case, size, sent) => {
        expect(
          resolveStableDiffusionSettings({
            ...sdcpp,
            defaults: { ...size, steps: 20 },
            target: { longestSide: 1000, multiple: 64 },
            bounds: bounds(1000),
          }),
        ).toEqual({ ok: true, value: { steps: 20, ...sent } });
      },
    );

    it('leaves an unaligned default to sd-server when rounding it up stays within the limit', () => {
      expect(
        resolveStableDiffusionSettings({
          ...sdcpp,
          defaults: { width: 1000, height: 1000, steps: 20 },
          target: { longestSide: 1000, multiple: 64 },
          bounds: bounds(1024),
        }),
      ).toEqual({ ok: true, value: { steps: 20 } });
    });

    it('sends the default side that would round up past its limit when one side is asked for', () => {
      expect(
        resolveStableDiffusionSettings({
          ...sdcpp,
          width: 512,
          defaults: { width: 1024, height: 1000, steps: 20 },
          target: { longestSide: 1024, multiple: 64 },
          bounds: bounds(2048, 50, 64, 1000),
        }),
      ).toEqual({ ok: true, value: { steps: 20, width: 512, height: 960 } });
    });

    it('refuses an unaligned default when no multiple fits the limits', () => {
      expect(
        resolveStableDiffusionSettings({
          ...sdcpp,
          defaults: { width: 1000, height: 1000, steps: 20 },
          target: { longestSide: 1000, multiple: 64 },
          bounds: bounds(1000, 50, 980),
        }),
      ).toMatchObject({ ok: false, error: { code: 'size_out_of_range' } });
    });

    it("scales a default size above the limits as a whole, keeping the recipe's shape", () => {
      expect(
        resolveStableDiffusionSettings({
          ...sdcpp,
          defaults: { width: 1280, height: 720, steps: 20 },
          bounds: bounds(1024),
        }),
      ).toEqual({ ok: true, value: { steps: 20, width: 1024, height: 576 } });
    });

    it("rounds the side asked for to the model's multiple and keeps the other default", () => {
      expect(resolveStableDiffusionSettings({ ...sdcpp, width: 500, bounds: bounds() })).toEqual({
        ok: true,
        value: { steps: 20, width: 512 },
      });
    });

    it('clamps a default side out of range next to the side asked for', () => {
      expect(
        resolveStableDiffusionSettings({
          ...sdcpp,
          width: 1536,
          bounds: bounds(2048, 50, 64, 768),
        }),
      ).toEqual({ ok: true, value: { steps: 20, width: 1536, height: 768 } });
    });

    it.each([
      ['steps', { width: 1024, height: 1024 }, {}],
      ['the height next to a width asked for', { steps: 20 }, { width: 512 }],
      ['the size', { steps: 20 }, {}],
    ])('refuses when %s cannot be checked', (_case, defaults, asked) => {
      expect(
        resolveStableDiffusionSettings({ ...sdcpp, defaults, ...asked, bounds: bounds() }),
      ).toEqual({
        ok: false,
        error: { code: 'defaults_unknown', message: STABLE_DIFFUSION_DEFAULTS_UNKNOWN },
      });
    });

    it('needs no defaults for the settings asked for', () => {
      expect(
        resolveStableDiffusionSettings({
          backend: 'sdcpp',
          width: 640,
          height: 640,
          steps: 4,
          bounds: bounds(),
        }),
      ).toEqual({ ok: true, value: { steps: 4, width: 640, height: 640 } });
    });
  });

  describe('img2img', () => {
    const img2img = (image: { width: number; height: number }) => ({
      backend: 'sdcpp' as const,
      image,
      defaults: recipe,
      target,
    });

    it.each([
      [{ width: 800 }, { width: 800, height: 448 }],
      [{ height: 900 }, { width: 1600, height: 896 }],
      [
        { width: 640, height: 640 },
        { width: 640, height: 640 },
      ],
      [{}, { width: 1024, height: 576 }],
    ])('keeps the input shape for %j', (asked, expected) => {
      expect(
        resolveStableDiffusionSettings({
          ...img2img({ width: 1600, height: 900 }),
          ...asked,
          bounds: bounds(),
        }),
      ).toEqual({ ok: true, value: { steps: 20, ...expected } });
    });

    it.each([
      [
        { width: 1024, height: 2048 },
        { width: 2048 },
        '"width" 2048 makes the height 4096',
        '"width" from 64 to 1024',
      ],
      [
        { width: 2048, height: 1024 },
        { height: 2048 },
        '"height" 2048 makes the width 4096',
        '"height" from 64 to 1024',
      ],
    ])(
      'refuses a side whose derived side leaves the limits, for a %j input',
      (image, asked, made, advice) => {
        const result = resolveStableDiffusionSettings({
          ...img2img(image),
          ...asked,
          bounds: bounds(),
        });
        expect(result).toMatchObject({ ok: false, error: { code: 'size_out_of_range' } });
        expect(!result.ok && result.error.message).toContain(made);
        expect(!result.ok && result.error.message).toContain(advice);
      },
    );

    it('scales the fitted size as a whole into different width and height limits', () => {
      expect(
        resolveStableDiffusionSettings({
          ...img2img({ width: 1600, height: 900 }),
          target: { longestSide: 2048, multiple: 32 },
          bounds: bounds(1024, 50, 64, 512),
        }),
      ).toEqual({ ok: true, value: { steps: 20, width: 896, height: 512 } });
    });

    it('sends the default size when the input size is unknown', () => {
      expect(
        resolveStableDiffusionSettings({
          backend: 'sdcpp',
          image: {},
          defaults: recipe,
          target,
          bounds: bounds(),
        }),
      ).toEqual({ ok: true, value: { steps: 20, width: 1024, height: 1024 } });
    });
  });

  describe('rounding within the limits', () => {
    it('rounds up to a multiple at or above the minimum', () => {
      expect(
        resolveStableDiffusionSettings({
          backend: 'sdcpp',
          width: 80,
          height: 128,
          steps: 4,
          target: { longestSide: 1024, multiple: 64 },
          bounds: bounds(2048, 50, 80),
        }),
      ).toEqual({ ok: true, value: { steps: 4, width: 128, height: 128 } });
    });

    it("refuses limits no multiple of the model's size fits", () => {
      const result = resolveStableDiffusionSettings({
        backend: 'sdcpp',
        width: 64,
        height: 64,
        steps: 4,
        target: { longestSide: 1024, multiple: 128 },
        bounds: bounds(64),
      });
      expect(result).toEqual({
        ok: false,
        error: {
          code: 'size_out_of_range',
          message:
            'No image width fits the limits: the model needs a multiple of 128 pixels from 64 to 64.',
        },
      });
    });
  });
});
