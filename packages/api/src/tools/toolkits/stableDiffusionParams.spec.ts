import type { SdcppCapabilities } from './sdcpp';
import { checkStableDiffusionParams, STABLE_DIFFUSION_LIMITS } from './stableDiffusionParams';

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
      },
      maxSize: { width: 2048, height: 2048 },
    });
  });

  it('gives nothing when no setting is asked for, and treats seed -1 as random', () => {
    expect(checkStableDiffusionParams({ seed: -1 })).toEqual({
      ok: true,
      value: {},
      maxSize: { width: STABLE_DIFFUSION_LIMITS.maxSize, height: STABLE_DIFFUSION_LIMITS.maxSize },
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
      maxSize: { width: 1536, height: 1024 },
    });
    expect(checkStableDiffusionParams({ width: 100 }, { capabilities: narrow })).toMatchObject({
      ok: false,
      error: { message: '"width" must be a whole number of pixels from 128 to 1536.' },
    });
  });
});
