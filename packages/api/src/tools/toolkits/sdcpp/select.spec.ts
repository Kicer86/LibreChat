import type { SdcppCatalog, SdcppModel } from './catalog';
import { allowedSdcppModels, chooseSdcppModel } from './select';

const model = (id: string, overrides: Partial<SdcppModel> = {}): SdcppModel => ({
  id,
  name: id,
  mode: 'image',
  acceptsImage: true,
  sizeMultiple: 32,
  ...overrides,
});

const qwen = model('qwen');
const flux = model('flux');
const textOnly = model('text-only', { acceptsImage: false });
const ltx = model('ltx', { mode: 'video' });
const wan = model('wan', { mode: 'video', acceptsImage: false });
const models = [textOnly, qwen, flux, ltx, wan];
const catalog = (loaded?: string): SdcppCatalog => ({ models, ...(loaded && { loaded }) });

describe('allowedSdcppModels', () => {
  it('allows every model without configuration', () => {
    expect(allowedSdcppModels(models)).toEqual(models);
  });

  it('keeps only included models and drops excluded ones', () => {
    expect(allowedSdcppModels(models, { include: ['qwen', 'flux'], exclude: ['flux'] })).toEqual([
      qwen,
    ]);
  });
});

describe('chooseSdcppModel', () => {
  it('uses a requested model', () => {
    expect(chooseSdcppModel({ catalog: catalog('qwen'), requested: 'flux' })).toEqual({
      ok: true,
      model: flux,
    });
  });

  it.each([
    ['missing', {}, undefined, 'unknown_model'],
    ['flux', { exclude: ['flux'] }, undefined, 'unknown_model'],
    ['flux', { include: ['qwen'] }, undefined, 'unknown_model'],
    ['ltx', {}, false, 'wrong_mode'],
    ['qwen', {}, true, 'wrong_mode'],
  ])('refuses the requested model %s with %j (video: %s)', (requested, config, video, code) => {
    expect(chooseSdcppModel({ catalog: catalog('qwen'), config, requested, video })).toMatchObject({
      ok: false,
      error: { code },
    });
  });

  it('makes a clip with a requested video model, also without "video"', () => {
    expect(chooseSdcppModel({ catalog: catalog('qwen'), requested: 'ltx' })).toEqual({
      ok: true,
      model: ltx,
    });
    expect(chooseSdcppModel({ catalog: catalog('qwen'), requested: 'ltx', video: true })).toEqual({
      ok: true,
      model: ltx,
    });
  });

  it('keeps a loaded video model for a clip', () => {
    expect(
      chooseSdcppModel({ catalog: catalog('wan'), config: { defaultVideo: 'ltx' }, video: true }),
    ).toEqual({ ok: true, model: wan });
  });

  it('switches from a loaded image model to the default video model', () => {
    expect(
      chooseSdcppModel({ catalog: catalog('qwen'), config: { defaultVideo: 'wan' }, video: true }),
    ).toEqual({ ok: true, model: wan });
    expect(chooseSdcppModel({ catalog: catalog('qwen'), video: true })).toEqual({
      ok: true,
      model: ltx,
    });
  });

  it('does not make a clip with a model outside the catalog', () => {
    expect(
      chooseSdcppModel({ catalog: catalog(), readyOutsideCatalog: true, video: true }),
    ).toEqual({ ok: true, model: ltx });
  });

  it('reports when no allowed model makes video', () => {
    expect(
      chooseSdcppModel({
        catalog: catalog('qwen'),
        config: { exclude: ['ltx', 'wan'] },
        video: true,
      }),
    ).toEqual({
      ok: false,
      error: { code: 'no_model', message: 'No available model can make video.' },
    });
  });

  it('refuses a requested model that cannot change an image', () => {
    expect(
      chooseSdcppModel({ catalog: catalog('qwen'), requested: 'text-only', withImage: true }),
    ).toMatchObject({ ok: false, error: { code: 'no_image_input' } });
  });

  it('keeps the loaded model when it can serve the request', () => {
    expect(
      chooseSdcppModel({ catalog: catalog('flux'), config: { defaultImage: 'qwen' } }),
    ).toEqual({ ok: true, model: flux });
  });

  it('switches from a loaded video model to the default image model', () => {
    expect(chooseSdcppModel({ catalog: catalog('ltx'), config: { defaultImage: 'flux' } })).toEqual(
      { ok: true, model: flux },
    );
  });

  it('switches from an excluded loaded model to the first allowed one', () => {
    expect(chooseSdcppModel({ catalog: catalog('qwen'), config: { exclude: ['qwen'] } })).toEqual({
      ok: true,
      model: textOnly,
    });
  });

  it('skips models that cannot change the given image', () => {
    expect(chooseSdcppModel({ catalog: catalog('text-only'), withImage: true })).toEqual({
      ok: true,
      model: qwen,
    });
    expect(
      chooseSdcppModel({
        catalog: catalog('ltx'),
        config: { defaultImage: 'text-only' },
        withImage: true,
      }),
    ).toEqual({ ok: true, model: qwen });
  });

  it('leaves a ready model loaded outside the catalog alone unless models are restricted', () => {
    expect(chooseSdcppModel({ catalog: catalog(), readyOutsideCatalog: true })).toEqual({
      ok: true,
    });
    expect(
      chooseSdcppModel({
        catalog: catalog(),
        config: { include: ['flux'] },
        readyOutsideCatalog: true,
      }),
    ).toEqual({ ok: true, model: flux });
  });

  it('loads a model when none is ready', () => {
    expect(chooseSdcppModel({ catalog: catalog(), config: { defaultImage: 'qwen' } })).toEqual({
      ok: true,
      model: qwen,
    });
    expect(chooseSdcppModel({ catalog: catalog() })).toEqual({ ok: true, model: textOnly });
  });

  it('reports when no allowed model fits', () => {
    expect(
      chooseSdcppModel({ catalog: catalog('ltx'), config: { include: ['ltx', 'text-only'] } }),
    ).toEqual({ ok: true, model: textOnly });
    expect(
      chooseSdcppModel({
        catalog: catalog('ltx'),
        config: { include: ['ltx', 'text-only'] },
        withImage: true,
      }),
    ).toMatchObject({ ok: false, error: { code: 'no_model' } });
  });
});
