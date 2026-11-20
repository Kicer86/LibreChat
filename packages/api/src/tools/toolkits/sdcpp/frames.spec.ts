import type { SdcppModel } from './catalog';
import { MAX_CLIP_PIXELS, clipFrameLimit, resolveSdcppClip } from './frames';

const ltx: SdcppModel = {
  id: 'ltx',
  name: 'LTX 2.5',
  mode: 'video',
  acceptsImage: true,
  sizeMultiple: 32,
  frames: 121,
  fps: 24,
  frameRule: { offset: 1, step: 8, rounding: 'floor', min: 1 },
};
const hyperflow: SdcppModel = {
  id: 'hyperflow',
  name: 'Hyperflow',
  mode: 'video',
  acceptsImage: false,
  sizeMultiple: 32,
  frames: 124,
  fps: 24,
  frameRule: { offset: 5, step: 17, rounding: 'ceil', min: 5, max: 260, fixedFps: 24 },
};

describe('resolveSdcppClip', () => {
  it("keeps the model's default clip within the limit", () => {
    expect(resolveSdcppClip({}, ltx, 121)).toEqual({ ok: true, value: { frames: 121, fps: 24 } });
    expect(resolveSdcppClip({}, ltx, 100)).toEqual({ ok: true, value: { frames: 97, fps: 24 } });
    expect(resolveSdcppClip({}, hyperflow, 121)).toEqual({
      ok: true,
      value: { frames: 107, fps: 24 },
    });
  });

  it("rounds frames to the model's rule, the way the model rounds", () => {
    expect(resolveSdcppClip({ frames: 30 }, ltx, 121)).toMatchObject({ value: { frames: 25 } });
    expect(resolveSdcppClip({ frames: 30 }, hyperflow, 260)).toMatchObject({
      value: { frames: 39 },
    });
    expect(resolveSdcppClip({ frames: 1 }, hyperflow, 260)).toMatchObject({
      value: { frames: 5 },
    });
  });

  it('rounds down instead of past the limit', () => {
    expect(resolveSdcppClip({ frames: 120 }, hyperflow, 121)).toMatchObject({
      value: { frames: 107 },
    });
  });

  it('turns a duration into frames at the requested or default frame rate', () => {
    expect(resolveSdcppClip({ duration: 2 }, ltx, 121)).toEqual({
      ok: true,
      value: { frames: 41, fps: 24 },
    });
    expect(resolveSdcppClip({ duration: 2, fps: 12 }, ltx, 121)).toEqual({
      ok: true,
      value: { frames: 17, fps: 12 },
    });
  });

  it('refuses more frames or seconds than fit, with the range that does', () => {
    expect(resolveSdcppClip({ frames: 200 }, ltx, 121)).toEqual({
      ok: false,
      error: { code: 'invalid_frames', message: '"frames" must be from 1 to 121 for model "ltx".' },
    });
    expect(resolveSdcppClip({ duration: 10 }, ltx, 121)).toEqual({
      ok: false,
      error: {
        code: 'invalid_frames',
        message: '"duration" can be at most 5 s for model "ltx" at 24 fps.',
      },
    });
    expect(resolveSdcppClip({ frames: 300 }, hyperflow, 1000)).toMatchObject({
      error: { message: '"frames" must be from 5 to 260 for model "hyperflow".' },
    });
  });

  it('refuses another frame rate on a model with a fixed one', () => {
    expect(resolveSdcppClip({ fps: 12 }, hyperflow, 260)).toMatchObject({
      ok: false,
      error: { code: 'invalid_fps' },
    });
    expect(resolveSdcppClip({ fps: 24 }, hyperflow, 260)).toMatchObject({ ok: true });
  });

  it('refuses a limit below the least clip the model makes', () => {
    expect(resolveSdcppClip({}, hyperflow, 4)).toMatchObject({
      ok: false,
      error: { code: 'invalid_frames' },
    });
  });

  it('asks for the settings it cannot read from the model', () => {
    const unknown: SdcppModel = { ...ltx, frames: undefined, fps: undefined };
    expect(resolveSdcppClip({}, unknown, 121)).toMatchObject({
      error: { code: 'defaults_unknown' },
    });
    expect(resolveSdcppClip({ duration: 2 }, unknown, 121)).toMatchObject({
      error: { code: 'defaults_unknown' },
    });
    expect(resolveSdcppClip({ frames: 9 }, unknown, 121)).toEqual({
      ok: true,
      value: { frames: 9 },
    });
  });

  it('takes any frame count from a model without a rule', () => {
    const plain: SdcppModel = { ...ltx, frameRule: undefined };
    expect(resolveSdcppClip({ frames: 30 }, plain, 121)).toMatchObject({ value: { frames: 30 } });
  });
});

describe('clipFrameLimit', () => {
  it('keeps the configured limit for clips that can be saved', () => {
    expect(clipFrameLimit({ width: 768, height: 512 }, 121)).toBe(121);
    expect(clipFrameLimit({}, 121)).toBe(121);
  });

  it('lowers it so all frames stay within the pixel limit for saving', () => {
    expect(clipFrameLimit({ width: 2048, height: 2048 }, 121)).toBe(63);
    expect(63 * 2048 * 2048).toBeLessThanOrEqual(MAX_CLIP_PIXELS);
    expect(64 * 2048 * 2048).toBeGreaterThan(MAX_CLIP_PIXELS);
  });

  it('makes the model rule fit the lowered limit', () => {
    expect(resolveSdcppClip({}, ltx, clipFrameLimit({ width: 2048, height: 2048 }, 121))).toEqual({
      ok: true,
      value: { frames: 57, fps: 24 },
    });
  });
});
