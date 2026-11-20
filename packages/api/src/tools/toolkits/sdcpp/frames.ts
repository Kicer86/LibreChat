import type { SdcppFrameRule, SdcppModel } from './catalog';

/** The clip length and frame rate the calling model asked for, already checked on their own. */
export interface SdcppClipRequest {
  frames?: number;
  fps?: number;
  /** Seconds; turned into frames at the clip's frame rate. Never given together with `frames`. */
  duration?: number;
}

/** What a video generation is sent: frames on the model's rule, and the frame rate when known. */
export interface SdcppClip {
  frames: number;
  fps?: number;
}

export type SdcppClipErrorCode = 'invalid_frames' | 'invalid_fps' | 'defaults_unknown';

export type SdcppClipResult =
  | { ok: true; value: SdcppClip }
  | { ok: false; error: { code: SdcppClipErrorCode; message: string } };

/** Any frame count, for a model that states no rule. */
const ANY_FRAME_COUNT: SdcppFrameRule = { offset: 0, step: 1, rounding: 'floor', min: 1 };

const failure = (code: SdcppClipErrorCode, message: string): SdcppClipResult => ({
  ok: false,
  error: { code, message },
});

/** A length in seconds, to one decimal. */
const seconds = (frames: number, fps: number) => Math.round((frames / fps) * 10) / 10;

/**
 * The frames and frame rate a video model renders a clip with, within `maxFrames`
 * (`librechat.yaml`) and the model's own frame rule:
 * - frames are `offset + step·n`, rounded the way the model rounds (sd-server would do it
 *   anyway), then kept between the model's least frame count and the most the limits allow;
 * - `duration` becomes frames at the requested frame rate, else the model's;
 * - without either, the model's default clip, shortened to the limit when it is longer;
 * - a model with a fixed frame rate refuses another one.
 * Asking for more frames (or seconds) than fit is refused with the range that does.
 */
export function resolveSdcppClip(
  request: SdcppClipRequest,
  model: SdcppModel,
  maxFrames: number,
): SdcppClipResult {
  const rule = model.frameRule ?? ANY_FRAME_COUNT;
  if (request.fps != null && rule.fixedFps != null && request.fps !== rule.fixedFps) {
    return failure(
      'invalid_fps',
      `Model "${model.id}" plays at a fixed ${rule.fixedFps} fps. Omit "fps".`,
    );
  }
  const fps = rule.fixedFps ?? request.fps ?? model.fps;
  const cap = Math.min(maxFrames, rule.max ?? maxFrames);
  const lowest =
    rule.offset + rule.step * Math.ceil(Math.max(0, rule.min - rule.offset) / rule.step);
  const highest = rule.offset + rule.step * Math.floor((cap - rule.offset) / rule.step);
  if (cap < rule.offset || highest < lowest) {
    return failure(
      'invalid_frames',
      `Model "${model.id}" cannot make a clip within the limit of ${maxFrames} frames.`,
    );
  }
  const onRule = (count: number) => {
    const steps = (count - rule.offset) / rule.step;
    const n = rule.rounding === 'ceil' ? Math.ceil(steps) : Math.floor(steps);
    return Math.min(highest, Math.max(lowest, rule.offset + rule.step * n));
  };
  const clip = (frames: number): SdcppClipResult => ({
    ok: true,
    value: { frames: onRule(frames), ...(fps != null && { fps }) },
  });

  if (request.frames != null) {
    if (request.frames > cap) {
      return failure(
        'invalid_frames',
        `"frames" must be from ${lowest} to ${highest} for model "${model.id}".`,
      );
    }
    return clip(request.frames);
  }
  if (request.duration != null) {
    if (fps == null) {
      return failure(
        'defaults_unknown',
        `The frame rate of model "${model.id}" is unknown. Pass "fps" with "duration", or pass "frames".`,
      );
    }
    const frames = Math.max(1, Math.round(request.duration * fps));
    if (frames > cap) {
      return failure(
        'invalid_frames',
        `"duration" can be at most ${seconds(highest, fps)} s for model "${model.id}" at ${fps} fps.`,
      );
    }
    return clip(frames);
  }
  if (model.frames == null) {
    return failure(
      'defaults_unknown',
      `The default clip length of model "${model.id}" is unknown. Pass "frames" or "duration".`,
    );
  }
  return clip(Math.min(model.frames, highest));
}
