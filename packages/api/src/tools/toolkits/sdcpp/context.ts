import type { TStableDiffusionConfig } from 'librechat-data-provider';
import type { SdcppCatalog, SdcppFrameRule, SdcppModel } from './catalog';
import { allowedSdcppModels } from './select';

/** A clip length in seconds, to one decimal: `5`, `1.4`. */
const seconds = (frames: number, fps: number) => `${Math.round((frames / fps) * 10) / 10} s`;

/** The settings a model renders with when a request gives none, e.g. `1024x1024, 20 steps`. */
function describeDefaults(model: SdcppModel): string {
  const fps = model.frameRule?.fixedFps ?? model.fps;
  let clip = '';
  if (model.frames) {
    clip = fps
      ? `${model.frames} frames at ${fps} fps (${seconds(model.frames, fps)})`
      : `${model.frames} frames`;
  }
  const defaults = [
    model.width && model.height ? `${model.width}x${model.height}` : '',
    clip,
    model.steps ? `${model.steps} steps` : '',
    model.cfgScale != null ? `cfg_scale ${model.cfgScale}` : '',
  ].filter(Boolean);
  return defaults.length > 0 ? `; defaults: ${defaults.join(', ')}` : '';
}

/** The frame counts a video model takes, e.g. `; frames: 8n+1, up to 257, fixed 24 fps`. */
function describeFrameRule(rule: SdcppFrameRule): string {
  const counts =
    rule.step > 1 ? `${rule.step}n${rule.offset > 0 ? `+${rule.offset}` : ''}` : 'any count';
  const parts = [
    `frames: ${counts}`,
    rule.min > 1 ? `at least ${rule.min}` : '',
    rule.max ? `up to ${rule.max}` : '',
    rule.fixedFps ? `fixed ${rule.fixedFps} fps` : '',
  ].filter(Boolean);
  return `; ${parts.join(', ')}`;
}

function describeModel(model: SdcppModel, loaded: string | undefined): string {
  const marker = model.id === loaded ? ' (loaded)' : '';
  let abilities = 'text to video';
  if (model.mode === 'image') {
    abilities = model.acceptsImage ? 'text to image, img2img' : 'text to image only';
  }
  const frames = model.frameRule ? describeFrameRule(model.frameRule) : '';
  const description = model.description ? ` ${model.description}` : '';
  return `\t- ${model.id}${marker}: ${model.name} (${abilities}${describeDefaults(model)}${frames}).${description}`;
}

/**
 * The models the `stable-diffusion` tool may use, as tool context for the model calling it: the
 * image models, then the video models, each with its ID (the tool's `model` argument), name, what
 * it can do, the settings it renders with by default, a video model's frame counts, and the
 * server's own description, with the loaded one marked. Empty when no model is allowed.
 */
export function formatSdcppModels(
  catalog: SdcppCatalog,
  config?: TStableDiffusionConfig['models'],
): string {
  const allowed = allowedSdcppModels(catalog.models, config);
  const sections: string[] = [];
  const images = allowed.filter((model) => model.mode === 'image');
  if (images.length > 0) {
    sections.push(
      [
        'Image models available to `stable-diffusion` (pass the ID in "model"; omit "model" to keep the loaded one, since loading another model takes extra time):',
        ...images.map((model) => describeModel(model, catalog.loaded)),
      ].join('\n'),
    );
  }
  const videos = allowed.filter((model) => model.mode === 'video');
  if (videos.length > 0) {
    sections.push(
      [
        'Video models available to `stable-diffusion` (set "video": true to make a video clip; pass the ID in "model" to choose one, or omit it to keep the loaded video model or use the default one):',
        ...videos.map((model) => describeModel(model, catalog.loaded)),
      ].join('\n'),
    );
  }
  return sections.join('\n\n');
}
