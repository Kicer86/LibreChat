import type { TStableDiffusionConfig } from 'librechat-data-provider';
import type { SdcppCatalog, SdcppModel } from './catalog';
import { allowedSdcppModels } from './select';

/** The settings a model renders with when a request gives none, e.g. `1024x1024, 20 steps`. */
function describeDefaults(model: SdcppModel): string {
  const defaults = [
    model.width && model.height ? `${model.width}x${model.height}` : '',
    model.steps ? `${model.steps} steps` : '',
    model.cfgScale != null ? `cfg_scale ${model.cfgScale}` : '',
  ].filter(Boolean);
  return defaults.length > 0 ? `; defaults: ${defaults.join(', ')}` : '';
}

/**
 * The image models the `stable-diffusion` tool may use, as tool context for the model calling
 * it: each model's ID (the tool's `model` argument), name, what it can do, the settings it renders
 * with by default and the server's own description, with the loaded one marked. Empty when no
 * allowed model makes images.
 */
export function formatSdcppModels(
  catalog: SdcppCatalog,
  config?: TStableDiffusionConfig['models'],
): string {
  const models = allowedSdcppModels(catalog.models, config).filter(
    (model) => model.mode === 'image',
  );
  if (models.length === 0) {
    return '';
  }
  const lines = models.map((model) => {
    const loaded = model.id === catalog.loaded ? ' (loaded)' : '';
    const abilities = model.acceptsImage ? 'text to image, img2img' : 'text to image only';
    const description = model.description ? ` ${model.description}` : '';
    return `\t- ${model.id}${loaded}: ${model.name} (${abilities}${describeDefaults(model)}).${description}`;
  });
  return [
    'Image models available to `stable-diffusion` (pass the ID in "model"; omit "model" to keep the loaded one, since loading another model takes extra time):',
    ...lines,
  ].join('\n');
}
