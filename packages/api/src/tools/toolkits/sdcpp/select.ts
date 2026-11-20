import type { TStableDiffusionConfig } from 'librechat-data-provider';
import type { SdcppCatalog, SdcppModel } from './catalog';

type SdcppModelsConfig = NonNullable<TStableDiffusionConfig['models']>;

/** Why no model could be chosen; the message is written for the model calling the tool. */
export type SdcppModelChoiceError = {
  code: 'unknown_model' | 'wrong_mode' | 'no_image_input' | 'no_model';
  message: string;
};

/**
 * The model to generate with. `model` is absent when the server's loaded model should be used
 * as it is: no catalog names it (a model loaded from the command line), and nothing excludes it.
 */
export type SdcppModelChoice =
  | { ok: true; model?: SdcppModel }
  | { ok: false; error: SdcppModelChoiceError };

/** The catalog's models that `librechat.yaml` (`stableDiffusion.models`) allows. */
export function allowedSdcppModels(models: SdcppModel[], config?: SdcppModelsConfig): SdcppModel[] {
  const include = config?.include ? new Set(config.include) : undefined;
  const exclude = new Set(config?.exclude ?? []);
  return models.filter((model) => (!include || include.has(model.id)) && !exclude.has(model.id));
}

export interface ChooseSdcppModelParams {
  catalog: SdcppCatalog;
  config?: SdcppModelsConfig;
  /** Model ID the tool was called with. */
  requested?: string;
  /** Whether the request starts from an input image (img2img). */
  withImage?: boolean;
}

const fits = (model: SdcppModel, withImage: boolean) =>
  model.mode === 'image' && (!withImage || model.acceptsImage);

/**
 * Chooses the image model for a request. A requested model must be allowed and able to serve the
 * request. Without one, the loaded model is kept when it can serve it, so no load is needed;
 * otherwise `defaultImage` is used, then the first allowed model that fits.
 */
export function chooseSdcppModel({
  catalog,
  config,
  requested,
  withImage = false,
}: ChooseSdcppModelParams): SdcppModelChoice {
  const allowed = allowedSdcppModels(catalog.models, config);

  if (requested) {
    const model = allowed.find((candidate) => candidate.id === requested);
    if (!model) {
      return {
        ok: false,
        error: {
          code: 'unknown_model',
          message: `Unknown model "${requested}". Use one of the listed model IDs, or omit "model".`,
        },
      };
    }
    if (model.mode !== 'image') {
      return {
        ok: false,
        error: {
          code: 'wrong_mode',
          message: `Model "${requested}" makes video, not images. Choose an image model.`,
        },
      };
    }
    if (!fits(model, withImage)) {
      return {
        ok: false,
        error: {
          code: 'no_image_input',
          message: `Model "${requested}" cannot change an existing image. Choose a model that supports img2img.`,
        },
      };
    }
    return { ok: true, model };
  }

  const loaded = catalog.loaded
    ? catalog.models.find((model) => model.id === catalog.loaded)
    : undefined;
  if (loaded && allowed.includes(loaded) && fits(loaded, withImage)) {
    return { ok: true, model: loaded };
  }
  const restricted = config?.include != null || (config?.exclude?.length ?? 0) > 0;
  if (!loaded && !catalog.loaded && !restricted) {
    return { ok: true };
  }

  const fallback =
    allowed.find((model) => model.id === config?.defaultImage && fits(model, withImage)) ??
    allowed.find((model) => fits(model, withImage));
  if (fallback) {
    return { ok: true, model: fallback };
  }
  return {
    ok: false,
    error: {
      code: 'no_model',
      message: withImage
        ? 'No available model can change an existing image.'
        : 'No available model can generate images.',
    },
  };
}
