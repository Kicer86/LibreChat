import type { TStableDiffusionConfig } from 'librechat-data-provider';
import type { SdcppCatalog, SdcppModel, SdcppModelMode } from './catalog';

type SdcppModelsConfig = NonNullable<TStableDiffusionConfig['models']>;

/** Why no model could be chosen; the message is written for the model calling the tool. */
export type SdcppModelChoiceError = {
  code: 'unknown_model' | 'wrong_mode' | 'no_image_input' | 'no_model';
  message: string;
};

/**
 * The model to generate with. `model` is absent when the server's loaded model should be used
 * as it is: it is ready, no recipe names it (a model loaded from the command line), and
 * `librechat.yaml` restricts no models.
 */
export type SdcppModelChoice =
  | { ok: true; model?: SdcppModel }
  | { ok: false; error: SdcppModelChoiceError };

/** Whether `librechat.yaml` (`stableDiffusion.models`) limits which models may be used. */
export function restrictsSdcppModels(config?: SdcppModelsConfig): boolean {
  return config?.include != null || (config?.exclude?.length ?? 0) > 0;
}

/** Whether `librechat.yaml` allows the model (recipe ID) `id`. */
export function allowsSdcppModel(id: string, config?: SdcppModelsConfig): boolean {
  return (!config?.include || config.include.includes(id)) && !config?.exclude?.includes(id);
}

/** The catalog's models that `librechat.yaml` (`stableDiffusion.models`) allows. */
export function allowedSdcppModels(models: SdcppModel[], config?: SdcppModelsConfig): SdcppModel[] {
  return models.filter((model) => allowsSdcppModel(model.id, config));
}

export interface ChooseSdcppModelParams {
  catalog: SdcppCatalog;
  config?: SdcppModelsConfig;
  /** Model ID the tool was called with. */
  requested?: string;
  /** Whether the request starts from an input image (img2img). */
  withImage?: boolean;
  /**
   * Whether the calling model asked for a video clip (`true`) or an image (`false`). Left out,
   * a requested model's own mode applies, and without one an image is made.
   */
  video?: boolean;
  /** The server has a ready model that no recipe names, such as one loaded from the command line. */
  readyOutsideCatalog?: boolean;
}

const fits = (model: SdcppModel, mode: SdcppModelMode, withImage: boolean) =>
  model.mode === mode && (!withImage || model.acceptsImage);

function wrongMode(requested: string, mode: SdcppModelMode): SdcppModelChoice {
  return {
    ok: false,
    error: {
      code: 'wrong_mode',
      message:
        mode === 'video'
          ? `Model "${requested}" makes video, not images. Choose an image model, or set "video": true for a clip.`
          : `Model "${requested}" makes images, not video. Choose a video model, or omit "video" for an image.`,
    },
  };
}

/**
 * Chooses the model for a request, an image or a video clip. A requested model must be allowed
 * and able to serve the request. Without one, the loaded model is kept when it can serve it, so no
 * load is needed; otherwise, and when no model is ready at all, the default for the mode
 * (`defaultImage`, `defaultVideo`) is used, then the first allowed model that fits.
 */
export function chooseSdcppModel({
  catalog,
  config,
  requested,
  withImage = false,
  video,
  readyOutsideCatalog = false,
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
    if (video != null && model.mode !== (video ? 'video' : 'image')) {
      return wrongMode(requested, model.mode);
    }
    if (!fits(model, model.mode, withImage)) {
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

  const mode: SdcppModelMode = video ? 'video' : 'image';
  const configured = mode === 'video' ? config?.defaultVideo : config?.defaultImage;

  const loaded = catalog.loaded
    ? catalog.models.find((model) => model.id === catalog.loaded)
    : undefined;
  if (loaded && allowed.includes(loaded) && fits(loaded, mode, withImage)) {
    return { ok: true, model: loaded };
  }
  /** A model outside the catalog cannot be told to make video, so it serves images only. */
  if (mode === 'image' && !catalog.loaded && readyOutsideCatalog && !restrictsSdcppModels(config)) {
    return { ok: true };
  }

  const fallback =
    allowed.find((model) => model.id === configured && fits(model, mode, withImage)) ??
    allowed.find((model) => fits(model, mode, withImage));
  if (fallback) {
    return { ok: true, model: fallback };
  }
  let message = 'No available model can generate images.';
  if (mode === 'video') {
    message = 'No available model can make video.';
  } else if (withImage) {
    message = 'No available model can change an existing image.';
  }
  return { ok: false, error: { code: 'no_model', message } };
}
