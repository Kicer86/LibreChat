import type { TStableDiffusionConfig } from 'librechat-data-provider';
import type { SdcppCatalog, SdcppModel, SdcppModelMode } from './catalog';

type SdcppModelsConfig = NonNullable<TStableDiffusionConfig['models']>;

/** Why no model could be chosen; the message is written for the model calling the tool. */
export type SdcppModelChoiceError = {
  code: 'unknown_model' | 'wrong_mode' | 'no_image_input' | 'no_end_image_input' | 'no_model';
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
  /** Whether the request starts from an input image (img2img, or a clip's first frame). */
  withImage?: boolean;
  /** Whether the clip ends on an input image (its last frame); only video models can. */
  withEndImage?: boolean;
  /**
   * Whether the calling model asked for a video clip (`true`) or an image (`false`). Left out,
   * a requested model's own mode applies, and without one an image is made.
   */
  video?: boolean;
  /** The server has a ready model that no recipe names, such as one loaded from the command line. */
  readyOutsideCatalog?: boolean;
}

/** The input images a request brings. */
interface ImageNeeds {
  start: boolean;
  end: boolean;
}

const fits = (model: SdcppModel, mode: SdcppModelMode, needs: ImageNeeds) =>
  model.mode === mode &&
  (!needs.start || model.acceptsImage) &&
  (!needs.end || model.acceptsEndImage === true);

/** Why the requested `model` cannot take the request's images, or `undefined` when it can. */
function imageInputError(model: SdcppModel, needs: ImageNeeds): SdcppModelChoiceError | undefined {
  if (needs.start && !model.acceptsImage) {
    return {
      code: 'no_image_input',
      message:
        model.mode === 'video'
          ? `Model "${model.id}" cannot start a clip from an image. Choose a video model that starts from an image, or omit "image_ids".`
          : `Model "${model.id}" cannot change an existing image. Choose a model that supports img2img.`,
    };
  }
  if (needs.end && model.acceptsEndImage !== true) {
    return {
      code: 'no_end_image_input',
      message: `Model "${model.id}" cannot end a clip on an image. Choose a video model that ends on an image, or omit "end_image_id".`,
    };
  }
  return undefined;
}

/** Why no allowed model can serve the request. */
function noModelMessage(mode: SdcppModelMode, needs: ImageNeeds): string {
  if (mode === 'image') {
    return needs.start
      ? 'No available model can change an existing image.'
      : 'No available model can generate images.';
  }
  if (needs.start && needs.end) {
    return 'No available video model can start a clip from an image and end it on another.';
  }
  if (needs.start) {
    return 'No available video model can start a clip from an image.';
  }
  return needs.end
    ? 'No available video model can end a clip on an image.'
    : 'No available model can make video.';
}

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
 * and able to serve the request, taking its start and end images. Without one, the loaded model is kept when it can serve it, so no
 * load is needed; otherwise, and when no model is ready at all, the default for the mode
 * (`defaultImage`, `defaultVideo`) is used, then the first allowed model that fits.
 */
export function chooseSdcppModel({
  catalog,
  config,
  requested,
  withImage = false,
  withEndImage = false,
  video,
  readyOutsideCatalog = false,
}: ChooseSdcppModelParams): SdcppModelChoice {
  const allowed = allowedSdcppModels(catalog.models, config);
  const needs: ImageNeeds = { start: withImage, end: withEndImage };

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
    const error = imageInputError(model, needs);
    return error ? { ok: false, error } : { ok: true, model };
  }

  const mode: SdcppModelMode = video || withEndImage ? 'video' : 'image';
  const configured = mode === 'video' ? config?.defaultVideo : config?.defaultImage;

  const loaded = catalog.loaded
    ? catalog.models.find((model) => model.id === catalog.loaded)
    : undefined;
  if (loaded && allowed.includes(loaded) && fits(loaded, mode, needs)) {
    return { ok: true, model: loaded };
  }
  /** A model outside the catalog cannot be told to make video, so it serves images only. */
  if (mode === 'image' && !catalog.loaded && readyOutsideCatalog && !restrictsSdcppModels(config)) {
    return { ok: true };
  }

  const fallback =
    allowed.find((model) => model.id === configured && fits(model, mode, needs)) ??
    allowed.find((model) => fits(model, mode, needs));
  if (fallback) {
    return { ok: true, model: fallback };
  }
  return { ok: false, error: { code: 'no_model', message: noModelMessage(mode, needs) } };
}
