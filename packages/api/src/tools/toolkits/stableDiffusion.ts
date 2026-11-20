import type { ImageInput } from './images';

/** Longest side of a generated image (txt2img renders a square of this size). */
const IMAGE_SIZE = 1024;
/** img2img dimensions are rounded to this multiple, which every common model accepts. */
const SIZE_MULTIPLE = 64;

const GENERATION_DEFAULTS = {
  cfg_scale: 4.5,
  steps: 22,
  width: IMAGE_SIZE,
  height: IMAGE_SIZE,
};

export interface StableDiffusionRequestParams {
  prompt: string;
  negative_prompt?: string;
  /** The starting image; switches the request from txt2img to img2img. */
  image?: ImageInput;
  /** How much img2img may change the image (0–1); omitted, the server default applies. */
  denoising_strength?: number;
}

export interface StableDiffusionRequest {
  /** Path on the Stable Diffusion WebUI-compatible API (`/sdapi/v1/...`). */
  path: '/sdapi/v1/txt2img' | '/sdapi/v1/img2img';
  payload: Record<string, unknown>;
}

/**
 * Scales an image's size so its longest side is `IMAGE_SIZE`, keeping the aspect ratio, with both
 * sides rounded to `SIZE_MULTIPLE`. img2img renders at this size, so the result keeps the shape of
 * the input instead of being squeezed into the txt2img square.
 */
export function fitStableDiffusionSize(
  width: number,
  height: number,
): { width: number; height: number } {
  const scale = IMAGE_SIZE / Math.max(width, height);
  const fit = (side: number) =>
    Math.max(SIZE_MULTIPLE, Math.round((side * scale) / SIZE_MULTIPLE) * SIZE_MULTIPLE);
  return { width: fit(width), height: fit(height) };
}

/**
 * Builds a Stable Diffusion WebUI API request (`txt2img`, or `img2img` when an image is given).
 * The same API is served by AUTOMATIC1111, Forge and stable-diffusion.cpp's `sd-server`.
 */
export function buildStableDiffusionRequest({
  prompt,
  negative_prompt,
  image,
  denoising_strength,
}: StableDiffusionRequestParams): StableDiffusionRequest {
  const payload: Record<string, unknown> = { prompt, negative_prompt, ...GENERATION_DEFAULTS };
  if (!image) {
    return { path: '/sdapi/v1/txt2img', payload };
  }

  const { width, height } = image.file;
  if (width && height) {
    Object.assign(payload, fitStableDiffusionSize(width, height));
  }
  payload.init_images = [image.buffer.toString('base64')];
  if (denoising_strength != null) {
    payload.denoising_strength = denoising_strength;
  }
  return { path: '/sdapi/v1/img2img', payload };
}
