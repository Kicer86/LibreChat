import { ContentTypes } from 'librechat-data-provider';
import type { StableDiffusionResponse } from './stableDiffusion';
import type { GeneratedImageRegistry } from './images';
import type { ServerRequest } from '~/types';

/** The settings a generation used, as the server's `info` reports them. */
export interface StableDiffusionUsedSettings {
  /** One seed per image, in the order of the images. */
  seeds: number[];
  width?: number;
  height?: number;
  steps?: number;
  cfg_scale?: number;
  sampler?: string;
  scheduler?: string;
}

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  value != null && typeof value === 'object' && !Array.isArray(value);

const finiteNumber = (value: unknown): number | undefined => {
  const number = typeof value === 'string' && value.trim() !== '' ? Number(value) : value;
  return typeof number === 'number' && Number.isFinite(number) ? number : undefined;
};

const nonEmptyString = (value: unknown): string | undefined =>
  typeof value === 'string' && value.length > 0 ? value : undefined;

/**
 * The settings in a WebUI API generation's `info` (JSON text). AUTOMATIC1111 and
 * stable-diffusion.cpp both report the seeds, the size actually rendered (sd.cpp rounds it up to
 * the model's multiple), steps, CFG scale and sampler; the scheduler comes from sd.cpp's
 * `generation_info` or from the WebUI's "Schedule type". Unreadable `info` gives no settings.
 */
export function readStableDiffusionSettings(info: unknown): StableDiffusionUsedSettings {
  let data: unknown;
  try {
    data = typeof info === 'string' ? JSON.parse(info) : info;
  } catch {
    return { seeds: [] };
  }
  if (!isPlainObject(data)) {
    return { seeds: [] };
  }

  const listed = Array.isArray(data.all_seeds) ? data.all_seeds.map(finiteNumber) : [];
  const allSeeds = listed.filter((value): value is number => value != null);
  const seed = finiteNumber(data.seed);
  /** A list with an unreadable seed would pair the others with the wrong images. */
  const listReadable = allSeeds.length > 0 && allSeeds.length === listed.length;
  const firstSeed = seed != null ? [seed] : [];
  const seeds = listReadable ? allSeeds : firstSeed;
  const [first] = Array.isArray(data.generation_info) ? data.generation_info : [];
  const sampleParams =
    isPlainObject(first) && isPlainObject(first.sample_params) ? first.sample_params : {};
  const extra = isPlainObject(data.extra_generation_params) ? data.extra_generation_params : {};

  const width = finiteNumber(data.width);
  const height = finiteNumber(data.height);
  const steps = finiteNumber(data.steps);
  const cfgScale = finiteNumber(data.cfg_scale);
  const sampler = nonEmptyString(data.sampler_name);
  const scheduler =
    nonEmptyString(sampleParams.scheduler) ?? nonEmptyString(extra['Schedule type']);
  return {
    seeds,
    ...(width != null && { width }),
    ...(height != null && { height }),
    ...(steps != null && { steps }),
    ...(cfgScale != null && { cfg_scale: cfgScale }),
    ...(sampler && { sampler }),
    ...(scheduler && { scheduler }),
  };
}

/**
 * The settings the images share, named as the tool's arguments so the calling model can pass
 * them back to make an image again or a variation of it, e.g.
 * `width: 512, height: 320, steps: 4, cfg_scale: 1, sampler: euler_a`. The seeds, which differ
 * per image, are listed with each image. Empty when the server reported none.
 */
export function formatStableDiffusionSettings(settings: StableDiffusionUsedSettings): string {
  const entries: Array<[string, string | number | undefined]> = [
    ['width', settings.width],
    ['height', settings.height],
    ['steps', settings.steps],
    ['cfg_scale', settings.cfg_scale],
    ['sampler', settings.sampler],
    ['scheduler', settings.scheduler],
  ];
  return entries
    .filter(([, value]) => value != null)
    .map(([name, value]) => `${name}: ${value}`)
    .join(', ');
}

export interface StableDiffusionAgentOutputParams {
  data: StableDiffusionResponse;
  /** The stable-diffusion.cpp model that generated the images, when known. */
  model?: string;
  /** Text shown first, telling the calling model the images are already displayed. */
  displayMessage: string;
  /** Creates the ID each image's file is saved under. */
  createFileId: () => string;
  req?: ServerRequest;
  generatedImages?: GeneratedImageRegistry;
}

type TextPart = { type: ContentTypes.TEXT; text: string };
type ImagePart = { type: ContentTypes.IMAGE_URL; image_url: { url: string } };

/**
 * The agent tool result for a generation: the text for the calling model and the artifact that
 * displays and saves the images. Every image gets the file ID it will be saved under, listed with
 * its seed as a `generated_image_id` the model can pass back in `image_ids`, and is remembered for
 * this request so img2img works before the file is saved. The settings the server reports follow,
 * so the model can repeat or vary the result.
 */
export function buildStableDiffusionAgentOutput({
  data,
  model,
  displayMessage,
  createFileId,
  req,
  generatedImages,
}: StableDiffusionAgentOutputParams): [TextPart[], { content: ImagePart[]; file_ids: string[] }] {
  const settings = readStableDiffusionSettings(data.info);
  const content: ImagePart[] = [];
  const fileIds: string[] = [];
  const lines: string[] = [];
  data.images.forEach((image, index) => {
    const fileId = createFileId();
    fileIds.push(fileId);
    content.push({
      type: ContentTypes.IMAGE_URL,
      image_url: { url: `data:image/png;base64,${image}` },
    });
    generatedImages?.remember(req, {
      file: { file_id: fileId, width: settings.width, height: settings.height },
      buffer: Buffer.from(image, 'base64'),
      mimeType: 'image/png',
    });
    const seed = settings.seeds[index];
    lines.push(`generated_image_id: "${fileId}"${seed != null ? ` (seed: ${seed})` : ''}`);
  });

  if (model) {
    lines.push(`model: "${model}"`);
  }
  const shared = formatStableDiffusionSettings(settings);
  if (shared) {
    lines.push(`settings: ${shared}`);
  }
  const text = `${displayMessage}\n\n${lines.join('\n')}`;
  return [[{ type: ContentTypes.TEXT, text }], { content, file_ids: fileIds }];
}
