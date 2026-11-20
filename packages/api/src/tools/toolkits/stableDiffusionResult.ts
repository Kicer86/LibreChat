import { ContentTypes } from 'librechat-data-provider';
import type { StableDiffusionResponse } from './stableDiffusion';
import type { GeneratedImageRegistry } from './images';
import type { ServerRequest } from '~/types';
import type { SdcppVideo } from './sdcpp';

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
  /** A video clip's frames and frame rate. */
  frames?: number;
  fps?: number;
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
 * The settings of a stable-diffusion.cpp clip, from its `generation_info`: the size, frames, frame
 * rate and seed, and the sampling settings nested under `sample_params`.
 */
export function readSdcppVideoSettings(video: SdcppVideo): StableDiffusionUsedSettings {
  const { info } = video;
  const sampleParams = isPlainObject(info.sample_params) ? info.sample_params : {};
  const guidance = isPlainObject(sampleParams.guidance) ? sampleParams.guidance : {};
  const seed = finiteNumber(info.seed);
  const width = finiteNumber(info.width);
  const height = finiteNumber(info.height);
  const frames = finiteNumber(info.video_frames) ?? video.frames;
  const fps = finiteNumber(info.fps) ?? video.fps;
  const steps = finiteNumber(sampleParams.sample_steps);
  const cfgScale = finiteNumber(guidance.txt_cfg);
  const sampler = nonEmptyString(sampleParams.sample_method);
  const scheduler = nonEmptyString(sampleParams.scheduler);
  return {
    seeds: seed != null ? [seed] : [],
    ...(width != null && { width }),
    ...(height != null && { height }),
    ...(frames != null && { frames }),
    ...(fps != null && { fps }),
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
    ['frames', settings.frames],
    ['fps', settings.fps],
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

/** Shown to the calling model first when a clip was made. */
export const STABLE_DIFFUSION_VIDEO_DISPLAYED =
  "Stable Diffusion displayed a video clip. It is already plainly visible and plays in the chat, so don't describe it in detail and do not list download links.";

export interface StableDiffusionAgentOutputParams {
  /** The images a generation made, or the clip. */
  generation: { data: StableDiffusionResponse } | { video: SdcppVideo };
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

type AgentOutput = [TextPart[], { content: ImagePart[]; file_ids: string[] }];

/** The text for the calling model: the generated IDs, then the model and the shared settings. */
function agentText(
  displayMessage: string,
  lines: string[],
  model: string | undefined,
  settings: StableDiffusionUsedSettings,
): TextPart[] {
  if (model) {
    lines.push(`model: "${model}"`);
  }
  const shared = formatStableDiffusionSettings(settings);
  if (shared) {
    lines.push(`settings: ${shared}`);
  }
  return [{ type: ContentTypes.TEXT, text: `${displayMessage}\n\n${lines.join('\n')}` }];
}

/**
 * The agent tool result for a clip: the animated WebP as the artifact that displays and saves it,
 * under the file ID listed as its `generated_video_id`, remembered for this request like an
 * image, and the settings it was made with.
 */
function buildVideoOutput(
  video: SdcppVideo,
  {
    model,
    createFileId,
    req,
    generatedImages,
  }: Omit<StableDiffusionAgentOutputParams, 'generation'>,
): AgentOutput {
  const settings = readSdcppVideoSettings(video);
  const fileId = createFileId();
  generatedImages?.remember(req, {
    file: { file_id: fileId, width: settings.width, height: settings.height },
    buffer: Buffer.from(video.base64, 'base64'),
    mimeType: video.mimeType,
  });
  const seed = settings.seeds[0];
  const lines = [`generated_video_id: "${fileId}"${seed != null ? ` (seed: ${seed})` : ''}`];
  return [
    agentText(STABLE_DIFFUSION_VIDEO_DISPLAYED, lines, model, settings),
    {
      content: [
        {
          type: ContentTypes.IMAGE_URL,
          image_url: { url: `data:${video.mimeType};base64,${video.base64}` },
        },
      ],
      file_ids: [fileId],
    },
  ];
}

/**
 * The agent tool result for a generation: the text for the calling model and the artifact that
 * displays and saves the images. Every image gets the file ID it will be saved under, listed with
 * its seed as a `generated_image_id` the model can pass back in `image_ids`, and is remembered for
 * this request so img2img works before the file is saved. The settings the server reports follow,
 * so the model can repeat or vary the result. A clip is returned the same way, as one animated
 * image with a `generated_video_id`.
 */
export function buildStableDiffusionAgentOutput({
  generation,
  ...params
}: StableDiffusionAgentOutputParams): AgentOutput {
  if ('video' in generation) {
    return buildVideoOutput(generation.video, params);
  }
  const { data } = generation;
  const { model, displayMessage, createFileId, req, generatedImages } = params;
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

  return [agentText(displayMessage, lines, model, settings), { content, file_ids: fileIds }];
}
