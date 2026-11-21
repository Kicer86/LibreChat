import sharp from 'sharp';
import { logger } from '@librechat/data-schemas';
import type { Metadata } from 'sharp';
import type { ImageInput } from './images';

/** Which frame of an animated image (a clip) stands in for it. */
export type StillFrame = 'first' | 'last';

export interface StillImageOptions {
  /** Most pixels the chosen frame may have; Sharp's default limit when left out. */
  limitInputPixels?: number;
}

/**
 * The input image as one still frame, with its size read from the image. Image servers take
 * stills only (sd-server refuses an animated WebP), so a clip, such as a `generated_video_id`,
 * gives its first or last frame as a PNG; a still image is passed on as it is. An uploaded
 * animation usually arrives as a still already: uploads are converted to `imageOutputType`, which
 * keeps only the first frame unless the file is already in that format.
 *
 * Only the header is read to tell a clip from a still, so a long clip is not held to a pixel
 * limit meant for one image; the chosen frame alone is decoded, within the limit. Bytes that are
 * not an image Sharp can read are passed on unchanged, for the image server to accept or refuse.
 * `undefined` when the input is a clip but its frame cannot be taken, e.g. one above the limit.
 */
export async function toStillImage(
  image: ImageInput,
  frame: StillFrame,
  { limitInputPixels }: StillImageOptions = {},
): Promise<ImageInput | undefined> {
  let metadata: Metadata;
  try {
    metadata = await sharp(image.buffer, { limitInputPixels: false }).metadata();
  } catch (error) {
    logger.warn(`[toStillImage] Image ${image.file.file_id} could not be read`, error);
    return image;
  }
  const size = { width: metadata.width, height: metadata.height };
  const pages = metadata.pages ?? 1;
  if (pages <= 1) {
    return { ...image, file: { ...image.file, ...size } };
  }
  try {
    const page = frame === 'first' ? 0 : pages - 1;
    const buffer = await sharp(image.buffer, {
      page,
      ...(limitInputPixels != null && { limitInputPixels }),
    })
      .png()
      .toBuffer();
    return { file: { ...image.file, ...size }, buffer, mimeType: 'image/png' };
  } catch (error) {
    logger.warn(
      `[toStillImage] Could not take the ${frame} frame of clip ${image.file.file_id}`,
      error,
    );
    return undefined;
  }
}
