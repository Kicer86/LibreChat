import sharp from 'sharp';
import { logger } from '@librechat/data-schemas';
import type { ImageInput } from './images';

/** Which frame of an animated image (a clip) stands in for it. */
export type StillFrame = 'first' | 'last';

/**
 * The input image as one still frame, with its size read from the image. Image servers take
 * stills only (sd-server refuses an animated WebP), so a clip, such as a `generated_video_id` or
 * an attached animated WebP or GIF, gives its first or last frame as a PNG; a still image is
 * passed on as it is. Bytes that cannot be read as an image are passed on unchanged too, for the
 * image server to accept or refuse.
 */
export async function toStillImage(image: ImageInput, frame: StillFrame): Promise<ImageInput> {
  try {
    const metadata = await sharp(image.buffer, { animated: true }).metadata();
    const pages = metadata.pages ?? 1;
    const size = { width: metadata.width, height: metadata.pageHeight ?? metadata.height };
    if (pages <= 1) {
      return { ...image, file: { ...image.file, ...size } };
    }
    const page = frame === 'first' ? 0 : pages - 1;
    return {
      file: { ...image.file, ...size },
      buffer: await sharp(image.buffer, { page }).png().toBuffer(),
      mimeType: 'image/png',
    };
  } catch (error) {
    logger.warn(`[toStillImage] Image ${image.file.file_id} could not be read`, error);
    return image;
  }
}
