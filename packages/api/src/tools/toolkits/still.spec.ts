import sharp from 'sharp';
import type { ImageInput } from './images';
import { toStillImage } from './still';

const COLORS = ['#ff0000', '#00ff00', '#0000ff'];

const frame = (background: string) =>
  sharp({ create: { width: 6, height: 4, channels: 3, background } })
    .png()
    .toBuffer();

const input = (buffer: Buffer, mimeType: string): ImageInput => ({
  file: { file_id: 'clip-1' },
  buffer,
  mimeType,
});

/** The dominant channel of the image's first pixel: 0 red, 1 green, 2 blue. */
async function dominantChannel(buffer: Buffer): Promise<number> {
  const pixel = [...(await sharp(buffer).raw().toBuffer()).subarray(0, 3)];
  return pixel.indexOf(Math.max(...pixel));
}

describe('toStillImage', () => {
  let frames: Buffer[];

  beforeAll(async () => {
    frames = await Promise.all(COLORS.map(frame));
  });

  it.each([
    ['webp', 'image/webp'],
    ['gif', 'image/gif'],
  ] as const)('takes the first or last frame of an animated %s', async (format, mimeType) => {
    const clip = await sharp(frames, { join: { animated: true } })
      .toFormat(format)
      .toBuffer();

    const first = await toStillImage(input(clip, mimeType), 'first');
    const last = await toStillImage(input(clip, mimeType), 'last');

    for (const still of [first, last]) {
      expect(still).toMatchObject({
        file: { file_id: 'clip-1', width: 6, height: 4 },
        mimeType: 'image/png',
      });
      const metadata = await sharp(still?.buffer).metadata();
      expect(metadata).toMatchObject({ format: 'png', width: 6, height: 4 });
      expect(metadata.pages ?? 1).toBe(1);
    }
    expect(await dominantChannel(first?.buffer as Buffer)).toBe(0);
    expect(await dominantChannel(last?.buffer as Buffer)).toBe(2);
  });

  it('takes one frame of a clip whose frames together exceed the pixel limit', async () => {
    const clip = await sharp(frames, { join: { animated: true } })
      .webp()
      .toBuffer();
    /** Each 6x4 frame fits within 30 pixels; the three together do not. */
    const options = { limitInputPixels: 30 };

    const first = await toStillImage(input(clip, 'image/webp'), 'first', options);
    const last = await toStillImage(input(clip, 'image/webp'), 'last', options);

    expect(first).toMatchObject({ file: { width: 6, height: 4 }, mimeType: 'image/png' });
    expect(last).toMatchObject({ file: { width: 6, height: 4 }, mimeType: 'image/png' });
    expect(await dominantChannel(first?.buffer as Buffer)).toBe(0);
    expect(await dominantChannel(last?.buffer as Buffer)).toBe(2);
  });

  it('gives nothing for a clip whose frame exceeds the pixel limit', async () => {
    const clip = await sharp(frames, { join: { animated: true } })
      .webp()
      .toBuffer();
    expect(
      await toStillImage(input(clip, 'image/webp'), 'last', { limitInputPixels: 20 }),
    ).toBeUndefined();
  });

  it('passes a still image on, with the size read from it', async () => {
    const image: ImageInput = {
      file: { file_id: 'photo', width: 1, height: 1 },
      buffer: frames[1],
      mimeType: 'image/png',
    };
    expect(await toStillImage(image, 'last')).toEqual({
      ...image,
      file: { file_id: 'photo', width: 6, height: 4 },
    });
  });

  it('passes bytes that are not an image on unchanged', async () => {
    const image = input(Buffer.from('not an image'), 'image/png');
    expect(await toStillImage(image, 'first')).toBe(image);
  });
});
