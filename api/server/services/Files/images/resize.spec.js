const sharp = require('sharp');
const { resizeImageBuffer } = require('./resize');

/** An animated WebP of `count` frames, each `width` x `height`. */
async function animatedWebp(width, height, count) {
  const frames = await Promise.all(
    Array.from({ length: count }, (_, index) =>
      sharp({ create: { width, height, channels: 3, background: { r: index * 60, g: 0, b: 0 } } })
        .png()
        .toBuffer(),
    ),
  );
  return sharp(frames, { join: { animated: true } })
    .webp({ delay: Array(count).fill(42), loop: 0 })
    .toBuffer();
}

describe('resizeImageBuffer', () => {
  it('keeps every frame of an animated image, sized per frame', async () => {
    const clip = await animatedWebp(1024, 640, 3);

    const resized = await resizeImageBuffer(clip, { percentage: 50 });

    expect(resized).toMatchObject({ width: 512, height: 320, type: 'image/webp' });
    const metadata = await sharp(resized.buffer).metadata();
    expect(metadata.pages).toBe(3);
    expect(metadata.delay).toEqual([42, 42, 42]);
  });

  it('leaves an animated image that already fits at its size', async () => {
    const clip = await animatedWebp(512, 320, 2);

    const resized = await resizeImageBuffer(clip, 'high');

    expect(resized).toMatchObject({ width: 512, height: 320 });
    expect((await sharp(resized.buffer).metadata()).pages).toBe(2);
  });

  it('still resizes a still image', async () => {
    const image = await sharp({
      create: { width: 2048, height: 2048, channels: 3, background: 'red' },
    })
      .png()
      .toBuffer();

    const resized = await resizeImageBuffer(image, 'high');

    expect(resized).toMatchObject({ width: 768, height: 768, type: 'image/png' });
    expect((await sharp(resized.buffer).metadata()).pages).toBeUndefined();
  });
});
