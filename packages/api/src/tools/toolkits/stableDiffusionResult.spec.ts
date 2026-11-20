import type { ServerRequest } from '~/types';
import {
  readStableDiffusionSettings,
  formatStableDiffusionSettings,
  buildStableDiffusionAgentOutput,
} from './stableDiffusionResult';
import { createGeneratedImageRegistry } from './images';

/** Trimmed from a live stable-diffusion.cpp `/sdapi/v1/txt2img` response (500x300 asked for). */
const sdcppInfo = JSON.stringify({
  all_seeds: [7, 8],
  cfg_scale: 1.0,
  extra_generation_params: { 'Schedule type': 'flux' },
  generation_info: [
    { height: 320, sample_params: { sample_method: 'euler', scheduler: 'flux' }, seed: '7' },
    { height: 320, sample_params: { sample_method: 'euler', scheduler: 'flux' }, seed: '8' },
  ],
  height: 320,
  sampler_name: 'euler',
  seed: 7,
  steps: 4,
  version: 'stable-diffusion.cpp',
  width: 512,
});

/** The shape AUTOMATIC1111 reports. */
const webuiInfo = JSON.stringify({
  seed: 1234,
  all_seeds: [1234],
  width: 1024,
  height: 1024,
  steps: 22,
  cfg_scale: 4.5,
  sampler_name: 'DPM++ 2M',
  extra_generation_params: { 'Schedule type': 'Karras' },
});

describe('readStableDiffusionSettings', () => {
  it('reads what stable-diffusion.cpp rendered, one seed per image', () => {
    expect(readStableDiffusionSettings(sdcppInfo)).toEqual({
      seeds: [7, 8],
      width: 512,
      height: 320,
      steps: 4,
      cfg_scale: 1,
      sampler: 'euler',
      scheduler: 'flux',
    });
  });

  it("reads a WebUI server's info and its schedule type", () => {
    expect(readStableDiffusionSettings(webuiInfo)).toEqual({
      seeds: [1234],
      width: 1024,
      height: 1024,
      steps: 22,
      cfg_scale: 4.5,
      sampler: 'DPM++ 2M',
      scheduler: 'Karras',
    });
  });

  it('falls back to the first seed when the seed list cannot be read', () => {
    expect(readStableDiffusionSettings({ seed: 5, all_seeds: [5, 'x'] }).seeds).toEqual([5]);
  });

  it.each([[undefined], ['not json'], ['[]'], [42]])('gives no settings for %j', (info) => {
    expect(readStableDiffusionSettings(info)).toEqual({ seeds: [] });
  });
});

describe('formatStableDiffusionSettings', () => {
  it("names the shared settings as the tool's arguments", () => {
    expect(formatStableDiffusionSettings(readStableDiffusionSettings(sdcppInfo))).toBe(
      'width: 512, height: 320, steps: 4, cfg_scale: 1, sampler: euler, scheduler: flux',
    );
  });

  it('is empty when nothing was reported', () => {
    expect(formatStableDiffusionSettings({ seeds: [] })).toBe('');
  });
});

describe('buildStableDiffusionAgentOutput', () => {
  const req = {} as ServerRequest;

  it('names each image with its seed, then the model and the settings', () => {
    const generatedImages = createGeneratedImageRegistry();
    let next = 0;
    const [response, artifact] = buildStableDiffusionAgentOutput({
      data: { images: ['Zmlyc3Q=', 'c2Vjb25k'], info: sdcppInfo },
      model: 'qwen',
      displayMessage: 'Shown.',
      createFileId: () => `file-${++next}`,
      req,
      generatedImages,
    });

    expect(response).toEqual([
      {
        type: 'text',
        text: [
          'Shown.',
          '',
          'generated_image_id: "file-1" (seed: 7)',
          'generated_image_id: "file-2" (seed: 8)',
          'model: "qwen"',
          'settings: width: 512, height: 320, steps: 4, cfg_scale: 1, sampler: euler, scheduler: flux',
        ].join('\n'),
      },
    ]);
    expect(artifact).toEqual({
      content: [
        { type: 'image_url', image_url: { url: 'data:image/png;base64,Zmlyc3Q=' } },
        { type: 'image_url', image_url: { url: 'data:image/png;base64,c2Vjb25k' } },
      ],
      file_ids: ['file-1', 'file-2'],
    });
    expect(generatedImages.find(req, 'file-2')).toEqual({
      file: { file_id: 'file-2', width: 512, height: 320 },
      buffer: Buffer.from('second'),
      mimeType: 'image/png',
    });
  });

  it('names the image alone when the server reports no settings', () => {
    const [response] = buildStableDiffusionAgentOutput({
      data: { images: ['aW1hZ2U='] },
      displayMessage: 'Shown.',
      createFileId: () => 'file-1',
    });
    expect(response[0].text).toBe('Shown.\n\ngenerated_image_id: "file-1"');
  });
});
