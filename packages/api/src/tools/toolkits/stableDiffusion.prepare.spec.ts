import axios from 'axios';
import type { TFile } from 'librechat-data-provider';
import type { ServerRequest } from '~/types';
import {
  prepareStableDiffusionRequest,
  clearStableDiffusionBackendCache,
  STABLE_DIFFUSION_IMAGE_NOT_LOADED,
} from './stableDiffusion';
import { applySSRFSafeAgentIfDirect } from '~/auth/agent';
import { createGeneratedImageRegistry } from './images';

jest.mock('axios', () => ({ get: jest.fn() }));
jest.mock('~/auth/agent', () => ({
  applySSRFSafeAgentIfDirect: jest.fn((config: object) => ({ ...config, guarded: true })),
}));

const mockGet = axios.get as jest.Mock;
const url = 'http://sd.local:7860';
const req = {
  user: { id: 'user-1' },
  config: { timeouts: { toolsMs: 1234, imageGenerationMs: 5678 } },
} as unknown as ServerRequest;
const sdcpp = { status: 200, data: { current_mode: 'img_gen', defaults: {} } };

describe('prepareStableDiffusionRequest', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    clearStableDiffusionBackendCache();
    mockGet.mockResolvedValue(sdcpp);
  });

  it('probes the backend within the tools budget and builds the generation request', async () => {
    const request = await prepareStableDiffusionRequest({ prompt: 'a cat' }, { url, req });

    expect(mockGet).toHaveBeenCalledWith(
      `${url}/sdcpp/v1/capabilities`,
      expect.objectContaining({ timeout: 1234, validateStatus: expect.any(Function) }),
    );
    expect(request).toEqual({
      url: `${url}/sdapi/v1/txt2img`,
      payload: { prompt: 'a cat' },
      config: expect.objectContaining({ timeout: 5678 }),
    });
    expect(applySSRFSafeAgentIfDirect).not.toHaveBeenCalled();
  });

  it('guards the probe and the generation of a user-provided endpoint', async () => {
    const request = await prepareStableDiffusionRequest(
      { prompt: 'a cat' },
      { url, req, isUserProvidedEndpoint: true },
    );

    expect(applySSRFSafeAgentIfDirect).toHaveBeenCalledWith(
      expect.any(Object),
      `${url}/sdcpp/v1/capabilities`,
    );
    expect(applySSRFSafeAgentIfDirect).toHaveBeenCalledWith({}, `${url}/sdapi/v1/txt2img`);
    expect(mockGet.mock.calls[0][1]).toMatchObject({ guarded: true });
    expect(request).toMatchObject({ config: { guarded: true, timeout: 5678 } });
  });

  it('changes an image generated earlier in the request', async () => {
    const generatedImages = createGeneratedImageRegistry();
    const buffer = Buffer.from('generated');
    generatedImages.remember(req, {
      file: { file_id: 'gen-1', width: 1600, height: 900 },
      buffer,
      mimeType: 'image/png',
    });

    const request = await prepareStableDiffusionRequest(
      { prompt: 'red', image_ids: ['gen-1'], denoising_strength: 0.7 },
      {
        url,
        req,
        generatedImages,
        findImageFiles: jest.fn(async () => [] as TFile[]),
        getStrategyFunctions: jest.fn(() => ({})),
      },
    );

    expect(request).toMatchObject({
      url: `${url}/sdapi/v1/img2img`,
      payload: {
        init_images: [buffer.toString('base64')],
        denoising_strength: 0.7,
        width: 1024,
        height: 576,
      },
    });
  });

  it.each([
    [
      'an unknown image',
      { findImageFiles: jest.fn(async () => []), getStrategyFunctions: jest.fn(() => ({})) },
    ],
    ['no storage access', {}],
  ])('reports %s without probing or generating', async (_case, imageContext) => {
    const request = await prepareStableDiffusionRequest(
      { prompt: 'p', image_ids: ['nope'] },
      { url, req, ...imageContext },
    );
    expect(request).toEqual({ error: STABLE_DIFFUSION_IMAGE_NOT_LOADED });
    expect(mockGet).not.toHaveBeenCalled();
  });
});
