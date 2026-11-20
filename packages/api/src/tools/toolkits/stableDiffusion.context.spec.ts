import axios from 'axios';
import type { AppConfig } from '@librechat/data-schemas';
import { buildStableDiffusionToolContext } from './stableDiffusion';
import { applySSRFSafeAgentIfDirect } from '~/auth/agent';
import { clearSdcppCatalogCache } from './sdcpp';

jest.mock('axios', () => ({ get: jest.fn() }));
jest.mock('~/auth/agent', () => ({
  applySSRFSafeAgentIfDirect: jest.fn((config: object) => ({ ...config, guarded: true })),
}));

const mockGet = axios.get as jest.Mock;
const url = 'http://sd.local:7860';
const appConfig = { timeouts: { imageModelsMs: 1500 } } as unknown as AppConfig;
const recipes = {
  recipes: [
    {
      id: 'qwen',
      name: 'Qwen Image',
      models: { diffusion_model: 'qwen.gguf' },
      effective_features: { img2img: true },
    },
  ],
};

describe('buildStableDiffusionToolContext', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    clearSdcppCatalogCache();
    mockGet.mockImplementation(async (requestUrl: string) =>
      requestUrl.endsWith('/recipes')
        ? { status: 200, data: recipes }
        : { status: 200, data: { recipe: 'qwen', models: {} } },
    );
  });

  it('lists the models within the imageModelsMs budget, then the attached images', async () => {
    const context = await buildStableDiffusionToolContext({
      url,
      appConfig,
      imageFiles: [{ file_id: 'img-1' }],
    });

    expect(mockGet).toHaveBeenCalledWith(
      `${url}/sdcpp/v1/recipes`,
      expect.objectContaining({ timeout: 1500 }),
    );
    expect(context).toMatch(
      /^Image models available[^]*- qwen \(loaded\)[^]*\n\nImage files[^]*img-1/,
    );
    expect(applySSRFSafeAgentIfDirect).not.toHaveBeenCalled();
  });

  it('guards a user-provided server', async () => {
    await buildStableDiffusionToolContext({ url, isUserProvidedEndpoint: true });
    expect(applySSRFSafeAgentIfDirect).toHaveBeenCalledWith(
      expect.any(Object),
      `${url}/sdcpp/v1/recipes`,
    );
    expect(mockGet.mock.calls[0][1]).toMatchObject({ guarded: true });
  });

  it('lists only the attached images when the server is unreachable or not sd.cpp', async () => {
    mockGet.mockRejectedValue(new Error('ECONNREFUSED'));
    await expect(
      buildStableDiffusionToolContext({ url, imageFiles: [{ file_id: 'img-1' }] }),
    ).resolves.toMatch(/^Image files/);
    await expect(buildStableDiffusionToolContext({ url })).resolves.toBe('');
  });

  it('does not reach any server without a URL', async () => {
    await expect(buildStableDiffusionToolContext({})).resolves.toBe('');
    expect(mockGet).not.toHaveBeenCalled();
  });
});
