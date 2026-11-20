const axios = require('axios');

const mockPrepareStableDiffusionRequest = jest.fn();

jest.mock('axios', () => ({ post: jest.fn() }), { virtual: true });
jest.mock('sharp', () => jest.fn(), { virtual: true });
jest.mock('uuid', () => ({ v4: jest.fn() }), { virtual: true });
jest.mock('@librechat/data-schemas', () => ({
  logger: { error: jest.fn() },
}));
jest.mock(
  '@librechat/agents/langchain/tools',
  () => ({
    Tool: class {},
  }),
  { virtual: true },
);
jest.mock(
  'librechat-data-provider',
  () => ({
    ContentTypes: {},
    FileContext: {},
  }),
  { virtual: true },
);
jest.mock('@librechat/api', () => ({
  prepareStableDiffusionRequest: (...args) => mockPrepareStableDiffusionRequest(...args),
  getBasePath: jest.fn(),
}));
jest.mock('~/config/paths', () => ({}), { virtual: true });

const StableDiffusionAPI = require('../StableDiffusion');

describe('StableDiffusionAPI', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('prepares the request with the endpoint and the image context it was created with', async () => {
    mockPrepareStableDiffusionRequest.mockResolvedValue({ error: 'no image' });
    const imageFiles = [{ file_id: 'img-1' }];
    const findImageFiles = jest.fn();
    const tool = new StableDiffusionAPI({
      SD_WEBUI_URL: 'http://127.0.0.1:9000',
      userProvidedAuthFields: new Set(['SD_WEBUI_URL']),
      imageFiles,
      findImageFiles,
    });
    const input = { prompt: 'test prompt', image_ids: ['img-1'] };

    await tool._call(input);

    expect(mockPrepareStableDiffusionRequest).toHaveBeenCalledWith(
      input,
      expect.objectContaining({
        url: 'http://127.0.0.1:9000',
        isUserProvidedEndpoint: true,
        imageFiles,
        findImageFiles,
      }),
    );
  });

  it('returns the reason a request could not be prepared without calling the API', async () => {
    mockPrepareStableDiffusionRequest.mockResolvedValue({
      error: 'The image could not be loaded.',
    });
    const tool = new StableDiffusionAPI({ SD_WEBUI_URL: 'http://sd.local' });

    const result = await tool._call({ prompt: 'p', image_ids: ['nope'] });

    expect(result).toBe('The image could not be loaded.');
    expect(axios.post).not.toHaveBeenCalled();
  });

  it('reports a request blocked by the SSRF guard as an API error', async () => {
    mockPrepareStableDiffusionRequest.mockRejectedValue(
      new Error('SSRF protection: blocked address'),
    );
    const tool = new StableDiffusionAPI({
      SD_WEBUI_URL: 'http://127.0.0.1:9000',
      userProvidedAuthFields: new Set(['SD_WEBUI_URL']),
    });

    const result = await tool._call({ prompt: 'test prompt', negative_prompt: 'test negative' });

    expect(axios.post).not.toHaveBeenCalled();
    expect(result).toBe('Error making API request.');
  });

  it('sends the prepared request', async () => {
    const config = { timeout: 1000 };
    mockPrepareStableDiffusionRequest.mockResolvedValue({
      url: 'http://sd.local/sdapi/v1/txt2img',
      payload: { prompt: 'p' },
      config,
    });
    axios.post.mockRejectedValue(new Error('stop after the request'));
    const tool = new StableDiffusionAPI({ SD_WEBUI_URL: 'http://sd.local' });

    await tool._call({ prompt: 'p' });

    expect(axios.post).toHaveBeenCalledWith(
      'http://sd.local/sdapi/v1/txt2img',
      { prompt: 'p' },
      config,
    );
  });
});
