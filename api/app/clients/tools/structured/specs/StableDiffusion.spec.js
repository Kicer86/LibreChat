const mockGenerateStableDiffusionImage = jest.fn();

jest.mock('sharp', () => jest.fn(), { virtual: true });
jest.mock('uuid', () => ({ v4: jest.fn(() => 'file-1') }), { virtual: true });
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
    ContentTypes: { TEXT: 'text', IMAGE_URL: 'image_url' },
    FileContext: {},
  }),
  { virtual: true },
);
jest.mock('@librechat/api', () => ({
  generateStableDiffusionImage: (...args) => mockGenerateStableDiffusionImage(...args),
  getBasePath: jest.fn(),
}));
jest.mock('~/config/paths', () => ({ imageOutput: '/tmp/sd-images', clientPath: '/tmp' }), {
  virtual: true,
});
jest.mock('fs', () => ({ existsSync: jest.fn(() => true), mkdirSync: jest.fn() }));

const StableDiffusionAPI = require('../StableDiffusion');

describe('StableDiffusionAPI', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('generates with the endpoint, the image context and the lock it was created with', async () => {
    mockGenerateStableDiffusionImage.mockResolvedValue({ error: 'no image' });
    const imageFiles = [{ file_id: 'img-1' }];
    const findImageFiles = jest.fn();
    const lock = { acquire: jest.fn(), extend: jest.fn(), release: jest.fn() };
    const tool = new StableDiffusionAPI({
      SD_WEBUI_URL: 'http://127.0.0.1:9000',
      userProvidedAuthFields: new Set(['SD_WEBUI_URL']),
      imageFiles,
      findImageFiles,
      lock,
    });
    const input = { prompt: 'test prompt', image_ids: ['img-1'] };

    await tool._call(input);

    expect(mockGenerateStableDiffusionImage).toHaveBeenCalledWith(
      input,
      expect.objectContaining({
        url: 'http://127.0.0.1:9000',
        isUserProvidedEndpoint: true,
        imageFiles,
        findImageFiles,
        lock,
      }),
    );
  });

  it('returns the reason an image could not be generated', async () => {
    mockGenerateStableDiffusionImage.mockResolvedValue({
      error: 'The image could not be loaded.',
    });
    const tool = new StableDiffusionAPI({ SD_WEBUI_URL: 'http://sd.local' });

    const result = await tool._call({ prompt: 'p', image_ids: ['nope'] });

    expect(result).toBe('The image could not be loaded.');
  });

  it('reports a request blocked by the SSRF guard as an API error', async () => {
    mockGenerateStableDiffusionImage.mockRejectedValue(
      new Error('SSRF protection: blocked address'),
    );
    const tool = new StableDiffusionAPI({
      SD_WEBUI_URL: 'http://127.0.0.1:9000',
      userProvidedAuthFields: new Set(['SD_WEBUI_URL']),
    });

    const result = await tool._call({ prompt: 'test prompt', negative_prompt: 'test negative' });

    expect(result).toBe('Error making API request.');
  });

  it('names the generated image and the model that made it to the agent', async () => {
    mockGenerateStableDiffusionImage.mockResolvedValue({
      data: { images: ['aW1hZ2U='], info: JSON.stringify({ width: 512, height: 512 }) },
      model: 'local-qwen-image-21-q8',
    });
    const generatedImages = { remember: jest.fn() };
    const tool = new StableDiffusionAPI({
      SD_WEBUI_URL: 'http://sd.local',
      userId: 'user-1',
      isAgent: true,
      generatedImages,
    });

    const [response, artifact] = await tool._call({ prompt: 'p' });

    expect(response[0].text).toContain('generated_image_id: "file-1"');
    expect(response[0].text).toContain('model: "local-qwen-image-21-q8"');
    expect(artifact).toMatchObject({ file_ids: ['file-1'] });
    expect(generatedImages.remember).toHaveBeenCalledWith(
      undefined,
      expect.objectContaining({ file: { file_id: 'file-1', width: 512, height: 512 } }),
    );
  });
});
