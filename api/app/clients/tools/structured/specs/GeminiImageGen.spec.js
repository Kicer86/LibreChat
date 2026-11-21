const { GoogleGenAI } = require('@google/genai');
const { logger } = require('@librechat/data-schemas');
const createGeminiImageTool = require('../GeminiImageGen');

jest.mock('@google/genai', () => ({ GoogleGenAI: jest.fn() }));
jest.mock('~/server/services/Files/strategies', () => ({ getStrategyFunctions: jest.fn() }));
jest.mock('~/models', () => ({
  spendTokens: jest.fn(),
  getUserImageFiles: jest.fn().mockResolvedValue([]),
}));

const SECRET = 'PRIVATE-PROVIDER-DETAIL';
const FALLBACK = 'Error: The tool request failed.';
const EMPTY_ARTIFACT = { content: [], file_ids: [] };

const providerError = (status) =>
  Object.assign(new Error(JSON.stringify({ error: { code: status, message: SECRET } })), {
    status,
  });

const failGeneration = (error) =>
  GoogleGenAI.mockImplementation(() => ({
    models: { generateContent: jest.fn().mockRejectedValue(error) },
  }));

const createTool = (fields = {}) =>
  createGeminiImageTool({
    isAgent: true,
    req: { user: { id: 'user-1' } },
    userId: 'user-1',
    GEMINI_API_KEY: 'gemini-key',
    ...fields,
  });

const run = async (tool) => {
  const [content, artifact] = await tool.func({ prompt: 'a cat' });
  return { text: content.map((part) => part.text).join('\n'), artifact };
};

/** True when any logged argument, including an error message, carries the text. */
const logged = (text) =>
  [logger.warn, logger.error, logger.info, logger.debug]
    .flatMap((method) => method.mock.calls.flat())
    .some((arg) => `${arg instanceof Error ? arg.message : JSON.stringify(arg)}`.includes(text));

describe('GeminiImageGen approved error text', () => {
  let originalEnv;

  beforeEach(() => {
    originalEnv = { ...process.env };
    jest.restoreAllMocks();
    for (const method of ['warn', 'error', 'info', 'debug']) {
      jest.spyOn(logger, method).mockImplementation(() => logger);
    }
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('words a failing generation from its status, never the provider message', async () => {
    failGeneration(providerError(403));

    const { text, artifact } = await run(createTool());

    expect(text).toBe(
      'Error: Gemini image generation rejected the configured credentials (status 403).',
    );
    expect(artifact).toEqual(EMPTY_ARTIFACT);
    expect(logger.warn).toHaveBeenCalledWith('[Gemini image generation] Tool request failed', {
      status: 403,
    });
    expect(logged(SECRET)).toBe(false);
  });

  it('returns the stable fallback for a generation failure without a status', async () => {
    failGeneration(new Error(`socket hang up ${SECRET}`));

    const { text, artifact } = await run(createTool());

    expect(text).toBe(FALLBACK);
    expect(artifact).toEqual(EMPTY_ARTIFACT);
    expect(logged(SECRET)).toBe(false);
  });

  it('returns the stable fallback when the client cannot be created', async () => {
    GoogleGenAI.mockImplementation(() => {
      throw new Error(`invalid credentials ${SECRET}`);
    });

    const { text, artifact } = await run(createTool());

    expect(text).toBe(FALLBACK);
    expect(artifact).toEqual(EMPTY_ARTIFACT);
    expect(logged(SECRET)).toBe(false);
  });

  it('keeps its configuration guidance without naming the service account path', async () => {
    process.env.GOOGLE_SERVICE_KEY_FILE = `/missing/${SECRET}/auth.json`;

    const { text } = await run(createTool({ GEMINI_API_KEY: undefined, GOOGLE_KEY: undefined }));

    expect(text).toBe(
      'Error: Gemini Image Generation requires one of: user-provided API key, GEMINI_API_KEY or GOOGLE_KEY env var, or a valid Google service account.',
    );
    expect(text).not.toContain('/missing/');
    expect(GoogleGenAI).not.toHaveBeenCalled();
  });
});

describe('GeminiImageGen input images', () => {
  const sharp = require('sharp');
  const { Readable } = require('stream');
  const { createGeneratedImageRegistry } = require('@librechat/api');
  const { getUserImageFiles } = require('~/models');
  const { getStrategyFunctions } = require('~/server/services/Files/strategies');
  let generateContent;

  beforeEach(async () => {
    jest.clearAllMocks();
    const png = await sharp({
      create: { width: 4, height: 4, channels: 3, background: { r: 255, g: 0, b: 0 } },
    })
      .png()
      .toBuffer();
    generateContent = jest.fn().mockResolvedValue({
      candidates: [{ content: { parts: [{ inlineData: { data: png.toString('base64') } }] } }],
    });
    GoogleGenAI.mockImplementation(() => ({ models: { generateContent } }));
    getStrategyFunctions.mockReturnValue({
      getDownloadStream: jest.fn(async () => Readable.from([Buffer.from('stored-bytes')])),
    });
  });

  /** The inline images the n-th request sent with its prompt. */
  const sentImages = (call) =>
    generateContent.mock.calls[call][0].contents.filter((part) => part.inlineData);

  it('uses an image generated earlier in the request before its file is saved', async () => {
    const generatedImages = createGeneratedImageRegistry();
    const tool = createTool({ generatedImages });
    const [content] = await tool.func({ prompt: 'a cat' });
    const generatedId = /generated_image_id: "([^"]+)"/.exec(content[0].text)[1];

    await tool.func({ prompt: 'the cat in a hat', image_ids: [generatedId] });

    const [image] = sentImages(1);
    expect(image.inlineData.mimeType).toBe('image/png');
    expect(Buffer.from(image.inlineData.data, 'base64').subarray(1, 4).toString()).toBe('PNG');
    expect(getUserImageFiles).not.toHaveBeenCalled();
  });

  it("looks other IDs up among the user's own files in their tenant", async () => {
    getUserImageFiles.mockResolvedValueOnce([
      { file_id: 'earlier', source: 'local', filepath: '/images/earlier.png', type: 'image/png' },
    ]);
    const tool = createTool({ req: { user: { id: 'user-1', tenantId: 'tenant-1' } } });

    await tool.func({ prompt: 'a cat', image_ids: ['earlier', 'unknown'] });

    expect(getUserImageFiles).toHaveBeenCalledWith(['earlier', 'unknown'], {
      userId: 'user-1',
      tenantId: 'tenant-1',
    });
    const images = sentImages(0);
    expect(images).toHaveLength(1);
    expect(Buffer.from(images[0].inlineData.data, 'base64').toString()).toBe('stored-bytes');
  });
});
