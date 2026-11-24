const mockCache = {
  set: jest.fn(),
};
const mockTitleStateStore = new Map();
const mockTitleStateCache = {
  get: jest.fn((key) => mockTitleStateStore.get(key)),
  set: jest.fn((key, value) => mockTitleStateStore.set(key, value)),
  delete: jest.fn((key) => mockTitleStateStore.delete(key)),
};
const mockSaveConvo = jest.fn();
const mockGetConvo = jest.fn();
const mockInitializeClient = jest.fn();

jest.mock('@librechat/api', () => ({
  ...jest.requireActual('@librechat/api'),
  isEnabled: (value) => value === true || value === 'true',
  sanitizeTitle: (title) => title,
}));

jest.mock('@librechat/data-schemas', () => ({
  logger: { debug: jest.fn(), info: jest.fn(), warn: jest.fn(), error: jest.fn() },
}));

jest.mock('~/cache/getLogStores', () =>
  jest.fn((key) => (key === 'TITLE_GENERATION' ? mockTitleStateCache : mockCache)),
);
jest.mock(
  './initalize',
  () =>
    (...args) =>
      mockInitializeClient(...args),
);
jest.mock('~/models', () => ({
  saveConvo: (...args) => mockSaveConvo(...args),
  getConvo: (...args) => mockGetConvo(...args),
}));

const addTitle = require('./title');

const getTitleState = (conversationId) => {
  const key = `user-1-${conversationId}`;
  const jobId = mockTitleStateStore.get(key);
  return JSON.parse(mockTitleStateStore.get(`${key}:${jobId}`) ?? '{}');
};

describe('assistants addTitle content policy', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockTitleStateStore.clear();
    mockTitleStateCache.get.mockReset().mockImplementation((key) => mockTitleStateStore.get(key));
    mockTitleStateCache.set
      .mockReset()
      .mockImplementation((key, value) => mockTitleStateStore.set(key, value));
    mockTitleStateCache.delete
      .mockReset()
      .mockImplementation((key) => mockTitleStateStore.delete(key));
    mockSaveConvo.mockImplementation(async (_ctx, data) => data);
    mockGetConvo.mockResolvedValue(null);
  });

  it('skips the title provider, cache, and save for a normalized temporary request', async () => {
    await addTitle(
      { user: { id: 'user-1' }, body: { isTemporary: true } },
      { text: 'temporary input', responseText: 'response', conversationId: 'conversation-1' },
    );

    expect(mockInitializeClient).not.toHaveBeenCalled();
    expect(mockCache.set).not.toHaveBeenCalled();
    expect(mockSaveConvo).not.toHaveBeenCalled();
    expect(getTitleState('conversation-1')).toMatchObject({ status: 'unavailable' });
  });

  it('replaces a blocked generated title before caching or saving it', async () => {
    const create = jest.fn().mockResolvedValue({
      choices: [{ message: { content: 'BLOCKED-GENERATED-TITLE' } }],
    });
    mockInitializeClient.mockResolvedValue({
      openai: { chat: { completions: { create } } },
    });
    const req = {
      user: { id: 'user-1' },
      body: {},
      config: {
        filters: {
          conversationTitles: {
            pii: {
              starterPatterns: [],
              customPatterns: [{ id: 'blocked', label: 'blocked', regex: 'BLOCKED' }],
            },
          },
        },
      },
    };

    await addTitle(req, {
      text: 'submitted text',
      responseText: 'response',
      conversationId: 'conversation-generated',
    });

    expect(mockCache.set).toHaveBeenCalledWith('user-1-conversation-generated', 'New Chat', 120000);
    expect(mockSaveConvo).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({
        conversationId: 'conversation-generated',
        title: 'New Chat',
      }),
      expect.objectContaining({ noUpsert: true }),
    );
    expect(getTitleState('conversation-generated')).toMatchObject({
      status: 'complete',
      title: 'New Chat',
    });
  });

  it('replaces a blocked submitted-text fallback before caching or saving it', async () => {
    const create = jest.fn().mockRejectedValue(new Error('title model unavailable'));
    mockInitializeClient.mockResolvedValue({
      openai: { chat: { completions: { create } } },
    });
    const req = {
      user: { id: 'user-1' },
      body: {},
      config: {
        filters: {
          conversationTitles: {
            pii: {
              starterPatterns: [],
              customPatterns: [{ id: 'blocked', label: 'blocked', regex: 'BLOCKED' }],
            },
          },
        },
      },
    };

    await addTitle(req, {
      text: 'BLOCKED-SUBMISSION',
      responseText: 'response',
      conversationId: 'conversation-1',
    });

    expect(mockCache.set).toHaveBeenCalledWith('user-1-conversation-1', 'New Chat', 120000);
    expect(mockSaveConvo).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({
        conversationId: 'conversation-1',
        title: 'New Chat',
      }),
      expect.objectContaining({ noUpsert: true }),
    );
    expect(getTitleState('conversation-1')).toMatchObject({
      status: 'complete',
      title: 'New Chat',
    });
  });

  it('publishes a terminal error when neither generation nor fallback produces a title', async () => {
    mockInitializeClient.mockRejectedValue(new Error('title model unavailable'));

    await addTitle(
      { user: { id: 'user-1' }, body: {} },
      { text: '', responseText: '', conversationId: 'empty-chat' },
    );

    expect(getTitleState('empty-chat')).toMatchObject({
      status: 'error',
      code: 'title_generation_failed',
    });
  });
  it.each(['database', 'cache'])('owns a detached fallback %s failure', async (boundary) => {
    mockInitializeClient.mockResolvedValue({
      openai: {
        chat: { completions: { create: jest.fn().mockRejectedValue(new Error('provider')) } },
      },
    });
    const failure = new Error('secret-provider-payload');
    if (boundary === 'database') mockSaveConvo.mockRejectedValueOnce(failure);
    else mockCache.set.mockRejectedValueOnce(failure);
    await expect(
      addTitle(
        { user: { id: 'user-1' }, body: {} },
        {
          text: 'fallback',
          responseText: 'response',
          conversationId: 'fallback-chat',
        },
      ),
    ).resolves.toBeUndefined();
    const { logger } = require('@librechat/data-schemas');
    expect(logger.error).toHaveBeenCalledWith('[addTitle] Fallback publication failed', {
      type: 'Error',
    });
  });
});
