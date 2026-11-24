import express from 'express';
import request from 'supertest';
import {
  createGeneratedTitleHandler,
  createTitleGenerationStreamHandler,
  createTitleGenerationTracker,
} from './title';

function setup(authenticated = true) {
  const cache = {
    get: jest.fn().mockResolvedValue('Generated'),
    delete: jest.fn().mockResolvedValue(true),
    set: jest.fn(),
  };
  const deps = {
    getCache: () => cache,
    getConvoTitleState: jest.fn().mockResolvedValue(null),
    logger: { error: jest.fn() },
    delay: jest.fn().mockResolvedValue(undefined),
  };
  const handler = createGeneratedTitleHandler(deps);
  const app = express();
  app.get<{ conversationId: string }>('/title/:conversationId', async (req, res) => {
    await handler({ params: req.params, user: authenticated ? { id: 'user-1' } : undefined }, res);
  });
  return { app, cache, deps };
}

describe('generated title polling', () => {
  it('keeps legacy string cache values and generated responses compatible', async () => {
    const { app, cache, deps } = setup();
    const response = await request(app).get('/title/chat');
    expect(response.status).toBe(200);
    expect(response.body).toEqual({ title: 'Generated' });
    expect(deps.getConvoTitleState).toHaveBeenCalledWith('user-1', 'chat');
    expect(cache.delete).toHaveBeenCalledWith('user-1-chat');
  });

  it.each(['Newest rename', 'New Chat', ''])(
    'returns current manual authority for %s',
    async (title) => {
      const { app, deps } = setup();
      deps.getConvoTitleState.mockResolvedValue({ title, titleSetByUser: true, titleRevision: 2 });
      const response = await request(app).get('/title/chat');
      expect(response.status).toBe(200);
      expect(response.body).toEqual({ title, titleSetByUser: true, titleRevision: 2 });
    },
  );

  it('preserves the bounded not-ready polling delays without database reads', async () => {
    const { app, cache, deps } = setup();
    cache.get.mockResolvedValue(undefined);
    const response = await request(app).get('/title/chat');
    expect(response.status).toBe(404);
    expect(deps.delay.mock.calls.map(([milliseconds]) => milliseconds)).toEqual([
      500, 1000, 2000, 4000, 8000,
    ]);
    expect(deps.getConvoTitleState).not.toHaveBeenCalled();
    expect(cache.delete).not.toHaveBeenCalled();
  });

  it('does not consume the cache or expose diagnostics on a database failure', async () => {
    const { app, cache, deps } = setup();
    deps.getConvoTitleState.mockRejectedValue(new Error('secret-provider-payload'));
    const response = await request(app).get('/title/chat');
    expect(response.status).toBe(500);
    expect(response.body).toEqual({ error: 'title_read_failed' });
    expect(cache.delete).not.toHaveBeenCalled();
    expect(deps.logger.error).toHaveBeenCalledWith('[gen_title] Title lookup failed', {
      type: 'Error',
    });
  });

  it('does not read cache state without an authenticated owner', async () => {
    const { app, cache } = setup(false);
    expect((await request(app).get('/title/chat')).status).toBe(401);
    expect(cache.get).not.toHaveBeenCalled();
  });
});

describe('title generation state', () => {
  it('does not let an older job replace a newer job state', async () => {
    const values = new Map<string, string>();
    const cache = {
      get: jest.fn(async (key: string) => values.get(key)),
      set: jest.fn(async (key: string, value: string) => {
        values.set(key, value);
        return true;
      }),
      delete: jest.fn(async (key: string) => values.delete(key)),
    };
    const older = createTitleGenerationTracker({
      cache,
      userId: 'user-1',
      conversationId: 'chat',
      jobId: 'older',
    });
    const newer = createTitleGenerationTracker({
      cache,
      userId: 'user-1',
      conversationId: 'chat',
      jobId: 'newer',
    });

    await older.pending();
    await newer.pending();
    await older.complete('Stale title');
    await newer.complete('Current title');

    expect(JSON.parse(values.get('user-1-chat') ?? '{}')).toMatchObject({
      jobId: 'newer',
      status: 'complete',
      title: 'Current title',
    });
  });

  it('streams pending and completed states independently of the chat stream', async () => {
    const states = [
      JSON.stringify({
        jobId: 'job-1',
        conversationId: 'chat',
        status: 'pending',
        updatedAt: 1,
      }),
      JSON.stringify({
        jobId: 'job-1',
        conversationId: 'chat',
        status: 'complete',
        title: 'Generated title',
        updatedAt: 2,
      }),
    ];
    const complete = states[1];
    const cache = {
      get: jest.fn(async () => states.shift() ?? complete),
      set: jest.fn(),
      delete: jest.fn(),
    };
    const handler = createTitleGenerationStreamHandler({
      getCache: () => cache,
      logger: { error: jest.fn() },
      delay: jest.fn().mockResolvedValue(undefined),
    });
    const app = express();
    app.get('/title/:conversationId', async (req, res) => {
      await handler({ params: req.params, user: { id: 'user-1' }, on: req.on.bind(req) }, res);
    });

    const response = await request(app).get('/title/chat').buffer(true);

    expect(response.status).toBe(200);
    expect(response.headers['content-type']).toContain('text/event-stream');
    expect(response.text).toContain('"status":"pending"');
    expect(response.text).toContain('"status":"complete"');
    expect(response.text).toContain('"title":"Generated title"');
  });

  it('rejects an unauthenticated title stream before opening SSE', async () => {
    const handler = createTitleGenerationStreamHandler({
      getCache: () => ({ get: jest.fn(), set: jest.fn(), delete: jest.fn() }),
      logger: { error: jest.fn() },
    });
    const app = express();
    app.get('/title/:conversationId', async (req, res) => {
      await handler({ params: req.params, on: req.on.bind(req) }, res);
    });

    const response = await request(app).get('/title/chat');

    expect(response.status).toBe(401);
    expect(response.body).toEqual({ error: 'unauthorized' });
  });

  it('bridges titles published by a legacy endpoint into the independent stream', async () => {
    const legacy = {
      get: jest.fn().mockResolvedValue('Legacy title'),
      set: jest.fn(),
      delete: jest.fn().mockResolvedValue(true),
    };
    const handler = createTitleGenerationStreamHandler({
      getCache: () => ({
        get: jest.fn().mockResolvedValue(undefined),
        set: jest.fn(),
        delete: jest.fn(),
      }),
      getLegacyCache: () => legacy,
      getConvoTitleState: jest.fn().mockResolvedValue(null),
      logger: { error: jest.fn() },
    });
    const app = express();
    app.get('/title/:conversationId', async (req, res) => {
      await handler({ params: req.params, user: { id: 'user-1' }, on: req.on.bind(req) }, res);
    });

    const response = await request(app).get('/title/chat').buffer(true);

    expect(response.text).toContain('"status":"complete"');
    expect(response.text).toContain('"title":"Legacy title"');
    expect(legacy.delete).toHaveBeenCalledWith('user-1-chat');
  });
});
