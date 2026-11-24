import { observeTitleGeneration } from './data-service';
import request from './request';

jest.mock('./request', () => ({
  __esModule: true,
  default: { authenticatedFetch: jest.fn() },
}));

const authenticatedFetch = request.authenticatedFetch as jest.Mock;

const response = (chunks: string[]): Response => {
  const encoder = new TextEncoder();
  return {
    ok: true,
    status: 200,
    body: new ReadableStream({
      start(controller) {
        for (const chunk of chunks) {
          controller.enqueue(encoder.encode(chunk));
        }
        controller.close();
      },
    }),
  } as Response;
};

describe('observeTitleGeneration', () => {
  beforeEach(() => {
    authenticatedFetch.mockReset();
  });

  it('ignores heartbeats and emits title states across chunk boundaries', async () => {
    authenticatedFetch.mockResolvedValue(
      response([
        'retry: 1000\n\n: heart',
        'beat\n\nevent: title\ndata: {"conversationId":"chat","jobId":"job","status":"pending","updatedAt":1}\n\n',
        'event: title\ndata: {"conversationId":"chat","jobId":"job","status":"complete","title":"Ready","updatedAt":2}\n\n',
      ]),
    );
    const onState = jest.fn();

    await observeTitleGeneration({ conversationId: 'chat', onState });

    expect(authenticatedFetch).toHaveBeenCalledWith('/api/convos/gen_title/chat/stream', {
      headers: { Accept: 'text/event-stream' },
      signal: undefined,
    });
    expect(onState.mock.calls.map(([state]) => state.status)).toEqual(['pending', 'complete']);
    expect(onState).toHaveBeenLastCalledWith(expect.objectContaining({ title: 'Ready' }));
  });

  it('rejects a connection that ends without a terminal state', async () => {
    authenticatedFetch.mockResolvedValue(
      response([
        'event: title\ndata: {"conversationId":"chat","jobId":"job","status":"pending","updatedAt":1}\n\n',
      ]),
    );

    await expect(
      observeTitleGeneration({ conversationId: 'chat', onState: jest.fn() }),
    ).rejects.toThrow('Title stream ended before a terminal state');
  });
});
