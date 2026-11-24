import { FakeChatModel } from '@librechat/agents';
import { HumanMessage } from '@librechat/agents/langchain/messages';
import {
  TITLE_GENERATION_BODY_TIMEOUT,
  TITLE_GENERATION_HEADERS_TIMEOUT,
  createTitleGenerationWatchdog,
} from './titleGeneration';

const deferred = <T>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
};

describe('createTitleGenerationWatchdog', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('allows a healthy title request to run longer than 45 seconds', async () => {
    const result = deferred<string>();
    const abortController = new AbortController();
    const watchdog = createTitleGenerationWatchdog({
      abortController,
      headersTimeout: 300_000,
      bodyTimeout: 900_000,
    });

    const pending = watchdog.run(() => result.promise);
    await jest.advanceTimersByTimeAsync(45_001);
    expect(abortController.signal.aborted).toBe(false);

    result.resolve('Ready');
    await expect(pending).resolves.toBe('Ready');
  });

  it('aborts when the provider sends no first chunk', async () => {
    const abortController = new AbortController();
    const watchdog = createTitleGenerationWatchdog({
      abortController,
      headersTimeout: 100,
      bodyTimeout: 200,
    });

    const pending = watchdog.run(() => new Promise<string>(() => undefined));
    const rejection = pending.catch((error) => error);
    await jest.advanceTimersByTimeAsync(100);

    await expect(rejection).resolves.toMatchObject({ code: TITLE_GENERATION_HEADERS_TIMEOUT });
    expect(abortController.signal.aborted).toBe(true);
  });

  it('resets the body timer for every streamed chunk', async () => {
    const result = deferred<string>();
    const abortController = new AbortController();
    const watchdog = createTitleGenerationWatchdog({
      abortController,
      headersTimeout: 100,
      bodyTimeout: 50,
    });

    const pending = watchdog.run(() => result.promise);
    await jest.advanceTimersByTimeAsync(90);
    watchdog.callback.handleLLMNewToken();
    await jest.advanceTimersByTimeAsync(49);
    watchdog.callback.handleLLMNewToken();
    await jest.advanceTimersByTimeAsync(49);
    expect(abortController.signal.aborted).toBe(false);

    result.resolve('Ready');
    await expect(pending).resolves.toBe('Ready');
  });

  it('aborts when an active stream stops producing chunks', async () => {
    const abortController = new AbortController();
    const watchdog = createTitleGenerationWatchdog({
      abortController,
      headersTimeout: 100,
      bodyTimeout: 50,
    });

    const pending = watchdog.run(() => new Promise<string>(() => undefined));
    watchdog.callback.handleLLMNewToken();
    const rejection = pending.catch((error) => error);
    await jest.advanceTimersByTimeAsync(50);

    await expect(rejection).resolves.toMatchObject({ code: TITLE_GENERATION_BODY_TIMEOUT });
    expect(abortController.signal.aborted).toBe(true);
  });

  it('makes LangChain invoke the model streaming path', async () => {
    const watchdog = createTitleGenerationWatchdog({
      abortController: new AbortController(),
      headersTimeout: 100,
      bodyTimeout: 100,
    });
    const onToken = jest.spyOn(watchdog.callback, 'handleLLMNewToken');
    const model = new FakeChatModel({ responses: ['Ready'] });

    const result = await watchdog.run(() =>
      model.invoke([new HumanMessage('Generate a title')], {
        callbacks: [watchdog.callback],
      }),
    );

    expect(result.content).toBe('Ready');
    expect(onToken).toHaveBeenCalled();
  });
});
