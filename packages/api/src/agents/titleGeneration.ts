export const TITLE_GENERATION_HEADERS_TIMEOUT = 'title_generation_headers_timeout';
export const TITLE_GENERATION_BODY_TIMEOUT = 'title_generation_body_timeout';

type TitleGenerationTimeoutCode =
  | typeof TITLE_GENERATION_HEADERS_TIMEOUT
  | typeof TITLE_GENERATION_BODY_TIMEOUT;

export class TitleGenerationTimeoutError extends Error {
  code: TitleGenerationTimeoutCode;

  constructor(code: TitleGenerationTimeoutCode) {
    super('Title generation transport timed out');
    this.name = 'TitleGenerationTimeoutError';
    this.code = code;
  }
}

type TitleGenerationWatchdogOptions = {
  abortController: AbortController;
  headersTimeout: number;
  bodyTimeout: number;
};

type TitleGenerationWatchdog = {
  callback: {
    name: string;
    lc_prefer_streaming: boolean;
    handleLLMNewToken: () => void;
  };
  run: <T>(task: () => Promise<T>) => Promise<T>;
};

export function createTitleGenerationWatchdog({
  abortController,
  headersTimeout,
  bodyTimeout,
}: TitleGenerationWatchdogOptions): TitleGenerationWatchdog {
  let timer: ReturnType<typeof setTimeout> | undefined;
  let rejectTimeout: ((error: TitleGenerationTimeoutError) => void) | undefined;
  let settled = false;

  const schedule = (timeout: number, code: TitleGenerationTimeoutCode) => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      abortController.abort();
      rejectTimeout?.(new TitleGenerationTimeoutError(code));
    }, timeout);
  };

  const timeout = new Promise<never>((_resolve, reject) => {
    rejectTimeout = reject;
  });

  const callback = {
    name: 'title-generation-watchdog',
    lc_prefer_streaming: true,
    handleLLMNewToken: () => {
      if (!settled) {
        schedule(bodyTimeout, TITLE_GENERATION_BODY_TIMEOUT);
      }
    },
  };

  const run = async <T>(task: () => Promise<T>): Promise<T> => {
    schedule(headersTimeout, TITLE_GENERATION_HEADERS_TIMEOUT);
    try {
      return await Promise.race([task(), timeout]);
    } finally {
      settled = true;
      clearTimeout(timer);
    }
  };

  return { callback, run };
}
