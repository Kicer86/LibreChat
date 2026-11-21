import { readWebuiProgress, toToolProgress, watchWebuiProgress } from './progress';

describe('toToolProgress', () => {
  it('reports a queued job with its place in the queue', () => {
    expect(toToolProgress({ stage: 'queued', queuePosition: 2 })).toEqual({
      stage: 'queued',
      progress: 0,
      queuePosition: 2,
    });
  });

  it.each(['vae_encode', 'conditioning', 'preparing'])('shows %s as preparing', (stage) => {
    expect(toToolProgress({ stage })).toEqual({ stage: 'preparing', progress: 0.025 });
  });

  it('fills most of the bar with sampling steps', () => {
    expect(toToolProgress({ stage: 'sampling', pass: 1, completed: 0, total: 4 })).toEqual({
      stage: 'sampling',
      progress: 0.05,
      completed: 0,
      total: 4,
    });
    expect(toToolProgress({ stage: 'sampling', pass: 1, completed: 4, total: 4 }).progress).toBe(
      0.9,
    );
  });

  it('counts a batch image by image', () => {
    const second = toToolProgress({ stage: 'sampling', pass: 2, completed: 2, total: 4 }, 2);
    expect(second).toEqual({
      stage: 'sampling',
      progress: expect.closeTo(0.05 + 0.85 * (6 / 8)),
      completed: 2,
      total: 4,
      item: 2,
      items: 2,
    });
    expect(toToolProgress({ stage: 'vae_decode', pass: 2 }, 2)).toEqual({
      stage: 'decoding',
      progress: 0.92,
      item: 2,
      items: 2,
    });
  });

  it('counts more passes than expected without going past the end', () => {
    expect(toToolProgress({ stage: 'sampling', pass: 2, completed: 4, total: 4 }).progress).toBe(
      0.9,
    );
  });

  it('reports the last stage and stages it does not know', () => {
    expect(toToolProgress({ stage: 'encoding', completed: 1, total: 1 })).toEqual({
      stage: 'encoding',
      progress: 0.97,
    });
    expect(toToolProgress({ stage: 'upscaling' })).toEqual({ stage: 'working' });
    expect(toToolProgress({ stage: 'sampling' })).toEqual({ stage: 'sampling' });
  });
});

describe('readWebuiProgress', () => {
  const answer = (state: Record<string, unknown>) => ({ progress: 0.5, state });

  it('reads the step and the image of a batch once sampling has started', () => {
    expect(
      readWebuiProgress(
        answer({ job: 'txt2img', job_no: 1, job_count: 2, sampling_step: 3, sampling_steps: 6 }),
      ),
    ).toEqual({ stage: 'sampling', pass: 2, completed: 3, total: 6 });
  });

  it('reports a generation before its first step as preparing, and no generation as none', () => {
    expect(readWebuiProgress(answer({ job: 'txt2img', sampling_steps: 0 }))).toEqual({
      stage: 'preparing',
    });
    expect(readWebuiProgress(answer({ job: '', sampling_steps: 0 }))).toBeUndefined();
    expect(readWebuiProgress({ progress: 0 })).toBeUndefined();
    expect(readWebuiProgress('busy')).toBeUndefined();
  });
});

describe('watchWebuiProgress', () => {
  it('reports each change until stopped and skips a failed read', async () => {
    const watch = new AbortController();
    const answers = [
      { status: 200, data: { state: { job: 'txt2img', sampling_steps: 0 } } },
      { status: 200, data: { state: { job: 'txt2img', sampling_steps: 0 } } },
      new Error('connection reset'),
      { status: 200, data: { state: { job: 'txt2img', sampling_step: 2, sampling_steps: 4 } } },
    ];
    const probe = jest.fn(async (requestUrl: string) => {
      expect(requestUrl).toBe('http://sd.local/sdapi/v1/progress?skip_current_image=true');
      const next = answers.shift();
      if (!next) {
        watch.abort();
        return { status: 200, data: {} };
      }
      if (next instanceof Error) {
        throw next;
      }
      return next;
    });
    const reported: unknown[] = [];
    await watchWebuiProgress({
      url: 'http://sd.local',
      probe,
      signal: watch.signal,
      onProgress: (progress) => reported.push(progress),
      sleep: async () => undefined,
    });
    expect(reported).toEqual([
      { stage: 'preparing' },
      { stage: 'sampling', completed: 2, total: 4 },
    ]);
  });
});
