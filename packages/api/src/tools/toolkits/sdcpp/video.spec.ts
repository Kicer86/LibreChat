import type { SdcppJobClient } from './video';
import { buildSdcppVideoBody, runSdcppVideoJob, SdcppVideoJobError } from './video';

const url = 'http://sd.local';
const jobUrl = `${url}/sdcpp/v1/jobs/job_1`;
const body = { prompt: 'a red car', video_frames: 9 };
const clip = {
  b64_json: 'UklGRg==',
  mime_type: 'image/webp',
  fps: 24,
  frame_count: 9,
  generation_info: { kind: 'video', seed: '7', width: 512, height: 320 },
};

/** A server whose job reports `states` in turn, then keeps the last one. */
function fakeServer(states: Array<Record<string, unknown>>, submit = { status: 202 }) {
  let polls = 0;
  const requests: string[] = [];
  const probe = jest.fn(async (requestUrl: string): Promise<{ status: number; data: unknown }> => {
    requests.push(`GET ${requestUrl}`);
    const state = states[Math.min(polls, states.length - 1)];
    polls++;
    return { status: 200, data: { id: 'job_1', ...state } };
  });
  const post = jest.fn(async (requestUrl: string): Promise<{ status: number; data: unknown }> => {
    requests.push(`POST ${requestUrl}`);
    if (requestUrl.endsWith('/cancel')) {
      return { status: 200, data: { cancel_requested: true } };
    }
    return { status: submit.status, data: { id: 'job_1', status: 'queued' } };
  });
  const client: SdcppJobClient = { probe, post };
  return { client, probe, post, requests };
}

/** A clock that moves forward only when the job runner sleeps. */
function fakeClock() {
  let time = 0;
  return {
    now: () => time,
    sleep: jest.fn(async (ms: number, signal?: AbortSignal) => {
      signal?.throwIfAborted();
      time += ms;
    }),
  };
}

describe('runSdcppVideoJob', () => {
  it('sends the job, waits for it and returns the clip', async () => {
    const server = fakeServer([
      { status: 'queued' },
      { status: 'generating' },
      { status: 'completed', result: clip },
    ]);
    const clock = fakeClock();
    const result = await runSdcppVideoJob({
      url,
      body,
      connect: () => server.client,
      cleanup: server.client,
      signal: new AbortController().signal,
      budgetMs: 60_000,
      ...clock,
    });
    expect(server.post).toHaveBeenCalledWith(`${url}/sdcpp/v1/vid_gen`, body);
    expect(result).toEqual({
      status: 'completed',
      video: {
        base64: 'UklGRg==',
        mimeType: 'image/webp',
        fps: 24,
        frames: 9,
        info: clip.generation_info,
      },
    });
    expect(server.probe).toHaveBeenCalledTimes(3);
    expect(server.requests).not.toContain(`POST ${jobUrl}/cancel`);
  });

  it("reports a job the server failed or cancelled with sd-server's code", async () => {
    const server = fakeServer([
      { status: 'failed', error: { code: 'out_of_memory', message: 'raw detail' } },
    ]);
    const result = await runSdcppVideoJob({
      url,
      body,
      connect: () => server.client,
      cleanup: server.client,
      signal: new AbortController().signal,
      budgetMs: 60_000,
      ...fakeClock(),
    });
    expect(result).toEqual({ status: 'failed', code: 'out_of_memory' });
  });

  it('reports a request the server refused', async () => {
    const server = fakeServer([], { status: 400 });
    const result = await runSdcppVideoJob({
      url,
      body,
      connect: () => server.client,
      cleanup: server.client,
      signal: new AbortController().signal,
      budgetMs: 60_000,
      ...fakeClock(),
    });
    expect(result).toEqual({ status: 'rejected', httpStatus: 400 });
    expect(server.probe).not.toHaveBeenCalled();
  });

  it('throws when the server answers without a job', async () => {
    const server = fakeServer([], { status: 503 });
    await expect(
      runSdcppVideoJob({
        url,
        body,
        connect: () => server.client,
        cleanup: server.client,
        signal: new AbortController().signal,
        budgetMs: 60_000,
        ...fakeClock(),
      }),
    ).rejects.toBeInstanceOf(SdcppVideoJobError);
  });

  it('cancels a job that outlasts the budget and waits for it to end', async () => {
    const server = fakeServer([
      { status: 'generating' },
      { status: 'generating' },
      { status: 'generating' },
      { status: 'generating' },
      { status: 'cancelled', error: { code: 'cancelled' } },
    ]);
    const result = await runSdcppVideoJob({
      url,
      body,
      connect: () => server.client,
      cleanup: server.client,
      signal: new AbortController().signal,
      budgetMs: 3_000,
      ...fakeClock(),
    });
    expect(result).toEqual({ status: 'timeout' });
    const cancelAt = server.requests.indexOf(`POST ${jobUrl}/cancel`);
    expect(cancelAt).toBeGreaterThan(0);
    expect(
      server.requests.slice(cancelAt + 1).every((request) => request === `GET ${jobUrl}`),
    ).toBe(true);
    expect(server.probe).toHaveBeenCalledTimes(5);
  });

  it('cancels the job when the generation is stopped, through the cleanup client', async () => {
    const controller = new AbortController();
    const reason = new Error('stopped');
    const server = fakeServer([{ status: 'generating' }]);
    const cleanup = fakeServer([{ status: 'cancelled' }]);
    const clock = fakeClock();
    server.probe.mockImplementationOnce(async () => {
      controller.abort(reason);
      return { status: 200, data: { status: 'generating' } };
    });
    await expect(
      runSdcppVideoJob({
        url,
        body,
        connect: () => server.client,
        cleanup: cleanup.client,
        signal: controller.signal,
        budgetMs: 60_000,
        ...clock,
      }),
    ).rejects.toBe(reason);
    expect(cleanup.requests).toEqual([`POST ${jobUrl}/cancel`, `GET ${jobUrl}`]);
  });

  it('cancels the job when its state cannot be read, then throws', async () => {
    const server = fakeServer([{ status: 'generating' }]);
    server.probe.mockResolvedValueOnce({ status: 404, data: { error: 'job not found' } });
    await expect(
      runSdcppVideoJob({
        url,
        body,
        connect: () => server.client,
        cleanup: server.client,
        signal: new AbortController().signal,
        budgetMs: 60_000,
        ...fakeClock(),
      }),
    ).rejects.toBeInstanceOf(SdcppVideoJobError);
    expect(server.requests).toContain(`POST ${jobUrl}/cancel`);
  });

  it('ends a request still waiting when the budget runs out, then cancels the job', async () => {
    const server = fakeServer([{ status: 'cancelled' }]);
    let requestSignal: AbortSignal | undefined;
    const hanging = {
      post: server.post,
      probe: jest.fn(
        (_requestUrl: string) =>
          new Promise<{ status: number; data: unknown }>((_resolve, reject) => {
            requestSignal?.addEventListener('abort', () => reject(new Error('canceled')));
          }),
      ),
    };
    const result = await runSdcppVideoJob({
      url,
      body,
      connect: (signal) => {
        requestSignal = signal;
        return hanging;
      },
      cleanup: server.client,
      signal: new AbortController().signal,
      budgetMs: 20,
    });
    expect(result).toEqual({ status: 'timeout' });
    expect(requestSignal?.aborted).toBe(true);
    expect(server.requests).toContain(`POST ${jobUrl}/cancel`);
  });

  it('ends a job request still waiting when the budget runs out', async () => {
    const server = fakeServer([]);
    const result = await runSdcppVideoJob({
      url,
      body,
      connect: (signal) => ({
        probe: server.probe,
        post: () =>
          new Promise((_resolve, reject) => {
            signal.addEventListener('abort', () => reject(new Error('canceled')));
          }),
      }),
      cleanup: server.client,
      signal: new AbortController().signal,
      budgetMs: 20,
    });
    expect(result).toEqual({ status: 'timeout' });
    expect(server.requests).toEqual([]);
  });

  it('keeps waiting for a cancelled job through a passing read error', async () => {
    const server = fakeServer([{ status: 'cancelled' }]);
    const generating = async () => ({ status: 200, data: { status: 'generating' } });
    server.probe
      .mockImplementationOnce(generating)
      .mockImplementationOnce(generating)
      .mockImplementationOnce(async () => {
        throw new Error('socket hang up');
      });
    const clock = fakeClock();
    let settled = false;
    const result = runSdcppVideoJob({
      url,
      body,
      connect: () => server.client,
      cleanup: server.client,
      signal: new AbortController().signal,
      budgetMs: 1,
      ...clock,
    }).then((value) => {
      settled = true;
      return value;
    });
    await expect(result).resolves.toEqual({ status: 'timeout' });
    expect(settled).toBe(true);
    expect(server.probe).toHaveBeenCalledTimes(4);
    expect(server.requests).toEqual([
      `POST ${url}/sdcpp/v1/vid_gen`,
      `POST ${jobUrl}/cancel`,
      `GET ${jobUrl}`,
    ]);
  });

  it('retries a cancel request that failed', async () => {
    const server = fakeServer([
      { status: 'generating' },
      { status: 'generating' },
      { status: 'cancelled' },
    ]);
    server.post
      .mockImplementationOnce(async () => ({ status: 202, data: { id: 'job_1' } }))
      .mockImplementationOnce(async () => {
        throw new Error('ECONNRESET');
      });
    const result = await runSdcppVideoJob({
      url,
      body,
      connect: () => server.client,
      cleanup: server.client,
      signal: new AbortController().signal,
      budgetMs: 1,
      ...fakeClock(),
    });
    expect(result).toEqual({ status: 'timeout' });
    expect(
      server.post.mock.calls.filter(([requestUrl]) => requestUrl.endsWith('/cancel')),
    ).toHaveLength(2);
  });

  it('stops waiting for a cancelled job that never ends', async () => {
    const server = fakeServer([{ status: 'generating' }]);
    const clock = fakeClock();
    const result = await runSdcppVideoJob({
      url,
      body,
      connect: () => server.client,
      cleanup: server.client,
      signal: new AbortController().signal,
      budgetMs: 1_000,
      ...clock,
    });
    expect(result).toEqual({ status: 'timeout' });
    expect(clock.now()).toBeGreaterThanOrEqual(120_000);
    expect(clock.now()).toBeLessThan(125_000);
  });
});

describe('buildSdcppVideoBody', () => {
  it('nests the sampling settings and asks for an animated WebP', () => {
    expect(
      buildSdcppVideoBody(
        {
          prompt: 'a red car',
          negative_prompt: 'blur',
          width: 512,
          height: 320,
          steps: 8,
          cfg_scale: 1,
          seed: 7,
          sampler_name: 'euler',
          scheduler: 'ltx2',
        },
        { frames: 9, fps: 24 },
      ),
    ).toEqual({
      prompt: 'a red car',
      negative_prompt: 'blur',
      width: 512,
      height: 320,
      video_frames: 9,
      fps: 24,
      seed: 7,
      output_format: 'webp',
      sample_params: {
        sample_steps: 8,
        sample_method: 'euler',
        scheduler: 'ltx2',
        guidance: { txt_cfg: 1 },
      },
    });
  });

  it('leaves out the settings not asked for, but asks for a random seed', () => {
    expect(buildSdcppVideoBody({ prompt: 'a cat', negative_prompt: '' }, { frames: 121 })).toEqual({
      prompt: 'a cat',
      video_frames: 121,
      seed: -1,
      output_format: 'webp',
    });
  });
});
