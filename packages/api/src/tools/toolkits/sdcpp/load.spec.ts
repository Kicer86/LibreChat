import {
  loadSdcppModel,
  hasReadySdcppModel,
  SdcppModelError,
  readSdcppModelStatus,
  withSdcppServerLock,
  findSdcppModelMismatch,
} from './load';
import { createDistributedLock } from '~/cache/lock';

const url = 'http://sd.local:7860';
const qwenStatus = {
  recipe: 'qwen',
  state: 'ready',
  models: {
    diffusion_model: 'diffusion_models/qwen-image-2.1-Q8_0.gguf',
    vae: 'vae/qwen-image-2.1-vae.safetensors',
  },
};

describe('readSdcppModelStatus', () => {
  it('reads the loaded model', async () => {
    const probe = jest.fn(async () => ({ status: 200, data: { ...qwenStatus, error: null } }));
    await expect(readSdcppModelStatus(url, probe)).resolves.toEqual(qwenStatus);
    expect(probe).toHaveBeenCalledWith(`${url}/sdcpp/v1/models/status`);
  });

  it('fails on an error status', async () => {
    const probe = jest.fn(async () => ({ status: 404, data: 'Not Found' }));
    await expect(readSdcppModelStatus(url, probe)).rejects.toBeInstanceOf(SdcppModelError);
  });
});

describe('hasReadySdcppModel', () => {
  it.each([
    [qwenStatus, true],
    [{ state: 'ready', models: { diffusion_model: 'model.gguf' } }, true],
    [{ state: 'ready', models: {} }, false],
    [{ state: 'loading', models: { diffusion_model: 'model.gguf' } }, false],
    [{ error: 'out of memory', models: { diffusion_model: 'model.gguf' } }, false],
  ])('tells whether %j can generate', (status, ready) => {
    expect(hasReadySdcppModel(status)).toBe(ready);
  });
});

describe('loadSdcppModel', () => {
  it('does not reload a model that is ready', async () => {
    const post = jest.fn();
    await expect(loadSdcppModel(url, 'qwen', qwenStatus, post)).resolves.toBe(qwenStatus);
    expect(post).not.toHaveBeenCalled();
  });

  it('loads another model and returns its status', async () => {
    const fluxStatus = { recipe: 'flux', state: 'ready', models: { diffusion_model: 'flux.gguf' } };
    const post = jest.fn(async () => ({ status: 200, data: fluxStatus }));
    await expect(loadSdcppModel(url, 'flux', qwenStatus, post)).resolves.toEqual(fluxStatus);
    expect(post).toHaveBeenCalledWith(`${url}/sdcpp/v1/models/load`, { recipe: 'flux' });
  });

  it.each([
    ['an error status', { status: 400, data: { error: 'unknown recipe', models: {} } }],
    ['a load error', { status: 200, data: { recipe: 'flux', error: 'out of memory' } }],
    ['another model', { status: 200, data: qwenStatus }],
    ['a model still loading', { status: 200, data: { recipe: 'flux', state: 'loading' } }],
  ])('fails on %s', async (_case, response) => {
    const post = jest.fn(async () => response);
    await expect(loadSdcppModel(url, 'flux', qwenStatus, post)).rejects.toBeInstanceOf(
      SdcppModelError,
    );
  });
});

describe('findSdcppModelMismatch', () => {
  const info = (models: Record<string, string>) =>
    JSON.stringify({ generation_info: [{ kind: 'image', models }] });

  it('matches the files by name', () => {
    expect(
      findSdcppModelMismatch(
        qwenStatus,
        info({
          diffusion_model: 'qwen-image-2.1-Q8_0.gguf',
          vae: 'qwen-image-2.1-vae.safetensors',
          llm: 'Qwen3VL-8B-Instruct-Q8_0.gguf',
        }),
      ),
    ).toBeUndefined();
  });

  it('names the slots that used other files', () => {
    expect(
      findSdcppModelMismatch(
        qwenStatus,
        info({
          diffusion_model: 'qwen-image-2.1-Q6_K.gguf',
          vae: 'qwen-image-2.1-vae.safetensors',
        }),
      ),
    ).toBe('diffusion_model: expected qwen-image-2.1-Q8_0.gguf, got qwen-image-2.1-Q6_K.gguf');
  });

  it.each([[undefined], ['not json'], [JSON.stringify({ seed: 1 })], [{ generation_info: [] }]])(
    'cannot tell from %j',
    (value) => {
      expect(findSdcppModelMismatch(qwenStatus, value)).toBeUndefined();
    },
  );
});

describe('withSdcppServerLock', () => {
  it('runs one task at a time per server', async () => {
    const lock = createDistributedLock();
    const order: string[] = [];
    let releaseFirst: () => void = () => undefined;
    const first = withSdcppServerLock({ lock, url, waitMs: 5000 }, async () => {
      order.push('first:start');
      await new Promise<void>((resolve) => (releaseFirst = resolve));
      order.push('first:end');
      return 1;
    });
    const second = withSdcppServerLock(
      { lock, url, waitMs: 5000, sleep: () => new Promise((resolve) => setTimeout(resolve, 1)) },
      async () => {
        order.push('second');
        return 2;
      },
    );
    await new Promise((resolve) => setTimeout(resolve, 5));
    expect(order).toEqual(['first:start']);
    releaseFirst();
    await expect(Promise.all([first, second])).resolves.toEqual([1, 2]);
    expect(order).toEqual(['first:start', 'first:end', 'second']);
  });

  it('releases the lock when the task fails', async () => {
    const lock = createDistributedLock();
    await expect(
      withSdcppServerLock({ lock, url, waitMs: 0 }, async () => {
        throw new Error('generation failed');
      }),
    ).rejects.toThrow('generation failed');
    await expect(withSdcppServerLock({ lock, url, waitMs: 0 }, async () => 'free')).resolves.toBe(
      'free',
    );
  });

  it('gives up when the lock stays held', async () => {
    const lock = { acquire: jest.fn(async () => null), extend: jest.fn(), release: jest.fn() };
    let time = 0;
    const task = jest.fn();
    await expect(
      withSdcppServerLock(
        { lock, url, waitMs: 1000, now: () => time, sleep: async (ms) => void (time += ms) },
        task,
      ),
    ).rejects.toBeInstanceOf(SdcppModelError);
    expect(task).not.toHaveBeenCalled();
    expect(lock.acquire).toHaveBeenCalledWith(`stable-diffusion:sdcpp:${url}`, 60_000);
  });
});
