import {
  getSdcppCatalog,
  parseSdcppRecipes,
  noteSdcppModelLoaded,
  parseSdcppModelStatus,
  clearSdcppCatalogCache,
} from './catalog';

/** Trimmed from a live `sd-server` `/sdcpp/v1/recipes` response. */
const recipesResponse = {
  errors: [],
  recipes: [
    {
      id: 'anima',
      name: 'Anima',
      source: 'builtin',
      models: null,
      size_multiple: 8,
      defaults: { width: 1024, height: 1024, steps: 20 },
      effective_features: { img2img: true },
    },
    {
      id: 'local-qwen-image-21-q8',
      name: 'Local · Qwen Image 2.1 · Q8',
      source: 'user',
      description: 'Qwen Image 2.1 with Q8 diffusion weights.',
      models: { diffusion_model: 'diffusion_models/qwen-image-2.1-Q8_0.gguf' },
      size_multiple: 32,
      video_constraints: null,
      defaults: { cfg_scale: 1, width: 1024, height: 1024, steps: 20 },
      effective_features: { img2img: true, init_image: false },
    },
    {
      id: 'local-ltx25-distilled-q4',
      name: 'Local · LTX 2.5 distilled · Q4',
      source: 'user',
      models: { diffusion_model: 'diffusion_models/ltx-2.5-22b-distilled-Q4_K_M.gguf' },
      size_multiple: 32,
      video_constraints: { frame_offset: 1, frame_step: 8, min_frames: 1 },
      defaults: { width: 768, height: 512, video_frames: 121, fps: 24 },
      effective_features: { img2img: false, init_image: true, end_image: true },
    },
    {
      id: 'local-wan22-t2v-q4',
      source: 'user',
      models: { diffusion_model: 'diffusion_models/Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf' },
      defaults: { width: 832, height: 480, video_frames: 33 },
      effective_features: { img2img: false, init_image: false },
    },
  ],
};

const statusResponse = {
  error: null,
  recipe: 'local-qwen-image-21-q8',
  state: 'ready',
  models: {
    diffusion_model: 'diffusion_models/qwen-image-2.1-Q8_0.gguf',
    vae: 'vae/qwen-image-2.1-vae.safetensors',
  },
};

describe('parseSdcppRecipes', () => {
  it('lists the recipes that name model files, with their mode and input support', () => {
    expect(parseSdcppRecipes(recipesResponse)).toEqual([
      {
        id: 'local-qwen-image-21-q8',
        name: 'Local · Qwen Image 2.1 · Q8',
        description: 'Qwen Image 2.1 with Q8 diffusion weights.',
        mode: 'image',
        acceptsImage: true,
        sizeMultiple: 32,
        width: 1024,
        height: 1024,
        steps: 20,
        cfgScale: 1,
      },
      {
        id: 'local-ltx25-distilled-q4',
        name: 'Local · LTX 2.5 distilled · Q4',
        mode: 'video',
        acceptsImage: true,
        sizeMultiple: 32,
        width: 768,
        height: 512,
      },
      {
        id: 'local-wan22-t2v-q4',
        name: 'local-wan22-t2v-q4',
        mode: 'video',
        acceptsImage: false,
        sizeMultiple: 64,
        width: 832,
        height: 480,
      },
    ]);
  });

  it.each([[null], [[]], [{}], [{ recipes: {} }], ['recipes']])(
    'rejects %j, which is not a recipes response',
    (data) => {
      expect(parseSdcppRecipes(data)).toBeUndefined();
    },
  );

  it('skips malformed recipes', () => {
    expect(
      parseSdcppRecipes({ recipes: [null, 'x', { id: '', models: {} }, { id: 'a', models: [] }] }),
    ).toEqual([]);
  });
});

describe('parseSdcppModelStatus', () => {
  it('reads the loaded recipe and its model files', () => {
    expect(parseSdcppModelStatus(statusResponse)).toEqual({
      recipe: 'local-qwen-image-21-q8',
      state: 'ready',
      models: statusResponse.models,
    });
  });

  it('reports a load error', () => {
    expect(parseSdcppModelStatus({ error: 'out of memory', models: {} })).toEqual({
      error: 'out of memory',
      models: {},
    });
  });
});

describe('getSdcppCatalog', () => {
  const url = 'http://sd.local:7860';
  const probe = jest.fn();
  const respond = (recipes: unknown, status: unknown = { status: 200, data: statusResponse }) =>
    probe.mockImplementation(async (requestUrl: string) => {
      const answer = requestUrl.endsWith('/recipes') ? recipes : status;
      if (answer instanceof Error) {
        throw answer;
      }
      return answer;
    });
  const listed = (loaded: string) => ({
    status: 'listed',
    catalog: expect.objectContaining({ loaded }),
  });

  beforeEach(() => {
    clearSdcppCatalogCache();
    probe.mockReset();
  });

  it('reads the models and the loaded one, then reuses the answer', async () => {
    respond({ status: 200, data: recipesResponse });
    const lookup = await getSdcppCatalog(url, probe, 0);
    expect(lookup.status).toBe('listed');
    const catalog = lookup.status === 'listed' ? lookup.catalog : undefined;
    expect(catalog?.loaded).toBe('local-qwen-image-21-q8');
    expect(catalog?.models.map((model) => model.id)).toEqual([
      'local-qwen-image-21-q8',
      'local-ltx25-distilled-q4',
      'local-wan22-t2v-q4',
    ]);
    expect(probe).toHaveBeenCalledWith(`${url}/sdcpp/v1/recipes`);
    expect(probe).toHaveBeenCalledWith(`${url}/sdcpp/v1/models/status`);

    probe.mockClear();
    await expect(getSdcppCatalog(url, probe, 1000)).resolves.toEqual(lookup);
    expect(probe).not.toHaveBeenCalled();
  });

  it('caches that a server without the recipes route has no catalog', async () => {
    respond({ status: 404, data: 'Not Found' });
    await expect(getSdcppCatalog(url, probe, 0)).resolves.toEqual({ status: 'unsupported' });
    probe.mockClear();
    await expect(getSdcppCatalog(url, probe, 1000)).resolves.toEqual({ status: 'unsupported' });
    expect(probe).not.toHaveBeenCalled();
  });

  it('keeps the last catalog when the server stops answering', async () => {
    respond({ status: 200, data: recipesResponse });
    const lookup = await getSdcppCatalog(url, probe, 0);

    respond(new Error('ECONNREFUSED'), new Error('ECONNREFUSED'));
    await expect(getSdcppCatalog(url, probe, 10 * 60 * 1000)).resolves.toEqual(lookup);
    respond({ status: 503, data: {} });
    await expect(getSdcppCatalog(url, probe, 20 * 60 * 1000)).resolves.toEqual(lookup);
  });

  it('reports a server that has never answered as unavailable, without remembering it', async () => {
    respond(new Error('ETIMEDOUT'), new Error('ETIMEDOUT'));
    await expect(getSdcppCatalog(url, probe, 0)).resolves.toEqual({ status: 'unavailable' });
    respond({ status: 503, data: {} });
    await expect(getSdcppCatalog(url, probe, 1)).resolves.toEqual({ status: 'unavailable' });
    respond({ status: 200, data: recipesResponse });
    await expect(getSdcppCatalog(url, probe, 2)).resolves.toEqual(listed('local-qwen-image-21-q8'));
  });

  it('keeps the last known loaded model when only the status fails', async () => {
    respond({ status: 200, data: recipesResponse });
    await getSdcppCatalog(url, probe, 0);
    respond({ status: 200, data: recipesResponse }, new Error('ECONNRESET'));
    await expect(getSdcppCatalog(url, probe, 10 * 60 * 1000)).resolves.toEqual(
      listed('local-qwen-image-21-q8'),
    );
  });

  it('names a model this process loaded', async () => {
    respond({ status: 200, data: recipesResponse });
    await getSdcppCatalog(url, probe, 0);
    noteSdcppModelLoaded(url, 'local-ltx25-distilled-q4');
    await expect(getSdcppCatalog(url, probe, 1000)).resolves.toEqual(
      listed('local-ltx25-distilled-q4'),
    );
  });
});
