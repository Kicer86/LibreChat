import { parseSdcppCapabilities, parseSdcppDefaults } from './capabilities';

describe('parseSdcppCapabilities', () => {
  it('reads the samplers, schedulers and limits', () => {
    expect(
      parseSdcppCapabilities({
        samplers: ['euler', 'dpm++2m'],
        schedulers: ['karras', 'simple'],
        limits: {
          min_width: 64,
          max_width: 4096,
          min_height: 64,
          max_height: 4096,
          max_batch_count: 8,
          max_queue_size: 64,
        },
      }),
    ).toEqual({
      samplers: ['euler', 'dpm++2m'],
      schedulers: ['karras', 'simple'],
      limits: { minWidth: 64, maxWidth: 4096, minHeight: 64, maxHeight: 4096, maxBatchCount: 8 },
    });
  });

  it('leaves out what the server does not state or states wrongly', () => {
    expect(
      parseSdcppCapabilities({
        samplers: 'euler',
        schedulers: ['karras', '', 3],
        limits: { max_width: -1, max_height: 1.5, max_batch_count: '8' },
      }),
    ).toEqual({ samplers: [], schedulers: ['karras'], limits: {} });
    expect(parseSdcppCapabilities({})).toEqual({ samplers: [], schedulers: [], limits: {} });
  });
});

describe('parseSdcppDefaults', () => {
  it("reads the loaded model's default size and steps", () => {
    expect(
      parseSdcppDefaults({
        defaults: {
          width: 1024,
          height: 768,
          sample_params: { sample_steps: 20, scheduler: 'default' },
        },
      }),
    ).toEqual({ width: 1024, height: 768, steps: 20 });
  });

  it('leaves out what is missing or malformed', () => {
    expect(parseSdcppDefaults({ defaults: { width: 0, sample_params: [] } })).toEqual({});
    expect(parseSdcppDefaults({})).toEqual({});
  });
});
