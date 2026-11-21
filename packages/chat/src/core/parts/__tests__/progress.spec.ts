import { formatToolProgress } from '../progress';

const localize = (key: string, values?: Record<string, string | number>) =>
  values ? `${key} ${JSON.stringify(values)}` : key;

describe('formatToolProgress', () => {
  it.each([
    [{ stage: 'queued' }, 'com_ui_tool_progress_queued'],
    [{ stage: 'queued', queuePosition: 2 }, 'com_ui_tool_progress_queued_behind {"0":2}'],
    [{ stage: 'loading_model', progress: 0 }, 'com_ui_tool_progress_loading_model'],
    [{ stage: 'preparing' }, 'com_ui_tool_progress_preparing'],
    [{ stage: 'sampling', completed: 3, total: 8 }, 'com_ui_tool_progress_step {"0":3,"1":8}'],
    [
      { stage: 'sampling', completed: 1, total: 4, item: 2, items: 2 },
      'com_ui_tool_progress_step_of_item {"0":1,"1":4,"2":2,"3":2}',
    ],
    [{ stage: 'decoding' }, 'com_ui_tool_progress_finishing'],
    [{ stage: 'encoding' }, 'com_ui_tool_progress_finishing'],
    [{ stage: 'working', progress: 0.42 }, '42%'],
  ])('words %j', (report, text) => {
    expect(formatToolProgress(report, localize)).toBe(text);
  });

  it('has nothing to say about a stage it does not know without a share done', () => {
    expect(formatToolProgress({ stage: 'working' }, localize)).toBeUndefined();
    expect(formatToolProgress({ stage: 'sampling' }, localize)).toBeUndefined();
  });
});
