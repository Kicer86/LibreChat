import { buildWebSearchContext, buildWebSearchDynamicContext } from './web';

jest.mock('librechat-data-provider', () => ({
  Tools: { web_search: 'web_search' },
  replaceSpecialVars: jest.fn(({ now }: { now?: string }) => now ?? 'NOW'),
}));

describe('web search context', () => {
  it('keeps static context free of volatile date replacements', () => {
    const context = buildWebSearchContext();

    expect(context).toContain('web_search');
    expect(context).not.toContain('NOW');
    expect(context).not.toContain('{{iso_datetime}}');
  });

  it('guides the model to answer directly when a search is not warranted', () => {
    const context = buildWebSearchContext();

    expect(context).toContain('respond directly without searching');
    expect(context).toContain('current, real-time, or otherwise beyond your own knowledge');
  });

  it('keeps dynamic context stable within a UTC day and advances on the next day', () => {
    const context = buildWebSearchDynamicContext('2024-01-02T03:04:05.000Z');
    const sameDayContext = buildWebSearchDynamicContext('2024-01-02T23:59:59.999Z');
    const nextDayContext = buildWebSearchDynamicContext('2024-01-03T00:00:00.000Z');

    expect(context).toBe('# `web_search` Runtime Context\nCurrent Date: 2024-01-02');
    expect(sameDayContext).toBe(context);
    expect(nextDayContext).toBe('# `web_search` Runtime Context\nCurrent Date: 2024-01-03');
  });
});
