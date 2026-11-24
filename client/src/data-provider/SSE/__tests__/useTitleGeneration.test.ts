let mockActiveJobIds: string[] = [];
let mockTiming: 'immediate' | 'final' = 'immediate';

const mockSetQueryData = jest.fn();
const mockGetQueryData = jest.fn();
const mockUpdateConvoInAllQueries = jest.fn();
const mockFindManualTitle = jest.fn();
const mockSetDocumentTitle = jest.fn();
const mockStreams = new Map<string, (state: Record<string, unknown>) => void>();
const mockObserveTitleGeneration = jest.fn(
  ({
    conversationId,
    onState,
  }: {
    conversationId: string;
    onState: (state: Record<string, unknown>) => void;
  }) => {
    mockStreams.set(conversationId, onState);
    return new Promise<void>(() => undefined);
  },
);

jest.mock('@tanstack/react-query', () => ({
  useQuery: jest.fn(() => ({ data: { activeJobIds: mockActiveJobIds } })),
  useQueryClient: jest.fn(() => ({
    setQueryData: mockSetQueryData,
    getQueryData: mockGetQueryData,
  })),
}));

jest.mock('../../Endpoints', () => ({
  useGetStartupConfig: () => ({ data: { titleGenerationTiming: mockTiming } }),
}));

jest.mock('~/utils', () => ({
  findManualConvoTitleInAllQueries: (...args: unknown[]) => mockFindManualTitle(...args),
  updateConvoInAllQueries: (...args: unknown[]) => mockUpdateConvoInAllQueries(...args),
  setDocumentTitle: (...args: unknown[]) => mockSetDocumentTitle(...args),
}));

jest.mock('librechat-data-provider', () => ({
  apiBaseUrl: () => '',
  QueryKeys: { conversation: 'conversation', activeJobs: 'activeJobs' },
  request: { get: jest.fn() },
  dataService: {
    observeTitleGeneration: (request: {
      conversationId: string;
      onState: (state: Record<string, unknown>) => void;
    }) => mockObserveTitleGeneration(request),
    getActiveJobs: jest.fn(),
  },
}));

import { act, renderHook, waitFor } from '@testing-library/react';
import {
  useTitleGeneration,
  queueTitleGeneration,
  clearTitleGenerationState,
  isTitleGenerationProcessed,
  markTitleGenerationProcessed,
} from '../queries';

const isSubscribed = (id: string) =>
  mockObserveTitleGeneration.mock.calls.some(([request]) => request.conversationId === id);

beforeEach(() => {
  act(() => clearTitleGenerationState());
  mockActiveJobIds = [];
  mockTiming = 'immediate';
  mockStreams.clear();
  jest.clearAllMocks();
  mockGetQueryData.mockReset();
  mockFindManualTitle.mockReset().mockImplementation((_client, id, incoming) => {
    const cached = mockGetQueryData(['conversation', id]);
    if (cached?.titleSetByUser) {
      return cached;
    } else if (incoming.titleSetByUser) {
      return incoming;
    }
  });
});

describe('useTitleGeneration eligibility', () => {
  it('subscribes in immediate mode while the chat stream is active', async () => {
    mockActiveJobIds = ['conv-immediate'];
    renderHook(() => useTitleGeneration(true));

    act(() => queueTitleGeneration('conv-immediate'));

    await waitFor(() => expect(isSubscribed('conv-immediate')).toBe(true));
  });

  it('waits for chat completion in final mode', async () => {
    mockTiming = 'final';
    mockActiveJobIds = ['conv-final'];
    const { rerender } = renderHook(() => useTitleGeneration(true));

    act(() => queueTitleGeneration('conv-final'));
    expect(isSubscribed('conv-final')).toBe(false);

    mockActiveJobIds = [];
    rerender();

    await waitFor(() => expect(isSubscribed('conv-final')).toBe(true));
  });

  it('does not subscribe while unauthenticated', () => {
    renderHook(() => useTitleGeneration(false));

    act(() => queueTitleGeneration('conv-disabled'));

    expect(isSubscribed('conv-disabled')).toBe(false);
  });

  it('stops observing a title completed by the main chat SSE', async () => {
    renderHook(() => useTitleGeneration(true));
    act(() => queueTitleGeneration('conv-main-sse'));
    await waitFor(() => expect(isSubscribed('conv-main-sse')).toBe(true));

    act(() => markTitleGenerationProcessed('conv-main-sse'));

    expect(isTitleGenerationProcessed('conv-main-sse')).toBe(true);
  });
});

describe('useTitleGeneration state handling', () => {
  it('applies a completed title to point and list caches', async () => {
    const id = 'conv-complete';
    mockGetQueryData.mockReturnValue({ title: 'New Chat' });
    renderHook(() => useTitleGeneration(true));
    act(() => queueTitleGeneration(id));
    await waitFor(() => expect(mockStreams.has(id)).toBe(true));

    act(() => {
      mockStreams.get(id)?.({
        conversationId: id,
        jobId: 'job',
        status: 'complete',
        title: 'Generated title',
        updatedAt: 1,
      });
    });

    const update = mockSetQueryData.mock.calls.find(([key]) => key[1] === id)?.[1];
    expect(update({ title: 'New Chat' })).toEqual({ title: 'Generated title' });
    expect(mockUpdateConvoInAllQueries).toHaveBeenCalledWith(
      expect.anything(),
      id,
      expect.any(Function),
    );
    expect(isTitleGenerationProcessed(id)).toBe(true);
  });

  it('preserves a manual title when a generated result arrives late', async () => {
    const id = 'conv-manual';
    const manual = { title: 'Manual title', titleSetByUser: true, titleRevision: 3 };
    mockGetQueryData.mockReturnValue(manual);
    renderHook(() => useTitleGeneration(true));
    act(() => queueTitleGeneration(id));
    await waitFor(() => expect(mockStreams.has(id)).toBe(true));

    act(() => {
      mockStreams.get(id)?.({
        conversationId: id,
        jobId: 'job',
        status: 'complete',
        title: 'Generated title',
        updatedAt: 1,
      });
    });

    const update = mockSetQueryData.mock.calls.find(([key]) => key[1] === id)?.[1];
    expect(update(manual)).toEqual(manual);
  });

  it.each(['error', 'cancelled', 'discarded', 'unavailable'])(
    'finishes observation for the %s state',
    async (status) => {
      const id = `conv-${status}`;
      renderHook(() => useTitleGeneration(true));
      act(() => queueTitleGeneration(id));
      await waitFor(() => expect(mockStreams.has(id)).toBe(true));

      act(() => {
        mockStreams.get(id)?.({
          conversationId: id,
          jobId: 'job',
          status,
          updatedAt: 1,
        });
      });

      expect(isTitleGenerationProcessed(id)).toBe(true);
      expect(mockSetQueryData).not.toHaveBeenCalled();
    },
  );
});
