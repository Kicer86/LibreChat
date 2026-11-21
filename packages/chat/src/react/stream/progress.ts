import { atom } from 'jotai';
import { atomFamily } from 'jotai/utils';
import type { ToolProgress } from 'librechat-data-provider';
import type { Family } from '../steer/family';

/**
 * The key of one tool call's progress: the response message it belongs to and the call's id.
 * Provider call ids repeat across runs (`call_0`), so the id alone would let one run's reports
 * overwrite or clear another's.
 */
export const toolProgressKey = (parentMessageId: string, toolCallId: string): string =>
  `${parentMessageId}\u0000${toolCallId}`;

/**
 * The latest progress a running tool call reported (`on_tool_progress` SSE event), keyed by
 * `toolProgressKey`; null when it reported none. Cleared when the tool call's run step completes
 * and at the end of a run, so a later call reusing the id starts without it.
 */
export const toolProgressByToolCallId: Family<ToolProgress | null> = atomFamily((_key: string) =>
  atom<ToolProgress | null>(null),
);
