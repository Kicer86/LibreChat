import { atom } from 'jotai';
import { atomFamily } from 'jotai/utils';
import type { ToolProgress } from 'librechat-data-provider';
import type { Family } from '../steer/family';

/**
 * The latest progress a running tool call reported (`on_tool_progress` SSE event), keyed by
 * `tool_call_id`; null when it reported none. Cleared when the tool call's run step completes and
 * at the end of a run, so a later call reusing the id starts without it.
 */
export const toolProgressByToolCallId: Family<ToolProgress | null> = atomFamily(
  (_toolCallId: string) => atom<ToolProgress | null>(null),
);
