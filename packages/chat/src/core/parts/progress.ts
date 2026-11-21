import type { ToolProgress } from 'librechat-data-provider';

type Localize = (key: string, values?: Record<string, string | number>) => string;

/**
 * The words for where a running tool call is, from the progress it reported: its place in a
 * queue, loading a model, preparing, the step (and the image of a batch), or finishing. A stage
 * without words of its own gives the share done, and nothing when that is unknown too.
 */
export function formatToolProgress(report: ToolProgress, localize: Localize): string | undefined {
  const { stage, completed, total, item, items, queuePosition, progress } = report;
  if (stage === 'queued') {
    return queuePosition
      ? localize('com_ui_tool_progress_queued_behind', { 0: queuePosition })
      : localize('com_ui_tool_progress_queued');
  }
  if (stage === 'loading_model') {
    return localize('com_ui_tool_progress_loading_model');
  }
  if (stage === 'preparing') {
    return localize('com_ui_tool_progress_preparing');
  }
  if (stage === 'sampling' && completed != null && total != null) {
    return item != null && items != null
      ? localize('com_ui_tool_progress_step_of_item', { 0: completed, 1: total, 2: item, 3: items })
      : localize('com_ui_tool_progress_step', { 0: completed, 1: total });
  }
  if (stage === 'decoding' || stage === 'encoding') {
    return localize('com_ui_tool_progress_finishing');
  }
  return progress != null ? `${Math.round(progress * 100)}%` : undefined;
}
