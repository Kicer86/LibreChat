import { logger } from '@librechat/data-schemas';
import type { TFile } from 'librechat-data-provider';
import type { ServerRequest } from '~/types';
import { resolveDownloadPath } from '~/storage/path';

type GetDownloadStream = (req: ServerRequest, filepath: string) => Promise<NodeJS.ReadableStream>;

/** An input image resolved from an image ID, ready to send to an image API. */
export interface ImageInput {
  file: TFile;
  buffer: Buffer;
  mimeType: string;
}

export interface LoadImageInputsParams {
  /** Image IDs chosen by the model, in the order the tool should use them. */
  imageIds?: string[] | null;
  /** Images attached to the current request (the `image_edit` tool resource). */
  requestFiles: TFile[];
  req?: ServerRequest;
  /** Storage used when a file record carries no `source`. */
  fileStrategy?: string;
  /**
   * Looks up the requesting user's own image files (with dimensions) by ID; used for IDs that
   * are not attached to this request, such as images generated earlier in the conversation.
   */
  findImageFiles: (userId: string, fileIds: string[]) => Promise<TFile[]>;
  getStrategyFunctions: (source: string) => { getDownloadStream?: GetDownloadStream };
  /** Prefix for log lines, e.g. the tool name. */
  logPrefix?: string;
}

/**
 * Resolves image IDs to image bytes, in the order given. IDs attached to the request are used as
 * is; other IDs are looked up among the requesting user's own files only. Unknown IDs and images
 * that cannot be read are skipped (and logged), so the result may be shorter than `imageIds`.
 */
export async function loadImageInputs({
  imageIds,
  requestFiles,
  req,
  fileStrategy,
  findImageFiles,
  getStrategyFunctions,
  logPrefix = '[loadImageInputs]',
}: LoadImageInputsParams): Promise<ImageInput[]> {
  if (!imageIds?.length) {
    return [];
  }

  const filesById = new Map(requestFiles.map((file) => [file.file_id, file]));
  const missing = imageIds.filter((id) => !filesById.has(id));
  const userId = req?.user?.id;
  if (missing.length && userId) {
    for (const file of await findImageFiles(userId, missing)) {
      filesById.set(file.file_id, file);
    }
  }

  const inputs: ImageInput[] = [];
  for (const id of imageIds) {
    const file = filesById.get(id);
    if (!file) {
      logger.warn(`${logPrefix} Image not found or not accessible: ${id}`);
      continue;
    }
    const source = file.source || fileStrategy;
    const getDownloadStream = source ? getStrategyFunctions(source).getDownloadStream : undefined;
    if (!getDownloadStream || !req) {
      logger.warn(`${logPrefix} No download stream for image ${id} (source: ${source})`);
      continue;
    }
    try {
      const stream = await getDownloadStream(req, resolveDownloadPath(file));
      const chunks: Uint8Array[] = [];
      for await (const chunk of stream as AsyncIterable<Uint8Array>) {
        chunks.push(chunk);
      }
      inputs.push({ file, buffer: Buffer.concat(chunks), mimeType: file.type || 'image/png' });
    } catch (error) {
      logger.error(`${logPrefix} Error reading image ${id}:`, error);
    }
  }
  return inputs;
}
