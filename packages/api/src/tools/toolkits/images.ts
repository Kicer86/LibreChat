import { logger } from '@librechat/data-schemas';
import type { TFile } from 'librechat-data-provider';
import type { ServerRequest } from '~/types';
import { resolveDownloadPath } from '~/storage/path';

type GetDownloadStream = (req: ServerRequest, filepath: string) => Promise<NodeJS.ReadableStream>;

/** An input image resolved from an image ID, ready to send to an image API. */
export interface ImageInput {
  file: Pick<TFile, 'file_id'> & Partial<Pick<TFile, 'width' | 'height'>>;
  buffer: Buffer;
  mimeType: string;
}

/**
 * Images generated during the current request, held until their file records exist.
 *
 * A generated image is saved as a file only after the tool returns (the tool-end callback stores
 * it in the background), but the model can pass the returned `generated_image_id` to the next
 * tool call right away. Tools are created per call, so the images are kept per request, the one
 * object every tool call of an agent run shares, and are released with it.
 */
export interface GeneratedImageRegistry {
  remember(req: ServerRequest | undefined, image: ImageInput): void;
  find(req: ServerRequest | undefined, fileId: string): ImageInput | undefined;
}

export function createGeneratedImageRegistry(): GeneratedImageRegistry {
  const byRequest = new WeakMap<ServerRequest, Map<string, ImageInput>>();
  return {
    remember(req, image) {
      if (!req) {
        return;
      }
      const images = byRequest.get(req) ?? new Map<string, ImageInput>();
      images.set(image.file.file_id, image);
      byRequest.set(req, images);
    },
    find(req, fileId) {
      return req ? byRequest.get(req)?.get(fileId) : undefined;
    },
  };
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
  /** Images generated earlier in this request whose files may not be saved yet. */
  generatedImages?: GeneratedImageRegistry;
  /** Prefix for log lines, e.g. the tool name. */
  logPrefix?: string;
}

/**
 * Resolves image IDs to image bytes, in the order given. IDs attached to the request or generated
 * earlier in it are used as is; other IDs are looked up among the requesting user's own files only. Unknown IDs and images
 * that cannot be read are skipped (and logged), so the result may be shorter than `imageIds`.
 */
export async function loadImageInputs({
  imageIds,
  requestFiles,
  req,
  fileStrategy,
  findImageFiles,
  getStrategyFunctions,
  generatedImages,
  logPrefix = '[loadImageInputs]',
}: LoadImageInputsParams): Promise<ImageInput[]> {
  if (!imageIds?.length) {
    return [];
  }

  const filesById = new Map(requestFiles.map((file) => [file.file_id, file]));
  const generated = new Map<string, ImageInput>();
  for (const id of imageIds) {
    const image = filesById.has(id) ? undefined : generatedImages?.find(req, id);
    if (image) {
      generated.set(id, image);
    }
  }
  const missing = imageIds.filter((id) => !filesById.has(id) && !generated.has(id));
  const userId = req?.user?.id;
  if (missing.length && userId) {
    for (const file of await findImageFiles(userId, missing)) {
      filesById.set(file.file_id, file);
    }
  }

  const inputs: ImageInput[] = [];
  for (const id of imageIds) {
    const image = generated.get(id);
    if (image) {
      inputs.push(image);
      continue;
    }
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
