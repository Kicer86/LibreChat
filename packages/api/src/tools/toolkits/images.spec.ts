import { Readable } from 'stream';
import { FileSources } from 'librechat-data-provider';
import type { TFile } from 'librechat-data-provider';
import type { ServerRequest } from '~/types';
import { loadImageInputs } from './images';

const image = (file_id: string, extra: Partial<TFile> = {}): TFile =>
  ({
    file_id,
    filepath: `/images/user-1/${file_id}.png`,
    type: 'image/png',
    width: 64,
    height: 64,
    ...extra,
  }) as TFile;

const req = { user: { id: 'user-1' } } as unknown as ServerRequest;

describe('loadImageInputs', () => {
  const getDownloadStream = jest.fn(async (_req: ServerRequest, filepath: string) =>
    Readable.from([Buffer.from(`bytes:${filepath}`)]),
  );
  const getStrategyFunctions = jest.fn(() => ({ getDownloadStream }));
  const findImageFiles = jest.fn(async (_userId: string, _ids: string[]): Promise<TFile[]> => []);

  beforeEach(() => jest.clearAllMocks());

  it('returns nothing without image ids', async () => {
    const result = await loadImageInputs({
      imageIds: [],
      requestFiles: [image('a')],
      req,
      findImageFiles,
      getStrategyFunctions,
    });
    expect(result).toEqual([]);
    expect(getDownloadStream).not.toHaveBeenCalled();
  });

  it('reads attached images in the requested order, looking up the rest for the user', async () => {
    findImageFiles.mockResolvedValueOnce([image('generated', { source: FileSources.s3 })]);
    const result = await loadImageInputs({
      imageIds: ['generated', 'attached'],
      requestFiles: [image('attached', { source: FileSources.local, type: 'image/jpeg' })],
      req,
      findImageFiles,
      getStrategyFunctions,
    });

    expect(findImageFiles).toHaveBeenCalledWith('user-1', ['generated']);
    expect(result.map((input) => input.file.file_id)).toEqual(['generated', 'attached']);
    expect(result[1].mimeType).toBe('image/jpeg');
    expect(result[0].buffer.toString()).toBe('bytes:/images/user-1/generated.png');
    expect(getStrategyFunctions).toHaveBeenCalledWith(FileSources.s3);
  });

  it('prefers the storage key over the URL path when downloading', async () => {
    await loadImageInputs({
      imageIds: ['a'],
      requestFiles: [
        image('a', { source: FileSources.s3, storageKey: 'key/a.png' } as Partial<TFile>),
      ],
      req,
      findImageFiles,
      getStrategyFunctions,
    });
    expect(getDownloadStream).toHaveBeenCalledWith(req, 'key/a.png');
  });

  it('falls back to the configured strategy and skips unknown or unreadable images', async () => {
    getDownloadStream.mockRejectedValueOnce(new Error('gone'));
    const result = await loadImageInputs({
      imageIds: ['broken', 'unknown', 'ok'],
      requestFiles: [image('broken'), image('ok')],
      req,
      fileStrategy: 'local',
      findImageFiles,
      getStrategyFunctions,
    });
    expect(getStrategyFunctions).toHaveBeenCalledWith('local');
    expect(result.map((input) => input.file.file_id)).toEqual(['ok']);
  });

  it('does not look up files without a user', async () => {
    const result = await loadImageInputs({
      imageIds: ['other'],
      requestFiles: [],
      findImageFiles,
      getStrategyFunctions,
    });
    expect(findImageFiles).not.toHaveBeenCalled();
    expect(result).toEqual([]);
  });
});
