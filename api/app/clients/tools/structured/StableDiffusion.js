// Generates image using stable diffusion webui's api (automatic1111)
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const { v4: uuidv4 } = require('uuid');
const { logger } = require('@librechat/data-schemas');
const { Tool } = require('@librechat/agents/langchain/tools');
const { FileContext } = require('librechat-data-provider');
const {
  getBasePath,
  stableDiffusionSchema,
  generateStableDiffusionImage,
  buildStableDiffusionAgentOutput,
} = require('@librechat/api');
const paths = require('~/config/paths');

const displayMessage =
  "Stable Diffusion displayed an image. All generated images are already plainly visible, so don't repeat the descriptions in detail. Do not list download links as they are available in the UI already. The user may download the images by clicking on them, but do not mention anything about downloading to the user.";

class StableDiffusionAPI extends Tool {
  constructor(fields) {
    super();
    /** @type {string} User ID */
    this.userId = fields.userId;
    /** @type {ServerRequest | undefined} Express Request object, only provided by ToolService */
    this.req = fields.req;
    /** @type {boolean} Used to initialize the Tool without necessary variables. */
    this.override = fields.override ?? false;
    /** @type {boolean} Necessary for output to contain all image metadata. */
    this.returnMetadata = fields.returnMetadata ?? false;
    /** @type {boolean} */
    this.isAgent = fields.isAgent;
    if (this.isAgent) {
      /** Ensures LangChain maps [content, artifact] tuple to ToolMessage fields instead of serializing it into content. */
      this.responseFormat = 'content_and_artifact';
    }
    if (fields.uploadImageBuffer) {
      /** @type {uploadImageBuffer} Necessary for output to contain all image metadata. */
      this.uploadImageBuffer = fields.uploadImageBuffer.bind(this);
    }
    /** @type {MongoFile[]} Images attached to the request, usable as img2img input */
    this.imageFiles = fields.imageFiles ?? [];
    /** @type {string | undefined} */
    this.fileStrategy = fields.fileStrategy;
    /** Looks up the user's own image files by ID; required for img2img */
    this.findImageFiles = fields.findImageFiles;
    /** Storage strategy resolver; required for img2img */
    this.getStrategyFunctions = fields.getStrategyFunctions;
    /** @type {import('@librechat/api').GeneratedImageRegistry | undefined} Images generated in this request, before their files are saved */
    this.generatedImages = fields.generatedImages;
    /** @type {import('@librechat/api').DistributedLock | undefined} Serializes model switches on a stable-diffusion.cpp server */
    this.lock = fields.lock;

    this.name = 'stable-diffusion';
    this.url = fields.SD_WEBUI_URL || this.getServerURL();
    this.isUserProvidedEndpoint = fields.userProvidedAuthFields?.has('SD_WEBUI_URL') === true;
    this.description_for_model = `// Generate images and visuals using text.
// Guidelines:
// - ALWAYS use {{"prompt": "7+ detailed keywords", "negative_prompt": "7+ detailed keywords"}} structure for queries; "negative_prompt" is optional.
// - ALWAYS include the markdown url in your final response to show the user: ![caption](${getBasePath()}/images/id.png)
// - Visually describe the moods, details, structures, styles, and/or proportions of the image. Remember, the focus is on visual attributes.
// - Craft your input by "showing" and not "telling" the imagery. Think in terms of what you'd want to see in a photograph or a painting.
// - Here's an example for generating a realistic portrait photo of a man:
// "prompt":"photo of a man in black clothes, half body, high detailed skin, coastline, overcast weather, wind, waves, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
// "negative_prompt":"semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, out of frame, low quality, ugly, mutation, deformed"
// - To change an existing image instead (img2img), pass its ID in "image_ids" and "denoising_strength" (0-1, how much it may change). img2img redraws the whole image from the prompt, so describe the full result; a color change needs about 0.85.
// - When the tool context lists image models, set "model" only if another model suits the request better than the loaded one; loading a model takes extra time.
// - "width", "height", "steps", "cfg_scale", "seed", "count", "sampler" and "scheduler" are optional; omit them unless the user asks for a size, a format, a seed or other settings, since each model has suitable defaults. To make a variation of an earlier image, reuse its seed and change the prompt or settings.
// - When the tool context lists video models and the user asks for a video or an animation, set "video": true (and "model" to choose a video model). Describe the motion as well as the scene. "frames" or "duration" and "fps" are optional; a clip takes minutes, so make one per request.
// - To animate an image, or to continue a clip made earlier, pass its ID (for a clip, its generated_video_id) in "image_ids" with "video": true; to end the clip on an image, pass that ID in "end_image_id". Use a video model the tool context lists as starting from or ending on an image.
// - Generate images only once per human query unless explicitly requested by the user`;
    this.description =
      "You can generate images using text with 'stable-diffusion', change an existing image (img2img) by passing its ID in image_ids, or make a short video clip when video models are listed. This tool is exclusively for visual content.";
    this.schema = stableDiffusionSchema;
  }

  static get jsonSchema() {
    return stableDiffusionSchema;
  }

  replaceNewLinesWithSpaces(inputString) {
    return inputString.replace(/\r\n|\r|\n/g, ' ');
  }

  getMarkdownImageUrl(imageName) {
    const imageUrl = path
      .join(this.relativePath, this.userId, imageName)
      .replace(/\\/g, '/')
      .replace('public/', '');
    return `![generated image](/${imageUrl})`;
  }

  returnValue(value) {
    if (this.isAgent === true && typeof value === 'string') {
      return [value, {}];
    } else if (this.isAgent === true && typeof value === 'object') {
      return [displayMessage, value];
    }

    return value;
  }

  getServerURL() {
    const url = process.env.SD_WEBUI_URL || '';
    if (!url && !this.override) {
      throw new Error('Missing SD_WEBUI_URL environment variable.');
    }
    return url;
  }

  /**
   * @param {object} data
   * @param {unknown} [_runManager]
   * @param {{ signal?: AbortSignal, toolCall?: { reportProgress?: import('@librechat/api').ToolProgressReporter } }} [runnableConfig]
   */
  async _call(data, _runManager, runnableConfig) {
    const { prompt, negative_prompt } = data;
    let generation;
    try {
      generation = await generateStableDiffusionImage(data, {
        url: this.url,
        req: this.req,
        isUserProvidedEndpoint: this.isUserProvidedEndpoint,
        imageFiles: this.imageFiles,
        fileStrategy: this.fileStrategy,
        findImageFiles: this.findImageFiles,
        getStrategyFunctions: this.getStrategyFunctions,
        generatedImages: this.generatedImages,
        lock: this.lock,
        signal: runnableConfig?.signal,
        allowVideo: this.isAgent === true,
        onProgress: runnableConfig?.toolCall?.reportProgress,
      });
      if ('error' in generation) {
        return this.returnValue(generation.error);
      }
    } catch (error) {
      logger.error('[StableDiffusion] Error while generating image:', error);
      return this.returnValue('Error making API request.');
    }
    if (this.isAgent) {
      return buildStableDiffusionAgentOutput({
        generation,
        model: generation.model,
        displayMessage,
        createFileId: uuidv4,
        req: this.req,
        generatedImages: this.generatedImages,
      });
    }

    const image = generation.data.images[0];

    /** @type {{ height: number, width: number, seed: number, infotexts: string[] }} */
    let info = {};
    try {
      info = JSON.parse(generation.data.info);
    } catch (error) {
      logger.error('[StableDiffusion] Error while getting image metadata:', error);
    }

    const file_id = uuidv4();
    const imageName = `${file_id}.png`;
    const { imageOutput: imageOutputPath, clientPath } = paths;
    const filepath = path.join(imageOutputPath, this.userId, imageName);
    this.relativePath = path.relative(clientPath, imageOutputPath);

    if (!fs.existsSync(path.join(imageOutputPath, this.userId))) {
      fs.mkdirSync(path.join(imageOutputPath, this.userId), { recursive: true });
    }

    try {
      const buffer = Buffer.from(image.split(',', 1)[0], 'base64');
      if (this.returnMetadata && this.uploadImageBuffer && this.req) {
        const file = await this.uploadImageBuffer({
          req: this.req,
          context: FileContext.image_generation,
          resize: false,
          metadata: {
            buffer,
            height: info.height,
            width: info.width,
            bytes: Buffer.byteLength(buffer),
            filename: imageName,
            type: 'image/png',
            file_id,
          },
        });

        const generationInfo = info.infotexts[0].split('\n').pop();
        return {
          ...file,
          prompt,
          metadata: {
            negative_prompt,
            seed: info.seed,
            info: generationInfo,
          },
        };
      }

      await sharp(buffer)
        .withMetadata({
          iptcpng: {
            parameters: info.infotexts[0],
          },
        })
        .toFile(filepath);
      this.result = this.getMarkdownImageUrl(imageName);
    } catch (error) {
      logger.error('[StableDiffusion] Error while saving the image:', error);
    }

    return this.returnValue(this.result);
  }
}

module.exports = StableDiffusionAPI;
