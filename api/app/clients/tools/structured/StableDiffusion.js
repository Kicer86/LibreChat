// Generates image using stable diffusion webui's api (automatic1111)
const fs = require('fs');
const path = require('path');
const axios = require('axios');
const sharp = require('sharp');
const { v4: uuidv4 } = require('uuid');
const { logger } = require('@librechat/data-schemas');
const { Tool } = require('@librechat/agents/langchain/tools');
const { FileContext, ContentTypes } = require('librechat-data-provider');
const {
  getBasePath,
  loadImageInputs,
  stableDiffusionSchema,
  configuredRequestBudget,
  applySSRFSafeAgentIfDirect,
  buildStableDiffusionRequest,
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

    this.name = 'stable-diffusion';
    this.url = fields.SD_WEBUI_URL || this.getServerURL();
    this.isUserProvidedEndpoint = fields.userProvidedAuthFields?.has('SD_WEBUI_URL') === true;
    this.description_for_model = `// Generate images and visuals using text.
// Guidelines:
// - ALWAYS use {{"prompt": "7+ detailed keywords", "negative_prompt": "7+ detailed keywords"}} structure for queries.
// - ALWAYS include the markdown url in your final response to show the user: ![caption](${getBasePath()}/images/id.png)
// - Visually describe the moods, details, structures, styles, and/or proportions of the image. Remember, the focus is on visual attributes.
// - Craft your input by "showing" and not "telling" the imagery. Think in terms of what you'd want to see in a photograph or a painting.
// - Here's an example for generating a realistic portrait photo of a man:
// "prompt":"photo of a man in black clothes, half body, high detailed skin, coastline, overcast weather, wind, waves, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
// "negative_prompt":"semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, out of frame, low quality, ugly, mutation, deformed"
// - To change an existing image instead (img2img), pass its ID in "image_ids" and optionally "denoising_strength" (0-1, how much it may change).
// - Generate images only once per human query unless explicitly requested by the user`;
    this.description =
      "You can generate images using text with 'stable-diffusion', or change an existing image (img2img) by passing its ID in image_ids. This tool is exclusively for visual content.";
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
   * Resolves the first requested image ID to image bytes for img2img.
   * @returns {Promise<import('@librechat/api').ImageInput | undefined>}
   */
  async loadInputImage(image_ids) {
    if (!this.findImageFiles || !this.getStrategyFunctions) {
      return undefined;
    }
    const [image] = await loadImageInputs({
      imageIds: image_ids.slice(0, 1),
      requestFiles: this.imageFiles,
      req: this.req,
      fileStrategy: this.fileStrategy,
      findImageFiles: this.findImageFiles,
      getStrategyFunctions: this.getStrategyFunctions,
      logPrefix: '[StableDiffusion]',
    });
    return image;
  }

  async _call(data) {
    const url = this.url;
    const { prompt, negative_prompt, image_ids, denoising_strength } = data;
    let inputImage;
    if (image_ids?.length) {
      inputImage = await this.loadInputImage(image_ids);
      if (!inputImage) {
        return this.returnValue(
          'The image to change could not be loaded. Use the ID of an image attached to this request or generated earlier.',
        );
      }
    }
    const { path: apiPath, payload } = buildStableDiffusionRequest({
      prompt,
      negative_prompt,
      image: inputImage,
      denoising_strength,
    });
    let generationResponse;
    try {
      const requestUrl = `${url}${apiPath}`;
      const requestConfig = {
        ...(this.isUserProvidedEndpoint ? applySSRFSafeAgentIfDirect({}, requestUrl) : {}),
        ...configuredRequestBudget(this.req?.config, 'imageGenerationMs'),
      };
      generationResponse = await axios.post(requestUrl, payload, requestConfig);
    } catch (error) {
      logger.error('[StableDiffusion] Error while generating image:', error);
      return this.returnValue('Error making API request.');
    }
    const image = generationResponse.data.images[0];

    /** @type {{ height: number, width: number, seed: number, infotexts: string[] }} */
    let info = {};
    try {
      info = JSON.parse(generationResponse.data.info);
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
      if (this.isAgent) {
        const content = [
          {
            type: ContentTypes.IMAGE_URL,
            image_url: {
              url: `data:image/png;base64,${image}`,
            },
          },
        ];

        const response = [
          {
            type: ContentTypes.TEXT,
            text: displayMessage,
          },
        ];
        return [response, { content }];
      }

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
