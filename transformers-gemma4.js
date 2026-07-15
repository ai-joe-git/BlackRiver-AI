import {
  AutoProcessor,
  Gemma4ForConditionalGeneration,
  InterruptableStoppingCriteria,
  TextStreamer,
  load_image,
  read_audio,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const MODEL_IDS = {
  "google/gemma-4-E2B-it-qat-mobile-transformers": "onnx-community/gemma-4-E2B-it-ONNX",
  "google/gemma-4-E4B-it-qat-mobile-transformers": "onnx-community/gemma-4-E4B-it-ONNX",
  "onnx-community/gemma-4-E2B-it-ONNX": "onnx-community/gemma-4-E2B-it-ONNX",
  "onnx-community/gemma-4-E4B-it-ONNX": "onnx-community/gemma-4-E4B-it-ONNX",
};

export class Gemma4Multimodal {
  static async load(requestedId, { onProgress = () => {} } = {}) {
    const modelId = MODEL_IDS[requestedId] ?? requestedId;
    onProgress({ status: "tokenizer", message: "Loading multimodal processor", fraction: 0 });
    const processor = await AutoProcessor.from_pretrained(modelId, {
      progress_callback: (info) => reportProgress(info, onProgress),
    });
    onProgress({ status: "weights", message: "Loading WebGPU model", fraction: 0 });
    const model = await Gemma4ForConditionalGeneration.from_pretrained(modelId, {
      dtype: "q4f16",
      device: "webgpu",
      progress_callback: (info) => reportProgress(info, onProgress),
    });
    onProgress({ status: "ready", message: "Ready", fraction: 1 });
    return new Gemma4Multimodal(modelId, processor, model);
  }

  constructor(modelId, processor, model) {
    this.modelId = modelId;
    this.processor = processor;
    this.model = model;
  }

  async *generate(messages, { maxNewTokens = 512, signal } = {}) {
    const prepared = await prepareInputs(this.processor, messages);
    const stopping = new InterruptableStoppingCriteria();
    const onAbort = () => stopping.interrupt();
    signal?.addEventListener("abort", onAbort, { once: true });
    if (signal?.aborted) stopping.interrupt();

    const queue = [];
    let wake = null;
    let done = false;
    let failure = null;
    let fullText = "";
    const push = (text) => {
      if (!text) return;
      fullText += text;
      queue.push({ delta: text, text: fullText });
      wake?.();
      wake = null;
    };

    const streamer = new TextStreamer(this.processor.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: push,
    });

    const generation = this.model.generate({
      ...prepared.inputs,
      max_new_tokens: maxNewTokens,
      do_sample: false,
      streamer,
      stopping_criteria: [stopping],
    }).catch((error) => { failure = error; }).finally(() => {
      done = true;
      wake?.();
      wake = null;
      prepared.cleanup();
      signal?.removeEventListener("abort", onAbort);
    });

    while (!done || queue.length) {
      if (!queue.length) await new Promise((resolve) => { wake = resolve; });
      while (queue.length) yield queue.shift();
    }
    await generation;
    if (failure) throw failure;
  }

  reset() {}

  dispose() {
    this.model?.dispose?.();
  }
}

async function prepareInputs(processor, messages) {
  const templated = [];
  const imageUrls = [];
  const audioSources = [];
  const temporaryUrls = [];
  // Transformers.js 4.2 currently merges one set of Gemma 4 multimodal
  // features per generation call. Re-inserting media from earlier turns makes
  // the number of placeholder tokens diverge from the newly encoded features.
  // Keep earlier turns as text; the assistant response already preserves their
  // conversational meaning. Only encode attachments from the newest media turn.
  const lastIndex = messages.length - 1;
  const activeMediaIndex = messages[lastIndex]?.role === "user" && messages[lastIndex]?.attachments?.length
    ? lastIndex
    : -1;

  for (let messageIndex = 0; messageIndex < messages.length; messageIndex++) {
    const message = messages[messageIndex];
    if (message.role !== "user" || !message.attachments?.length || messageIndex !== activeMediaIndex) {
      const previousMedia = message.role === "user" && message.attachments?.length
        ? describePreviousMedia(message.attachments)
        : "";
      templated.push({ role: message.role, content: message.content || "" });
      if (previousMedia && !message.content) templated[templated.length - 1].content = previousMedia;
      continue;
    }

    const content = [];
    for (const attachment of message.attachments.filter((item) => item.type === "image")) {
      content.push({ type: "image" });
      imageUrls.push(mediaUrl(attachment, temporaryUrls));
    }
    if (message.content) content.push({ type: "text", text: message.content });
    for (const attachment of message.attachments.filter((item) => item.type === "audio")) {
      content.push({ type: "audio" });
      audioSources.push(attachment.pcm ?? mediaUrl(attachment, temporaryUrls));
    }
    templated.push({ role: "user", content });
  }

  const prompt = processor.apply_chat_template(templated, {
    enable_thinking: false,
    add_generation_prompt: true,
  });
  const images = await Promise.all(imageUrls.map((url) => load_image(url)));
  const audios = await Promise.all(audioSources.map((source) =>
    source instanceof Float32Array ? source : read_audio(source, 16000)
  ));
  const inputs = await processor(
    prompt,
    images.length ? (images.length === 1 ? images[0] : images) : null,
    audios.length ? (audios.length === 1 ? audios[0] : audios) : null,
    { add_special_tokens: false },
  );
  if (audioSources.length && !inputs.input_features) {
    throw new Error("The Gemma 4 processor did not create audio features (input_features is missing).");
  }
  return {
    inputs,
    cleanup: () => temporaryUrls.forEach((url) => URL.revokeObjectURL(url)),
  };
}

function describePreviousMedia(attachments) {
  const images = attachments.filter((item) => item.type === "image").length;
  const audios = attachments.filter((item) => item.type === "audio").length;
  const parts = [];
  if (images) parts.push(`[${images} previous image${images > 1 ? "s" : ""}]`);
  if (audios) parts.push(`[${audios} previous audio message${audios > 1 ? "s" : ""}]`);
  return parts.join(" ");
}

function mediaUrl(attachment, temporaryUrls) {
  if (attachment.url) return attachment.url;
  const source = attachment.file ?? attachment.blob;
  if (!source) throw new Error(`Missing ${attachment.type} data`);
  const url = URL.createObjectURL(source);
  temporaryUrls.push(url);
  return url;
}

function reportProgress(info, callback) {
  if (info.status === "progress_total") {
    callback({
      status: "weights",
      kind: "bytes",
      loaded: info.loaded,
      total: info.total,
      fraction: Number.isFinite(info.progress) ? info.progress / 100 : undefined,
      message: info.file ? `Loading ${info.file}` : "Loading model files",
    });
  } else if (info.status === "progress") {
    callback({
      status: "weights",
      kind: "bytes",
      loaded: info.loaded,
      total: info.total,
      fraction: Number.isFinite(info.progress) ? info.progress / 100 : undefined,
      message: info.file ? `Loading ${info.file}` : "Loading model files",
    });
  }
}
