import * as ort from "onnxruntime-web/webgpu";

export class OnnxModel {
  constructor(modelUrl) {
    this.modelUrl = modelUrl;
    this.session = null;
    this.isLoading = true;
    this.error = null;
    this.usingWebGPU = false;
    this.MODEL_INPUT_SIZE = { width: 1024, height: 1024 };

    // Set WASM paths
    ort.env.wasm.wasmPaths = {
      "ort-wasm.wasm": "/ort-wasm.wasm",
      "ort-wasm-simd.wasm": "/ort-wasm-simd.wasm",
      "ort-wasm-threaded.wasm": "/ort-wasm-threaded.wasm",
    };

    this.initialize();
  }

  async checkWebGPUSupport() {
    if (!navigator.gpu) {
      console.log("WebGPU is not supported in this browser");
      return false;
    }

    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        console.log("No WebGPU adapter found");
        return false;
      }
      const device = await adapter.requestDevice();
      const adapterInfo = await adapter.requestAdapterInfo();
      console.log("WebGPU Adapter Info:", adapterInfo);
      return !!device;
    } catch (error) {
      console.log("WebGPU device creation failed:", error);
      return false;
    }
  }

  async initialize() {
    try {
      // Try WebGPU first
      if (await this.checkWebGPUSupport()) {
        console.log("Attempting WebGPU initialization...");
        try {
          const webGPUOptions = {
            executionProviders: ["webgpu"],
            graphOptimizationLevel: "all",
            enableCpuMemArena: true,
            webgpuFlags: {
              preferWebGPU: true,
              enableFloat16: true,
              allowFloat16: true,
            },
          };

          this.session = await ort.InferenceSession.create(
            this.modelUrl,
            webGPUOptions
          );
          this.usingWebGPU = true;
          console.log("WebGPU initialization successful");
        } catch (webgpuError) {
          console.warn(
            "WebGPU initialization failed, falling back to WASM:",
            webgpuError
          );
          await this.initializeWASM();
        }
      } else {
        await this.initializeWASM();
      }

      console.log("Final execution provider:", this.session.handler?._ep?.name);
      this.isLoading = false;
      this.updateUI();
    } catch (err) {
      console.error("Initialization error:", err);
      this.error = err;
      this.isLoading = false;
      this.updateUI();
    }
  }

  async initializeWASM() {
    console.log("Attempting WASM initialization...");
    const wasmOptions = {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
      enableCpuMemArena: true,
    };

    this.session = await ort.InferenceSession.create(
      this.modelUrl,
      wasmOptions
    );
    this.usingWebGPU = false;
    console.log("WASM initialization successful");
  }

  async preprocessImage(imageElement) {
    console.log("Preprocessing image...", {
      originalWidth: imageElement.naturalWidth,
      originalHeight: imageElement.naturalHeight,
    });

    // Calculate dimensions maintaining aspect ratio
    let targetWidth = this.MODEL_INPUT_SIZE.width;
    let targetHeight = this.MODEL_INPUT_SIZE.height;
    const aspectRatio = imageElement.naturalWidth / imageElement.naturalHeight;

    if (aspectRatio > 1) {
      // Image is wider than tall
      targetHeight = Math.round(this.MODEL_INPUT_SIZE.width / aspectRatio);
    } else {
      // Image is taller than wide
      targetWidth = Math.round(this.MODEL_INPUT_SIZE.height * aspectRatio);
    }

    // Create a canvas with padding to reach MODEL_INPUT_SIZE
    const canvas = document.createElement("canvas");
    canvas.width = this.MODEL_INPUT_SIZE.width;
    canvas.height = this.MODEL_INPUT_SIZE.height;
    const ctx = canvas.getContext("2d");

    // Fill with black (will be zero after normalization)
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Calculate positioning to center the image
    const xOffset = Math.round((this.MODEL_INPUT_SIZE.width - targetWidth) / 2);
    const yOffset = Math.round(
      (this.MODEL_INPUT_SIZE.height - targetHeight) / 2
    );

    // Draw the image centered and scaled
    ctx.drawImage(imageElement, xOffset, yOffset, targetWidth, targetHeight);

    // Get the image data
    const imageData = ctx.getImageData(
      0,
      0,
      this.MODEL_INPUT_SIZE.width,
      this.MODEL_INPUT_SIZE.height
    );

    // Convert to float32 and normalize
    const float32Data = new Float32Array(
      this.MODEL_INPUT_SIZE.width * this.MODEL_INPUT_SIZE.height * 3
    );
    for (let i = 0, j = 0; i < imageData.data.length; i += 4, j += 3) {
      // Normalize and apply mean/std normalization
      float32Data[j] = (imageData.data[i] / 255.0 - 0.5) / 1.0; // R
      float32Data[j + 1] = (imageData.data[i + 1] / 255.0 - 0.5) / 1.0; // G
      float32Data[j + 2] = (imageData.data[i + 2] / 255.0 - 0.5) / 1.0; // B
    }

    // Convert from HWC to CHW format
    const tensorData = new Float32Array(
      this.MODEL_INPUT_SIZE.width * this.MODEL_INPUT_SIZE.height * 3
    );
    for (let c = 0; c < 3; c++) {
      for (let h = 0; h < this.MODEL_INPUT_SIZE.height; h++) {
        for (let w = 0; w < this.MODEL_INPUT_SIZE.width; w++) {
          tensorData[
            c * this.MODEL_INPUT_SIZE.height * this.MODEL_INPUT_SIZE.width +
              h * this.MODEL_INPUT_SIZE.width +
              w
          ] = float32Data[h * this.MODEL_INPUT_SIZE.width * 3 + w * 3 + c];
        }
      }
    }

    // Save the dimensions for post-processing
    this.lastProcessedDimensions = {
      targetWidth,
      targetHeight,
      xOffset,
      yOffset,
      aspectRatio,
    };

    return new ort.Tensor("float32", tensorData, [
      1,
      3,
      this.MODEL_INPUT_SIZE.height,
      this.MODEL_INPUT_SIZE.width,
    ]);
  }

  async postprocessMask(outputTensor, originalWidth, originalHeight) {
    console.log("Postprocessing mask...");

    if (!outputTensor || !outputTensor.data) {
      throw new Error("Invalid output tensor");
    }

    const maskData = outputTensor.data;
    const maskHeight = outputTensor.dims[2];
    const maskWidth = outputTensor.dims[3];

    // Get the dimensions we used during preprocessing
    const { targetWidth, targetHeight, xOffset, yOffset } =
      this.lastProcessedDimensions;

    console.log("Mask dimensions:", {
      maskWidth,
      maskHeight,
      targetWidth,
      targetHeight,
      xOffset,
      yOffset,
    });

    // Calculate min/max from the relevant part of the mask only
    let min = Infinity;
    let max = -Infinity;
    for (let y = yOffset; y < yOffset + targetHeight; y++) {
      for (let x = xOffset; x < xOffset + targetWidth; x++) {
        const value = maskData[y * maskWidth + x];
        min = Math.min(min, value);
        max = Math.max(max, value);
      }
    }

    // Create final image data
    const resizedMaskData = new Uint8ClampedArray(
      originalWidth * originalHeight * 4
    );
    const scale = max !== min ? 255 / (max - min) : 0;

    // Calculate scaling factors from the padded area to original image
    const scaleX = targetWidth / originalWidth;
    const scaleY = targetHeight / originalHeight;

    // Process each pixel of the output image
    for (let y = 0; y < originalHeight; y++) {
      for (let x = 0; x < originalWidth; x++) {
        // Map original image coordinates to padded mask coordinates
        const maskX = Math.floor(x * scaleX) + xOffset;
        const maskY = Math.floor(y * scaleY) + yOffset;

        // Get mask value
        const maskValue = maskData[maskY * maskWidth + maskX];
        const value = Math.round((maskValue - min) * scale);

        // Set RGBA values
        const dstIdx = (y * originalWidth + x) * 4;
        resizedMaskData[dstIdx] = value; // R
        resizedMaskData[dstIdx + 1] = value; // G
        resizedMaskData[dstIdx + 2] = value; // B
        resizedMaskData[dstIdx + 3] = 255; // A
      }
    }

    return new ImageData(resizedMaskData, originalWidth, originalHeight);
  }

  async removeBackground(imageElement) {
    if (!this.session) {
      throw new Error("Model not loaded");
    }

    try {
      console.log("Starting background removal...");

      // Preprocess image
      const tensor = await this.preprocessImage(imageElement);
      console.log("Input tensor shape:", tensor.dims);

      // Run inference
      console.log("Running inference...");
      const results = await this.session.run({
        input: tensor,
      });

      console.log("Model outputs:", results);

      // Get the output tensor
      if (!results.output) {
        throw new Error(
          "Model output not found. Available outputs: " +
            Object.keys(results).join(", ")
        );
      }

      const outputTensor = results.output;
      console.log("Output tensor:", outputTensor);

      // Create output canvas
      const canvas = document.createElement("canvas");
      canvas.width = imageElement.naturalWidth;
      canvas.height = imageElement.naturalHeight;
      const ctx = canvas.getContext("2d");

      // Draw original image
      ctx.drawImage(imageElement, 0, 0);

      // Process mask and update alpha channel
      const mask = await this.postprocessMask(
        outputTensor,
        imageElement.naturalWidth,
        imageElement.naturalHeight
      );
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

      // Use the mask as alpha channel
      for (let i = 0; i < mask.data.length / 4; i++) {
        imageData.data[i * 4 + 3] = mask.data[i * 4]; // Use R channel as alpha
      }

      ctx.putImageData(imageData, 0, 0);
      return canvas;
    } catch (error) {
      console.error("Background removal error:", error);
      throw error;
    }
  }

  updateUI() {
    const statusElement = document.getElementById("modelStatus");
    const inferenceButton = document.getElementById("inferenceButton");

    if (this.isLoading) {
      statusElement.textContent = "Loading model...";
      if (inferenceButton) inferenceButton.disabled = true;
    } else if (this.error) {
      statusElement.innerHTML = `
                <div class="error">
                    Error loading model: ${this.error.message}<br>
                    <small>Using fallback if available</small>
                </div>
            `;
      if (inferenceButton) inferenceButton.disabled = true;
    } else {
      const provider = this.usingWebGPU ? "WebGPU" : "WASM";
      statusElement.innerHTML = `
                <div class="success">
                    Model loaded successfully<br>
                    <small>Using ${provider} backend</small>
                </div>
            `;
      if (inferenceButton) inferenceButton.disabled = false;
    }
  }
}
