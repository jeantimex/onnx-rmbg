import * as ort from 'onnxruntime-web';

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
            'ort-wasm.wasm': '/ort-wasm.wasm',
            'ort-wasm-simd.wasm': '/ort-wasm-simd.wasm',
            'ort-wasm-threaded.wasm': '/ort-wasm-threaded.wasm'
        };

        this.initialize();
    }

    async checkWebGPUSupport() {
        if (!navigator.gpu) {
            console.log('WebGPU is not supported in this browser');
            return false;
        }
        
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                console.log('No WebGPU adapter found');
                return false;
            }
            const device = await adapter.requestDevice();
            return !!device;
        } catch (error) {
            console.log('WebGPU device creation failed:', error);
            return false;
        }
    }

    async initialize() {
        try {
            // Try WASM first
            try {
                console.log('Attempting WASM initialization...');
                const wasmOptions = {
                    executionProviders: ['wasm'],
                    graphOptimizationLevel: 'all',
                    enableCpuMemArena: true
                };
                
                this.session = await ort.InferenceSession.create(this.modelUrl, wasmOptions);
                console.log('WASM initialization successful');
                
                // Now try WebGPU if available
                if (await this.checkWebGPUSupport()) {
                    console.log('WebGPU support detected, attempting WebGPU initialization...');
                    try {
                        const webGPUOptions = {
                            executionProviders: ['webgpu'],
                            graphOptimizationLevel: 'all',
                            enableCpuMemArena: true,
                            webgpuFlags: {
                                preferWebGPU: true,
                                enableFloat16: true
                            }
                        };
                        
                        const webGPUSession = await ort.InferenceSession.create(this.modelUrl, webGPUOptions);
                        this.session = webGPUSession;
                        this.usingWebGPU = true;
                        console.log('WebGPU initialization successful');
                    } catch (webgpuError) {
                        console.log('WebGPU initialization failed, keeping WASM session:', webgpuError);
                    }
                }
            } catch (wasmError) {
                console.error('WASM initialization failed:', wasmError);
                throw wasmError;
            }

            console.log('Final execution provider:', this.session.handler?._ep?.name);
            this.isLoading = false;
            this.updateUI();

        } catch (err) {
            console.error('Initialization error:', err);
            this.error = err;
            this.isLoading = false;
            this.updateUI();
        }
    }

    async preprocessImage(imageElement) {
        console.log('Preprocessing image...');
        // Create a canvas to get image data
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = this.MODEL_INPUT_SIZE.width;
        canvas.height = this.MODEL_INPUT_SIZE.height;
        
        // Draw and resize image
        ctx.drawImage(imageElement, 0, 0, this.MODEL_INPUT_SIZE.width, this.MODEL_INPUT_SIZE.height);
        const imageData = ctx.getImageData(0, 0, this.MODEL_INPUT_SIZE.width, this.MODEL_INPUT_SIZE.height);
        
        // Convert to float32 and normalize
        const float32Data = new Float32Array(this.MODEL_INPUT_SIZE.width * this.MODEL_INPUT_SIZE.height * 3);
        for (let i = 0, j = 0; i < imageData.data.length; i += 4, j += 3) {
            // Normalize and apply mean/std normalization
            float32Data[j] = (imageData.data[i] / 255.0 - 0.5) / 1.0;     // R
            float32Data[j + 1] = (imageData.data[i + 1] / 255.0 - 0.5) / 1.0; // G
            float32Data[j + 2] = (imageData.data[i + 2] / 255.0 - 0.5) / 1.0; // B
        }

        // Convert from HWC to CHW format
        const tensorData = new Float32Array(this.MODEL_INPUT_SIZE.width * this.MODEL_INPUT_SIZE.height * 3);
        for (let c = 0; c < 3; c++) {
            for (let h = 0; h < this.MODEL_INPUT_SIZE.height; h++) {
                for (let w = 0; w < this.MODEL_INPUT_SIZE.width; w++) {
                    tensorData[c * this.MODEL_INPUT_SIZE.height * this.MODEL_INPUT_SIZE.width + h * this.MODEL_INPUT_SIZE.width + w] =
                        float32Data[h * this.MODEL_INPUT_SIZE.width * 3 + w * 3 + c];
                }
            }
        }

        return new ort.Tensor('float32', tensorData, [1, 3, this.MODEL_INPUT_SIZE.height, this.MODEL_INPUT_SIZE.width]);
    }

    async postprocessMask(outputTensor, originalWidth, originalHeight) {
        console.log('Postprocessing mask...');
        
        // Ensure we have the tensor data
        if (!outputTensor || !outputTensor.data) {
            throw new Error('Invalid output tensor');
        }

        const maskData = outputTensor.data;
        const maskHeight = outputTensor.dims[2];
        const maskWidth = outputTensor.dims[3];
        
        console.log('Mask dimensions:', { width: maskWidth, height: maskHeight });
        
        // Calculate min/max more efficiently
        let min = maskData[0];
        let max = maskData[0];
        for (let i = 1; i < maskData.length; i++) {
            if (maskData[i] < min) min = maskData[i];
            if (maskData[i] > max) max = maskData[i];
        }
        
        // Create final image data
        const resizedMaskData = new Uint8ClampedArray(originalWidth * originalHeight * 4);
        const scale = max !== min ? 255 / (max - min) : 0;

        console.log('Mask value range:', { min, max, scale });

        // Simple nearest neighbor scaling for efficiency
        const xRatio = maskWidth / originalWidth;
        const yRatio = maskHeight / originalHeight;

        for (let y = 0; y < originalHeight; y++) {
            for (let x = 0; x < originalWidth; x++) {
                const px = Math.floor(x * xRatio);
                const py = Math.floor(y * yRatio);
                const value = Math.round((maskData[py * maskWidth + px] - min) * scale);
                
                const dstIdx = (y * originalWidth + x) * 4;
                resizedMaskData[dstIdx] = value;     // R
                resizedMaskData[dstIdx + 1] = value; // G
                resizedMaskData[dstIdx + 2] = value; // B
                resizedMaskData[dstIdx + 3] = 255;   // A
            }
        }

        return new ImageData(resizedMaskData, originalWidth, originalHeight);
    }

    async removeBackground(imageElement) {
        if (!this.session) {
            throw new Error('Model not loaded');
        }

        try {
            console.log('Starting background removal...');
            
            // Preprocess image
            const tensor = await this.preprocessImage(imageElement);
            console.log('Input tensor shape:', tensor.dims);
            
            // Run inference
            console.log('Running inference...');
            const results = await this.session.run({
                'input': tensor
            });
            
            console.log('Model outputs:', results);
            
            // Get the output tensor
            if (!results.output) {
                throw new Error('Model output not found. Available outputs: ' + Object.keys(results).join(', '));
            }

            const outputTensor = results.output;
            console.log('Output tensor:', outputTensor);

            // Create output canvas
            const canvas = document.createElement('canvas');
            canvas.width = imageElement.naturalWidth;
            canvas.height = imageElement.naturalHeight;
            const ctx = canvas.getContext('2d');

            // Draw original image
            ctx.drawImage(imageElement, 0, 0);

            // Process mask and update alpha channel
            const mask = await this.postprocessMask(outputTensor, imageElement.naturalWidth, imageElement.naturalHeight);
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            
            // Use the mask as alpha channel
            for (let i = 0; i < mask.data.length / 4; i++) {
                imageData.data[i * 4 + 3] = mask.data[i * 4]; // Use R channel as alpha
            }
            
            ctx.putImageData(imageData, 0, 0);
            return canvas;

        } catch (error) {
            console.error('Background removal error:', error);
            throw error;
        }
    }

    updateUI() {
        const statusElement = document.getElementById('modelStatus');
        const inferenceButton = document.getElementById('inferenceButton');

        if (this.isLoading) {
            statusElement.textContent = 'Loading model...';
            if (inferenceButton) inferenceButton.disabled = true;
        } else if (this.error) {
            statusElement.textContent = `Error loading model: ${this.error.message}`;
            statusElement.classList.add('error');
            if (inferenceButton) inferenceButton.disabled = true;
        } else {
            const provider = this.usingWebGPU ? 'WebGPU' : 'WASM';
            statusElement.textContent = `Model loaded successfully (using ${provider})`;
            if (inferenceButton) inferenceButton.disabled = false;
        }
    }
}