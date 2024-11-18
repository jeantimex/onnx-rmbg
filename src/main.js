import './style.css';
import { OnnxModel } from './onnxModel';

// Example image URL for testing
const EXAMPLE_URL = 'https://images.pexels.com/photos/5965592/pexels-photo-5965592.jpeg';

async function getGPUInfo() {
    if (!navigator.gpu) {
        return 'WebGPU is not supported in this browser';
    }

    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            return 'WebGPU adapter not found';
        }

        // Get adapter info
        const info = await adapter.requestAdapterInfo();
        
        // Request device to get limits and features
        const device = await adapter.requestDevice();
        
        return `
            <div class="gpu-info">
                <h3>GPU Information:</h3>
                <ul>
                    <li>Vendor: ${info.vendor}</li>
                    <li>Architecture: ${info.architecture}</li>
                    <li>Description: ${info.description || 'Not available'}</li>
                    <li>Device: ${info.device || 'Not available'}</li>
                    <li>Max Buffer Size: ${device.limits.maxBufferSize / (1024 * 1024)} MB</li>
                    <li>Max Compute Invocations: ${device.limits.maxComputeInvocationsPerWorkgroup}</li>
                    <li>Max Storage Buffer Binding Size: ${device.limits.maxStorageBufferBindingSize / (1024 * 1024)} MB</li>
                </ul>
            </div>
        `;
    } catch (error) {
        console.error('Error getting GPU info:', error);
        return `WebGPU Error: ${error.message}`;
    }
}

async function processImage(model, imageUrl, isFile = false) {
    const output = document.getElementById('output');
    const originalImage = document.getElementById('originalImage');
    const processedImage = document.getElementById('processedImage');

    try {
        output.textContent = 'Processing image...';

        // Load image
        const image = new Image();
        if (!isFile) {
            image.crossOrigin = 'anonymous'; // Needed for external URLs
        }
        
        await new Promise((resolve, reject) => {
            image.onload = resolve;
            image.onerror = reject;
            image.src = imageUrl;
        });

        // Display original image
        originalImage.innerHTML = '';
        originalImage.appendChild(image.cloneNode());

        // Process image
        const resultCanvas = await model.removeBackground(image);
        
        // Display processed image
        processedImage.innerHTML = '';
        processedImage.appendChild(resultCanvas);

        output.textContent = 'Processing complete!';

        // Cleanup if it was a file URL
        if (isFile) {
            URL.revokeObjectURL(imageUrl);
        }

    } catch (error) {
        output.innerHTML = `<div class="error">Error processing image: ${error.message}</div>`;
        console.error('Error:', error);
    }
}

async function initializeApp() {
    const container = document.querySelector('.container');

    // Add GPU info section
    const gpuInfo = document.createElement('div');
    gpuInfo.id = 'gpuInfo';
    gpuInfo.textContent = 'Loading GPU information...';
    container.insertBefore(gpuInfo, container.firstChild);
    gpuInfo.innerHTML = await getGPUInfo();

    // Initialize model
    const modelUrl = '/models/briaai/RMBG-1.4/onnx/model.onnx'; // Make sure this path is correct
    const model = new OnnxModel(modelUrl);

    // Handle file upload
    const imageUpload = document.getElementById('imageUpload');
    const fileName = document.getElementById('fileName');

    imageUpload.addEventListener('change', async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        fileName.textContent = file.name;
        const imageUrl = URL.createObjectURL(file);
        await processImage(model, imageUrl, true);
    });

    // Add example button
    const exampleButton = document.createElement('button');
    exampleButton.textContent = 'Try Example Image';
    exampleButton.className = 'example-button';
    exampleButton.onclick = async () => {
        await processImage(model, EXAMPLE_URL);
    };
    
    const uploadSection = document.querySelector('.upload-section');
    uploadSection.appendChild(exampleButton);
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', initializeApp);