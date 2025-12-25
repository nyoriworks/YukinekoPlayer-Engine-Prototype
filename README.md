# YukinekoPlayer-Engine-Prototype

# YukinekoPlayer - Research Engine Prototypes

This repository hosts experimental image processing engines and Metal shaders developed during the creation of **[YukinekoPlayer](https://github.com/nyoriworks/YukinekoPlayer)**.

These algorithms were evaluated for high-fidelity video upscaling and post-processing on macOS. While the final production version of YukinekoPlayer employs different optimization strategies, these prototypes demonstrate pure Metal Compute implementations of advanced image processing techniques.

## 📂 Contents

### 1. RAVU Upscaler (Custom Algorithm)
* **Files:** `RAVU.metal`, `RAVUStage.swift`
* **Description:** A custom upscaling algorithm designed to eliminate "ringing" artifacts while maintaining edge sharpness.
* **Tech Specs:**
    * Implements a custom S-Curve interpolation.
    * Uses gradient-based weighting (Luma analysis) to preserve edges.
    * Features an anti-ringing clamp mechanism based on 6-point sampling.
    * Includes a Swift driver with triple-buffering (texture pool) to prevent screen tearing.

### 2. Kuwahara Filter
* **Files:** `Kuwahara.metal`, `KuwaharaStage.swift`
* **Description:** A GPU-accelerated implementation of the Kuwahara filter.
* **Tech Specs:**
    * Calculates mean and variance across 4 sub-regions per pixel.
    * Selects the region with the lowest variance to preserve edges while smoothing textures.
    * Fully optimized for Metal Compute Shaders.

### 3. ESPCN Super Resolution
* **Files:** `ESPCN.metal`
* **Models:** Includes 4 pre-trained model weights (in `/models` directory).
* **Description:** A lightweight Super Resolution implementation running directly on Metal Shaders (without CoreML overhead).
* **Tech Specs:**
    * **Layer 1:** 5x5 Convolution (Input -> 24ch)
    * **Layer 2:** 3x3 Convolution (24ch -> 12ch)
    * **Layer 3:** 3x3 Convolution + Pixel Shuffle (Sub-pixel convolution)
    * Manual convolution implementation for maximum control over the pipeline.

## 📦 Included Models
This repository includes 4 variations of trained weights for the ESPCN engine:
* `espcn_x2_v1.bin`
* `espcn_x2_v2.bin`
* `espcn_x3_v1.bin`
* `espcn_x4_v1.bin`

*(Note: These are raw binary weights intended to be loaded into the `weights` buffer in `ESPCN.metal`.)*

## ⚠️ Disclaimer
* **Status:** **Archived / Prototype**
* These codes are provided "as is" for educational and research purposes.
* They are not maintained and may require modification to run in a standalone environment.

## 📄 License
MIT License. Feel free to use these codes for your own projects or research.

---

### 🎵 About the Main Project
If you are interested in the final product, check out **YukinekoPlayer** on the Mac App Store.

[![Download on the Mac App Store](https://devimages-cdn.apple.com/app-store/marketing/guidelines/images/badge-download-on-the-mac-app-store.svg)](https://apps.apple.com/app/yukinekoplayer/id6756008211)

**[Visit Official Repository](https://github.com/nyoriworks/YukinekoPlayer)**
