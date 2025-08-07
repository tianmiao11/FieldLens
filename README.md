# FieldLens: A Private, Offline AI Assistant for Detecting Corn Abnormalities

## Video

This is the YouTube video link demonstrating the FieldLens application:

[![Watch on YouTube](https://img.youtube.com/vi/YT33z4GDt5U/0.jpg)](https://www.youtube.com/watch?v=YT33z4GDt5U)

## Introduction

Corn farming, especially in remote and rural areas, often faces a critical challenge: identifying plant growth problems like sparse emergence or pest damage at an early stage—before it's too late. These issues typically emerge during sensitive phases such as tasseling, when quick intervention is crucial for protecting yields. However, many farmers lack the tools, expertise, or connectivity to detect and interpret these problems in time. What makes this challenge more severe is that visual abnormalities often go unnoticed when observed from the ground. Early-stage pest outbreaks or missing plants are hard to identify without an aerial perspective—and even when drone footage is available, interpreting it remains a technical barrier for non-experts.

To solve this, we developed **FieldLens**: an offline-first, on-device AI assistant that analyzes cornfield drone videos, detects abnormal regions (such as sparse growth), and guides the user to generate intelligent questions about the observed issues. These questions are then routed to a local language model interface powered by **Gemma 3n**, which produces structured, multimodal answers based on retrieved context and visual evidence. Unlike cloud-based AI tools, **FieldLens** is designed for low-connectivity and privacy-sensitive environments. Everything runs on the phone—video analysis, image segmentation, prompt generation, and ultimately, language-based insights. This ensures real-time, private, and actionable support for farmers, right in the field, without needing the internet.

In this project, we demonstrate how the unique capabilities of **Gemma 3n**—such as on-device performance, multimodal understanding, and flexible prompt-based reasoning—enable a new kind of practical, deployable agricultural intelligence.

## Approach

The **FieldLens** pipeline begins with video input—typically drone or handheld footage captured in the field. To maintain full offline capability, all subsequent processing is performed on-device. From each video, representative frames are extracted and passed through a lightweight visual pipeline to detect crop anomalies, generate domain-specific questions, and construct multimodal prompts for reasoning with **Gemma 3n**.

Our approach includes three components:

1. **Anomaly Detection** (thresholding + morphology)  
2. **Question Generation** (template-based, RAG-style)  
3. **Multimodal Reasoning** (Gemma 3n-compatible prompts)

### 1. Anomaly Detection (Thresholding + Morphology)

Sparse crop regions typically exhibit lower green intensity and spatial discontinuity in aerial imagery. An Excess Green Index (ExG) is computed to enhance vegetation contrast and suppress non-plant background. Based on the ExG output, a rule-based segmentation pipeline extracts large, non-continuous low-green areas as anomaly candidates.

Each extracted frame is processed using the following steps:

- 'Input': RGB image ('.jpg'), resolution ~'1920×1080'
- 'ExG computation': 'ExG = 2G - R - B'
- 'Normalization': scaled to '8-bit grayscale' ('uint8', range '0–255')
- 'Thresholding': binary inverse, 'T = 60'
- 'Morphology': open → close, kernel = elliptical ('7×7')
- 'Connected Components': area ≥ '2000 px', '8-connectivity'
- 'Masking': zero out top '50%' of image ('mask[0:H//2, :] = 0')
- 'Output': binary mask highlighting sparse regions

**Output**: Binary mask highlighting sparse growth areas.

### 2. Question Generation (Template-based, RAG-style)

Visual outputs are enriched with structured language prompts to produce actionable insights. A RAG-style method retrieves related domain knowledge and injects it into the prompt.

#### Semantic Retrieval:

- **Embedding Model**: Qwen3-Embedding-0.6B  
- **Vector Store**: Milvus or FAISS  
- **Top-k Retrieval**: 2  
- **Similarity**: Cosine  

#### Prompt Template:

```
Based on <retrieved_text>, answer <question>
```

All operations are **fully offline** and optimized for Android deployment.

### 3. Multimodal Reasoning (Gemma 3n-compatible Prompts)

We use **Gemma-3n-E4b-it-int4** for local reasoning with visual + textual input.

- **Prompt format**:  
  ```
  <image_soft_token> In this image, <question>
  ```
- **Inputs**: Image + Text  
- **Outputs**: Natural language answers grounded in both visual and semantic evidence  
- **Deployment**: Google Edge Gallery, ONNX/mobile-compatible

## System Demo

We simulate the entire pipeline step-by-step using real aerial footage:

### 1. Input Image

Raw drone frame captured from a cornfield.

![Figure 1: Original RGB Image](data/images/original.jpg)  
**Figure 1**: Original RGB Image

### 2. Anomaly Detection

ExG-based segmentation and morphological filtering reveal sparse areas.

![Figure 2: ExG Vegetation Index Map](data/images/exg.jpg)  
**Figure 2**: ExG Vegetation Index Map

![Figure 3: Binary Anomaly Mask](data/images/mask.jpg)  
**Figure 3**: Binary Anomaly Mask

![Figure 4: Annotated Image with Red Boxes](data/images/box.jpg)  
**Figure 4**: Annotated Image with Red Boxes

### 3. Question Generation

Structured prompts built for each anomaly.

- **Question**: What causes sparse crop growth here?  
- **Retrieved Text**: "According to agronomic guidelines, pest activity increases in the tasseling stage..."  
- **Prompt**:  
  ```
  Please answer <question> based on <retrieved_text>
  ```

### 4. Multimodal Reasoning

![Figure 5: Multimodal reasoning example using a Gemma 3n-compatible prompt](data/images/gemma.jpg)  
**Figure 5**: Multimodal reasoning example using a Gemma 3n-compatible prompt

### Interactive Demo

Due to offline design (local Android deployment), we do not provide a live cloud demo.  
However, we created a **web-based walkthrough** simulating the app flow:

🔗 [https://tianmiao11.github.io/FieldLens/](https://tianmiao11.github.io/FieldLens/)

## Future Improvements

### 1. Enhanced Anomaly Classification  
Support multiple crop stress types (drought, disease, lodging) using multi-class segmentation and phenological phase adaptation.

### 2. Multimodal Reasoning on Drones  
Embed quantized models onto agricultural drones for **fully autonomous** and **offline** decision-making.

### 3. Field-Level Reporting  
Aggregate frame-level detections into a full-field diagnostic report with spatial analytics and drone-action guidance (e.g. spraying, replanting).

## Outro

The pipeline—**anomaly detection**, **question generation**, and **multimodal reasoning**—is **crop-agnostic** and expandable to other domains like forestry or conservation.

This project represents a **new class of edge AI systems**, combining **multimodal models like Gemma 3n** with traditional agricultural hardware. The result: private, offline, field-deployable intelligence that helps farmers act—without needing the cloud.
