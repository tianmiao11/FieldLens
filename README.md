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

- `Input`: RGB image (`.jpg`), resolution ~`1920×1080`
- `ExG computation`: `ExG = 2G - R - B`
- `Normalization`: scaled to `8-bit grayscale` (`uint8`, range `0–255`)
- `Thresholding`: binary inverse, `T = 60`
- `Morphology`: open → close, kernel = elliptical (`7×7`)
- `Connected Components`: area ≥ `2000 px`, `8-connectivity`
- `Masking`: zero out top `50%` of image (`mask[0:H//2, :] = 0`)
- `Output`: binary mask highlighting sparse regions

### 2. Question Generation (Template-based, RAG-style)

Once abnormal crop regions are detected, visual results alone are often insufficient for practical decision-making. To address this, FieldLens incorporates a language-based layer that transforms each anomaly into a relevant, structured question. This enables users to not only see the problem, but also understand it in agronomic terms.

RAG-style prompting enhances question relevance by retrieving semantically related content from local text corpora. This method avoids model fine-tuning and supports offline operation.

#### Semantic Representation via Embedding
The corpus is segmented into sentences and encoded as dense vectors to enable efficient similarity matching.

- Embedding Model: `Qwen3-Embedding-0.6B`
- Output Dimensionality: `1024`
- Pooling Strategy: `Last-token pooling` (padding-aware)

#### Indexing: Offline Semantic Retrieval
All document embeddings are stored locally. For each query, the most relevant content is retrieved based on vector similarity.

- Vector Store: `Milvus or FAISS`
- Similarity Metric: `Cosine similarity`
- Retrieval Top-k: `2`

#### Prompt Construction: Structured Template
Retrieved texts and user questions are combined into a structured prompt for language model inference.

- Prompt Format: `"Based on <retrieved_text>, answer <question>"`

#### Deployment: Fully Offline Execution
The entire pipeline runs locally without internet access, optimized for lightweight deployment on Android devices.

- Deployment Mode: `Offline-capable`
- Adaptation Options: `ONNX` / `Faiss` (mobile support)
 
### 3. Multimodal Reasoning (Gemma 3n-compatible Prompts)

Multimodal reasoning is employed to synthesize visual observations with retrieved domain knowledge. After anomaly detection and question construction, a structured prompt is created, combining selected video frames with textual queries. These inputs are passed to a vision-language model to generate contextual answers grounded in both visual and semantic signals. The prompt design is explicitly tailored to match the input format expected by Gemma 3n, ensuring compatibility and optimal inference.

- Model: `Gemma-3n-E4b-it-int4` (Vision-Language model)
- Input Types: `Image` + `Text` (formatted prompt)
- Prompt Format: `"<image_soft_token> In this image, <question>"`
- Output: Textual answer grounded in image + query
- Deployment Mode: Offline-capable (via `Google Edge Gallery`)
- Adaptation Options: `ONNX` / `Faiss` (mobile-compatible)

## System Demo

To illustrate the full pipeline of **FieldLens**, we provide a step-by-step demonstration using real-world aerial footage. Each stage of the offline process is visualized as follows:

### 1. Input Image (from agricultural drone video)

Raw frame captured from a cornfield using an agricultural drone.

![Figure 1: Original RGB Image](data/images/original.jpg)  
**Figure 1**: Original RGB Image

### 2. Anomaly Detection

Binary segmentation highlights sparse growth regions based on ExG index and morphological filtering.

![Figure 2: ExG Vegetation Index Map](data/images/exg.jpg)  
**Figure 2**: ExG Vegetation Index Map

![Figure 3: Binary Anomaly Mask](data/images/mask.jpg)  
**Figure 3**: Binary Anomaly Mask

![Figure 4: Annotated Image with Red Boxes](data/images/box.jpg)  
**Figure 4**: Annotated Image with Red Boxes

### 3. Question Generation

Based on detected anomaly, a RAG-style prompt is constructed to formulate an agronomic question.

```
Question: What causes sparse crop growth here?
Retrieved: “According to agronomic guidelines, pest activity increases in the tasseling stage...”
Prompt: Please answer <question> based on <retrieved_text>
```

### 4. Multimodal Reasoning (Gemma 3n output)
Final reasoning result generated by Gemma 3n using image + prompt.

![Figure 5: Multimodal reasoning example using a Gemma 3n-compatible prompt](data/images/gemma.jpg)  
**Figure 5**: Multimodal reasoning example using a Gemma 3n-compatible prompt

### Interactive Demo

Due to the offline architecture of the application — which is designed to run entirely on local Android devices — a traditional cloud-based live demo is not available. To illustrate the user interaction flow and highlight the core features, a web-based walkthrough has been developed to simulate the app experience.
Interactive mockup: 
[https://tianmiao11.github.io/FieldLens/](https://tianmiao11.github.io/FieldLens/)

## Future Improvements

### 1. Enhanced Anomaly Classification  
The current system focuses solely on detecting sparse vegetation. However, anomaly patterns vary significantly across different crop growth stages. Future versions may incorporate multi-class segmentation to identify various stressors such as disease, drought, or lodging, with adaptive modes tailored to different phenological phases.

### 2. Deploy Multimodal Inference on Agricultural Drones
Rather than relying on mobile or desktop devices, future iterations may embed quantized vision-language models directly onto agricultural drones. This would enable fully offline, on-device reasoning—allowing autonomous, real-time decision-making in the field without human intervention or internet connectivity.

### 3. Field-Level Report Generation & Actionable Feedback 
Instead of analyzing frames in isolation, the system can aggregate outputs into field-level diagnostic reports. These reports would include spatial summaries and recommendations for targeted interventions. The insights could be used to guide drones for site-specific actions such as localized spraying, fertilization, or replanting, enabling precision agriculture at the plant level.

## Outro

The underlying pipeline—anomaly detection, question generation, and multimodal reasoning—is crop-agnostic and adaptable. Its potential applications extend beyond cornfields to other agricultural domains, or even forest monitoring and environmental conservation.
This project showcases not just a functional prototype, but a glimpse into the future of edge intelligence. By deploying multimodal models like Gemma 3n in conjunction with traditional hardware such as agricultural drones, we begin to see the contours of a new class of field-deployable, offline-capable robotic systems.
