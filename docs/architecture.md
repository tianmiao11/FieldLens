# 🧠 FieldLens Architecture Overview

This document describes the internal architecture and workflow of the **FieldLens** system, designed for offline crop abnormality detection and multimodal analysis using AI models like Gemma 3n.

---

## 🔁 Pipeline Overview

FieldLens is divided into three modular components:

### 1. 🚨 Anomaly Detection
- Input: RGB image (e.g., drone footage frame)
- Process:
  - Compute ExG (Excess Green Index)
  - Detect sparse regions using thresholding and morphology
  - Generate a binary mask and bounding boxes
- Output: `output/box.jpg`

### 2. ❓ Question Generation (RAG-style)
- Input: A user question
- Process:
  - Use Qwen3 Embedding model to vectorize guidebook sentences
  - Store embeddings in Milvus vector DB
  - Retrieve top-matching sentence
  - Combine with the question to form a natural-language prompt
- Output: A prompt for multimodal reasoning

### 3. 🔍 Multimodal Reasoning
- Input: Image (`box.jpg`) + Prompt
- Process:
  - Use Gemma 3n to generate answers conditioned on image + text
- Output: Explanatory answer (fully offline, on-device ready)

---

## 📁 Directory Structure

```
FieldLens/
├── data/                 # Input samples
│   ├── images/
│   ├── texts/
├── output/               # Generated masks, boxes, results
├── src/
│   ├── anomaly_detection/
│   ├── question_generation/
│   ├── multimodal_reasoning/
├── main.py               # Full pipeline runner
├── docs/
│   └── architecture.md   # You are here
```

---

## ▶️ Run Example

To run the entire pipeline end-to-end:

```bash
python main.py
```

Ensure:
- `data/images/original.jpg` exists
- Milvus is initialized with guidebook vectors
- `box.jpg` is generated and used for multimodal reasoning

---

## 🔧 Dependencies

- `transformers` (for Qwen & Gemma 3n)
- `pymilvus` (for vector DB)
- `torch`, `opencv-python`, `matplotlib`

---

## ✨ Future Improvements

- Add batch processing support
- Extend to other crops and domains
- Quantize for low-end Android deployment