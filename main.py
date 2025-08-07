"""
main.py - Unified Entry Point for FieldLens Pipeline
1. Detect abnormal regions in the image
2. Generate question + retrieved context (RAG-style)
3. Perform multimodal reasoning using Gemma 3n
"""

from src.anomaly_detection.exg_utils import generate_exg
from src.anomaly_detection.mask_generator import generate_mask
from src.anomaly_detection.box_drawer import draw_boxes_from_mask

from src.question_generation.retrieve_and_prompt import retrieve_text
from pymilvus import MilvusClient

from src.multimodal_reasoning.run_multimodal_reasoning import run_multimodal_reasoning

# === Step 1: Anomaly Detection ===
print("🚨 Step 1: Detecting anomalies...")
original_path = "data/images/original.jpg"
exg_path = "output/exg.jpg"
mask_path = "output/mask.jpg"
box_path = "output/box.jpg"

generate_exg(original_path, exg_path)
generate_mask(exg_path, mask_path)
draw_boxes_from_mask(original_path, mask_path, box_path)

# === Step 2: Question Generation ===
print("❓ Step 2: Generating question and retrieving context...")
question = "When should I spray fertilizer?"
client = MilvusClient("milvus_demo.db")  # assumes pre-built vector DB
prompt = retrieve_text(question, client)
print("📝 Prompt:", prompt)

# === Step 3: Multimodal Reasoning ===
print("🔍 Step 3: Running Gemma 3n for multimodal reasoning...")
answer = run_multimodal_reasoning(image_path=box_path)
print("🧠 Final Answer:", answer)