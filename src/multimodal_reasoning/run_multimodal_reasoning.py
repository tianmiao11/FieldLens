from transformers import AutoProcessor, AutoModelForImageTextToText, GenerationConfig
from PIL import Image
import torch

def run_multimodal_reasoning(image_path: str):
    GEMMA_PATH = "google/gemma-3n/transformers/gemma-3n-e2b-it"

    # Load processor and model
    processor = AutoProcessor.from_pretrained(GEMMA_PATH)
    model = AutoModelForImageTextToText.from_pretrained(
        GEMMA_PATH,
        torch_dtype="auto",
        device_map="auto"
    )

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Compose message with prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Answer the questions in easy to understand language. Do not output irrelevant content. Please refer to the document: Foliar fertilizer application timing should preferably be morning or evening when temperature is lower and light intensity is weaker, facilitating foliar absorption. Avoid application during high temperature, strong light, rainy or windy conditions to prevent fertilizer loss or leaf burn. Question: When should I spray fertilizer?"}
            ]
        }
    ]

    # Tokenize input
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=model.dtype)

    input_len = inputs["input_ids"].shape[-1]

    # Load and set generation config
    generation_config = GenerationConfig.from_pretrained(GEMMA_PATH)
    generation_config.cache_implementation = "static"
    generation_config.max_new_tokens = 512
    generation_config.do_sample = False

    # Generate output
    outputs = model.generate(**inputs, generation_config=generation_config, disable_compile=True)
    text = processor.batch_decode(outputs[:, input_len:], skip_special_tokens=True)[0]
    return text