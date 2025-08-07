from transformers import AutoTokenizer, AutoModel
from embed_model import last_token_pool
import torch
from pymilvus import MilvusClient

def retrieve_text(question: str, client: MilvusClient, collection_name: str = "demo_collection", top_k: int = 2) -> str:
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', padding_side='left')
    model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B')
    model.eval()

    batch_dict = tokenizer(
        [question],
        padding=True,
        truncation=True,
        max_length=8192,
        return_tensors="pt"
    )
    batch_dict.to(model.device)

    with torch.no_grad():
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        query_vectors = embeddings.cpu().numpy()

    results = client.search(
        collection_name=collection_name,
        data=query_vectors,
        limit=top_k,
        output_fields=["text", "subject"]
    )
    retrieved_text = results[0][0]['text']
    prompt = f"Based on<{retrieved_text}>,Answer<{question}>"
    return prompt