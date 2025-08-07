from transformers import AutoTokenizer, AutoModel
from embed_model import last_token_pool
import torch
import torch.nn.functional as F
from pymilvus import MilvusClient

def build_vector_store(txt_path: str, collection_name: str = "demo_collection"):
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', padding_side='left')
    model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B')
    model.eval()

    with open(txt_path, 'r', encoding='utf-8') as file:
        guidebook_lines = [line.strip() for line in file if line.strip()]

    batch_dict = tokenizer(
        guidebook_lines,
        padding=True,
        truncation=True,
        max_length=8192,
        return_tensors="pt"
    )
    batch_dict.to(model.device)

    with torch.no_grad():
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)

    client = MilvusClient("milvus_demo.db")
    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)
    client.create_collection(collection_name=collection_name, dimension=embeddings.shape[1])

    data = [
        {"id": i, "vector": embeddings[i].cpu().numpy(), "text": guidebook_lines[i], "subject": "guidebook"}
        for i in range(len(embeddings))
    ]
    client.insert(collection_name=collection_name, data=data)
    return client