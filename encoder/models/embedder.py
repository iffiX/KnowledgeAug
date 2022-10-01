import torch as t
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from encoder.utils.settings import (
    model_cache_dir,
    proxies,
    huggingface_mirror,
    local_files_only,
)


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return t.sum(token_embeddings * input_mask_expanded, 1) / t.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class Embedder:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.model = AutoModel.from_pretrained(
            "sentence-transformers/all-mpnet-base-v2",
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            local_files_only=local_files_only,
        ).to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-mpnet-base-v2",
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            local_files_only=local_files_only,
        )

    def embed(self, inputs, max_length=256, batch_size=256):
        with t.no_grad():
            result = []
            for start in range(0, len(inputs), batch_size):
                batch = self.tokenizer(
                    inputs[start : start + batch_size],
                    padding="longest",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                embedding = mean_pooling(
                    self.model(**batch)[0], batch["attention_mask"]
                )
                result.append(F.normalize(embedding, p=2, dim=1))
            return t.cat(result)
