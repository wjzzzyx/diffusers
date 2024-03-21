import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel


class CLIPEncoderUnlimited(nn.Module):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]

    def __init__(self, version="openai/clip-vit-large-patch14", max_length=77,
                 layer="last", layer_idx=None):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.max_length = max_length
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12
        
        self.comma_token_id = self.tokenizer.vocab.get(',</w>', None)
        self.stop_token_id = self.tokenizer.vocab.get('.</w>', None)
    
    @property
    def device(self):
        return self.transformer.text_model.embeddings.token_embedding.weight.device
    
    def forward(self, texts):
        tokens = self.get_tokens(texts)
        chunks = self.split_chunks(tokens)    # list(batch) of list(chunk)
        chunks_input = [x for x in zip(chunks)]    # list(chunk) of list(batch)

        text_embs = list()
        for chunk in chunks_input:
            emb = self.encode_with_transformers(chunk)
            text_embs.append(emb)
        
        return torch.stack(text_embs, dim=1)

    def get_tokens(self, texts):
        return self.tokenizer(texts, truncation=False, add_special_tokens=False).input_ids

    def split_chunks(self, batch_tokens):
        batch_chunks = list()
        for tokens in batch_tokens:
            batch_chunks.append(self._split_chunks(tokens))
        return batch_chunks

    def _split_chunks(self, tokens):
        split_points = list()
        pos = 0
        last_comma = -1
        i = 0
        while i < len(tokens):
            if tokens[i] == self.stop_token_id:
                split_points.append(i)
                pos = 0
                i += 1
            elif tokens[i] == self.comma_token_id:
                last_comma = i
                i += 1
            elif pos == self.tokenizer.model_max_length - 2:
                split_points.append(last_comma)
                pos = 0
                i = last_comma + 1
        
        chunks = list()
        for j in range(len(split_points)):
            p = split_points[j]
            if j == 0:
                chunks.append(tokens[0:p])
            else:
                last_p = split_points[j-1]
                chunks.append(tokens[last_p + 1:p])
                if j == len(split_points) - 1:
                    chunks.append(tokens[p:])
        chunks = [chunk for chunk in chunks if chunk != []]

        for ichunk in range(len(chunks)):
            while len(chunks[ichunk]) < self.tokenizer.model_max_length - 2:
                chunks[ichunk].append(self.tokenizer.eos_token_id)
        
        for ichunk in range(len(chunks)):
            chunks[ichunk] = [self.tokenizer.bos_token_id] + chunks[ichunk] + [self.tokenizer.eos_token_id]
        
        return chunks

    def encode_with_transformers(self, batch_tokens):
        batch_tokens = torch.asarray(batch_tokens, device=self.device)
        outputs = self.transformer(input_ids=batch_tokens, output_hidden_states=(self.layer == 'hidden'))
        if self.layer == 'hidden':
            h = outputs.hidden_states[self.layer_idx]
            h = self.transformer.text_model.final_layer_norm(h)
        else:
            h = outputs.last_hidden_state
        return h