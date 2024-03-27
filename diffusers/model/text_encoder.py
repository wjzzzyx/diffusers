import re
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
        
        vocab = self.tokenizer.get_vocab()
        self.comma_token_id = vocab.get(',</w>', None)
        self.stop_token_id = vocab.get('.</w>', None)
    
    @property
    def device(self):
        return self.transformer.text_model.embeddings.token_embedding.weight.device
    
    def forward(self, texts):
        tokens = self.get_tokens(texts)
        chunks = self.split_chunks(tokens)    # list(batch) of list(chunk)
        chunks_input = list(map(list, zip(*chunks)))    # list(chunk) of list(batch)

        text_embs = list()
        for chunk in chunks_input:
            emb = self.encode_with_transformers(chunk)
            text_embs.append(emb)
        
        return torch.concatenate(text_embs, dim=1)

    def get_tokens(self, texts):
        return self.tokenizer(texts, truncation=False, add_special_tokens=False).input_ids

    def split_chunks(self, batch_tokens):
        batch_chunks = list()
        for tokens in batch_tokens:
            batch_chunks.append(self._split_chunks(tokens))
        
        max_num_chunks = max([len(chunks) for chunks in batch_chunks])
        empty_chunk = [self.tokenizer.bos_token_id] + [self.tokenizer.eos_token_id] * (self.tokenizer.model_max_length - 1)
        for chunks in batch_chunks:
            while len(chunks) < max_num_chunks:
                chunks.append(empty_chunk)
        return batch_chunks

    def _split_chunks(self, tokens):
        split_points = self.locate_split_points(tokens)
        chunks = list()
        for j in range(len(split_points)):
            p = split_points[j]
            if j == 0:
                chunks.append(tokens[0:p+1])
            else:
                last_p = split_points[j-1]
                chunks.append(tokens[last_p+1:p+1])
            if j == len(split_points) - 1:
                chunks.append(tokens[p+1:])
        chunks = [chunk for chunk in chunks if chunk != []]

        for ichunk in range(len(chunks)):
            while len(chunks[ichunk]) < self.tokenizer.model_max_length - 2:
                chunks[ichunk].append(self.tokenizer.eos_token_id)
        
        for ichunk in range(len(chunks)):
            chunks[ichunk] = [self.tokenizer.bos_token_id] + chunks[ichunk] + [self.tokenizer.eos_token_id]
        
        return chunks
    
    def _locate_split_points(self, tokens):
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
                pos += 1
                i += 1
            elif pos == self.tokenizer.model_max_length - 2:
                split_points.append(last_comma)
                pos = 0
                i = last_comma + 1
            else:
                pos += 1
                i += 1
        return split_points

    def encode_with_transformers(self, batch_tokens):
        batch_tokens = torch.asarray(batch_tokens, device=self.device)
        outputs = self.transformer(input_ids=batch_tokens, output_hidden_states=(self.layer == 'hidden'))
        if self.layer == 'hidden':
            h = outputs.hidden_states[self.layer_idx]
            h = self.transformer.text_model.final_layer_norm(h)
        else:
            h = outputs.last_hidden_state
        return h


class CLIPEncoderWeighted(CLIPEncoderUnlimited):
    def forward(self, texts):
        texts_with_weights = self.parse_weighted_text(texts)
        tokens, weights = self.get_tokens(texts_with_weights)
        chunks, chunk_weights = self.split_chunks(tokens, weights)    # list(batch) of list(chunk)
        chunks_input = list(map(list, zip(*chunks)))    # list(chunk) of list(batch)

        text_embs = list()
        for chunk in chunks_input:
            emb = self.encode_with_transformers(chunk)
            text_embs.append(emb)
        text_embs = torch.concatenate(text_embs, dim=1)    # shape (batch, seq, emb)
        
        weights = torch.asarray(chunk_weights, device=self.device).view(len(chunk_weights), -1)
        original_mean = text_embs.mean(dim=(1, 2))
        text_embs = text_embs * weights.unsqueeze(-1)
        new_mean = text_embs.mean(dim=(1, 2))
        text_embs = text_embs * (original_mean / new_mean)[:, None, None]

        return text_embs

    def parse_weighted_text(self, texts):
        pattern = re.compile(r'\((.*?)\:(\d+(?:\.\d+)?)\)')
        texts_with_weights = list()

        for text in texts:
            text_with_weights = list()
            last = 0
            matches = pattern.finditer(text)
            for m in matches:
                m_text, m_weight = m.group(0).strip('()').split(':')
                m_text = m_text.strip()
                m_weight = float(m_weight.strip())
                text_with_weights.append((text[last:m.start()], 1.0))
                text_with_weights.append((m_text, m_weight))
                last = m.end()
            text_with_weights.append((text[last:], 1.0))

            texts_with_weights.append(text_with_weights)
        
        return texts_with_weights

    def get_tokens(self, texts_with_weights):
        batch_tokens = list()
        batch_weights = list()
        for text_with_weights in texts_with_weights:
            tokens = list()
            weights = list()
            for phrase, weight in text_with_weights:
                phrase_tokens = self.tokenizer(phrase, truncation=False, add_special_tokens=False).input_ids
                tokens.extend(phrase_tokens)
                weights.extend([weight] * len(phrase_tokens))
            batch_tokens.append(tokens)
            batch_weights.append(weights)
        return batch_tokens, batch_weights
    
    def split_chunks(self, batch_tokens, batch_weights):
        batch_chunks = list()
        batch_chunk_weights = list()

        for tokens, weights in zip(batch_tokens, batch_weights):
            chunks, chunk_weights = self._split_chunks(tokens, weights)
            batch_chunks.append(chunks)
            batch_chunk_weights.append(chunk_weights)
        
        max_num_chunks = max([len(chunks) for chunks in batch_chunks])
        empty_chunk = [self.tokenizer.bos_token_id] + [self.tokenizer.eos_token_id] * (self.tokenizer.model_max_length - 1)
        empty_chunk_weight = [1.0] * self.tokenizer.model_max_length
        for chunks, chunk_weights in zip(batch_chunks, batch_chunk_weights):
            while len(chunks) < max_num_chunks:
                chunks.append(empty_chunk)
                chunk_weights.append(empty_chunk_weight)
        
        return batch_chunks, batch_chunk_weights

    def _split_chunks(self, tokens, weights):
        split_points = self._locate_split_points(tokens)
        chunks = list()
        chunk_weights = list()
        last = 0

        for p in split_points:
            chunks.append(tokens[last:p+1])
            chunk_weights.append(weights[last:p+1])
            last = p + 1
        chunks.append(tokens[last:])
        chunk_weights.append(weights[last:])

        chunks = [chunk for chunk in chunks if chunk != []]
        chunk_weights = [w for w in chunk_weights if w != []]

        # pad each chunk to be model_max_length
        for ichunk in range(len(chunks)):
            while len(chunks[ichunk]) < self.tokenizer.model_max_length - 2:
                chunks[ichunk].append(self.tokenizer.eos_token_id)
                chunk_weights[ichunk].append(1.0)
        
        # add bos and eos tokens to each chunk
        for ichunk in range(len(chunks)):
            chunks[ichunk] = [self.tokenizer.bos_token_id] + chunks[ichunk] + [self.tokenizer.eos_token_id]
            chunk_weights[ichunk] = [1.0] + chunk_weights[ichunk] + [1.0]
        
        return chunks, chunk_weights