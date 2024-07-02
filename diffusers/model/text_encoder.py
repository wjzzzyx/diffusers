import open_clip
import re
import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel


class TextEncoderUnlimited(nn.Module):
    "A base class for unlimited text encoder. Need to inherit and specify tokenizer and model."
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]
    
    def forward(self, texts):
        tokens = self.get_tokens(texts)
        chunks = self.split_chunks(tokens)    # list(batch) of list(chunk)
        chunks_input = list(map(list, zip(*chunks)))    # list(chunk) of list(batch)

        text_embs = list()
        for chunk in chunks_input:
            emb = self.encode_batch_tokens(chunk)
            text_embs.append(emb)
        
        return torch.concatenate(text_embs, dim=1)

    def get_tokens(self, texts):
        return self.tokenize(texts)
    
    def tokenize(self, texts):
        raise NotImplementedError()

    def split_chunks(self, batch_tokens):
        batch_chunks = list()
        for tokens in batch_tokens:
            batch_chunks.append(self._split_chunks(tokens))
        
        max_num_chunks = max([len(chunks) for chunks in batch_chunks])
        empty_chunk = [self.bos_token_id] + [self.eos_token_id] * (self.model_max_length - 1)
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
            while len(chunks[ichunk]) < self.model_max_length - 2:
                chunks[ichunk].append(self.eos_token_id)
        
        for ichunk in range(len(chunks)):
            chunks[ichunk] = [self.bos_token_id] + chunks[ichunk] + [self.eos_token_id]
        
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
            elif pos == self.model_max_length - 2:
                split_points.append(last_comma)
                pos = 0
                i = last_comma + 1
            elif tokens[i] == self.comma_token_id:
                last_comma = i
                pos += 1
                i += 1
            else:
                pos += 1
                i += 1
        return split_points

    def encode_batch_tokens(self, batch_tokens):
        raise NotImplementedError()


class TextEncoderWeighted(TextEncoderUnlimited):
    def forward(self, texts):
        texts_with_weights = self.parse_weighted_text(texts)
        tokens, weights = self.get_tokens(texts_with_weights)
        chunks, chunk_weights = self.split_chunks(tokens, weights)    # list(batch) of list(chunk)
        chunks_input = list(map(list, zip(*chunks)))    # list(chunk) of list(batch)

        text_embs = list()
        for chunk in chunks_input:
            emb = self.encode_batch_tokens(chunk)
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
                phrase_tokens = self.tokenize(phrase)
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
        empty_chunk = [self.bos_token_id] + [self.eos_token_id] * (self.model_max_length - 1)
        empty_chunk_weight = [1.0] * self.model_max_length
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

        chunks = [chunk for ichunk, chunk in enumerate(chunks) if ichunk == 0 or chunk != []]
        chunk_weights = [w for ichunk, w in enumerate(chunk_weights) if ichunk == 0 or w != []]

        # pad each chunk to be model_max_length
        for ichunk in range(len(chunks)):
            while len(chunks[ichunk]) < self.model_max_length - 2:
                chunks[ichunk].append(self.eos_token_id)
                chunk_weights[ichunk].append(1.0)
        
        # add bos and eos tokens to each chunk
        for ichunk in range(len(chunks)):
            chunks[ichunk] = [self.bos_token_id] + chunks[ichunk] + [self.eos_token_id]
            chunk_weights[ichunk] = [1.0] + chunk_weights[ichunk] + [1.0]
        
        return chunks, chunk_weights


class CLIPTextEncoder(TextEncoderWeighted):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    def __init__(self, version="openai/clip-vit-large-patch14", layer="last", layer_idx=-1):
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.layer = layer
        self.layer_idx = layer_idx
        
        vocab = self.tokenizer.get_vocab()
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.model_max_length = self.tokenizer.model_max_length
        self.comma_token_id = vocab.get(',</w>', None)
        self.stop_token_id = vocab.get('.</w>', None)
    
    @property
    def device(self):
        return self.transformer.text_model.embeddings.token_embedding.weight.device
    
    def tokenize(self, text):
        # text is a single string
        return self.tokenizer(text, truncation=False, add_special_tokens=False).input_ids
    
    def encode_batch_tokens(self, batch_tokens):
        batch_tokens = torch.asarray(batch_tokens, device=self.device)
        outputs = self.transformer(input_ids=batch_tokens, output_hidden_states=(self.layer == 'hidden'))
        if self.layer == 'hidden':
            h = outputs.hidden_states[self.layer_idx]
            h = self.transformer.text_model.final_layer_norm(h)
        else:
            h = outputs.last_hidden_state
        return h


class OpenCLIPTextEncoder(TextEncoderWeighted):
    def __init__(self, arch, version, layer='last', layer_idx=-1):
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = open_clip.get_tokenizer(arch)
        self.model = open_clip.create_model(arch, pretrained=version)
        del self.model.visual
        self.layer = layer
        self.layer_idx = layer_idx if layer_idx >= 0 else self.model.transformer.layers + layer_idx
    
        self.bos_token_id = self.tokenizer.sot_token_id
        self.eos_token_id = self.tokenizer.eot_token_id
        self.model_max_length = self.tokenizer.context_length
        self.comma_token_id = self.tokenizer.encoder[',</w>']
        self.stop_token_id = self.tokenizer.encoder['.</w>']
    
    @property
    def device(self):
        return self.model.token_embedding.weight.device
    
    def tokenize(self, text):
        # text is a single string
        return self.tokenizer.encode(text)
    
    def encode_batch_tokens(self, batch_tokens):
        batch_tokens = torch.asarray(batch_tokens, device=self.device)
        x = self.model.token_embedding(batch_tokens)
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)    # shape (seq, batch, emb)
        for i_block, block in enumerate(self.model.transformer.resblocks):
            x = block(x, attn_mask=self.model.attn_mask)
            if i_block == self.layer_idx:
                break
        x = x.permute(1, 0, 2)
        x = self.model.ln_final(x)
        return x


class OpenCLIPTextEncoderPooled(OpenCLIPTextEncoder):
    def forward(self, texts):
        texts_with_weights = self.parse_weighted_text(texts)
        tokens, weights = self.get_tokens(texts_with_weights)
        chunks, chunk_weights = self.split_chunks(tokens, weights)    # list(batch) of list(chunk)
        chunks_input = list(map(list, zip(*chunks)))    # list(chunk) of list(batch)

        text_embs = list()
        last_embs_for_pool = list()
        for chunk in chunks_input:
            emb, last = self.encode_batch_tokens(chunk)
            text_embs.append(emb)
            last_embs_for_pool.append(last)
        text_embs = torch.concatenate(text_embs, dim=1)    # shape (batch, seq, emb)
        
        weights = torch.asarray(chunk_weights, device=self.device).view(len(chunk_weights), -1)
        original_mean = text_embs.mean(dim=(1, 2))
        text_embs = text_embs * weights.unsqueeze(-1)
        new_mean = text_embs.mean(dim=(1, 2))
        text_embs = text_embs * (original_mean / new_mean)[:, None, None]

        batch_size = len(texts)
        eos_position = torch.asarray(chunks_input[0]).argmax(dim=-1)
        pooled_embs = last_embs_for_pool[0][
            torch.arange(batch_size, device=self.device),
            eos_position
        ]
        pooled_embs = pooled_embs @ self.model.text_projection

        return text_embs, pooled_embs

    def encode_batch_tokens(self, batch_tokens):
        batch_tokens = torch.asarray(batch_tokens, device=self.device)
        x = self.model.token_embedding(batch_tokens)
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)    # shape (seq, batch, emb)
        for i_block, block in enumerate(self.model.transformer.resblocks):
            x = block(x, attn_mask=self.model.attn_mask)
            if i_block == self.layer_idx:
                out = x
        out = out.permute(1, 0, 2)
        # out = self.model.ln_final(out)    # should we normalize this?
        x = x.permute(1, 0, 2)
        x = self.model.ln_final(x)
        return out, x


class CLIPTextEncoder_TextualInversion(CLIPTextEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface). Allow additional tokens for textual inversion."""

    def __init__(
        self, version="openai/clip-vit-large-patch14", layer="last", layer_idx=-1,
        ti_name2numtoken=None
    ):
        super().__init__(version, layer, layer_idx)
        self.original_num_tokens = len(self.tokenizer)
        self.ti_name2numtoken = dict(ti_name2numtoken) if ti_name2numtoken else None

        # freeze all parameters except for token embeddings
        self.transformer.requires_grad_(True)
        self.transformer.text_model.encoder.requires_grad_(False)
        self.transformer.text_model.final_layer_norm.requires_grad_(False)
        self.transformer.text_model.embeddings.position_embedding.requires_grad_(False)
    
    def expand_vocab(self):
        for name, num_tokens in self.ti_name2numtoken.items():
            name_repeats = [f'{name}{i}' for i in range(len(num_tokens))]
            num_added_tokens = self.tokenizer.add_tokens(name_repeats)
            assert(num_added_tokens == num_tokens)
            token_ids = self.tokenizer.convert_tokens_to_ids(name_repeats)
            assert(min(token_ids) == token_ids[0])
            assert(token_ids[-1] == token_ids[0] + len(token_ids) - 1)
            assert(len(self.tokenizer) - 1 == token_ids[-1])
        self.original_token_embedding = self.transformer.text_model.embeddings.token_embedding.clone()
        self.transformer.resize_token_embeddings(len(self.tokenizer))
        # TODO initialize new embeddings
    
    def expand_vocab_from_weight(self, names, embeddings):
        self.ti_name2numtoken = {name: emb.size(0) for name, emb in zip(names, embeddings)}
        self.expand_vocab()
        index = self.original_num_tokens
        with torch.no_grad():
            for name, embedding in zip(names, embeddings):
                num_token = embedding.size(0)
                self.transformer.text_model.embeddings.token_embedding[index:index+num_token] = embedding
                index += num_token
    
    def trainable_parameters(self):
        return self.transformer.text_model.embeddings.token_embedding.parameters()

    def forward(self, texts):
        texts = self.repeat_ti_names(texts)
        return super().forward(texts)
    
    def repeat_ti_names(self, texts):
        for i in range(len(texts)):
            text = texts[i]
            for name, num_token in self.ti_name2numtoken.items():
                if name in text:
                    text = text.replace(name, ' '.join([f'{name}{j}' for j in range(num_token)]))
            texts[i] = text
        return texts
    
    def freeze_original_embedding(self):
        with torch.no_grad():
            self.transformer.text_model.embeddings.token_embedding.weight[:self.original_num_tokens] = self.original_token_embedding

    def get_ti_embedding(self, state_dict):
        token_embedding_weight = state_dict['model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight']
        ti_embedding_weight = token_embedding_weight[self.original_num_tokens:]
        ti_embeddings_dict = dict()
        i = 0
        for name, num_token in self.ti_name2numtoken.items():
            ti_embeddings_dict[name] = ti_embedding_weight[i:i + num_token]
            i += num_token
        return ti_embeddings_dict
