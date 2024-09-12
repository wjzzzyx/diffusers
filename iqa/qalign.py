import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, CLIPImageProcessor


class QAlign(nn.Module):
    def __init__(self, dtype='fp16') -> None:
        super().__init__()
        assert dtype in ['fp16', '4bit', '8bit']

        self.model = AutoModelForCausalLM.from_pretrained(
            "q-future/one-align",
            trust_remote_code=True,
            load_in_4bit = True if dtype == "4bit" else False,
            load_in_8bit = True if dtype == "8bit" else False,
            torch_dtype=torch.float16 if dtype == "fp16" else None,
            device_map="cpu"
        )
        self.image_processor = CLIPImageProcessor.from_pretrained("q-future/one-align")
    
    def forward(self, x: torch.Tensor, task_: str):
        """
        x: image tensor
        task_: choices=[quality, aesthetic]
        """
        x = self.image_processor.preprocess(x, return_tensors="pt")["pixel_values"]
        x = x.half()
        score = self.model.score(images=None, image_tensor=x, task_=task_, input_="image")
        return score