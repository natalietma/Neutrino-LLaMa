import torch
from transformers import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info

def collate_fn(examples, processor):
    texts = [processor.apply_chat_template(e["messages"], tokenize=False) for e in examples]
    image_inputs = [process_vision_info(e["messages"])[0] for e in examples]

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    if isinstance(processor, Qwen2VLProcessor):
        image_tokens = [151652, 151653, 151655]
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]

    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100

    batch["labels"] = labels
    return batch
