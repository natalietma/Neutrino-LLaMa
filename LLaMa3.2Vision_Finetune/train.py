import glob
import torch
from tqdm import tqdm
from PIL import Image, ImageOps
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from dataset import load_neutrino_dataset
from collator import collate_fn

PROMPT = "Classify the attached pixel maps as NuE CC, NuMu CC, or Neutral Current."
SYSTEM_MESSAGE = "You are a neutrino physicist who is working with Fermilab Deep Underground Neutrino Experiment (DUNE).Your are given simulated DUNE Near-Detector-like readouts (as pairs of 2D pixel maps) in the form of two images in the zx plane and the zy plane. In the pixel maps as images , the z-axis is the beam direction. Your task is to identify if the event in the pixel maps is Electron Neutrino Charged Current interaction (NuE CC) or Muon Neutrino Charged Current interaction (NuMu CC) or Neutral Current interaction. For NuE CC there should be a fuzzy electron shower, For NuMu CC the muon track is usually longer and narrow, and for Neutral Current there is no significant muon track or electron shower."
DATASET_ROOT = "../data" #"/baldig/physicsprojects2/dikshans/datasets/bigDataset"
MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
OUTPUT_DIR = "./checkpoints/Llama3.2-11B-Vision-Instruct-Neutrino"

def main():
    # ---------------- Dataset ---------------- #
    train_dataset = load_neutrino_dataset(
        DATASET_ROOT, system_message=SYSTEM_MESSAGE, prompt=PROMPT, split="train"
    )

    # ---------------- Model ---------------- #
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # ---------------- PEFT Config ---------------- #
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                        "up_proj", "down_proj"],
        task_type="VISION_MODEL",
    )

    # ---------------- Training Config ---------------- #
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=0.3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        learning_rate=2e-4,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        push_to_hub=False,
        report_to="tensorboard",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    training_args.remove_unused_columns = False

    # ---------------- Trainer ---------------- #
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda x: collate_fn(x, processor),
        dataset_text_field="",
        peft_config=peft_config,
        tokenizer=processor.tokenizer,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)

    del model
    del trainer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
