import glob
import torch
from tqdm import tqdm
from PIL import Image, ImageOps
from dataset import load_neutrino_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, PhrasalConstraint
from qwen_vl_utils import process_vision_info

# ---------------- Config ---------------- #
PROMPT = "Classify the attached pixel maps as NuE CC, NuMu CC, or Neutral Current."
SYSTEM_MESSAGE = '''You are a neutrino physicist who is working with Fermilab Deep Underground Neutrino Experiment (DUNE).Your are given simulated DUNE Near-Detector-like readouts (as pairs of 2D pixel maps)
                    in the form of two images in the zx plane and the zy plane. In the pixel maps as images , the z-axis is the beam direction. Your task is to identify if the event in the pixel maps is Electron
                    Neutrino Charged Current interaction (NuE CC) or Muon Neutrino Charged Current interaction (NuMu CC) or Neutral Current interaction. For NuE CC there should be a fuzzy electron shower,
                    For NuMu CC the muon track is usually longer and narrow, and for Neutral Current there is no significant muon track or electron shower.'''

MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
ADAPTER_PATH = "/mnt/ironwolf_12t/users/shared_llama_adapters/Llama3.2-11B-Vision-Instruct-Neutrino"
DATASET_ROOT = "../data" #`"/baldig/physicsprojects2/dikshans/datasets/bigDataset"
LOG_FILE = "./predictions/finetune_predictions.log"

CLASSES = {"nuecc": "NuE CC", "numucc": "NuMu CC", "nc": "Neutral Current"}
CLASSES_REV = {v: k for k, v in CLASSES.items()}


# ---------------- Inference ---------------- #
def run_inference():
    # Load dataset
    val_dataset = load_neutrino_dataset(
        DATASET_ROOT, system_message=SYSTEM_MESSAGE, prompt=PROMPT, split="test"
    )

    # Load model + adapter
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    model.load_adapter(ADAPTER_PATH)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # Add phrasal constraint
    constraints = [
        PhrasalConstraint(
            processor.tokenizer("I classify the pixel maps as", add_special_tokens=False).input_ids
        )
    ]

    with open(LOG_FILE, "a") as log_file:
        for data in tqdm(val_dataset, desc="Running Inference:"):
            message = data["messages"][:2]
            gt = CLASSES_REV[data["messages"][2]["content"][0]["text"]]

            text = processor.apply_chat_template(message, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(message)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                num_beams=2,
                return_dict_in_generate=True,
                output_logits=True,
                constraints=constraints,
            )

            # Decode output
            trimmed_ids = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, outputs.sequences)
            ]
            decoded = processor.batch_decode(
                trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # Map output to class
            if "NuMu" in decoded:
                answer = "numucc"
            elif "NuE" in decoded:
                answer = "nuecc"
            elif "Neutral" in decoded:
                answer = "nc"
            else:
                answer = decoded

            # Confidence scores
            scores = outputs.logits
            conf = []
            for i, e in enumerate(scores):
                e[0] = torch.log_softmax(e[0], dim=-1)
                if torch.argmax(e[0]) in [33424, 59794]:
                    conf = [
                        e[0][33424] + scores[i+1][0][40220], # Nu + Mu
                        e[0][33424] + scores[i+1][0][36],    # Nu + E
                        e[0][59794] + scores[i+1][0][9303],  # Neutral + Current
                    ]

            if conf:
                conf = torch.exp(torch.tensor(conf)).tolist()

            # Log prediction
            log_file.write(f"{gt}|{answer}|{conf}|{decoded}\n")


if __name__ == "__main__":
    run_inference()
