import glob
from tqdm import tqdm
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

CLASSES = {"nuecc": "NuE CC", "numucc": "NuMu CC", "nc": "Neutral Current"}

def format_data(zx, zy, label, system_message, prompt, split):
    if split == 'train':
        return {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": system_message}]},
                {"role": "user", "content": [
                    {"type": "image", "image": zx},
                    {"type": "image", "image": zy},
                    {"type": "text", "text": prompt},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": f"I classify the pixel maps as {label}."}
                ]},
            ]
        }
    elif split == "test":
        return {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": system_message}]},
                {"role": "user", "content": [
                    {"type": "image", "image": zx},
                    {"type": "image", "image": zy},
                    {"type": "text", "text": prompt},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": label}
                ]},
            ]
        }

def load_neutrino_dataset(root, system_message, prompt, split="train", test_size=0.05, seed=77):
    
    assert split == 'train' or split == "test", "split should be either train or test"
    class_paths = glob.glob(f"{root}/*/*")
    train_paths, test_paths = train_test_split(
        class_paths, test_size=test_size, random_state=seed, shuffle=True
    )

    def make_dataset(paths, desc):
        dataset = []
        for path in tqdm(paths, desc=f"Creating {desc} Dataset"):
            with open(f"{path}/{path.split('/')[-1]}_pid.txt") as f:
                gt = f.read().strip()
            if gt in CLASSES:
                zx = ImageOps.invert(Image.open(f"{path}/{path.split('/')[-1]}_xz.png"))
                zy = ImageOps.invert(Image.open(f"{path}/{path.split('/')[-1]}_yz.png"))
                dataset.append(format_data(zx, zy, CLASSES[gt], system_message, prompt, split=split))
        return dataset

    if split == "train":
        return make_dataset(train_paths, "Train")
    elif split == "test":
        return make_dataset(test_paths, "Validation")
