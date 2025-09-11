# Neutrino Interaction Classification with LLaMA 3.2 Vision

This repository contains the code to fine-tune **LLaMA 3.2-11B Vision-Instruct** on simulated DUNE neutrino pixel maps.

## Dataset
- Input: pairs of **2D pixel maps** in `xz` and `yz` planes
- Classes: `NuE CC`, `NuMu CC`, `Neutral Current`

<b>Note:</b> Raw event h5py files were converted to .png after event crops and corresponding pid.txt files were generated. Data available on reasonable request.

## Setup
We recommend using [conda](https://docs.conda.io/) for environment setup.  
```bash
git clone https://github.com/Neutrino-LLaMa.git
cd LLaMa3.2Vision_Finetune
conda create -n neutrino python=3.10
conda activate neutrino
pip install -r requirements.txt
```

## Finetune LLaMa 3.2 Vision

```python
python train.py
```

## Run Inference with Finetuned LLaMa 3.2 Vision
```python
python inference.py
```

