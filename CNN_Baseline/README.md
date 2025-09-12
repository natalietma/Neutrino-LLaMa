# Neutrino Interaction Classification with CNN Baseline

This repository contains the **Baseline CNN** code on simulated DUNE neutrino pixel maps.

## Dataset
- Input: pairs of **2D pixel maps** in `xz` and `yz` planes
- Classes: `NuE CC`, `NuMu CC`, `Neutral Current`

<b>Note:</b> Raw event h5py files were converted to .png after event crops and corresponding pid.txt files were generated. Data available on reasonable request.

## Setup
We recommend using [conda](https://docs.conda.io/) for environment setup.  
```bash
git clone https://github.com/Neutrino-LLaMa.git
cd CNN_Baseline
conda create -n neutrino python=3.10
conda activate neutrino
pip install -r requirements.txt
```

## Train Baseline CNN Model

```python
python model_train.py separate_train
```
