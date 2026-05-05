# Neutrino Event Classification with LLaMA 3.2 Vision

This repository contains the code to fine-tune **LLaMA 3.2-11B Vision-Instruct** . This tutorial based on Tau server @tau-neutrino.ps.uci.edu

## Connect to Group Server
To connect the group server, you must be either on campus with UCI network, or use UCI VPN to connect to the server. More info about UCI VPN: https://www.oit.uci.edu/services/security/vpn/

Connect to the server with your password: ```bash
ssh username@tau-neutrino.ps.uci.edu

If you don’t have an account on Tau server, please contact Jiaxi Liu.

*Note: Be sure your password is safe and correct. Your account will be locked when you input wrong password twice. You are not entitled to run at root or sudo.*

## Dataset
The dataset is a custom simulation of a modular LArTPC with square 5 mm pixel-based readout. The detector is 2 m ×2 m ×7 m in x,y,z with anodes at x= {−0.9 m,−0.3 m,0.3 m,0.9 m}and cathodes at x= {−0.6 m,0.0 m,0.6 m} resulting in 0.3 m drift lengths along x. 

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

