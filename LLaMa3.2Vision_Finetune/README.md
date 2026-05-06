# Neutrino Event Classification with LLaMA 3.2 Vision

This repository contains the code to fine-tune **LLaMA 3.2-11B Vision-Instruct** . This tutorial based on Tau server @tau-neutrino.ps.uci.edu

## Connect to Group Server
To connect the group server, you must be either on campus with UCI network, or use UCI VPN to connect to the server. More info about UCI VPN: https://www.oit.uci.edu/services/security/vpn/

Run `ssh username@tau-neutrino.ps.uci.edu` to connect to the server.

*Replace username with your own username set for this server.
If you don’t have an account on Tau server, please contact Jiaxi Liu.*

*Note: Be sure your password is safe and correct. Your account will be locked when you input wrong password twice. You are not entitled to run at root or sudo.*

## Obtain the codebase
Clone the repository onto the server:
git clone `git clone https://github.com/natalietma/Neutrino-LLaMa.git`

You will need your github username and a github token for the "password"

Your github token can be found under Settings → Developer settings → Personal access tokens → Tokens (classic).

## Dataset
The dataset is a custom simulation of a modular liquid Argon time projection chamber (LArTPC) with square 5 mm pixel-based readout. 

- Input: pairs of **2D pixel maps** in the `xz` and `yz` planes  
- Classes: `NuE CC`, `NuMu CC`, `Neutral Current`  

Dataset location used in this tutorial is at:
`/mnt/ironwolf_20t/users/dikshans/preprocessed_NewDataset/asImages`

Each event is stored in its own subdirectory and includes:

*_xz.png
*_yz.png
*_pid.txt

Note: The raw HDF5 event files were preprocessed into cropped .png pixel maps, and the corresponding pid.txt label files were generated during preprocessing.

If the dataset is not already available on your server, you can copy it with rsync: `rsync -avP username@tau-neutrino.ps.uci.edu:/mnt/ironwolf_20t/users/dikshans/preprocessed_NewDataset/asImages/ /path/to/local/asImages/`

Replace /path/to/local/asImages/ with your target directory on the destination machine.

## Setup
We recommend using [conda](https://docs.conda.io/) to create the Python environment for inference.

‘cd Neutrino-LLaMa/LLaMa3.2Vision_Finetune’

Create and activate the conda environment:

‘conda create -n llm python=3.10’
‘conda activate llm’

Install the required Python packages:
`pip install -r requirements.txt`

**Environment**

Python 3.10.20
torch 2.4.1+cu121
torchvision 0.19.1+cu121
transformers 4.46.2
peft 0.13.0
accelerate 1.13.0
Pillow 12.2.0

*This setup documents the environment used for inference. The original training environment may differ.*

## Finetune LLaMa 3.2 Vision

```python
python train.py
```

## Run Inference with Finetuned LLaMa 3.2 Vision
```python
python inference.py
```

