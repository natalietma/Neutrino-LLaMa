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


## Prepare directories

After cloning the repository, move into the LLM directory:
`cd Neutrino-LLaMa/LLaMa3.2Vision_Finetune`

Create the output directory for prediction logs: `mkdir -p predictions`

## Setup
We recommend using [conda](https://docs.conda.io/) to create the Python environment for inference.

`cd Neutrino-LLaMa/LLaMa3.2Vision_Finetune`

Create and activate the conda environment:

```conda create -n llm python=3.10

conda activate llm
```

## Environment

- Python `3.10.20`
- PyTorch `2.4.1+cu121`
- torchvision `0.19.1+cu121`
- transformers `4.46.2`
- peft `0.13.0`
- accelerate `1.13.0`
- Pillow `12.2.0`

Install the required Python packages:
`pip install -r requirements.txt`

To match with specific dependency version:
`pip install dependency==version`. For example: `pip install transformers==4.46.2` 

**Notes**
- `qwen_vl_utils` was available in the tested environment.
- `+cu121` indicates the CUDA 12.1 build of PyTorch / torchvision.

*This setup documents the environment used for inference. The training environment may differ.*

## Dataset
The dataset is a custom simulation of a modular liquid Argon time projection chamber (LArTPC) with square 5 mm pixel-based readout. 

- Input: pairs of **2D pixel maps** in the `xz` and `yz` planes  
- Classes: `NuE CC`, `NuMu CC`, `Neutral Current`  

Dataset location used in this tutorial is at:
`/mnt/ironwolf_20t/users/dikshans/preprocessed_NewDataset/asImages`

Note: The raw HDF5 event files were preprocessed into cropped .png pixel maps, and the corresponding pid.txt label files were generated during preprocessing.

If the dataset is not already available on your server, you can copy it with rsync: `rsync -avP username@tau-neutrino.ps.uci.edu:/mnt/ironwolf_20t/users/dikshans/preprocessed_NewDataset/asImages/ /path/to/local/asImages/`

Replace /path/to/local/asImages/ with your target directory on the destination machine.

## Run Inference with Finetuned LLaMa 3.2 Vision

**1. Activate the environment**

`conda activate llm`

**2. Check GPU availability**


Before launching inference, verify that the target GPUs are visible and have sufficient free memory: `nvidia-smi`

When reading the nvidia-smi output:

If a GPU already has a large amount of memory in use, it is likely being used by another job.
If another job is already occupying the GPU, inference may fail to start or may run out of memory before completion.

This project typically requires **two GPUs** for inference. In the tested setup, running on a single 24 GB GPU was not sufficient, so inference was launched with two GPUs.

**3. Update file path**

Before running inference, verify the following paths in `inference.py`:

```python
MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
ADAPTER_PATH = "./checkpoints/Llama3.2-11B-Vision-Instruct-Neutrino"
DATASET_ROOT = "../data"
```

MODEL_ID should point to the base vision-language model.
ADAPTER_PATH should point to the local fine-tuned adapter checkpoint.
DATASET_ROOT should point to the dataset root directory containing the preprocessed event folders.

If your dataset is stored in a different location, either update DATASET_ROOT directly or create a symbolic link so that ../data points to the correct dataset path.

**4. Run in a persistent terminal session**

For long-running inference jobs on a remote server, it is recommended to use tmux so that your job continue running after disconnecting from SSH (e.g. when your screen is locked or you are disconnected from the server).

Create a new session:
`tmux new -s session_name` *replace session_name with how you'd like to call this window*

Detach from the session: Ctrl-b, release, then d

Re-attach later: `tmux attach -t session_name`

**5. Launch inference**
Create the output directory:

`mkdir -p predictions`

Then run inference, for example on GPUs 0 and 1:

`CUDA_VISIBLE_DEVICES=0,1 python inference.py | tee inference_output.out`

