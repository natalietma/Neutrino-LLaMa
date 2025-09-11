# Neutrino Event Classification with LLaMA 3.2-11B Vision-Instruct

This repository contains the training and inference code for our paper:  
**"Adapting Vision-Language Models for Neutrino Event Classification in High-Energy Physics"** (submitted to *Nature Communications*).  

We fine-tune the multimodal large language model **LLaMA 3.2-11B Vision-Instruct** on simulated **DUNE Near Detector(LArTPC) pixel maps** to classify neutrino interactions into three categories:

- **NuE CC** — Electron Neutrino Charged Current interaction  
- **NuMu CC** — Muon Neutrino Charged Current interaction  
- **Neutral Current (NC)**  

The model takes as input **2D projections** (`xz`, `yz`) of 3D detector events and learns to recognize distinctive features such as fuzzy electron showers (NuE CC) or long muon tracks (NuMu CC).  


Code for running LLama 3.2 Vision and the CNN Baseline are provided in their respective directories, please follow instructions in their respective README.md files.



<!-- 
## ✏️ Citation [DRAFT]
If you use this code, please cite our paper:

```bibtex
@article{sagar202xneutrino,
  title   = {Neutrino Event Classification with Multimodal Large Language Models},
  author  = {Sagar, Dikshant and ...},
  journal = {TBD},
  year    = {202x}
}
``` -->

## 📜 License
This code is released under the MIT License (See the [LICENSE](LICENSE))

