# 🎵 DiffSynth-Studio - Setup & Inference Guide

This guide provides a step-by-step walkthrough for setting up the environment and running inference using **DiffSynth-Studio** with the pre-trained **Wan2.1 I2V** model.

---

## 📦 Environment Setup

### 1. Clone the Repository
First, ensure you're inside the correct working directory:
the `dit_models/wow_wan/` directory before cloning the repo:

```bash
cd dit_models/wow_wan
```
Then clone the DiffSynth-Studio repository, and checkout a specific commit for compatibility (recommended):
```bash
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
git checkout 3edf3583b1f08944cee837b94d9f84d669c2729c
```
ℹ️ For the official setup instructions, refer to the official README in https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/wanvideo/README.md



## 📁 Model Preparation
Make sure your directory structure looks like this:
```bash
ditmodels/
├── wow_wan/models--Wan-AI--Wan2.1-I2V-14B-480P/
└── checkpoints/wan-epoch=95-train_loss=0.0265.ckpt
```
You can find the checkpoint in https://huggingface.co/WoW-world-model/WoW-1-Wan-14B-2M. We will keep updating more checkpoints.

## ▶️ How to Run
Run the following command from within the directory:
```bash
python demo/infer_demo.py --gpus 0 --port 7862 
```