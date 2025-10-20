# 🎵 DiffSynth-Studio - Setup & Inference Guide

This guide provides a step-by-step walkthrough for setting up the environment and running inference using **DiffSynth-Studio** with the pre-trained **Wan2.1 I2V** model.

---

## 📦 Environment Setup

### 1. Clone the Repository
First, ensure you're inside the correct working directory:
the `dit_models` directory before cloning the repo:

```bash
cd dit_models
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
dit_models/
├── wow_wan/DiffSynth-Studio/
└── checkpoints/<checkpoint_filename>.ckpt
```

To download the checkpoint from Hugging Face and place it in the correct directory, run:
```bash
mkdir -p dit_models/checkpoints
cd dit_models/checkpoints
git lfs install
git clone https://huggingface.co/WoW-world-model/WoW-1-DiT-2B-600k
# Move the checkpoint file to the expected location (rename if necessary)
mv WoW-1-DiT-2B-600k/<checkpoint_filename>.ckpt wow-wan-600k-epoch=55-mp_rank_00_model_states.pt
```
Replace `<checkpoint_filename>.ckpt` with the actual filename from the Hugging Face repo.

You can find more checkpoints at [WoW-world-model on Hugging Face](https://huggingface.co/WoW-world-model). We will keep updating more checkpoints.

## ▶️ How to Run
Run the following command from within the direDiffSynth-Studioctory:
```bash
python demo/infer_demo.py --gpus 0 --port 7862 
```
