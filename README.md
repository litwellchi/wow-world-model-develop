# 🌍 WoW: World-Omniscient World Model

> Towards an Embodied, Physically-Consistent Generative World Model

[![arXiv](https://img.shields.io/badge/arXiv-2509.22642v1-b31b1b.svg)](https://arxiv.org/abs/2509.22642)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Website](https://img.shields.io/badge/demo-wow--world--model.github.io-green.svg)](https://wow-world-model.github.io)

---

WoW (World-Omniscient World Model) is a 14B-parameter generative world model trained on **2 million robotic interaction trajectories**, designed to **imagine**, **reason**, and **act** in the physical world. Unlike passive video models, WoW learns from **real-world, causally rich data**, enabling robust physical reasoning and real-world planning.

---

## 🔬 Key Features

- **📽️ Diffusion-Based Video Generation**  
  High-fidelity, physically grounded future prediction from images + instructions.
- **🧠 SOPHIA Framework**  
  Self-optimizing, closed-loop system integrating a VLM critic and DiT generator for iterative refinement.
- **🤖 Imagination-to-Action Loop**  
  FM-IDM translates predicted videos into 7-DoF robot actions executed in the physical world.
- **📊 WoWBench Benchmark**  
  Comprehensive evaluation suite across 4 dimensions and 20+ physical tasks.
- **🧪 Advanced Reasoning**  
  Supports counterfactual planning, logical parsing, tool use, and compositional reasoning.
- **🧩 Applications**  
  Novel-view synthesis, trajectory-guided generation, action-to-video generation, visual style transfer, VLM task planning via simulation

---

## 📖 Paper & Demo

<div align="center">
  <img src="wow-world-model.github.io/figs/teaser.png" alt="WoW Teaser" style="max-width: 600px; border-radius: 12px;">
</div>

**WoW world model** generates high-quality, physically consistent robot action videos in Out-of-Distribution (OOD) scenarios, enabling closed-loop corrections and real-world robotic execution. The illustration shows the model's strong generalization across diverse tasks and environments.

**Authors:**

Xiaowei Chi<sup>1,2,3</sup>†, Peidong Jia<sup>1,2</sup>†, Chun-Kai Fan<sup>1,2</sup>†, Xiaozhu Ju<sup>1</sup>†, Weishi Mi<sup>1</sup>†, Kevin Zhang<sup>2</sup>, Zhiyuan Qin<sup>1</sup>, Wanxin Tian<sup>1</sup>, Kuangzhi Ge<sup>2</sup>, Hao Li<sup>1</sup>, Zezhong Qian<sup>1,2</sup>, Anthony Chen<sup>2</sup>, Qiang Zhou<sup>1,2</sup>, Yueru Jia<sup>2</sup>, Jiaming Liu<sup>2</sup>, Yong Dai<sup>1</sup>, Qingpo Wuwu<sup>2</sup>, Chengyu Bai<sup>2</sup>, Yu-Kai Wang<sup>2</sup>, Ying Li<sup>2</sup>, Lizhang Chen<sup>1,2</sup>, Yong Bao<sup>1</sup>, Zhiyuan Jiang<sup>1</sup>, Jiacheng Zhu<sup>1</sup>, Kai Tang<sup>2</sup>, Ruichuan An<sup>2</sup>, Yulin Luo<sup>2</sup>, Qiuxuan Feng<sup>1,2</sup>, Siyuan Zhou<sup>3</sup>, Chi-min Chan<sup>3</sup>, Chengkai Hou<sup>1,2</sup>, Wei Xue<sup>3</sup>, Sirui Han<sup>3</sup>, Yike Guo<sup>3</sup>, [Shanghang Zhang]()<sup>2</sup>✉, [Jian Tang]()<sup>1</sup>✉

<sup>1</sup> Beijing Innovation Center of Humanoid Robotics  
<sup>2</sup> State Key Laboratory of Multimedia Information Processing, School of Computer Science, Peking University  
<sup>3</sup> Hong Kong University of Science and Technology



## 🚀 Open Source Roadmap

> **WoW (World-Omniscient World Model)** is a physically grounded generative world model designed to advance general-purpose robot intelligence. To promote transparency, collaboration, and progress in the community, we are releasing our components in phases:

<div style="background: #f8f8f8; border-radius: 12px; padding: 18px; margin-bottom: 18px; border-left: 6px solid #EC707D;">

### ✅ Phase 1 – *Published*
- [x] Paper released on [arXiv:2509.22642](https://arxiv.org/abs/2509.22642)
- [x] Project website launched: [wow-world-model.github.io](https://wow-world-model.github.io)

### 🚧 Phase 2 – *Planned for Oct. 2025*
- [ ] **Model Weights (2B, 7B, 14B WoW-DiT)**
- [ ] **Baseline Model Weights (SVD, CogVideoX, Cosmos1&2)**
- [ ] **Inference Scripts & Colab Demo**
- [ ] **Flow-Mask Inverse Dynamics Model (FM-IDM)**

### 🚀 Phase 3 – *Planned for Dec. 2025*
- [ ] **Training Pipeline**
- [ ] **SOPHIA Framework Code**
- [ ] **WoWBench benchmark design & evaluation metrics released**

### 🌐 Phase 4 – *2026 Onward*
- [ ] Continuous release of real/simulated trajectory data
- [ ] Expansion to multimodal inputs (e.g., audio, tactile)
- [ ] Universal fine-tuning API for downstream tasks
- [ ] Community challenges and leaderboard integration

</div>

---

## 🧰 Installation

Coming soon. We plan to release:

- [ ] Pretrained WoW models (2B, 7B, 14B)
- [ ] WoWBench dataset & evaluation scripts
- [ ] SOPHIA framework codebase
- [ ] FM-IDM module for robot control

---

## 🚀 Quick Demo

Visit our interactive demo at:  
👉 [**wow-world-model.github.io**](https://wow-world-model.github.io)

---

## 🤝 How You Can Get Involved

- 📥 Submit issues or suggest features  
- 🔧 Improve code or documentation with pull requests  
- 📊 Run experiments and submit results to WoWBench  
- 🤖 Contribute real-world robot data

---

## 📬 Contact

- Project website: [wow-world-model.github.io](https://wow-world-model.github.io)  

---

> “We don’t just generate videos — we generate grounded imagination, reasoning, and embodied action.”

---

## 📖 Citation

If you use WoW in your research, please cite:

```bibtex
@misc{chi2025wowworldomniscientworld,
      title={WoW: Towards a World omniscient World model Through Embodied Interaction}, 
      author={Xiaowei Chi and Peidong Jia and Chun-Kai Fan and Xiaozhu Ju and Weishi Mi and Kevin Zhang and Zhiyuan Qin and Wanxin Tian and Kuangzhi Ge and Hao Li and Zezhong Qian and Anthony Chen and Qiang Zhou and Yueru Jia and Jiaming Liu and Yong Dai and Qingpo Wuwu and Chengyu Bai and Yu-Kai Wang and Ying Li and Lizhang Chen and Yong Bao and Zhiyuan Jiang and Jiacheng Zhu and Kai Tang and Ruichuan An and Yulin Luo and Qiuxuan Feng and Siyuan Zhou and Chi-min Chan and Chengkai Hou and Wei Xue and Sirui Han and Yike Guo and Shanghang Zhang and Jian Tang},
      year={2025},
      eprint={2509.22642},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2509.22642}, 
}
```