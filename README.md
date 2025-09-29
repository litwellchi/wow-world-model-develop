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
  A self-optimizing, closed-loop system that integrates a VLM critic and DiT generator for iterative refinement.

- **🤖 Imagination-to-Action Loop**  
  FM-IDM translates predicted videos into 7-DoF robot actions executed in the physical world.

- **📊 WoWBench Benchmark**  
  A comprehensive evaluation suite across 4 dimensions and 20+ physical tasks.

- **🧪 Advanced Reasoning**  
  Supports counterfactual planning, logical parsing, tool use, and compositional reasoning.

- **🧩 Applications**  
  - Novel-view synthesis  
  - Trajectory-guided generation  
  - Action-to-video generation  
  - Visual style transfer  
  - VLM task planning via simulation

---

## 📖 Paper

> **Title**: WOW: Towards a World-Omniscient World Model Through Embodied Interaction  
> **Authors**: Xiaowei Chi, Peidong Jia, Chun-Kai Fan, Xiaozhu Ju, et al.  
> **arXiv**: [2509.22642](https://arxiv.org/pdf/2509.22642)

---

## 🌟 Demo & Abstract

<div align="center">
  <img src="wow-world-model.github.io/figs/teaser.png" alt="WoW Teaser" style="max-width: 600px; border-radius: 12px;">
</div>

**WoW world model** generates high-quality, physically consistent robot action videos in Out-of-Distribution (OOD) scenarios, enabling closed-loop corrections and real-world robotic execution. The illustration shows the model's strong generalization across diverse tasks and environments.

**Authors:**

Xiaowei Chi<sup>1,2,3</sup>†, Peidong Jia<sup>1,2</sup>†, Chun-Kai Fan<sup>1,2</sup>†, Xiaozhu Ju<sup>1</sup>†, Weishi Mi<sup>1</sup>†, Kevin Zhang<sup>2</sup>, Zhiyuan Qin<sup>1</sup>, Wanxin Tian<sup>1</sup>, Kuangzhi Ge<sup>2</sup>, Hao Li<sup>1</sup>, Zezhong Qian<sup>1,2</sup>, Anthony Chen<sup>2</sup>, Qiang Zhou<sup>1,2</sup>, Yueru Jia<sup>2</sup>, Jiaming Liu<sup>2</sup>, Yong Dai<sup>1</sup>, Qingpo Wuwu<sup>2</sup>, Chengyu Bai<sup>2</sup>, Yu-Kai Wang<sup>2</sup>, Ying Li<sup>2</sup>, Lizhang Chen<sup>1,2</sup>, Yong Bao<sup>1</sup>, Zhiyuan Jiang<sup>1</sup>, Jiacheng Zhu<sup>1</sup>, Kai Tang<sup>2</sup>, Ruichuan An<sup>2</sup>, Yulin Luo<sup>2</sup>, Qiuxuan Feng<sup>1,2</sup>, Siyuan Zhou<sup>3</sup>, Chi-min Chan<sup>3</sup>, Chengkai Hou<sup>1,2</sup>, Wei Xue<sup>3</sup>, Sirui Han<sup>3</sup>, Yike Guo<sup>3</sup>, [Shanghang Zhang](https://www.shanghangzhang.com)<sup>2</sup>✉, [Jian Tang](https://jian-tang.com)<sup>1</sup>✉

<sup>1</sup> Beijing Innovation Center of Humanoid Robotics  
<sup>2</sup> State Key Laboratory of Multimedia Information Processing, School of Computer Science, Peking University  
<sup>3</sup> Hong Kong University of Science and Technology

**Abstract:**

Humans develop an understanding of intuitive physics through active interaction with the world. In stark contrast, current video models such as Sora rely solely on passive observation and therefore struggle with grasping physical causality. This motivates our central hypothesis: *authentic physical intuition in a world model must be grounded in extensive, causally rich interactions with the real world*. To test this, we introduce **WoW**, a 14B-parameter generative world model trained on 2 million real-world robot interaction trajectories. We find that the model’s understanding of physics emerges as a probabilistic distribution of plausible outcomes, which can lead to stochastic instabilities and physical hallucinations. To mitigate these, we propose *SOPHIA*, a novel vision-language agent that evaluates the output of the DiT model and iteratively refines the language instructions to steer generation toward physical realism. Complementing this, a co-trained *Inverse Dynamics Model* translates refined plans into executable robotic actions, effectively closing the imagination-to-action loop. We further establish **WoWBench**, a new benchmark focused on physical consistency and causal reasoning in video, where WoW achieves state-of-the-art performance in both human and autonomous evaluations, excelling in physical causality, collision dynamics, and object permanence. Our work provides systematic evidence that large-scale, real-world interaction is essential for developing physical intuition in AI.

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

## 📈 Benchmark Results (WoWBench)

| Model         | Instruction ↑ | Physical Law ↑ | Planning ↑ | Overall ↑ |
|---------------|----------------|----------------|------------|------------|
| WoW (Ours)    | **96.53%**     | **80.16%**     | **78.53%** | **46.11**  |
| Cosmos-Predict | 45.29%         | 16.85%         | 7.47%      | 16.30      |
| CogVideoX     | 5.91%          | 44.13%         | 2.32%      | 10.34      |

---

## 🧠 Architecture Overview

```text
Perception → Imagination → Reflection → Action

1. Vision-Language Model (Critic)
2. Diffusion Transformer (Generator)
3. Refiner Agent (Prompt Optimization)
4. Inverse Dynamics Model (Execution)

# 🚀 Open Source Roadmap

> **WoW (World-Omniscient World Model)** is a physically grounded generative world model designed to advance general-purpose robot intelligence. To promote transparency, collaboration, and progress in the community, we are releasing our components in phases:

<div style="background: #f8f8f8; border-radius: 12px; padding: 18px; margin-bottom: 18px; border-left: 6px solid #EC707D;">

### ✅ Phase 1 – *Published*
- [x] Paper released on [arXiv:2509.22642](https://arxiv.org/abs/2509.22642)
- [x] Project website launched: [wow-world-model.github.io](https://wow-world-model.github.io)
- [x] WoWBench benchmark design & evaluation metrics released

### 🚧 Phase 2 – *Planned for Q4 2025*
- [ ] **WoWBench Dataset Release**
- [ ] **Model Weights (2B, 7B WoW-DiT)**
- [ ] **Inference Scripts & Colab Demo**

### 🚀 Phase 3 – *Planned for Q1 2026*
- [ ] **SOPHIA Framework Code**
- [ ] **Flow-Mask Inverse Dynamics Model (FM-IDM)**
- [ ] **Training Pipeline**

### 🌐 Phase 4 – *2026 Onward*
- [ ] Continuous release of real/simulated trajectory data
- [ ] Expansion to multimodal inputs (e.g., audio, tactile)
- [ ] Universal fine-tuning API for downstream tasks
- [ ] Community challenges and leaderboard integration

</div>

---

## 🤝 How You Can Get Involved

- 📥 Submit issues or suggest features  
- 🔧 Improve code or documentation with pull requests  
- 📊 Run experiments and submit results to WoWBench  
- 🤖 Contribute real-world robot data

---

## 📬 Contact

- Project website: [wow-world-model.github.io](https://wow-world-model.github.io)  
- Email: wow.world.model@pku.edu.cn

---

> “We don’t just generate videos — we generate grounded imagination, reasoning, and embodied action.”