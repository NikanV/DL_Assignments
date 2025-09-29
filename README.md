# Deep Learning Assignments

This repository contains my solutions and implementations for a series of deep learning assignments.
Each assignment explores key concepts in deep learning — from fundamentals of neural networks and optimization,
to cutting-edge topics such as large language models (LLMs), generative models, and diffusion-based architectures.

The repository is organized into subdirectories `A1`, `A2`, `A3`, etc., each representing an assignment.
Within each subdirectory are multiple Jupyter notebooks tackling specific problems.

---

## 📂 Repository Structure

```
.
├── A1/                 # Assignment 1 - Fundamentals
│   ├── 1_Basics.ipynb
│   ├── 2-NN_Scratch.ipynb
│   ├── 3-Optimization.ipynb
│   └── 4_Lazy_Gradient.ipynb
│
├── A2/                 # Assignment 2 - Computer Vision Applications
│   ├── Q1_Classification.ipynb
│   ├── Q2_Segmentation.ipynb
│   └── Q3_Object_Detection.ipynb
|
├── A3/                 # Assignment 3 - Sequence Models & LLMs
│   ├── Q1_RNN.ipynb
│   ├── Q2_GPT2.ipynb
│   ├── Q3_PEFT.ipynb
│   └── Q4_Reasoning.ipynb
|
├── A4/                 # Assignment 4 - Generative Modeling
│   ├── DDPM.ipynb
│   └── VAE.ipynb
|
├── A5/                 # Assignment 5 - Vision Transformers & Diffusion Models
│   ├── DINO.ipynb
│   ├── image_generation_with_clip.ipynb
│   └── StableDiffusion.ipynb
│
└── README.md
```

---

## 📝 Assignment Details

### **A1 – Fundamentals of Deep Learning**

| Notebook            | Description                                                                                                                                                       |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1_Basics**        | Introduction to PyTorch and its ecosystem. Covers tensors, autograd, modules, and basic workflows for building and training deep learning models.                 |
| **2_NN_Scratch**    | Implementation of a Multi-Layer Perceptron (MLP) from scratch, including forward and backward propagation, activation functions, and manual gradient computation. |
| **3_Optimization**  | Experiments with various optimization algorithms (SGD, Momentum, Adam, etc.) on custom 1D and 2D functions to understand convergence properties and behaviors.    |
| **4_Lazy_Gradient** | Exploration of gradient checkpointing techniques to reduce memory usage during training, enabling efficient backpropagation in memory-constrained settings.       |

---

### **A2 – Computer Vision with Deep Learning**

| Notebook                | Description                                                                                                                                                                                                                                                   |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Q1_Classification**   | Image classification on CIFAR-10 dataset using CNNs, followed by applying transfer learning techniques on CIFAR-100 for improved performance.                                                                                                                 |
| **Q2_Segmentation**     | End-to-end implementation of a **U-Net** for semantic segmentation. Trains the model on a self-driving car dataset to label every pixel in the image.                                                                                                         |
| **Q3_Object_detection** | Development of a two-stage license plate recognition pipeline: <br> **Stage 1:** License Plate Detection (LPD) using a YOLO-based model. <br> **Stage 2:** License Plate Recognition (LPR) by classifying individual characters in the detected plate region. |

---

### **A3 – Sequence Models & Large Language Models (LLMs)**

| Notebook         | Description                                                                                                                                                                                                                                               |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Q1_RNN**       | Time-series modeling for **oil price prediction** using RNN, LSTM, and ARIMA models. Compares performance of classical statistical models vs neural sequence models.                                                                                      |
| **Q2_GPT2**      | Scaled-down **GPT-2 implementation from scratch** using PyTorch, trained on Snappfood comments with sentiment labels. Generates synthetic Persian text with controllable sentiment.                                                                       |
| **Q3_PEFT**      | Fine-tuning GPT-2 on the [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext#wikitext-2-v1) dataset using **Parameter-Efficient Fine-Tuning (PEFT)** techniques such as LoRA, Adapters, and Prefix-Tuning to adapt large models efficiently. |
| **Q4_Reasoning** | Exploration of **inference-time reasoning** methods on the Math Benchmark, including Chain-of-Thought prompting, Best-of-n sampling, Beam Search, and Self-Refinement, to evaluate LLM reasoning capabilities.                                            |

---

### **A4 – Generative Modeling**

| Notebook | Description                                                                                                                                                  |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **DDPM** | Implementation of **Denoising Diffusion Probabilistic Models** (DDPM) and **classifier-free guidance** for controllable generative modeling.                 |
| **VAE**  | Implementation of a **Variational Autoencoder (VAE)** and exploration of latent space manipulations and downstream tasks enabled by learned representations. |

---

### **A5 – Vision Transformers & Advanced Generative Models**

| Notebook                       | Description                                                                                                                                                                                                                                    |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **DINO**                       | Exploration of **DINO** (self-supervised vision transformer) and **Grounding DINO** for zero-shot object detection. Includes attention map visualization, bounding box drawing, and text-prompt-driven detection.                              |
| **image_generation_with_clip** | Optimization-based image generation using CLIP representations. Demonstrates reversing discriminative models to synthesize images that maximize similarity with a given text prompt.                                                           |
| **StableDiffusion**            | Step-by-step implementation and experimentation with **Stable Diffusion** for text-to-image generation. Includes attention map visualization and noise optimization techniques (e.g., CLIP-score loss) for better alignment and image quality. |

---

## 🚀 How to Run

Clone this repository and open the desired assignment directory in Jupyter Lab/Notebook:

```bash
git clone https://github.com/yourusername/deep-learning-assignments.git
cd deep-learning-assignments/A3
jupyter notebook
```

Run notebooks sequentially for reproducible results. GPU acceleration is recommended for training LLMs, diffusion models, and CNNs.

---

## 📌 Notes

* Some notebooks (e.g., YOLO, GPT-2, Stable Diffusion) may require downloading pre-trained weights or datasets. Follow the instructions within each notebook.
* Training times may vary — for large models (GPT-2, diffusion), running on a GPU or TPU is highly recommended.
* Each notebook contains explanations, visualizations, and experiment results for easy reproducibility.

---
