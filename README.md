# 🌾 AgriVision-Gemma3n: The Gemma 3n Impact Challenge

**An offline-first, AI-powered field agronomist to help farmers diagnose crop diseases using their mobile phones.**

This project is a submission for the **Google - The Gemma 3n Impact Challenge**. It leverages Gemma 3n, fine-tuned with Unsloth, to provide a tangible solution for farmers in low-connectivity areas, promoting environmental sustainability and economic empowerment.

## The Problem

Farmers worldwide lose a significant portion of their crops to pests and diseases. In remote areas without reliable internet access, getting timely and accurate diagnoses is nearly impossible, leading to crop loss and financial hardship.

## Solution: AgriVision-Gemma3n

AgriVision-Gemma3n is a multimodal AI assistant that runs entirely on-device. A farmer can simply take a photo of a diseased plant, and the model will:
1.  Identify the crop.
2.  Diagnose the disease.
3.  Provide clear, actionable treatment advice.

This is all done privately and without needing an internet connection, directly addressing the core capabilities of Gemma 3n.

## How to Run the Demo

The primary demo can be found in the `3_interactive_demo.ipynb` notebook. You can run it directly on Kaggle or a capable local machine.

## Fine-Tuning Process

Our model was created using a novel two-stage fine-tuning process documented in these notebooks:
1.  **`1_cddm_finetuning.ipynb`**: Built the foundational knowledge base by fine-tuning on the CDDM dataset.
2.  **`2_agrillava_finetuning.ipynb`**: Enhanced the model's conversational abilities by continuing the fine-tuning on the Agri-LLaVA dataset.

This project proudly uses **Unsloth** for fast, memory-efficient fine-tuning, making it eligible for the Unsloth Special Technology Prize.
