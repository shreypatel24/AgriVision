# 🌱 AgriVision-Gemma

**Vision-Language Fine-Tuning for Crop Disease Diagnosis**\
This project focuses on fine-tuning the `Gemma-3n-e2b-it` vision-language model using QLoRA (via Unsloth) to create a specialized multimodal assistant for identifying crop diseases from leaf images and conversation context.

---

## 🔖 Table of Contents

1. [🧰 Setup & Installation](#setup--installation)
2. [📦 Importing Libraries](#importing-libraries)
3. [⚙️ Configuration](#configuration)
4. [🗂️ Data Preparation](#data-preparation)
5. [🤖 Model + QLoRA Configuration](#model--qlora-configuration)
6. [🚀 Training](#training)
7. [💾 Saving Model](#saving-model)
8. [🧪 Inference](#inference)
9. [🛠️ Tech Stack](#tech-stack)
10. [📘️ Usage](#usage)
11. [🙏 Contributing](#contributing)
12. [📄 License](#license)

---

## 🧰 Setup & Installation

```bash
pip install --upgrade unsloth
pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.53.0 datasets timm huggingface_hub
```

---

## 📦 Importing Libraries

```python
from unsloth import FastModel
from datasets import Dataset
from transformers import TrainingArguments, AutoProcessor
from huggingface_hub import login, HfApi
from PIL import Image
import torch, os, json, re
```

---

## ⚙️ Configuration

```python
model_id       = "unsloth/gemma-3n-e2b-it-unsloth-bnb-4bit"
output_dir     = "outputs/gemma-qlora"
hub_model_id   = "your-username/gemma3n-cddm-finetune"
hf_token       = "YOUR_HF_TOKEN"
hub_private    = False
login(hf_token)
```

---

## 🗂️ Data Preparation

Custom preprocessing pipeline transforms conversational JSON data from CDDM dataset into multi-turn, vision-language compatible format, with image paths correctly resolved for training.

---

## 🤖 Model + QLoRA Configuration

Using Unsloth's FastModel to efficiently load and prepare the vision-language model with 4-bit quantization:

```python
model, processor = FastModel.from_pretrained(model_id, load_in_4bit=True, dtype=torch.float16, max_seq_length=2048)
model = FastModel.get_peft_model(model, target_modules="all-linear", r=16, lora_alpha=32)
```

---

## 🚀 Training

The model is being fine-tuned using HuggingFace Trainer with Unsloth’s optimizations:

```python
args = TrainingArguments(
  output_dir=output_dir,
  num_train_epochs=1,
  per_device_train_batch_size=1,
  gradient_accumulation_steps=8,
  max_steps=8750,
  optim="adamw_8bit",
  logging_steps=25,
  save_steps=250,
  push_to_hub=True,
  hub_model_id=hub_model_id,
  hub_token=hf_token
)
trainer = Trainer(model=model, args=args, train_dataset=train_dataset, data_collator=collate_fn)
trainer.train(resume_from_checkpoint=True)
```

---

## 💾 Saving Model

Intermediate and final checkpoints will be saved locally and optionally pushed to the Hugging Face Hub.

---

## 🧪 Inference

Sample inference code is included using the trained model and preprocessed vision-text prompts.

---

## 🛠️ Tech Stack

**Frameworks:** PyTorch, HuggingFace Transformers, Unsloth\
**Libraries:** datasets, PIL, timm, numpy, huggingface\_hub, re

---

## 📘️ Usage

- Run the notebook step-by-step to reproduce the fine-tuning pipeline.
- Pre-trained model and inference notebook will be updated once training completes.

---

## 🙏 Contributing

Feel free to fork and contribute! Fine-tuning is ongoing and improvements, suggestions, and bugfixes are welcome.

