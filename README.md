# 🏥 Medical Reasoning Fine-Tune (DeepSeek-R1-Distill-Llama-8B)

> **A real-world experiment in fine-tuning a clinical reasoning assistant under strict hardware constraints (6GB VRAM).**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Library](https://img.shields.io/badge/Unsloth-Fast_Fine--Tuning-green)

---

## 🧠 Problem Statement

Generalist LLMs, even when combined with RAG, often produce **safe but shallow** medical advice.  
They behave like a *Wikipedia summary* — technically correct, but lacking clinical intuition.

**In healthcare, “generally speaking” is a failure.**

RAG provides the *information*,  
but it does **not** teach a model *how to think* like a doctor.

This project fine-tunes **DeepSeek-R1-Distill-Llama-8B** to bridge that gap, focusing on **structured clinical reasoning**, not just factual recall.

---

## 🎯 Objective

- **Structure:** Move from generic paragraphs to clinical reasoning  
  *(Symptoms → Causes → Precautions → Red Flags)*  
- **Constraint:** Run the full training pipeline on consumer hardware  
  *(RTX 4050 / 6GB VRAM)*  
- **Outcome:** Reduce hallucinations and increase diagnostic relevance  

---


## 🛠️ Tech Stack

- **Base Model:** `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- **Dataset:** `FreedomIntelligence/medical-o1-reasoning-SFT`
- **Library:** [Unsloth](https://github.com/unslothai/unsloth)
- **Hardware:** NVIDIA RTX 4050 (Laptop GPU, 6GB VRAM)

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Rajesh-007-dl/Fine-Tuned-Medical-Reasoning-Model.git
cd Fine-Tuned-Medical-Reasoning-Model
```

### 2. Install Dependencies
> **Note:** Unsloth requires a CUDA-aware install.

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
```

### 3. Environment Setup
Create a `.env` file or export variables:

```bash
HF_TOKEN="your_hugging_face_token"
WANDB_API_TOKEN="your_wandb_token"
```

---

## ⚙️ Training Configuration

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)
```

---

## 🧠 Inference & Prompting

```python
prompt_style = """
### Task:
You are a medical expert specializing in clinical reasoning.

### Query:
{question}

### Answer:
"""
```
## ⚠️ The 6GB VRAM Challenge (Engineering the “Impossible”)

Training a reasoning model on **6GB VRAM** is a constant fight against OOM errors.  
Standard Hugging Face `Trainer` pipelines were not viable.

### Critical Optimizations for Low-VRAM Stability

1. **Unsloth Framework**  
   Enables ~2× faster fine-tuning with ~60% lower memory usage.

2. **Disabled Packing**  
   `packing = False` prevents sequence concatenation, a major memory spike risk.

3. **Strict Context Limits**  
   `max_seq_length = 2048` (or 1024 for tighter constraints) to fit within VRAM.

4. **Gradient Accumulation**  
   `gradient_accumulation_steps = 4` with small batch sizes to simulate scale safely.

---

---

## 📈 Observed Outcomes

**Before Fine-Tuning:** Generic, safe, unhelpful responses.  
**After Fine-Tuning:** Structured reasoning, red-flag detection, and clinical relevance.

---

## 📌 Philosophy: RAG vs. Fine-Tuning

> **RAG gives a model a library.**  
> **Fine-tuning gives it the expertise to use it.**

---

