# Training

Follow the notebook to train and evaluate your model.

We first continued pretrained a base model, then did instruction finetuning to get a chat model.


## Hardware

- Colab A100, 40G

## Training Config

- base model: `unsloth/Mistral-Nemo-Base-2407` (12B, 4bit)
- training method: LoRA
    - rank: 128
    - alpha: 32
    - target modules: `"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj","embed_tokens", "lm_head"`

### Continued pretrianing
- dataset:
  - size: 60k
  - max length: ~2k tokens
- training config:
  - epochs: 1~2
  - batch size: 10
  - gradient accumulation steps: 16
  - learning rate: 5e-5
  - optim: `adamw_8bit`

### Instruction Finetuning
- dataset:
  - size: 70k (50k alpaca + 20k custom dataset)
- training config:
    - epochs: 1~2
    - batch size: 12
    - gradient accumulation steps: 16
    - learning rate: 5e-5
    - optim: `adamw_8bit`

## Training Time

- Continued pretraining:
  - ~12h/epoch
- Instruction finetuning:
  - ~4h/epoch