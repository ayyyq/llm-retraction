# When Do LLMs Admit Their Mistakes? Understanding the Role of Model Belief in Retraction
This is the official repository for our paper:

[When Do LLMs Admit Their Mistakes? Understanding the Role of Model Belief in Retraction](https://arxiv.org/abs/2505.16170).

## 🛠️ Environment Setup
Please refer to `retraction.yml` for setting up the conda environment.

**Note:**
* To successfully run **Olmo2-7B**, install `transformers>=4.51.3`.
* To extract attention weights and value vectors from **Qwen2.5-7B**, follow [this issue](https://github.com/QwenLM/Qwen3/issues/722) and apply the corresponding transformers patch.
* Be cautious with Qwen2.5's precision—`bf16` is recommended as discussed in [this issue](https://github.com/QwenLM/Qwen3/issues/761).

Most experiments were conducted on a single A6000 GPU. For evaluation with Llama3.3-70B-Instruct, we used 4 A6000 GPUs.

## 📂 Data
We provide:
* **Original datasets** `wikidata_{train,test}_free.jsonl`, `celebrity_{train,test}_free.jsonl`.
* **Continuation datasets** for Llama3.1-8B, Qwen2.5-7B, and Olmo2-7B.

For the construction of the original datasets and continuation datasets, please refer to [data/README.md]().

## 🔍 Main Experiments for Probing and Steering
* **Extract Activations:** Please run `scripts/get_activations.sh` first to get the hidden states for each layer of the model, for the following probing and steering experiments.
* **Run Probing:** For probing, we train a single linear probe per layer. Hyperparameters are provided in `scripts/probing.sh`.
* **Apply Steering:** To steer the model toward different belief directions, please run `scripts/steering.sh`.

## 🔧 Patching Experiments
* To perform attention weights and value vectors patching, please run `scripts/patching.sh` to first extract the patching values and then do patching to the original model without steering.

## 🧪 Supervised Fine-Tuning
To replicate our SFT experiments:
1. Generate training data using `data/generate_sft_data.py`.
2. Train the model using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/releases/tag/v0.9.2). As detailed in the paper, we use the following config:
```bash
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
per_device_train_batch_size: 8
gradient_accumulation_steps: 1
learning_rate: 1.0e-4
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
report_to: wandb
```

## 🥳Citation
If you find our work useful, please consider citing:
```bibtex
@misc{yang2025llmsadmitmistakesunderstanding,
      title={When Do LLMs Admit Their Mistakes? Understanding the Role of Model Belief in Retraction}, 
      author={Yuqing Yang and Robin Jia},
      year={2025},
      eprint={2505.16170},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.16170}, 
}
```