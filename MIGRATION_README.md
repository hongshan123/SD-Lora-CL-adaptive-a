# SD-Lora-CL Migration Notes

This package is prepared for moving `/home/zhaoyang/SD-Lora-CL` to another server.

## 1. Unpack

```bash
cd /path/to/workspace
tar -xzf SD-Lora-CL_code_20260622.tar.gz
cd SD-Lora-CL
```

If you also need the prepared datasets:

```bash
cd /path/to/workspace/SD-Lora-CL
tar -xzf /path/to/SD-Lora-CL_data_runtime_20260622.tar.gz
```

The data package restores:

```text
data/imagenet-r/
data/cifar-100-python/
```

The raw parquet cache `data/imagenet-r-parquet/` is intentionally not included.

## 2. Environment

Recommended:

```bash
conda create -n sdlora python=3.10 -y
conda activate sdlora
pip install -r requirements.txt
```

Install the PyTorch/CUDA build that matches the target server separately if needed.

## 3. HuggingFace Mirror

On servers in China, keep:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 4. Output Directories

Many configs save LoRA weights under `filepath`. Create the output directory before running if it does not exist. Examples:

```bash
mkdir -p ImageNetR3
mkdir -p ImageNetR_K_CMS_KMERGEPP_S02_RECENT0_K4_V2
mkdir -p CF100_K_CMS_KMERGEPP_S02_RECENT0_K4_CLEAN
```

## 5. Example Commands

Official SD-LoRA ImageNet-R config:

```bash
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 main.py \
--config=./exps/sdlora_inr.json \
> sdlora_inr.log 2>&1
```

K-CMS ImageNet-R k=4 config:

```bash
mkdir -p ImageNetR_K_CMS_KMERGEPP_S02_RECENT0_K4_V2
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 main.py \
--config=./exps/k_cms_sdlora_kmergepp_s02_recent0_k4_inr_v2.json \
> k_cms_kmergepp_s02_recent0_k4_inr_v2.log 2>&1
```

CIFAR-100 K-CMS clean k=4 config:

```bash
mkdir -p CF100_K_CMS_KMERGEPP_S02_RECENT0_K4_CLEAN
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 main.py \
--config=./exps/k_cms_sdlora_kmergepp_s02_recent0_k4_c100.json \
> k_cms_kmergepp_s02_recent0_k4_clean.log 2>&1
```

## 6. Package Contents

The code archive includes source code, configs, scripts, README, paper notes, and small project metadata.
It excludes generated outputs, logs, caches, and datasets.

The data runtime archive includes processed runtime datasets only.
