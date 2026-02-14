# :robot: Bilingual (Hindi–English) LLM From Scratch 

Train your own bilingual Large Language Model — from tokenizer to chat model — built entirely from scratch in pure PyTorch.

This repository is a complete, end-to-end blueprint for building a Hindi–English GPT-style model **without relying on external NLP frameworks or model libraries**.

No Hugging Face.
No Transformers package.
No pretrained tokenizers.
No RoPE utilities.
No external model abstractions.

Everything is implemented manually in raw PyTorch — including:

- A custom byte-level BPE tokenizer (written from scratch)
- Transformer architecture built manually
- Rotary Position Embeddings (RoPE) implemented from first principles
- Attention, masking, and training loop logic
- Pretraining pipeline
- Supervised fine-tuning (SFT) for chat capability

The goal of this project is to demonstrate that it is entirely possible to build a coherent, functional bilingual LLM with a clean, reproducible pipeline — fully independent of high-level NLP ecosystems.

## :toolbox: What This Repository Provides

### :one: Tokenizer

- Byte-Level BPE
- Vocabulary size: 32007 (32000 + 7 special tokens)
- Average tokens per word:
  - English: ~1.38  
  - Hindi: ~1.30  

The tokenizer is trained jointly on Hindi and English data, ensuring both languages are well represented and efficiently encoded.

### :two: Language Model

- GPT-style Transformer (with RoPE)
- ~163M parameters
- Pretrained on bilingual Hindi–English corpus
- Supervised fine-tuned on IndicInstruct for conversational ability

### :zap: Capabilities

The fine-tuned model can:

- Respond in Hindi to Hindi prompts  
- Respond in English to English prompts  
- Perform simple translation when prompted (e.g., prefix with  
  `"Translating from hindi to english"` or vice versa)  

*Despite its compact 160M parameter scale, the model generates coherent, well-structured outputs and exhibits emerging semantic understanding*

Pretrained tokenizer states, model checkpoints, and SFT checkpoints are available [here (G-drive)](https://drive.google.com/drive/folders/1fm_sqBZgxeiyqiBDQyKpqTw_IAW0zhpe?usp=sharing).

You can explore the model interactively via `demo.ipynb`.



# :wrench: Installation

## :clipboard: Prerequisites
- Python 3.12 or higher
- Pytorch 2.7 or higher
- CUDA-compatible GPU with CUDA 12.6 or higher

1. **Create a virtual environment**:  Create a virtual environment 
```
> conda create -n lfs python=3.13
```

2. **Instally PyTorch with CUDA**:  For windows and linux
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
For MacOS, 
```
pip3 install torch torchvision
```

3. **Clone this repo and install the dependencies**
```
git clone 
cd llm_from_scratch
pip install -e .
```

4. **Install additional dependencies for running notebooks**
```
pip install -e ".[jupyter]"
```

# :muscle: Training your own bi-lingual LLM
## :inbox_tray: Downloading the Pretraining Datasets

For bilingual pretraining, we use the following datasets:

- **FineWeb-Edu (10BT sample)** — English corpus  
  https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/10BT

- **Sangraha (Hindi – verified split)** — Hindi corpus  
  https://huggingface.co/datasets/ai4bharat/sangraha/tree/main/verified/hin

You may download either the full datasets or selected subsets depending on the scale of the model you intend to train. The key objective is to gather enough total tokens to adequately train your model while keeping the Hindi and English token contributions roughly balanced.

### How Much Data Do You Need?

A commonly used heuristic for estimating the required number of training tokens is:

> **Total tokens ≈ 20 × number of model parameters**

This guideline comes from the scaling law analysis in  
*Training Compute-Optimal Large Language Models* (Chinchilla scaling laws):  
https://arxiv.org/abs/2203.15556

For example:

- A 160M parameter model would ideally require approximately 3.2B tokens.

## :gear: Training the Tokenizer

The script `train_tokenizer.py` trains a joint Hindi–English **Byte-Level BPE** tokenizer from scratch.

It works as follows:

1. Selects a specified number of parquet files from the Hindi and English raw datasets.
2. Cleans the documents.
3. Streams the text through an iterator.
4. Trains a byte-level BPE tokenizer with the specified vocabulary size.
5. Saves the tokenizer state to the given directory.

When selecting files, ensure that **both languages contribute approximately the same number of tokens**. Since tokenizer training does not require billions of tokens, you only need a moderate subset of each dataset.

A practical approach:

- Check how many documents are present in each parquet file.
- Estimate the average token count per file.
- Select number of files from each, such that Hindi and English contribute roughly equal token volume.

This balance helps the tokenizer learn fair and efficient representations for both languages. You can train the tokenizer on approximately 5-10% of total tokens that you are planning to use for pre-training. 

Example:

```
python llm_from_scratch/tokenizer/train_tokenizer.py \
    --hin_data_dir /path/to/data/raw_data/sangraha \
    --eng_data_dir /path/to/data/raw_data/fineweb \
    --num_eng_files 1 \
    --num_hin_files 3 \
    --vocab_size 32000 \
    --save_tokenizer_state /path/to/tokenizer_state
```

## :file_folder: Creating `.bin` Files and Metadata for Pretraining

Instead of tokenizing documents on-the-fly during training (which is slow and inefficient), we preprocess the entire dataset in advance. This step:

- Cleans each document  
- Tokenizes it using the trained tokenizer  
- Stores all tokens in binary `.bin` files  
- Saves structured metadata for efficient retrieval  

### Why This Design?

Each parquet file is converted into:

1. A `.bin` file containing all tokenized documents stored sequentially.
2. A corresponding `.json` metadata file containing:
   - `dtype`
   - `source_file`
   - `bin_file`
   - `lang`
   - `total_tokens`
   - `documents` — a list of `{offset, length}` entries for every document

Additionally, a `global_metadata.npz` file is created by merging all metadata files.  
In this global metadata:

- All documents are shuffled.
- Offsets and lengths are preserved.
- Documents can be accessed from any binary file at arbitrary positions.

This structure allows training to use `numpy.memmap`, enabling:

- Streaming from disk without loading the entire dataset into memory  
- Efficient sequence packing across document boundaries  
- No wasted computation on padding tokens  
- Better GPU utilization  

### Choosing How Many Files to Process

To decide how many parquet files to include:

1. Estimate the average number of tokens per file in each dataset.
2. Select enough number files such that:
   - Your total token count meets your target.
   - Hindi and English contribute approximately equal proportions of the total tokens.

Maintaining balance between languages is important for bilingual model quality.

Example:

```
python llm_from_scratch/data/prepare_pretraining_data.py \
    --eng_data_dir /path/to/data/raw_data/fineweb \
    --num_eng_files 5 \
    --hin_data_dir /path/to/data/raw_data/sangraha \
    --num_hin_files 30 \
    --save_dir /path/to/data/cleaned/pretraining \
    --num_proc 8 \
    --random_seed 42 \
    --tokenizer_state /path/to/tokenizer_state
```

## :gear: Training the Language Model

Once the dataset has been prepared, you can begin pretraining the GPT model.

Model architecture and hyperparameters are defined in the `GPT_CFG` variable inside `traingpt.py`.  
Adjust this configuration according to the scale of the model you wish to train (e.g., number of layers, hidden size, number of heads, context length).

Training is implemented using PyTorch Distributed Data Parallel (DDP), and can be launched on a single node with one or more GPUs using `torchrun`.


```
torchrun --standalone --nproc_per_node=<NUM_GPUS> traingpt.py \
    --max_train_steps 50000 \
    --val_after_k_optim_steps 128 \
    --lr 3e-4 \
    --accum_grad_after_k_steps 8 \
    --val_batches_per_device 16 \
    --batch_size_per_device 4 \
    --metric_log_dir /path/to/log_dir/pretraining_run \
    --save_ckpt_after_k_steps 2500 \
    --start_saving_ckpt_after_k_steps 40000 \
    --ckpt_dir /path/to/ckpt_dir \
    --tokenizer_state /path/to/tokenizer_state \
    --clean_data_dir /path/to/data/cleaned/pretraining \
    --random_seed 42 \
    --num_docs_for_validation 10000 \
    --num_dl_workers 4
```

Gradient accumulation (`--accum_grad_after_k_steps`) allows you to simulate larger effective batch sizes. Validation runs periodically to track training progress. Checkpoints are saved after a specified number of steps, with optional delayed checkpoint saving to avoid storing early unstable states.

At the end of training, you will have:

- Model checkpoints (including the best validation checkpoint)
- Training logs for analysis
- A fully pretrained bilingual GPT model ready for supervised fine-tuning 

# :speech_balloon: Supervised Fine-Tuning (SFT)

After pretraining, the base language model must be adapted to follow instructions and engage in conversational dialogue. This is done via supervised fine-tuning (SFT) on curated instruction-following datasets.


## :inbox_tray: Downloading the SFT Dataset

For this project, we use the **IndicInstruct** dataset:

https://huggingface.co/datasets/BhabhaAI/indic-instruct-data-v0.2-filtered/tree/main

IndicInstruct is a multilingual instruction dataset containing both Hindi and English conversational data. It includes several high-quality sources such as:

- `anudesh`
- `dolly`
- `flan_v2`
- `hh-rlhf`
- `lm_sys`
- `nmt-seed`
- `oasst1`
- `wikihow`

The dataset contains both single-turn and multi-turn conversations, making it well-suited for training a bilingual chat model.



## :file_folder: Preparing `.bin` Files for SFT

Similar to the pretraining pipeline, we preprocess the dataset into:

- A single `.bin` file containing tokenized conversations
- A corresponding metadata file

During this step, additional special tokens such as `<user>` and `<assistant>` are added to structure conversational turns properly.

You may choose to:
- Overwrite the existing tokenizer state, or
- Save the updated tokenizer (with additional special tokens) in a new directory

Example:
```
python llm_from_scratch/data/prepare_indic_instruct_data.py \
    --sft_data_dir /path/to/data/raw_data/indicinstruct \
    --save_dir /path/to/data/cleaned/sft \
    --tokenizer_state /path/to/tokenizer_state \
    --save_tokenizer_state /path/to/tokenizer_state
```

## :gear: Running Supervised Fine-Tuning
To fine-tune the pretrained model on conversational data, run:

```
python trainsft.py \
    --num_epochs 1 \
    --val_every_k_steps 200 \
    --lr 1e-5 \
    --batch_size 4 \
    --metric_log_dir /path/to/log_dir/sft_run \
    --save_ckpt_after_k_steps 2000 \
    --ckpt_dir /path/to/sft_ckpt_dir \
    --tokenizer_state_sft /path/to/tokenizer_state \
    --sft_data_dir /path/to/data/cleaned/sft \
    --random_seed 42 \
    --num_samples_for_validation 1000 \
    --num_dl_workers 4 \
    --load_ckpt /path/to/ckpt_dir/GPTwithRoPE_best_val.pth
```

Notes: `--load_ckpt` should point to the best pretrained checkpoint. A lower learning rate (e.g., 1e-5) is recommended for fine-tuning.Validation runs periodically to monitor instruction-following performance.

After completion, you will have a bilingual GPT model capable of conversational responses in both Hindi and English!

## :package: Miscellaneous Utilities
Some auxiliary scripts are not documented in detail here to keep the README concise. The codebase is intentionally structured to be straightforward and self-explanatory. Readers are encouraged to explore and run these utilities directly.

- `llm_from_scratch/data/validate_tokenized_data.py` verifies that the generated `.bin` files are correctly constructed and ensures alignment between document offsets, lengths in metadata, and the underlying binary data.
- `llm_from_scratch/tokenizer/tokenizer_eval.py` Evaluates tokenizer integrity by performing round-trip checks

```
document ----encode---->  tokens  ----decode----> document
```
It allows selecting:
- A specific parquet file
- A target language
- A configurable number of rows

The script outputs a JSON report containing tokenizer statistics and any round-trip mismatches, for example:

```
{ "language": "eng",
  "num_samples": 10000,
  "metrics": {
    "avg_tokens_per_word": 1.3836537860492968,
    "num_round_trip_failures": 3
  },
  "failure_examples": [
    {
      "index": 5606,
      "original": "Biblical Commentary ..."
    },
    ...
  ]
} 
```
These utilities are designed to make data validation and tokenizer diagnostics transparent and reproducible.

---
If you find this project useful, consider giving it a ⭐!

---