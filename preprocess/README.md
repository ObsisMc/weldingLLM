# Data Preprocessing

This step contains preparing two datasets, one for continued pretraining and the other for instruction finetuning.

For the continued pretraining dataset, there are 3 steps, PDF extraction, data cleaning and chunking.

For the instruction dataset, alpaca template is used and a knowledge-based Question-Answering dataset is curated.

Dataset size:
- Continued Pretraining: ~60k (max length ~2k tokens)
- Instruction Finetuning: ~70k (50k alpaca + 20k custom QA)


## Continued Pretraining

There are roughly 3 steps to prepare pretraining datasets:

1. PDF extraction: extract texts from PDFs
2. Data Cleaning: clean extracted texts
3. Chunking: split cleaned texts into chunks


### PDF Extraction
[MinerU](https://github.com/opendatalab/MinerU), an open-source PDF extractor, is utilzed, which also has an [online PDF extractor](https://mineru.org.cn/OpenSourceTools/Extractor) (login needed).

MinerU convert PDFs into markdowns:
- Only body will be kept
- Figures, tables and other non-text elements will be saved as images
- Not perfect, captions of figures may be recognized as body texts


### Data Cleaning

Main removed parts:
1. preface, contents (removed during PDF extraction)
2. non-text elements (images, tables, etc.)
3. bibliography
4. section titles

Please refer to `clean.py` for more details. 

```shell

# run in the root dir
# by default, input dir is "data/raw/md", output dir is "data/processed/md"
python preprocess/clean.py

```

### Chunking

Chunks the cleaned texts into chunks. 

```shell
# run in the root dir
# by default, the cleaned data is in "data/processed/md" and the output file is "data/pretrain/pretrain_books_all.json"
python preprocess/chunk.py

```

After chunking, the pretraining dataset is well prepared, which is like:

```json
[
    {"text": "...."},
    {"text": "......."}
]

```

## Instruction Fine-tuning

Instruction template used is [Alpaca](https://github.com/tatsu-lab/stanford_alpaca).

```txt
### Instruction:
{instruction}

### Input:
{input}

### Response:
```

Two types of instruction datasets are used:

1. General QA: [Alpaca dataset](https://huggingface.co/datasets/tatsu-lab/alpaca) 
2. Knowledge-based QA: generated from reference books using ChatGPT. There are some methods used to generate high-quality QA datasets like [self-instruct](https://github.com/yizhongw/self-instruct).

The format of the instruction dataset is like:

```json
[
    {
        "instruction": "Create a classification task by clustering the given list of items.",
        "input": "Apples, oranges, bananas, strawberries, pineapples",
        "output": "Class 1: Apples, Oranges\nClass 2: Bananas, Strawberries\nClass 3: Pineapples",
    },
    {
        "instruction": "...",
        "input": "...",
        "output": "...."
    }
]

```