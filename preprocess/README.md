# Data Preprocessing


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

Please refer to `pretrain.py` for more details.

```shell

# run in the root dir
python preprocess/pretrain.py

```


###


## Instruction Fine-tuning



# Section Heading

Some body text of this section.

<a name="my-custom-anchor-point"></a>
Some text I want to provide a direct link to, but which doesn't have its own heading.

(… more content…)

[A link to that custom anchor](#my-custom-anchor-point)