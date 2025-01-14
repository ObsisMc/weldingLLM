# Welding LLM

This project further pre-trained the Mistral model and fine-tuned it with instructions using LoRA, based on five welding reference textbooks, aiming to develop a large language model specialized in the welding domain.

- model: `unsloth/Mistral-Nemo-Base-2407`
- Parameter-Efficient Fine-Tuning : LoRA
- Hardware: Colab A100, 40G

## Install

For data preparation,

```shell
pip install langchain
```

Model training will run on Colab so you don't need to install dependencies locally.

## Data Preparation <a name="data preparation"></a>

The preprocessed data is [here](https://drive.google.com/drive/folders/1XMb1OACE_Ww1xPY_cSpuA9DL-9vomNg8?usp=drive_link), which contains training and evaluation data.

If you want to use the shared evaluation data, which only contains multiple-choice questions, please see [`eval/README.md`](https://drive.google.com/file/d/1kBvlTifjtF4ZDBcCKowQ4JD8rLJ-Xvpr/view?usp=drive_link) before using it.

If you want to prepare your own training data, please see [`preprocess/README.md`](https://github.com/ObsisMc/weldingLLM/tree/main/preprocess#readme) for more details. The data will be generated in `data` folder and the `data_example` folder is provided for your reference.


## Finetuning

See [`finetune/README.md`](https://github.com/ObsisMc/weldingLLM/tree/main/finetune#readme) for more details.


## Evaluation

This project uses a dataset with multiple-choice questions to evaluate the model as mentioned in [Data Preparation](#data-preparation) section.

The initial result is [here](https://drive.google.com/drive/folders/1OPa0XTPesiumaL8A0WP4rGu_AhgEohgf?usp=drive_link).


## Future Improvement

- Higher-Quality training and evaluation data
- Larger LLM / Train more parameters
