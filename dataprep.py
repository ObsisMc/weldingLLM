import pymupdf
import os
import json
import re
import random
import numpy as np

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import dotenv
from typing import List

dotenv.load_dotenv()

def extract_text(path: str, start_page: int=0, end_page: int=-1, save_path=None,):
    root_path, file_name = os.path.split(path)
    fname, suffix = os.path.splitext(file_name)
    save_path = save_path if save_path is not None else os.path.join(root_path, fname + ".txt")

    if os.path.exists(save_path):
        os.remove(save_path)
    
    doc = pymupdf.open(path)
    page_n = doc.page_count
    end_page = end_page if end_page > -1 else page_n + end_page
    with open(save_path, "a+") as f:
        for i, page in enumerate(doc):
            if start_page <= i <= end_page:
                text = page.get_text()
                f.write(text)



def chunk_text(path: str, chunk_size=15000, chunk_overlap=150):
    with open(path, "r") as f:
        data = f.read()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    texts = text_splitter.create_documents([data])
    # print(texts[-1])
    # print(len(texts))
    return texts


def generate_qas(content: str):

    client = OpenAI()
    system_propmt = "You are an expert in welding and proficient in building dataset of question-answering pairs."
    requirements = (
        "Requirements:\n"
        "1. Create question-answer pairs based on the given text."
        "2. Each question should be based on a specific part of the text."
        "3. Each answer should be a specific answer to the question."
        "4. The types of questions should be diverse. The list should include diverse types of tasks like open-ended generation, classification, editing, multiple choice, true/false, fill in the blanks, etc."
        "5. The language used for the questions also should be diverse, use different types of language like formal, informal, technical, etc."
    )
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_question_answer_pairs",
                "description": "get question answer pairs from the text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pairs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "question": {
                                        "type": "string"
                                    },
                                    "answer": {
                                        "type": "string"
                                    }
                                }
                            },
                            "description": "Question-answering pairs",
                        },
                    },
                    "required": ["pairs"],
                },
            }
        }
    ]
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_propmt},
            {
                "role": "user",
                "content": "Create a list of question-answering pairs based on the following text from a welding handbook: \n\n" + content + f"\n\n{'-' * 10}" + requirements
            }
        ],
        tools=tools
    )
    tool_call = completion.choices[0].message.tool_calls[0]
    arguments = tool_call.function.arguments
    arguments = json.loads(arguments)
    pairs = arguments['pairs']

    return pairs


def create_qa_jsonl(path: str, save_path: str):
    if os.path.exists(save_path):
        os.remove(save_path)
    texts = chunk_text(path)

    with open(save_path, "a+") as f:
        for i, text in enumerate(texts):
            if 2 <= i <= 5:
                data = text.page_content
                pairs = generate_qas(data)
                for pair in pairs:
                    question = pair['question']
                    answer = pair['answer']
                    f.write(json.dumps({"question": question, "answer": answer}) + "\n")
            elif i > 5:
                break

def create_pretrain_json(path: str, save_path: str, chunk_sizes: list=None, chunk_overlaps: list=None,
                         remove_duplicate: bool=True):
    if os.path.exists(save_path):
        os.remove(save_path)
    if chunk_sizes is None:
        texts = [chunk_text(path)]
    else:
        texts = []
        for i, (chunk_size, chunk_overlap) in enumerate(zip(chunk_sizes, chunk_overlaps)):
            texts.append(chunk_text(path, chunk_size, chunk_overlap))


    ret = []
    for i, text in enumerate(texts):
        for j, page in enumerate(text):
            data = page.page_content
            ret.append(data)
    
    seen = set()
    # remove duplicates while keeping the order
    ret, original_len = ret if not remove_duplicate else [x for x in ret if not (x in seen or seen.add(x))], len(ret)
    ret = [{"text": t } for t in ret]
    print(f"data size: {len(ret)} (remove {original_len - len(ret)} duplicates)")
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(ret, f, ensure_ascii=False)


def build_openai_ft_dataset(jsonl_path: str):
    save_path = jsonl_path.replace(".jsonl", "_openai_ft.jsonl")
    if os.path.exists(save_path):
        os.remove(save_path)

    with open(save_path, "a+") as save_f:
        with open(jsonl_path, "r") as f:
            datas = f.readlines()
        
        for sample in datas:
            sample = json.loads(sample)

            sample = {"messages": [
                {"role": "system", "content": "You are a welding expert."}, 
                {"role": "user", "content": sample['question']}, 
                {"role": "assistant", "content": sample['answer']}
                ]}
            
            save_f.write(json.dumps(sample) + "\n")
    

def markdown2text(path: str, output: str, without_images: bool=True):
    with open(path, "r") as f:
        data = f.read()
    
    data = re.sub(r"!\[.*\]\(.*\)", "", data) if without_images else data
    with open(output, "w") as f:
        f.write(data)


def extract_lines_from_txt(path: str, start_line: int=0, end_line: int=-1, save_path=None):
    with open(path, "r") as f:
        lines = f.readlines()
    
    end_line = end_line if end_line > -1 else len(lines)
    ret = lines[start_line:end_line+1]

    if save_path is not None:
        with open(save_path, "w") as f:
            f.writelines(ret)
    return ret


def openai2llamafactory_format(input_paths: List[str], output_path: str):
    if os.path.exists(output_path):
        os.remove(output_path)
    
    data = []
    for input_path in input_paths:
        with open(input_path, "r") as input_f:
            while line := input_f.readline():
                line = json.loads(line)
                messages = line['messages']

                system_prompt = messages[0]['content']
                instruction = messages[1]["content"]
                answer = messages[2]["content"]

                data.append({
                    "instruction": instruction,
                    "input": "",
                    "output": answer,
                })

    print(f"total data size: {len(data)}")
    with open(output_path, "w") as output_f:
        json.dump(data, output_f, ensure_ascii=False)


def split_train_test(data_json: str, train_ratio: float=0.8):
    with open(data_json, "r") as f:
        data = np.array(json.load(f))

    N = len(data)
    train_n = int(N * train_ratio)

    np.random.seed(42)
    shuffle_idx = np.random.permutation([i for i in range(N)])
    train_data = data[shuffle_idx[:train_n]].tolist()
    test_data = data[shuffle_idx[train_n:]].tolist()

    file_name_prefix = os.path.splitext(os.path.split(data_json)[-1])[0]
    with open(os.path.join("./data/train", file_name_prefix + "_train.json"), "w") as f:
        json.dump(train_data, f)
    
    with open(os.path.join("./data/test", file_name_prefix + "_test.json"), "w") as f:
        json.dump(test_data, f)
    
    print(f"train data size: {len(train_data)}")
    print(f"test data size: {len(test_data)}")
    
    

if __name__ == "__main__":
    # extract_text("data/weldinghandbook_1.pdf", 2, 871)

    # texts = chunk_text("data/weldinghandbook_1.txt")
    # print(type(texts[0].page_content))


    # create_qa_jsonl("data/weldinghandbook_1.txt", "data/weldinghandbook_1_qa_2.jsonl")
    # build_openai_ft_dataset("data/weldinghandbook_1_qa.jsonl")


    ## workflow
    # extract_text("data/weldinghandbook_1.pdf", 2, 871)
    # create_qa_jsonl("data/weldinghandbook_1.txt", "data/weldinghandbook_1_qa.jsonl")
    # build_openai_ft_dataset("data/weldinghandbook_1_qa.jsonl")



    ## preprocessing: markdown to text (w/o images)
    # markdown2text("data/volume1_10-609.md", "data/volume1_10-609.txt", without_images=True)

    ## extract lines
    # extract_lines_from_txt("data/volume1_10-609.txt", 0, 880, "data/volume1_chapter1.txt")
    
    ## data preprocessing: chunking text
    # chunk_sizes=[256, 512, 800, 1024, 1600, 2048, 3000, 4096, 8192]
    # chunk_sizes = [256 + i for i in range(0, 8000, 256)]
    # chunk_overlaps=[0] * len(chunk_sizes)
    # create_pretrain_json(
    #     "data/chinese_welding_vol1-33-632.txt", 
    #     "data/chinese_welding_vol1-33-632_pretrain.json",
    #     chunk_sizes=chunk_sizes,
    #     chunk_overlaps=chunk_overlaps
    #     # chunk_overlaps=[100] * 6
    # )


    # chunk_sizes = [256 + i for i in range(0, 16000, 128)]
    # chunk_overlaps=[int(i * 0.2) for i in chunk_sizes]

    # create_pretrain_json(
    #     "data/volume1_chapter1.txt", 
    #     "data/volume1_chapter1_pretrain.json",
    #     chunk_sizes=chunk_sizes,
    #     chunk_overlaps=chunk_overlaps,
    #     # chunk_overlaps=[100] * 6
    #     remove_duplicate=True
    # )


    ## openai sft format -> llama factory format
    # input_paths = [
    #     "data/openai_ft_data/vol1_1.jsonl",
    #     "data/openai_ft_data/vol1_2.jsonl",
    #     "data/openai_ft_data/vol1_3.jsonl",
    #     "data/openai_ft_data/vol1_4.jsonl",
    #     "data/openai_ft_data/vol1_5.jsonl",
    # ]
    # openai2llamafactory_format(input_paths, "data/volume1_chapter1_llamafactory_sft.json")

    ## split train test from the entire dataset
    split_train_test("data/entire/volume1_chapter1_llamafactory_sft.json", train_ratio=0.8)
