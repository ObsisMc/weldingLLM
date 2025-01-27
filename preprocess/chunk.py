import os
import json
import re
import random

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List


def chunk_text(path: str, chunk_size=15000, chunk_overlap=150) -> List[Document]:
    with open(path, "r") as f:
        data = f.read()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # separators=["\n\n", ". ", "! ", "? ", "\n", " ", ""]
    )
    texts = text_splitter.create_documents([data])
    # print(texts[-1])
    # print(len(texts))
    return texts


def create_pretrain_json(
        path: str, save_path: str, chunk_sizes: List[int]=None, 
        chunk_overlaps: List[int]=None, remove_duplicate: bool=True):
    
    if save_path is not None and os.path.exists(save_path):
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
    print(f"File: {path} => data size: {len(ret)} (remove {original_len - len(ret)} duplicates)")
    
    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(ret, f, ensure_ascii=False, indent=4)
    return ret


## chunk text in some directories
def chunk_text_in_dirs(
        root_dir: str, 
        chunk_sizes: list, 
        chunk_overlaps: list,
        save_path: str,
        depth: int=2,
        shuffle: bool=False):
    assert depth >= 1, "depth must be greater than or equal to 1"
    
    if save_path is not None and os.path.exists(save_path):
        os.remove(save_path)

    def dfs(root_dir, d):
        if d == 1:
            ret = []
            file_n = 0
            input_paths = [os.path.join(root_dir, file) for file in os.listdir(root_dir)]
            for input_path in input_paths:
                chunked_text = create_pretrain_json(
                    input_path,
                    chunk_sizes=chunk_sizes,
                    chunk_overlaps=chunk_overlaps,
                    remove_duplicate=True,
                    save_path=None
                )
                file_n += 1
                ret.extend(chunked_text)
            return ret, file_n
        
        sub_dirs = os.listdir(root_dir)
        ret = []
        file_n = 0
        for sub_dir in sub_dirs:
            sub_dir = os.path.join(root_dir, sub_dir)
            texts, n = dfs(sub_dir, d - 1)
            ret.extend(texts)
            file_n += n
        return ret, file_n
    
    ret_chunked_texts, file_n = dfs(root_dir, depth)
    if shuffle:
        random.shuffle(ret_chunked_texts)
    
    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(ret_chunked_texts, f, ensure_ascii=False, indent=4)
    print(f"total file number: {file_n}, total data size: {len(ret_chunked_texts)}")
    return ret_chunked_texts


if __name__ == "__main__":

    # handle a directory with depth 2
    chunk_sizes = [1024, 2048, 4096, 8192]
    chunk_overlaps=[int(i * 0.2) for i in chunk_sizes]
    input_dir = "data/processed/md"
    output_file = "data/pretrain/pretrain_books_all.json"
    chunk_text_in_dirs(
        input_dir,
        chunk_sizes=chunk_sizes,
        chunk_overlaps=chunk_sizes,
        save_path=output_file,
        depth=2,
        shuffle=True
    )

    # # handle a directory with depth 1
    # chunk_sizes = [2048, 4096, 8192]
    # chunk_overlaps=[int(i * 0.2) for i in chunk_sizes]
    # input_dir = "data/processed/md/vol1"
    # output_file = "data/pretrain/pretrain_vol1.json"
    # chunk_text_in_dirs(
    #     input_dir,
    #     chunk_sizes=chunk_sizes,
    #     chunk_overlaps=chunk_sizes,
    #     save_path=output_file,
    #     depth=1,
    #     shuffle=True
    # )