import json
import pandas
import os
from collections import defaultdict
import re
from termcolor import colored


def rm_image_fc(content):
    """Removes images from markdown files"""
    return re.sub(r"!\[.*\]\(.*\) *", "", content)


def rm_addition_fc(content):
    """Removes bibliography and supplementary reading list from markdown files"""
    bib_match = re.search(r"# *BIBLIOGRAPHY", content, re.IGNORECASE)
    if bib_match:
        addition_start_idx = bib_match.span()[0]
    else:
        supplement_match = re.search(r"# *SUPPLEMENTARY *READING *LIST", content, re.IGNORECASE)
        addition_start_idx = supplement_match.span()[0]

    content = content[:addition_start_idx]
    return content


def rm_intro_fc(content):
    """Removes introduction content from markdown files"""
    intro_match = re.search(r"INTRODUCTION[^#]*", content)
    start_idx = intro_match.span()[1]
    content = content[start_idx:]
    return content


def rm_sec_title_fc(content):
    """Remove all section titles from markdown files"""
    content = re.sub(r"#\s*.*\n", "", content)
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content


def clean_markdown(dir_path, output_dir,
                   cleaners=[
                        rm_image_fc,
                        rm_addition_fc,
                        rm_intro_fc,
                        rm_sec_title_fc
                   ]
                   ):
    """Cleans markdown files in a directory using the provided cleaning functions

    Args:
        dir_path (str): input directory path
        output_dir (str): output directory path
        cleaners (List[str]): fc(str) -> str. Defaults to [ rm_image_fc, rm_addition_fc, rm_intro_fc, rm_sec_title_fc ].
    """
    file_names = os.listdir(dir_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in file_names:
        path = os.path.join(dir_path, file_name)

        with open(path, "r") as f:
            content = f.read()
        
        try:
            for cleaner in cleaners:
                content = cleaner(content)
        except Exception as e:
            print(f"Error at {path} in {cleaner.__name__} cleaning function")
            raise

        with open(os.path.join(output_dir, file_name), "w") as f:
            f.write(content)
        

if __name__ == "__main__":

    # data cleaning
    data_root = "data/raw/md"
    output_dir = "data/processed/md"

    dirs = os.listdir(data_root)
    for d in dirs:
        clean_markdown(os.path.join(data_root, d), os.path.join(output_dir, d))