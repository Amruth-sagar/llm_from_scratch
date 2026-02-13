import pyarrow.parquet as pq
from llm_from_scratch.tokenizer.bpe_tokenizer import BytelevelBPE
from llm_from_scratch.data.utils import clean_text
import numpy as np
import json
import argparse
from glob import glob
import os
import random

def verify(pq_file, tokenizer_state, bin_path, metadata_path, lang, num_docs):

    tokenizer = BytelevelBPE()
    tokenizer.load(tokenizer_state)

    eos_token = tokenizer.special_token_dict['<eos>']
    
    pf = pq.ParquetFile(pq_file)

    batch = next(pf.iter_batches(batch_size=num_docs*3))
    data = batch.to_pandas()

    if lang == "en":
        batch = batch.select(["text", "language_score"])
    elif lang == "hi":
        batch = batch.select(["text"])

    data = batch.to_pandas()

    if lang == "en":
        data = data[
            (data["language_score"]>0.85) &
            (data["text"].str.len() >= 100)
        ]
    elif lang == "hi":
        data = data[
            (data["text"].str.len() >= 100)
        ]

    random.seed(42)
    indices = random.sample(range(data.shape[0]), num_docs)

    data = data.iloc[indices]

    with open(metadata_path, 'r', encoding='utf-8') as meta_f:
        metadata = json.load(meta_f)

    print(f'\tRAW_TEXT : {pq_file}\n\tBIN_FILE : {bin_path}\n\tMETADATA : {metadata_path}\n\tLANG : {lang}')

    metadata = [metadata["documents"][i] for i in indices]
    
    bin_data = np.memmap(
        bin_path,
        dtype=np.uint16,
        mode="r"
    )

    mis_matches = 0
    for i, text in enumerate(data["text"]):
        text = clean_text(text)

        tokens = tokenizer.encode(text)
        tokens.append(eos_token)

        uint16_tokens = np.asarray(tokens, dtype=np.uint16)
        uint16_tokens_from_file = bin_data[metadata[i]['offset']:metadata[i]['offset'] + metadata[i]['length']]

        try: 
            if np.all(uint16_tokens != np.array(uint16_tokens_from_file)):
                mis_matches += 1
        except:
            mis_matches += 1

    print(f"Num mis-matches: {mis_matches}\n\n")
    

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--eng_data_dir', type=str, required=True)
    argument_parser.add_argument('--hin_data_dir', type=str, required=True)
    argument_parser.add_argument('--cleaned_data_dir', type=str, required=True)
    argument_parser.add_argument('--tokenizer_state', type=str, required=True)
    argument_parser.add_argument('--num_docs_to_test', type=int, required=True)

    args = argument_parser.parse_args()

    eng_pq_files = dict([(os.path.basename(x).split('.')[0], x) for x in glob(f'{args.eng_data_dir}/*.parquet')])
    hin_pq_files = dict([(os.path.basename(x).split('.')[0], x) for x in glob(f'{args.hin_data_dir}/*.parquet')])

    bin_file_paths = dict([(os.path.basename(x).split('.')[0], x) for x in glob(f'{args.cleaned_data_dir}/*.bin')])
    metadata_paths = dict([(os.path.basename(x).split('.')[0], x) for x in glob(f'{args.cleaned_data_dir}/*.json')])
    clean_data_filenames = list(bin_file_paths.keys())

    for filename in clean_data_filenames:
        if filename in eng_pq_files:
            lang = 'en'
            pq_file = f'{args.eng_data_dir}/{filename}.parquet'
        elif filename in hin_pq_files:
            lang = 'hi'
            pq_file = f'{args.hin_data_dir}/{filename}.parquet'
        
        print(f"Validating random {args.num_docs_to_test} docs of {filename}.bin ...")
        verify(
            pq_file=pq_file,
            tokenizer_state=args.tokenizer_state,
            bin_path=bin_file_paths[filename],
            metadata_path=metadata_paths[filename],
            lang=lang,
            num_docs=args.num_docs_to_test
        )

    





