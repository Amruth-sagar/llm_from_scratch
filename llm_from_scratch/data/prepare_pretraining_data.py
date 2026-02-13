import pyarrow.parquet as pq
from llm_from_scratch.tokenizer.bpe_tokenizer import BytelevelBPE
from llm_from_scratch.data.utils import clean_text
import argparse
import numpy as np
import json
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
import random
import pickle

def text_to_bin(pq_file, tokenizer_state, bin_path, metadata_path, lang):

    tokenizer = BytelevelBPE()
    tokenizer.load(tokenizer_state)

    eos_token = tokenizer.special_token_dict['<eos>']
    
    pf = pq.ParquetFile(pq_file)
    # Local offset (local to this file) for metadata
    offset = 0
    metadata = []

    batch_size = 10000
    total_batches = (pf.metadata.num_rows + batch_size - 1) // batch_size

    with open(bin_path, "wb") as bin_f:
        for batch in tqdm(
            pf.iter_batches(batch_size=10000),
            total=total_batches,
            desc=f"Tokenizing {pq_file} ..."):

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

            
            for text in data["text"]:
                text = clean_text(text)

                tokens = tokenizer.encode(text)
                tokens.append(eos_token)

                uint16_tokens = np.asarray(tokens, dtype=np.uint16)
                
                # write the tokens to bin file
                bin_f.write(uint16_tokens.tobytes())

                metadata.append({
                    "offset": offset,
                    "length": len(tokens),
                })

                offset += len(tokens)

    meta_out = {
        "dtype" : "uint16",
        "source_file" : pq_file,
        "bin_file" : bin_path,
        "lang": lang,
        "total_tokens" : offset,
        "documents": metadata
    }

    with open(metadata_path, "w", encoding='utf-8') as f:
        json.dump(meta_out, f, ensure_ascii = False)


def create_global_idx(clean_data_dir, random_seed=42):
    metadata_files = [(os.path.basename(x).split('.')[0], x)
                      for x in glob(f"{clean_data_dir}/*.json")]

    bin_file_mapping = {
        os.path.basename(x).split('.')[0]: (i, os.path.abspath(x))
        for i, x in enumerate(glob(f"{clean_data_dir}/*.bin"))
    }

    total_docs = 0
    for _, path in metadata_files:
        with open(path, "r", encoding="utf-8") as f:
            total_docs += len(json.load(f)["documents"])

    rng = np.random.default_rng(random_seed)
    perm = rng.permutation(total_docs)

    global_offsets = np.empty(total_docs, dtype=np.uint64)
    bin_ids = np.empty(total_docs, dtype=np.uint32)
    local_offset = np.empty(total_docs, dtype=np.uint64)
    local_length = np.empty(total_docs, dtype=np.uint64)

    write_idx = 0
    for basename, path in metadata_files:
        with open(path, "r", encoding="utf-8") as infile:
            data = json.load(infile)

        bin_file_id = bin_file_mapping[basename][0]
        for doc in data["documents"]:
            i = perm[write_idx]
            bin_ids[i] = bin_file_id
            local_offset[i] = doc["offset"]
            local_length[i] = doc["length"]
            write_idx += 1

    g_offset = np.uint64(0)
    total_token_count = np.uint64(0)
    for i in range(total_docs):
        global_offsets[i] = g_offset
        g_offset += local_length[i]
        total_token_count += local_length[i]

    np.savez(
        f"{clean_data_dir}/global_metadata.npz",
        global_offsets=global_offsets,
        bin_ids=bin_ids,
        local_offset=local_offset,
        local_length=local_length,
        total_token_count=total_token_count,
    )

    bin_fileid_path = dict(bin_file_mapping.values())
    with open(f"{clean_data_dir}/bin_fileid_path.pkl", "wb") as outfile:
        pickle.dump(bin_fileid_path, outfile)




if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--eng_data_dir', type=str, required=True)
    argument_parser.add_argument('--num_eng_files', type=int, required=True)
    argument_parser.add_argument('--hin_data_dir', type=str, required=True)
    argument_parser.add_argument('--num_hin_files', type=int, required=True)
    argument_parser.add_argument('--save_dir', type=str, required=True)
    argument_parser.add_argument('--num_proc', type=int, required=True)
    argument_parser.add_argument('--random_seed', type=int, default=42)
    argument_parser.add_argument('--tokenizer_state', type=str, required=True)

    args = argument_parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)



    eng_pq_files = glob(f'{args.eng_data_dir}/*.parquet')
    hin_pq_files = glob(f'{args.hin_data_dir}/*.parquet')

    random.seed(args.random_seed)
    eng_pq_files = random.sample(eng_pq_files, args.num_eng_files)
    hin_pq_files = random.sample(hin_pq_files, args.num_hin_files)

    eng_fn_args = [
        (
            filename,
            args.tokenizer_state,
            f"{args.save_dir}/{os.path.basename(filename).split('.')[0]}.bin",
            f"{args.save_dir}/{os.path.basename(filename).split('.')[0]}.json",
            "en"
        )
        for filename in eng_pq_files
    ]

    hin_fn_args = [
        (
            filename,
            args.tokenizer_state,
            f"{args.save_dir}/{os.path.basename(filename).split('.')[0]}.bin",
            f"{args.save_dir}/{os.path.basename(filename).split('.')[0]}.json",
            "hi"
        )
        for filename in hin_pq_files
    ]

    all_fn_args = eng_fn_args + hin_fn_args

    with ProcessPoolExecutor(max_workers=args.num_proc) as executor:
        futures = [
            executor.submit(text_to_bin, *fun_args) for fun_args in all_fn_args
        ]

        for fut in as_completed(futures):
            fut.result()

    create_global_idx(args.save_dir, args.random_seed)




