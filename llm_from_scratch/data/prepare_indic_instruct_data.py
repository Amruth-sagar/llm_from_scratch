from llm_from_scratch.tokenizer.bpe_tokenizer import BytelevelBPE
from llm_from_scratch.data.utils import clean_text
import argparse
import numpy as np
import json
from tqdm import tqdm
import os
from glob import glob
import pandas as pd
from collections import defaultdict
from faker import Faker
import re

fake = Faker("en_IN")

NAME_RE = re.compile(r"NAME_\d+")

def replace_names(text):
    mapping = {}
    def repl(match):
        key = match.group(0)
        if key not in mapping:
            mapping[key] = fake.name()
        return mapping[key]
    return NAME_RE.sub(repl, text)

def get_conv_from_messages(messages):
    full_conversation = ""
    prev_role = None
    for message in messages:
        if prev_role == "assistant":
            # adding an <eos> after assistant's message
            # the last eos_token_id will be added directly after tokenization
            full_conversation = full_conversation + "<eos>"
        full_conversation = full_conversation + f"<{message['role']}>" + message['content']
        prev_role = message['role']
    return full_conversation

def get_conv_dolly(row):
    instruction, context, response = row.instruction, row.context, row.response
    full_conversation = f"<user>{instruction}" + ("" if context == "" else f" Context: {context}") + f"<assistant>{response}"
    return full_conversation

def get_conv_flan_v2(row):
    input_text, target_text = row.inputs, row.targets
    full_conversation = f"<user>{input_text}<assistant>{target_text}"
    return full_conversation

def get_translations_nmt_seed(row):
    input_text, output_text = row.input_text, row.output_text
    english_to_hindi = f"<user>Translate from english to hindi: {input_text}<assistant>{output_text}"
    hindi_to_english = f"<user>Translate from hindi to english: {output_text}<assistant>{input_text}"
    return [english_to_hindi, hindi_to_english]


def tokenize_sft_data(tokenizer, datasets_and_files, save_dir):

    eos_token = tokenizer.special_token_dict['<eos>']
    metadata = []
    global_offset = 0
    unique_id = 0

    with open(save_dir+"/sft.bin", "wb") as bin_f:
        for dataset_name, dataset_files in datasets_and_files.items():

            print(f"Dataset: {dataset_name} ...")
            all_convs_in_dataset = []

            if dataset_name in ['anudesh', 'hh-rlhf', 'wikihow', 'oasst1']:
                for filename in dataset_files:
                    data = pd.read_json(filename, lines=True)
                    conversations = data['messages'].apply(get_conv_from_messages).tolist()
                    all_convs_in_dataset.extend(conversations)

            elif dataset_name == 'dolly':
                for filename in dataset_files:
                    data = pd.read_json(filename, lines=True)
                    conversations = data.apply(get_conv_dolly, axis=1).tolist()
                    all_convs_in_dataset.extend(conversations)

            elif dataset_name == 'flan_v2':
                for filename in dataset_files:
                    data = pd.read_json(filename, lines=True)
                    conversations = data.apply(get_conv_flan_v2, axis=1).tolist()
                    all_convs_in_dataset.extend(conversations)
            
            elif dataset_name == 'nmt-seed':
                for filename in dataset_files:
                    data = pd.read_json(filename, lines=True)
                    conversations = data.apply(get_translations_nmt_seed, axis=1).explode().tolist()
                    all_convs_in_dataset.extend(conversations)

            elif dataset_name == 'lm_sys':
                for filename in dataset_files:
                    data = pd.read_json(filename, lines=True)
                    conversations = data['messages'].apply(get_conv_from_messages).tolist()
                    conversations = [replace_names(conv) for conv in conversations]         # Swapping NAME_1 and NAME_2 in lm_sys with fake names
                    all_convs_in_dataset.extend(conversations)

            all_convs_in_dataset = [clean_text(conversation) for conversation in all_convs_in_dataset]
            
            print('Tokenizing all conversations ...')
            for conversation in tqdm(all_convs_in_dataset):
                tokens = tokenizer.encode(conversation)
                tokens.append(eos_token)
                uint16_tokens = np.asarray(tokens, dtype=np.uint16)
                
                # write the tokens to bin file
                bin_f.write(uint16_tokens.tobytes())
                
                metadata.append({'id':unique_id, 'dataset':dataset_name, 'offset':global_offset, 'length':len(tokens)})
                global_offset += len(tokens)
                unique_id += 1

    with open(save_dir+"/sft.json", "w", encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii = False)

         

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--sft_data_dir', type=str, required=True)
    argument_parser.add_argument('--save_dir', type=str, required=True)
    argument_parser.add_argument('--tokenizer_state', type=str, required=True)
    argument_parser.add_argument('--save_tokenizer_state', type=str, required=True)

    args = argument_parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    
    tokenizer = BytelevelBPE()
    tokenizer.load(args.tokenizer_state)

    tokenizer.add_special_tokens(
        [
            '<user>',
            '<assistant>',
        ]
    )
    tokenizer.save(args.save_tokenizer_state)

    all_jsonl_files = glob(args.sft_data_dir + "/**/*.jsonl", recursive=True)

    datasets_and_files = defaultdict(list)
    for jsonl_file in all_jsonl_files:
        dataset_name = jsonl_file.split('/')[-2]
        datasets_and_files[dataset_name].append(jsonl_file)

    tokenize_sft_data(tokenizer, datasets_and_files, args.save_dir)









