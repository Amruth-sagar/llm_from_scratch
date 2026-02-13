import pandas as pd
from glob import glob
from llm_from_scratch.data.utils import clean_text
import random


def prepare_sampled_eng_hin(raw_eng_data_dir, raw_hin_data_dir, num_eng_files, num_hin_files, random_seed = 42, min_chars = 100, eng_lang_score=0.80):

    english_parquet_files = glob(raw_eng_data_dir+'/*.parquet')
    hindi_parquet_files = glob(raw_hin_data_dir+'/*.parquet')
    random.seed(random_seed)

    sampled_eng_pq_files = random.sample(english_parquet_files, num_eng_files)
    sampled_hin_pq_files = random.sample(hindi_parquet_files, num_hin_files) 
    
    english_text_cleaned = []
    hindi_text_cleaned = []

    for pq_file in sampled_eng_pq_files:
        data = pd.read_parquet(pq_file)
        filtered = data[
            (data["language_score"]>eng_lang_score) &
            (data["text"].str.len() >= min_chars)
        ]
        english_text_cleaned.extend(filtered["text"].apply(clean_text).tolist())

    for pq_file in sampled_hin_pq_files:
        data = pd.read_parquet(pq_file)
        filtered = data[
            (data["text"].str.len() >= min_chars)
        ]
        hindi_text_cleaned.extend(filtered["text"].apply(clean_text).tolist())

    
    return english_text_cleaned + hindi_text_cleaned



    


