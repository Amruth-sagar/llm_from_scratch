import json
import numpy as np
import argparse
from llm_from_scratch.tokenizer.bpe_tokenizer import BytelevelBPE
import pandas as pd
from tqdm import tqdm
from llm_from_scratch.data.utils import clean_text


def round_trip_correctness(tokenizer, texts):
    failures = []

    for i, text in  enumerate(tqdm(texts, desc="Round-trip check")):
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        if decoded != text:
            failures.append((i, text, decoded))
    
    return failures


def token_per_word(tokenizer, texts):
    ratios = []
    
    for text in tqdm(texts, desc="Tokens per word"):
        words = text.split()
        if not words:
            continue

        num_words = len(words)
        num_tokens = len(tokenizer.encode(text))

        ratios.append(num_tokens / num_words)

    return float(np.mean(ratios)), ratios



if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--parquet_file', required=True, type=str)
    argument_parser.add_argument('--num_rows', required=True, type=int)
    argument_parser.add_argument('--lang', required=True, type=str)
    argument_parser.add_argument('--tokenizer_state', required=True, type=str)
    argument_parser.add_argument('--output_json', required=True, type=str)

    args = argument_parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = BytelevelBPE()
    tokenizer.load(args.tokenizer_state)
    print("Tokenizer loaded.")

    assert args.lang in ['eng', 'hin'], "Language must be either \'eng\' or \'hin\'"
    if args.lang == 'eng':
        print("\nLoading English data...")
        data = pd.read_parquet(args.parquet_file)
        filtered = data[
            (data["language_score"] > 0) &
            (data["text"].str.len() >= 100)
        ]
        text_cleaned = (
            filtered["text"]
            .apply(clean_text)
            .tolist()[:args.num_rows]
        )
        print(f"English samples loaded: {len(text_cleaned)}")

    elif args.lang == 'hin':
        print("\nLoading Hindi data...")
        data = pd.read_parquet(args.parquet_file)
        filtered = data[
            (data["text"].str.len() >= 100)
        ]
        text_cleaned = (
            filtered["text"]
            .apply(clean_text)
            .tolist()[:args.num_rows]
        )
        print(f"Hindi samples loaded: {len(text_cleaned)}")

    all_texts = text_cleaned
    print(f"\nTotal evaluation samples: {len(all_texts)}")


    failures = round_trip_correctness(tokenizer, all_texts)
    print(f"\nRound-trip failures in {args.lang} : {len(failures)}")

    avg_word, _ = token_per_word(tokenizer, all_texts)
    print(f"\n\nAverage tokens per word for {args.lang} : {avg_word:.2f}")


    failure_examples = []
    for i, orig, dec in failures[:5]:
        failure_examples.append({
            "index": i,
            "original": orig,
            "decoded": dec,
        })

    results = {
        "language": args.lang,
        "num_samples": len(text_cleaned),
        "metrics": {
            "avg_tokens_per_word": avg_word,
            "num_round_trip_failures": len(failures)
        },
        "failure_examples": failure_examples
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {args.output_json}")
