from data_prep import prepare_sampled_eng_hin
from llm_from_scratch.tokenizer.bpe_tokenizer import BytelevelBPE
import argparse

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--eng_data_dir', required=True, type=str)
    argument_parser.add_argument('--hin_data_dir', required=True, type=str)
    argument_parser.add_argument('--num_eng_files', required=True, type=int)
    argument_parser.add_argument('--num_hin_files', required=True, type=int)
    argument_parser.add_argument('--vocab_size', required=True, type=int)
    argument_parser.add_argument('--save_tokenizer_state', required=True, type=str)

    args = argument_parser.parse_args()

    tokenizer = BytelevelBPE()
    tokenizer.train_from_iterator(
        prepare_sampled_eng_hin(
            raw_eng_data_dir=args.eng_data_dir,
            raw_hin_data_dir=args.hin_data_dir,
            num_eng_files=args.num_eng_files,
            num_hin_files=args.num_hin_files,
        ),
        vocab_size=args.vocab_size
    )

    tokenizer.add_special_tokens(
        [
            '<eos>',
            '<bos>',
            '<pad>',
            '<unk>',
            '<mask>'
        ]
    )

    tokenizer.save(args.save_tokenizer_state)
