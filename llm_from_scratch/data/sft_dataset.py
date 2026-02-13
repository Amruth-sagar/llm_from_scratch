from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import json
import torch

class SFTDataset(Dataset):
    def __init__(self, clean_data_dir, ctx_len, pad_token_id, user_token_id, asst_token_id, eos_token_id, split='TRAIN', num_samples_for_validation=1000, random_seed=42):
        super().__init__()
        
        with open(f"{clean_data_dir}/sft.json", 'r') as infile:
            metadata = json.load(infile)

        self.ctx_len = ctx_len
        self.pad_token_id = pad_token_id
        self.user_token_id = user_token_id
        self.asst_token_id = asst_token_id
        self.eos_token_id = eos_token_id

        # only focusing on samples that fit in context len for simplicity
        self.metadata = [x for x in metadata if x['length'] <= ctx_len + 1]
        random.seed(random_seed)
        random.shuffle(self.metadata)

        if split == "TRAIN":
            self.metadata = self.metadata[:-num_samples_for_validation]
        else:
            self.metadata = self.metadata[-num_samples_for_validation:]
        
        print(f"Num samples for {split}:{len(self.metadata)}")

        self.memmapped_sft_data = np.memmap(f'{clean_data_dir}/sft.bin', dtype=np.uint16, mode='r')
    
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, i):
        offset, length = self.metadata[i]['offset'], self.metadata[i]['length']
        tokens = self.memmapped_sft_data[offset : offset+length]

        tokens = np.array(tokens, dtype=np.int32)
        
        input_tokens = tokens[:-1].copy()
        output_tokens = tokens[1:].copy()

        # masking everything related to user
        loss_mask = np.cumsum((tokens == self.user_token_id)*-1 + (tokens == self.asst_token_id)*1)[1:]
        loss_mask[output_tokens == self.asst_token_id] = -1
        # masking where <eos> is input ( in multi-turn convsersations, this appears just before <user> )
        loss_mask[input_tokens == self.eos_token_id] = -1

        output_tokens[loss_mask == -1] = -100

        return input_tokens, output_tokens
    

def collate_fn(batch, pad_token_id):
    input_tokens = [x[0] for x in batch]
    output_tokens = [x[1] for x in batch]
    
    max_input_len = max([len(x) for x in input_tokens])

    padded_input = []
    padded_output = []

    for input_seq in input_tokens:
        padded_input.append(np.concatenate((input_seq, np.array([pad_token_id]*(max_input_len - len(input_seq))))))
    for output_seq in output_tokens:
        padded_output.append(np.concatenate((output_seq, np.array([-100]*(max_input_len - len(output_seq))))))
        
    padded_input = torch.from_numpy(np.array(padded_input)).long()
    padded_output = torch.from_numpy(np.array(padded_output)).long()
    pad_mask = (padded_input == pad_token_id)

    return padded_input, padded_output, pad_mask


def get_sft_dataloader(dataset, batch_size, num_workers=4, **kwargs):
    pad_token_id = dataset.pad_token_id
    return DataLoader(
        dataset=dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id=pad_token_id),
        **kwargs
    )