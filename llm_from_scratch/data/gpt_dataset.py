from torch.utils.data import IterableDataset, DataLoader
from functools import partial
import pickle
import torch
import numpy as np

class GPTDataset(IterableDataset):
    def __init__(self, clean_data_dir, ctx_len, eos_token_id, split='TRAIN', num_docs_for_validation=10000, random_seed=42, return_doc_ids=True):
        super().__init__()
        
        with open(f"{clean_data_dir}/bin_fileid_path.pkl", 'rb') as infile:
            self.bin_fileid_path = pickle.load(infile)

        self.memmapped_binfiles = dict([
            (bin_fileid, np.memmap(filename, dtype=np.uint16, mode='r'))
            for bin_fileid, filename in self.bin_fileid_path.items()
        ])
        
        # Has keys: global_offsets, bin_ids, local_offset, local_length, total_token_count
        self.global_metadata = np.load(f"{clean_data_dir}/global_metadata.npz")
        
        # np.load(.npz) returns a Zip-backed object; we eagerly copy arrays so workers
        # don’t inherit an invalid ZIP/file handle when the Dataset is pickled.
        self.bin_ids = self.global_metadata['bin_ids']
        self.local_offset = self.global_metadata['local_offset']
        self.local_length = self.global_metadata['local_length']
        self.total_token_count = self.global_metadata['total_token_count']
        self.global_offsets = self.global_metadata['global_offsets']

        self.total_docs_count = self.global_offsets.shape[0]

        if split == 'TRAIN':
            last_train_doc = self.total_docs_count-num_docs_for_validation-1
            self.start_idx = 0
            self.end_idx = self.global_offsets[last_train_doc] + self.local_length[last_train_doc] - ctx_len - 1
        else:
            self.start_idx = self.global_offsets[self.total_docs_count-num_docs_for_validation]
            self.end_idx = self.total_token_count - ctx_len - 1

        # The dataloader workers will update it if we are using many workers
        self.rng = np.random.default_rng(random_seed)
        self.ctx_len = ctx_len
        self.eos_token_id = eos_token_id
        self.return_doc_ids = return_doc_ids

    def __iter__(self):
        while True:
            start = self.rng.integers(low=self.start_idx, high=self.end_idx, dtype=np.uint64)

            current_doc = np.searchsorted(self.global_offsets, start, side="right") - 1

            tokens = []
            total_tokens = 0

            while total_tokens < self.ctx_len + 1:
                global_offset = self.global_offsets[current_doc]
                bin_id, local_offset, local_length = (self.bin_ids[current_doc], 
                                                    self.local_offset[current_doc], 
                                                    self.local_length[current_doc])
                

                difference_in_start = start - global_offset
                local_start = local_offset + difference_in_start

                remaining_tokens = self.ctx_len + 1 - total_tokens
                available_tokens = local_length - difference_in_start

                take = min(remaining_tokens, available_tokens)

                tokens.append(self.memmapped_binfiles[bin_id][local_start : local_start + take])

                total_tokens += take
                
                # If we didnt get enought tokens from first doc, we start from next
                start = self.global_offsets[current_doc + 1]
                current_doc += 1

            tokens = np.concatenate(tokens)

            input_tokens = tokens[:-1]
            output_tokens = tokens[1:]

            if self.return_doc_ids:
                curr_doc_id = 1
                doc_ids = []
                for token_id in input_tokens:
                    doc_ids.append(curr_doc_id)
                    if token_id == self.eos_token_id:
                        curr_doc_id += 1
                        
                yield torch.from_numpy(input_tokens).long(), torch.from_numpy(output_tokens).long(), torch.tensor(doc_ids).long()
            else:
                yield torch.from_numpy(input_tokens).long(), torch.from_numpy(output_tokens).long()


def worker_init_function(worker_id, random_seed=42):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset

    dataset.rng = np.random.default_rng(random_seed + worker_id)
    dataset.memmapped_binfiles = dict([
        (bin_fileid, np.memmap(filename, dtype=np.uint16, mode='r'))
        for bin_fileid, filename in dataset.bin_fileid_path.items()
    ])

def get_dataloader(dataset, batch_size, num_workers=4, random_seed=42, **kwargs):
    worker_init = partial(worker_init_function, random_seed=random_seed)
    return DataLoader(
        dataset=dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        worker_init_fn=worker_init,
        **kwargs
    )