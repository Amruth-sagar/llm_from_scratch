import unicodedata
import regex
from collections import Counter, defaultdict

from tqdm import tqdm
import pickle
import os

class BytelevelBPE:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.special_token_dict = {}

        # Merges: (id_i, id_j) -> id_k
        self.bpe_merges = {}

        # Ranks: (id_i, id_j) -> rank (int)
        # helpful in finding best pair of bytes to merge during tokenization
        self.bpe_ranks = {}

        # regex for pre-tokenization
        # [ ]?\p{L}[\p{L}\p{M}] --> this is for grouping unicode letters and markings
        # [ ]?\d+ --> for digits
        # [ ]?[^\s\p{L}\p{M}\d]+ --> one or more NON space, letter, marking or digits (covers punctuations)
        # \s+(?!\S) --> handles all \n \t and redundant spaces.
        #  '[ ]?' allows leading space

        self.pattern = regex.compile(r"""[ ]?\p{L}[\p{L}\p{M}]*|[ ]?\d+|[ ]?[^\s\p{L}\p{M}\d]+|\s+(?!\S)|\s+""")
        self.special_token_pattern = None
    
    def train_from_iterator(self, text_iterator, vocab_size):
        if vocab_size < 256:
            raise ValueError("Vocab size must be at least 256 (to cover all base bytes).")
        # Reset state to ensure clean training 
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.inverse_vocab = {bytes([i]): i for i in range(256)}
        self.bpe_merges = {}
        self.bpe_ranks = {}

        
        word_counts = Counter()
        # word frequencies, and each word is a tuple of BYTES
        # Example: it store somethis like this
                    # (32, 109, 100): 1654,
                    # (32, 99, 111, 100, 101): 4551,
                    # (32, 96): 187,
                    # (99, 111, 100, 101): 1544,
                    # (32, 105, 115): 157,
                    # (32, 105, 116): 178,


        print("Obtaining all word frequencies...")

        for text in tqdm(text_iterator):
            # Unicode normalization (crucial for non-english)
            # converts decomposed characters into composed form (NFC)
            text = unicodedata.normalize('NFC', text)
            words = regex.findall(self.pattern, text)

            for w in words:
                w_bytes = tuple(w.encode("utf-8"))      # Ex: "is" becomes (105, 115) 
                word_counts[w_bytes] += 1
    

        split_words = {
            tuple(w_bytes): count
            for w_bytes, count in word_counts.items()
        }
        
        pair_counts = Counter()
        # tells what pairs exist in what words
        pair_to_words = defaultdict(set)

        # Counting all adjacent pairs
        for ids, count in split_words.items():
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                pair_counts[pair] += count
                pair_to_words[pair].add(ids)
        

        pbar = tqdm(total=vocab_size - len(self.vocab), desc="Training BPE ...")

        while len(self.vocab) < vocab_size and pair_counts:

            best_pair = pair_counts.most_common(1)[0][0]
            
            # Register new token
            new_id = len(self.vocab)
            token1_bytes = self.vocab[best_pair[0]]
            token2_bytes = self.vocab[best_pair[1]]
            new_token_bytes = token1_bytes + token2_bytes
            
            self.bpe_merges[best_pair] = new_id
            self.bpe_ranks[best_pair] = new_id
            self.vocab[new_id] = new_token_bytes
            self.inverse_vocab[new_token_bytes] = new_id

            # Only merging in affected words which contain the 'best_pair'
            affected_words = list(pair_to_words[best_pair])

            # deleting the best_pair since its no longer
            # required to maintain
            pair_counts.pop(best_pair)
            pair_to_words.pop(best_pair)
            
            for ids in affected_words:
                count = split_words.pop(ids)


                for i in range(len(ids)-1):
                    p = (ids[i], ids[i + 1])

                    if p == best_pair:
                        continue    # we already removed this.

                    pair_counts[p] -= count
                    pair_to_words[p].discard(ids)
                    if pair_counts[p] <= 0:
                        pair_counts.pop(p, None)
                        pair_to_words.pop(p, None)

                # Merging the best pair in affected words
                # and updating the split_words.
                new_ids = []
                i = 0
                while i < len(ids):
                    if i + 1 < len(ids) and (ids[i], ids[i + 1]) == best_pair:
                        new_ids.append(new_id)
                        i += 2
                    else:
                        new_ids.append(ids[i])
                        i += 1

                new_ids = tuple(new_ids)
                split_words[new_ids] = split_words.get(new_ids, 0) + count

                # adding new pairs
                for i in range(len(new_ids) - 1):
                    p = (new_ids[i], new_ids[i + 1])
                    pair_counts[p] += count
                    pair_to_words[p].add(new_ids)


            pbar.update(1)
        
        pbar.close()

    def add_special_tokens(self, special_tokens):

        if self.special_token_dict is None:
            self.special_token_dict = {}
        
        for token in special_tokens:
            if token in self.special_token_dict:
                continue
            new_id = len(self.vocab)

            self.special_token_dict[token] = new_id
            self.vocab[new_id] = token.encode("utf-8")
            self.inverse_vocab[self.vocab[new_id]] = new_id
        
        # updating the pattern if an update happens
        self.update_special_token_pattern()
        
    def update_special_token_pattern(self):
        if self.add_special_tokens is None:
            return
        
        escaped = sorted(
            (regex.escape(tok) for tok in self.special_token_dict.keys()),
            key=len,
            reverse=True, # Longest special token should be found first.
        )

        pattern = "|".join(escaped)
        self.special_token_pattern = regex.compile(pattern)

    def _encode_text(self, text):
        text = unicodedata.normalize("NFC", text)
        words = regex.findall(self.pattern, text)
        
        token_ids = []
        for word in words:
            word_bytes = list(word.encode("utf-8"))
            
            word_ids = self.tokenize_with_bpe(word_bytes)
            token_ids.extend(word_ids)
            
        return token_ids
    
    def encode(self, text):

        if self.special_token_pattern is None:
            return self._encode_text(text)
        
        tokens = []
        pos = 0
        for str_match in self.special_token_pattern.finditer(text):
            start, end = str_match.span()

            if start > pos:
                chunk = text[pos:start]
                tokens.extend(self._encode_text(chunk))
            
            special_token = str_match.group(0)
            tokens.append(self.special_token_dict[special_token])

            pos = end

        # Handles both no special tokens case
        # and text after last special token
        if pos < len(text):
            tokens.extend(self._encode_text(text[pos:]))

        return tokens

    def tokenize_with_bpe(self, ids):
        while len(ids) >= 2:
            min_rank = float('inf')
            best_pair = None
            
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i+1])
                rank = self.bpe_ranks.get(pair, float('inf'))
                if rank < min_rank:
                    min_rank = rank
                    best_pair = pair
            
            if best_pair is None:
                break
                
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i+1]) == best_pair:
                    new_ids.append(self.bpe_merges[best_pair])
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
            
        return ids

    def decode(self, token_ids):
        # Concatenate all byte sequences
        byte_stream = b"".join([self.vocab[tid] for tid in token_ids])
        
        # Decode UTF-8 back to string
        # errors="replace" ensures we don't crash on partial bytes
        text = byte_stream.decode("utf-8", errors="replace")
        return text
    
    def save(self, dir):
        os.makedirs(dir, exist_ok=True)
        with open(f'{dir}/vocab.pkl', 'wb') as outfile:
            pickle.dump(self.vocab, outfile)
        with open(f'{dir}/bpe_merges.pkl', 'wb') as outfile:
            pickle.dump(self.bpe_merges, outfile)
        with open(f'{dir}/bpe_ranks.pkl', 'wb') as outfile:
            pickle.dump(self.bpe_ranks, outfile)
        with open(f'{dir}/special_token_dict.pkl', 'wb') as outfile:
            pickle.dump(self.special_token_dict, outfile)

    def load(self, dir):
        with open(f'{dir}/vocab.pkl', 'rb') as infile:
            self.vocab = pickle.load(infile)
        with open(f'{dir}/bpe_merges.pkl', 'rb') as infile:
            self.bpe_merges = pickle.load(infile)
        with open(f'{dir}/bpe_ranks.pkl', 'rb') as infile:
            self.bpe_ranks = pickle.load(infile)
        with open(f'{dir}/special_token_dict.pkl', 'rb') as infile:
            self.special_token_dict = pickle.load(infile)

        self.inverse_vocab = {value:key for key, value in self.vocab.items()}
        self.update_special_token_pattern()