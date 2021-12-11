import argparse

from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast
import pytorch_lightning as pl
import pandas as pd
import numpy as np


class RapDataset(Dataset):
    def __init__(self, filepath, tok_vocab, max_seq_len=128) -> None:
        self.filepath = filepath
        self.data = self.cos_threshold(pd.read_csv(self.filepath))
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.max_seq_len = max_seq_len
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tok_vocab,
            bos_token=self.bos_token, eos_token=self.eos_token, unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')

    def __len__(self):
        return len(self.data)

    def make_input_id_mask(self, tokens, index):
        input_id = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_id)
        if len(input_id) < self.max_seq_len:
            while len(input_id) < self.max_seq_len:
                input_id += [self.tokenizer.pad_token_id]
                attention_mask += [0]
        else:
            # logging.warning(f'exceed max_seq_len for given article : {index}')
            input_id = input_id[:self.max_seq_len - 1] + [
                self.tokenizer.eos_token_id]
            attention_mask = attention_mask[:self.max_seq_len]
        return input_id, attention_mask

    def cos_threshold(self, data):

        label_col = 'cos_sim'
        threshold_start = 0.7
        threshold_end = 0.9
        if label_col in data.columns:
            return data[(data[label_col] >= threshold_start) & (data[label_col] <= threshold_end)]

        else:
            return data

    def __getitem__(self, index):
        record = self.data.iloc[index]
        q, a = record['query'], record['corpus_idx']
        q_tokens = [self.bos_token] + \
            self.tokenizer.tokenize(q) + [self.eos_token]
        a_tokens = [self.bos_token] + \
            self.tokenizer.tokenize(a) + [self.eos_token]
        encoder_input_id, encoder_attention_mask = self.make_input_id_mask(
            q_tokens, index)
        decoder_input_id, decoder_attention_mask = self.make_input_id_mask(
            a_tokens, index)
        labels = self.tokenizer.convert_tokens_to_ids(
            a_tokens[1:(self.max_seq_len + 1)])
        if len(labels) < self.max_seq_len:
            while len(labels) < self.max_seq_len:
                # for cross entropy loss masking
                labels += [-100]
        return {'input_ids': np.array(encoder_input_id, dtype=np.int_),
                'attention_mask': np.array(encoder_attention_mask, dtype=np.float_),
                'decoder_input_ids': np.array(decoder_input_id, dtype=np.int_),
                'decoder_attention_mask': np.array(decoder_attention_mask, dtype=np.float_),
                'labels': np.array(labels, dtype=np.int_)}


class RapDataModule(pl.LightningDataModule):
    def __init__(self, train_file,
                 test_file, tok_vocab,
                 max_seq_len=128,
                 batch_size=32,
                 num_workers=5):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.train_file_path = train_file
        self.test_file_path = test_file
        self.tok_vocab = tok_vocab
        self.num_workers = num_workers

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers',
                            type=int,
                            default=5,
                            help='num of worker for dataloader')
        return parser

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        self.train = RapDataset(self.train_file_path,
                                 self.tok_vocab,
                                 self.max_seq_len)
        self.test = RapDataset(self.test_file_path,
                                self.tok_vocab,
                                self.max_seq_len)

    def train_dataloader(self):
        train = DataLoader(self.train,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers, shuffle=True)
        return train

    def val_dataloader(self):
        val = DataLoader(self.test,
                         batch_size=self.batch_size,
                         num_workers=self.num_workers, shuffle=False)
        return val

    def test_dataloader(self):
        test = DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)
        return test