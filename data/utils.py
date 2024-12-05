import torch
from typing import Sequence, Tuple, Union


RawMSA = Sequence[Tuple[str, str]]


class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        batch_labels, seq_str_list = zip(*raw_batch)
        seq_encoded_list = [self.tokenizer(self._tokenize(seq_str)).input_ids for seq_str in seq_str_list]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        tokens = torch.empty(
            (batch_size, max_len),
            dtype=torch.int64,
        )
        tokens.fill_(self.tokenizer.pad_token_id)
        labels = []
        strs = []
        for i, (label, seq_str, seq_encoded) in enumerate(zip(batch_labels, seq_str_list, seq_encoded_list)):
            labels.append(label)
            strs.append(seq_str)
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[i, 0 : len(seq_encoded)] = seq

        return labels, strs, tokens

    def _tokenize(self, sequence):
        return " ".join(list(sequence))


class DataCollatorForMSA(BatchConverter):
    def msa_batch_convert(self, inputs: Union[Sequence[RawMSA], RawMSA]):
        # RawMSA: Sequence[label:str,acid_seq:str]
        if isinstance(inputs[0][0], str):
            # Input is a single MSA
            raw_batch: Sequence[RawMSA] = [inputs]  # type: ignore
        else:
            raw_batch = inputs  # type: ignore

        batch_size = len(raw_batch)
        max_alignments = max(len(msa) for msa in raw_batch)  # MSA的数量
        max_seqlen = max(len(msa[0][1]) for msa in raw_batch)  # MSA的每个序列长度

        tokens = torch.empty(
            (
                batch_size,
                max_alignments,
                max_seqlen + 1,
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.tokenizer.pad_token_id)
        labels = []
        strs = []

        for i, msa in enumerate(raw_batch):
            msa_seqlens = set(len(seq) for _, seq in msa)
            if not len(msa_seqlens) == 1:
                raise RuntimeError("Received unaligned sequences for input to MSA, all sequence " "lengths must be equal.")
            msa_labels, msa_strs, msa_tokens = super().__call__(msa)

            labels.append(msa_labels)
            strs.append(msa_strs)
            tokens[i, : msa_tokens.size(0), : msa_tokens.size(1)] = msa_tokens
        return tokens

    def __call__(self, batch):
        input_ids = self.msa_batch_convert([example["src"] for example in batch])
        labels = self.msa_batch_convert([example["tgt"] for example in batch])
        labels[labels == self.tokenizer.pad_token_id] = -100
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).type_as(input_ids)
        decoder_attention_mask = labels.ne(self.tokenizer.pad_token_id).type_as(input_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
        }
