# =============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from __future__ import annotations

import json
import tqdm
from pathlib import Path


class QwenTokenizer:
    def __init__(self, dir_model: Path, export_path: Path | None = None, export_tokenizer_json: bool = False) -> None:
        self.dir_model = dir_model
        self.export_path = export_path
        self.export_tokenizer_json = export_tokenizer_json

    # tiktoken allows representation of tokens as byte arrays and does not guarantee tokens to be valid UTF-8 bytes
    @staticmethod
    def token_bytes_to_string(b: bytes):
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
        byte_encoder = bytes_to_unicode()
        return ''.join([byte_encoder[ord(char)] for char in b.decode('latin-1')])

    # to generate BPE merges from tiktoken, for each token in vocab we iteratively try to merge consecutive sub-tokens
    # (initially starting with all consecutive byte pairs). If the newly merged sub-token is present in the vocab and
    # its token id is less than the original token (of which this merge is a sub-token) we add it to the merge list. If
    # at a given stage multiple sub-token pairs in consideration are present in the vocab, then we take the pair whose
    # merge gives us a token with the lowest token id
    @staticmethod
    def _extract(mergeable_ranks: dict[bytes, int], disable: bool = True) -> tuple[dict[str, int], list[tuple]]:
        merges = []
        vocab = {}
        for token, rank in tqdm.tqdm(mergeable_ranks.items(), total = len(mergeable_ranks), disable = disable):
            vocab[QwenTokenizer.token_bytes_to_string(token)] = rank
            if len(token) == 1:
                continue
            max_rank = rank
            pieces = [bytes([byte]) for byte in token]
            from itertools import count
            for _ in count():
                min_idx  = None
                min_rank = None
                current_merges = [(piece_l, piece_r) for piece_l, piece_r in zip(pieces[:-1], pieces[1:])]
                for idx in range(len(current_merges)):
                    merge      = current_merges[idx][0] + current_merges[idx][1]
                    rank_merge = mergeable_ranks.get(merge, None)
                    if rank_merge:
                        if min_rank is None or rank_merge < min_rank:
                            min_idx = idx
                            min_rank = rank_merge
                if min_rank is None:
                    break
                elif min_rank >= max_rank:
                    break
                assert min_idx is not None
                pieces[min_idx:min_idx + 2] = [pieces[min_idx] + pieces[min_idx + 1]]
            assert len(pieces) == 2
            merges.append((pieces[0], pieces[1], mergeable_ranks.get(pieces[0] + pieces[1])))
        merges = sorted(merges, key = lambda merge: merge[2])
        merges = [(QwenTokenizer.token_bytes_to_string(piece_l), QwenTokenizer.token_bytes_to_string(piece_r)) for (piece_l, piece_r, _) in merges]
        return vocab, merges

    def _create_qwen_bpe(self, disable: bool = True) -> dict[str, any]:
        dir_model = self.dir_model
        tokens: list[str] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(dir_model), trust_remote_code = True)
        vocab_size = json.loads(open(str(dir_model / "config.json"), "rb").read())["vocab_size"]
        assert max(tokenizer.get_vocab().values()) < vocab_size

        vocab, merges = QwenTokenizer._extract(tokenizer.mergeable_ranks, disable = disable)

        added_vocab = tokenizer.special_tokens
        added_vocab = dict(sorted(added_vocab.items(), key = lambda x : x[1]))
        added_vocab = list(added_vocab.keys())

        if (len(added_vocab) + len(vocab)) < vocab_size:
            for i in range(vocab_size - (len(added_vocab) + len(vocab))):
                added_vocab.append(f"[PAD{i}]")

        # Create a Tokenizer object
        from tokenizers import Tokenizer, Regex, models, normalizers, decoders, pre_tokenizers, processors
        custom_tokenizer = Tokenizer(models.BPE(vocab = vocab, merges = merges))
        custom_tokenizer.add_special_tokens(added_vocab)

        custom_normalizer = normalizers.NFC()
        PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
        custom_pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(pattern = Regex(PAT_STR), behavior = "isolated", invert = False),
                                                        pre_tokenizers.ByteLevel(add_prefix_space = False, use_regex = False)])
        custom_post_processor = processors.ByteLevel(trim_offsets = False)
        custom_decoder = decoders.ByteLevel()

        custom_tokenizer.normalizer = custom_normalizer
        custom_tokenizer.pre_tokenizer = custom_pre_tokenizer
        custom_tokenizer.post_processor = custom_post_processor
        custom_tokenizer.decoder = custom_decoder

        custom_tokenizer = json.loads(custom_tokenizer.to_str())
        custom_tokenizer["pre_tokenizer"]["pretokenizers"][1]["trim_offsets"] = False
        custom_tokenizer["post_processor"]["add_prefix_space"] = False
        custom_tokenizer["post_processor"]["use_regex"] = False
        custom_tokenizer["decoder"]["add_prefix_space"] = False
        custom_tokenizer["decoder"]["trim_offsets"] = False
        custom_tokenizer["decoder"]["use_regex"] = False

        if self.export_tokenizer_json:
            if self.export_path is not None:
                json.dump(custom_tokenizer, open(str(self.export_path / "tokenizer.json"), "w", encoding = "utf-8"), indent = 2, ensure_ascii = False)
            else:
                raise NotADirectoryError('Output Directory not Specified')

        return custom_tokenizer

class BaichuanTokenizer:
    def __init__(self, dir_model: Path, export_path: Path | None = None, export_tokenizer_json: bool = False) -> None:
        self.export_path = export_path
        self.export_tokenizer_json = export_tokenizer_json
        self.model_path = dir_model / "tokenizer.model"
        from sentencepiece import SentencePieceProcessor
        self.sp = SentencePieceProcessor(str(self.model_path))

    # to generate BPE merges from sentencepiece, we take a cartesian product of the token list with itself. We then
    # eliminate all the tokens not in the vocab from the token list generated by the cartesian product. The list is
    # then sorted by token id to generate the final merges
    @staticmethod
    def _extract(mergeable_ranks: dict[str, int]) -> tuple[dict[str, int], list[tuple]]:
        # Create the BPE merges
        vocab = mergeable_ranks
        merges = []
        for piece_l in tqdm.tqdm(mergeable_ranks.keys(), total = len(mergeable_ranks)):
            merges.extend([(piece_l, piece_r, rank_merge) for piece_r in mergeable_ranks.keys() if (rank_merge := mergeable_ranks.get(piece_l + piece_r)) is not None])
        merges = sorted(merges, key = lambda merge: merge[2])
        merges = [(piece_l, piece_r) for (piece_l, piece_r, _) in merges]

        return vocab, merges

    def _create_baichuan_bpe(self) -> dict[str, any]:
        mergeable_ranks = {self.sp.id_to_piece(index): index for index in range(self.sp.GetPieceSize())}
        vocab, merges = BaichuanTokenizer._extract(mergeable_ranks)

        # Create the Tokenizer
        from tokenizers import Tokenizer, models, normalizers, decoders
        custom_tokenizer = Tokenizer(models.BPE(vocab = vocab, merges = merges))

        # Create the Normalizer and Decoder pipelines
        custom_normalizer = normalizers.Replace(" ", "▁")
        custom_decoder = decoders.Replace("▁", " ")

        custom_tokenizer.normalizer = custom_normalizer
        custom_tokenizer.decoder = custom_decoder

        custom_tokenizer = json.loads(custom_tokenizer.to_str())

        if self.export_tokenizer_json:
            if self.export_path is not None:
                json.dump(custom_tokenizer, open(str(self.export_path / "tokenizer.json"), "w", encoding = "utf-8"), indent = 2, ensure_ascii = False)
            else:
                raise NotADirectoryError('Output Directory not Specified')

        return custom_tokenizer
