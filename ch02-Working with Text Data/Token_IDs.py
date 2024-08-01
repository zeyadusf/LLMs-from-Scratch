from Tokenizing_text import Tokenizing,read_txt
from typing import List


class Token_IDs:
    def __init__(self, text: str, remove_whitespace: bool = True, add_special_chars: bool = False):
        self.tokenizing_obj = Tokenizing(text, remove_whitespace)
        self.vocab = self.build_vocab(add_special_chars)

        self.add_special_chars = add_special_chars
        self.str_to_int = {s: i for i, s in enumerate(self.vocab)}
        self.int_to_str = {i: s for i, s in enumerate(self.vocab)}

    def build_vocab(self, add_special_chars: bool) -> List[str]:
        vocab_set = sorted(list(self.tokenizing_obj.get_set_of_vocab().keys()))

        return vocab_set

    def encode(self, text: str) -> List[int]:
        tokenized_text = self.tokenizing_obj.tokenize(text, remove_whitespace=self.tokenizing_obj.remove_whitespace)
        return [self.str_to_int[token] for token in tokenized_text]

    def decode(self, ids: List[int]) -> str:
        return " ".join([self.int_to_str[i] for i in ids])
if __name__ == "__main__":
    path_txt = r'E:\Courses\LLMs\LLMs-from-Scratch\ch02-Working with Text Data\data\the-verdict.txt'
    text=read_txt(path_txt)
    token_ids_obj = Token_IDs(text, remove_whitespace=True, add_special_chars=False)

    txt = "It's the last he painted, "
    ids = token_ids_obj.encode(txt)
    print("Encoded IDs:", ids)
    decoded_text = token_ids_obj.decode(ids)
    print("Decoded Text:", decoded_text)