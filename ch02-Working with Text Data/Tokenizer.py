import re
from typing import List
from tqdm import tqdm

class Tokenizer:
    def __init__(self, text: str, remove_whitespace: bool = True):
        self.text = text
        self.remove_whitespace = remove_whitespace
        self.tokenized_txt = self.tokenize(self.text, self.remove_whitespace)
        self.vocab = self.build_vocab()
        self.str_to_int = {token: idx for idx, token in enumerate(self.vocab)}
        self.int_to_str = {idx: token for token, idx in self.str_to_int.items()}

    def tokenize(self, text: str, remove_whitespace: bool) -> List[str]:
        tokens = re.split(r'([,.:;?_!@"()\']|--|\s)', text)
        if remove_whitespace:
            return [token.strip() for token in tqdm(tokens) if token.strip()]
        return tokens

    def build_vocab(self) -> List[str]:
        return sorted(set(self.tokenized_txt))

    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> List[int]:
        tokenized_text = self.tokenize(text, self.remove_whitespace)
        return [self.str_to_int.get(token, self.str_to_int.get('<UNK>')) for token in tokenized_text]

    def decode(self, token_ids: List[int]) -> str:
        return " ".join([self.int_to_str.get(id, '<UNK>') for id in token_ids])

    def get_token_count(self, token: str) -> int:
        return self.tokenized_txt.count(token)

    def __getitem__(self, idx):
        return self.tokenized_txt[idx]

    def __len__(self):
        return len(self.tokenized_txt)

def read_txt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        return raw_text
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return ""
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""

if __name__ == "__main__":
    path_txt = r'E:\Courses\LLMs\LLMs-from-Scratch\ch02-Working with Text Data\data\the-verdict.txt'
    text = read_txt(path_txt)
    tokenizer = Tokenizer(text, remove_whitespace=True)

    sample_text = "It's the last he painted, HEllo"
    encoded_ids = tokenizer.encode(sample_text)
    print("Encoded IDs:", encoded_ids)
    decoded_text = tokenizer.decode(encoded_ids)
    print("Decoded Text:", decoded_text)

    print(f"Vocabulary Size: {tokenizer.vocab_size()}")
    print(f"Token Count for '--': {tokenizer.get_token_count('--')}")
    print(f"First 30 Tokens: {tokenizer[:30]}")
