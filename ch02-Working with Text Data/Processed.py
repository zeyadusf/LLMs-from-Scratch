import re
from typing import List
from tqdm import tqdm


class Processed:
    def __init__(self, text: str, remove_whitespace: bool = True):
        self.text = text
        self.remove_whitespace = remove_whitespace
        self.tokenized_txt = self.tokenize(self.text, self.remove_whitespace)

    def __getitem__(self, item):
        return self.tokenized_txt[item]

    def __len__(self):
        return len(self.tokenized_txt)

    def get_set_of_vocab(self):
        all_vocab = sorted(list(set(self.tokenized_txt)))
        return {token: integer for integer, token in enumerate(all_vocab)}

    def vocab_size(self):
        return len(self.get_set_of_vocab())

    def tokenize(self, text: str, remove_whitespace: bool) -> List[str]:
        txt = re.split(r'([,.:;?_!@"()\']|--|\s)', text)
        if remove_whitespace:
            return [item.strip() for item in tqdm(txt) if item.strip()]
        else:
            return txt

    def count(self, token: str) -> int:
        return self.tokenized_txt.count(token)


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
    text=read_txt(path_txt)
    tokenize_txt = Processed(text, remove_whitespace=True)
    print(tokenize_txt[:30])
    print(tokenize_txt.count('--'))
    print(len(tokenize_txt))
    print(tokenize_txt.get_set_of_vocab())
    print(tokenize_txt.vocab_size())

