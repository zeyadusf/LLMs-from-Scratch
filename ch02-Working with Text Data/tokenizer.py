import re
from typing import List, Any
from tqdm import tqdm

class Tokenizer:
    def __init__(self, path_text: str, remove_whitespace: bool = True):
        self.path_text = path_text
        self.remove_whitespace = remove_whitespace
        self.raw_text = self.read_txt(self.path_text)
        self.tokenized_txt = self.tokenize(self.raw_text, self.remove_whitespace)

    def __getitem__(self, item):
        return self.tokenized_txt[item]

    def __len__(self):
        print(f'Length of characters: {len(self.raw_text)}')
        print(f'Length of tokens: {len(self.tokenized_txt)}')
        return len(self.tokenized_txt)

    def read_txt(self, path: str) -> str:
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

    def tokenize(self, text: str, remove_whitespace: bool) -> List[str]:
        txt = re.split(r'([,.:;?_!@"()\']|--|\s)', text)
        if remove_whitespace:
            return [item.strip() for item in tqdm(txt) if item.strip()]
        else:
            return txt
    def count(self, token: str) -> int:
        return self.tokenized_txt.count(token)

if __name__ == "__main__":
    path_txt = r'LLMs-from-Scratch\ch02-Working with Text Data\data\the-verdict.txt'
    tokenize_txt = Tokenizer(path_text=path_txt, remove_whitespace=True)
    print(tokenize_txt[:30])
    print(tokenize_txt.count('--'))
    print(len(tokenize_txt))
