import re
from typing import List
from tqdm import tqdm

class Tokenizer:
    """
    A class for tokenizing text and building a vocabulary from the tokens.

    Attributes:
    -----------
    text : str
        The input text to be tokenized.
    remove_whitespace : bool
        Flag to indicate whether to remove whitespace from tokens.
    add_special_chars : bool
        Flag to indicate whether to add special characters to the vocabulary.
    special_chars : List[str]
        A list of special characters to be added to the vocabulary.
    tokenized_txt : List[str]
        The tokenized representation of the input text.
    vocab : List[str]
        The vocabulary built from the tokenized text.
    str_to_int : Dict[str, int]
        A mapping from tokens to their integer indices in the vocabulary.
    int_to_str : Dict[int, str]
        A mapping from integer indices to tokens in the vocabulary.
    """

    def __init__(self, text: str, remove_whitespace: bool = True, add_special_chars: bool = False):
        """
        Initializes the Tokenizer with the given text and options.

        Parameters:
        -----------
        text : str
            The input text to be tokenized.
        remove_whitespace : bool
            If True, whitespace will be removed from tokens. Default is True.
        add_special_chars : bool
            If True, special characters will be added to the vocabulary. Default is False.
        """
        self.text = text
        self.remove_whitespace = remove_whitespace
        self.add_special_chars = add_special_chars
        self.special_chars = ['<UNK>', '<EOS>']
        self.tokenized_txt = self.tokenize(self.text, self.remove_whitespace)
        self.vocab = self.build_vocab()
        self.str_to_int = {token: idx for idx, token in enumerate(self.vocab)}
        self.int_to_str = {idx: token for token, idx in self.str_to_int.items()}

    def tokenize(self, text: str, remove_whitespace: bool) -> List[str]:
        """
        Tokenizes the input text.

        Parameters:
        -----------
        text : str
            The input text to be tokenized.
        remove_whitespace : bool
            If True, whitespace will be removed from tokens.

        Returns:
        --------
        List[str]
            The tokenized representation of the text.
        """
        tokens = re.split(r'([,.:;?_!@"()\']|--|\s)', text)
        if remove_whitespace:
            return [token.strip() for token in tqdm(tokens) if token.strip()]
        return tokens

    def build_vocab(self) -> List[str]:
        """
        Builds the vocabulary from the tokenized text.

        Returns:
        --------
        List[str]
            The vocabulary built from the tokenized text.
        """
        if self.add_special_chars:
            return sorted(set(self.tokenized_txt + self.special_chars))
        else:
            return sorted(set(self.tokenized_txt))

    def vocab_size(self) -> int:
        """
        Returns the size of the vocabulary.

        Returns:
        --------
        int
            The size of the vocabulary.
        """
        return len(self.vocab)

    def set_special_chars(self, special_char: List[str]):
        """
        Sets additional special characters to be included in the vocabulary.

        Parameters:
        -----------
        special_char : List[str]
            A list of special characters to add to the vocabulary.
        """
        self.special_chars.extend(special_char)

    def encode(self, text: str) -> List[int]:
        """
        Encodes the input text into a list of integers based on the vocabulary.

        Parameters:
        -----------
        text : str
            The input text to be encoded.

        Returns:
        --------
        List[int]
            The list of integers representing the tokens in the text.
        """
        tokenized_text = self.tokenize(text, self.remove_whitespace)
        if self.add_special_chars:
            return [self.str_to_int.get(token, self.str_to_int.get('<UNK>')) for token in tokenized_text]
        else:
            return [self.str_to_int[token] for token in tokenized_text]

    def decode(self, token_ids: List[int]) -> str:
        """
        Decodes a list of integers into a string using the vocabulary.

        Parameters:
        -----------
        token_ids : List[int]
            The list of integers to be decoded.

        Returns:
        --------
        str
            The decoded string.
        """
        if self.add_special_chars:
            return " ".join([self.int_to_str.get(id, '<UNK>') for id in token_ids])
        else:
            return " ".join([self.int_to_str[id] for id in token_ids])

    def get_token_count(self, token: str) -> int:
        """
        Returns the count of a specific token in the tokenized text.

        Parameters:
        -----------
        token : str
            The token to count in the tokenized text.

        Returns:
        --------
        int
            The count of the token in the tokenized text.
        """
        return self.tokenized_txt.count(token)

    def __getitem__(self, idx: int) -> str:
        """
        Gets the token at the specified index.

        Parameters:
        -----------
        idx : int
            The index of the token.

        Returns:
        --------
        str
            The token at the specified index.
        """
        return self.tokenized_txt[idx]

    def __len__(self) -> int:
        """
        Returns the number of tokens in the tokenized text.

        Returns:
        --------
        int
            The number of tokens.
        """
        return len(self.tokenized_txt)


def read_txt(path: str) -> str:
    """
    Reads the content of a text file.

    Parameters:
    -----------
    path : str
        The path to the text file.

    Returns:
    --------
    str
        The content of the text file as a string. Returns an empty string if the file is not found.
    """
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
    tokenizer = Tokenizer(text, remove_whitespace=True, add_special_chars=True)

    sample_text = "It's the last he painted, HEllo"
    encoded_ids = tokenizer.encode(sample_text)
    print("Encoded IDs:", encoded_ids)
    decoded_text = tokenizer.decode(encoded_ids)
    print("Decoded Text:", decoded_text)
    print('-*' * 10)
    print(f"Vocabulary Size: {tokenizer.vocab_size()}")
    print(f"Token Count for '--': {tokenizer.get_token_count('--')}")
    print(f"First 30 Tokens: {tokenizer[:30]}")
