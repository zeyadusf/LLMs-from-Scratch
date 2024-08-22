import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class LLMDataset(Dataset):
    """
    A custom Dataset for processing text data into input and target sequences for language modeling.

    Args:
        txt (str): The input text to be tokenized and processed.
        tokenizer (Tokenizer): The tokenizer to be used for encoding the text.
        max_length (int): The maximum length of each input sequence.
        stride (int): The number of tokens to skip between sequences.
    """

    def __init__(self, txt, tokenizer, max_length: int, stride: int):
        self.tokenizer = tokenizer
        token_ids = tokenizer.encode(txt)
        self.input_ids = []
        self.target_ids = []

        for i in tqdm(range(0, len(token_ids) - max_length, stride)):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        Retrieves the input and target sequence at the specified index.

        Args:
            idx (int): The index of the sequence to retrieve.

        Returns:
            tuple: (input_ids, target_ids) where both are tensors.
        """
        return self.input_ids[idx], self.target_ids[idx]


def llm_dataloader(txt, tokenizer, batch_size: int, max_length: int, stride: int,
                   shuffle: bool = True, drop_last: bool = True):
    """
    Creates a DataLoader for the LLMDataset.

    Args:
        txt (str): The input text to be tokenized and processed.
        tokenizer (Tokenizer): The tokenizer to be used for encoding the text.
        batch_size (int): The number of samples per batch to load.
        max_length (int): The maximum length of each input sequence.
        stride (int): The number of tokens to skip between sequences.
        shuffle (bool, optional): Whether to shuffle the data at every epoch. Defaults to True.
        drop_last (bool, optional): Whether to drop the last incomplete batch. Defaults to True.

    Returns:
        DataLoader: A DataLoader instance for the LLMDataset.
    """
    llmdataset = LLMDataset(txt, tokenizer, max_length, stride)
    llmdataloader = DataLoader(llmdataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return llmdataloader
