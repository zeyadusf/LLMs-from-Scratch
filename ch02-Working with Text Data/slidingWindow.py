import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class LLMDataset(Dataset):

    def __init__(self, txt, tokenizer, max_length: int, stride: int):
        self.tokenizer = tokenizer
        token_ids = tokenizer.encode(txt)
        self.input_ids = []
        self.target_ids = []

        for i in tqdm(range(0, len(token_ids)-max_length, stride)):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def llm_dataloader(txt, tokenizer, batch_size, max_length, stride,
                   shuffle: bool = True, drop_last: bool = True):

    llmdataset = LLMDataset(txt, tokenizer, max_length, stride)
    llmdataloader = DataLoader(llmdataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return llmdataloader

