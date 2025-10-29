from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import config

#定义Dataset
class My_Dataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_json(data_path, lines=True, orient='records').to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        input_tensor = torch.tensor(self.data[idx]['review'],dtype=torch.long)
        target_tensor = torch.tensor(self.data[idx]['label'],dtype=torch.float)
        return input_tensor, target_tensor

#定义dataloader
def get_dataloader():
    train_datapath = config.PROCESSED_DATA_DIR/'train_dataset.json'
    eval_datapath = config.PROCESSED_DATA_DIR/'test_dataset.json'
    train_dataset = My_Dataset(train_datapath)
    eval_dataset = My_Dataset(eval_datapath)
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    return train_dataloader, eval_dataloader

