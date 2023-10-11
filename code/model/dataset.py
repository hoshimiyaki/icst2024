import gzip
import pathlib
import pickle

import numpy as np
import torch
import random

def gen_pad_collate(batch):
    l_batch=list(zip(*batch))
    X,Y=l_batch
    x1,x2=list(zip(*X))
    single=[17, 59, 15, 69, 44, 47, 64] 
    xidx=[]
    yidx=[]
    for code in x1:
        for row in code:
            if 27 in row or row[:7]==single:
                xidx.append(code.index(row))
                break
    if isinstance(Y[0],int):
        return torch.LongTensor(x1),torch.FloatTensor(x2),torch.FloatTensor(Y),torch.LongTensor(xidx)
    y1,y2=list(zip(*Y))
    for code in y1:
        for row in code:
            if 27 in row or row[:7]==single:
                yidx.append(code.index(row))
                break
    return torch.LongTensor(x1),torch.FloatTensor(x2),torch.LongTensor(y1),torch.FloatTensor(y2),torch.LongTensor(xidx),torch.LongTensor(yidx)


class FusionDataset(torch.utils.data.Dataset):
    """Defines a Dataset of unsupervised programs stored in pickle format."""

    def __init__(
        self,
        pos_path,
        neg_path,
        random_seed,
        re_shuffle=False
    ):
        """Create a JSONLinesDataset given a path and field mapping dictionary.
        Arguments:
            path (str): Path to the data file. Must be in .pickle format.
        """
        super().__init__()
        with open(pos_path, "rb") as file:
            self.pos_data= pickle.load(file)
        with open(neg_path, "rb") as file:
            self.neg_data= pickle.load(file)
        assert len(self.pos_data)==len(self.neg_data)
        if re_shuffle:
            random.seed(random_seed)
            random.shuffle(self.pos_data)
            random.seed(random_seed)
            random.shuffle(self.neg_data)
        

    def __len__(self):
        return len(self.pos_data)

    def __getitem__(self, index):
        # print("{},{},{}".format(len(self.pos_data[index]),len(self.pos_data[index][0]),len(self.pos_data[index][0][0])))
        return  self.pos_data[index],self.neg_data[index]

class CombineDataset(torch.utils.data.Dataset):
    """Defines a Dataset of unsupervised programs stored in pickle format."""

    def __init__(
        self,
        pos_path,
        neg_path,
        random_seed,
        re_shuffle=False
    ):
        """Create a JSONLinesDataset given a path and field mapping dictionary.
        Arguments:
            path (str): Path to the data file. Must be in .pickle format.
        """
        super().__init__()
        with open(pos_path, "rb") as file:
            self.pos_data= pickle.load(file)
        with open(neg_path, "rb") as file:
            self.neg_data= pickle.load(file)
        
        self.examples=self.pos_data+self.neg_data
        if re_shuffle:
            random.seed(random_seed)
            random.shuffle(self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return  self.examples[index]