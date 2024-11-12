import torch
import os
import re
import random
import json
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import functional as F

from multiprocessing import Pool

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

MAX_TOKEN_LENGTH = 15

class VideoRecord():
    def __init__(self, data, root_datapath):
        self._data = data
        self._path = os.path.join(root_datapath, 
                                  data["participant_id"], 
                                  "rgb_frames", 
                                  data["video_id"]
                                  )

    @property
    def path(self) -> str:
        return self._path

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame + 1  # +1 because end frame is inclusive

    @property
    def start_frame(self) -> int:
        return int(self._data["start_frame"])

    @property
    def end_frame(self) -> int:
        return int(self._data["stop_frame"])

    @property
    def label(self):
        return self._data["label"]

    @property
    def activity(self):
        temp = self._data["activity"]
        temp = temp.replace("-", " ")
        return temp

class EpicKitchen100(Dataset):
    def __init__(self, 
                split = "train",
                activity_set = None,
                vocab_encoder = None,
                tokenized_vocab = None,
                mean = [0., 0., 0.],
                std = [1., 1., 1.]
                ) -> None:
        super(Dataset, self).__init__()
        self.split = split
        self.activity_set = activity_set
        self.vocab_encoder = vocab_encoder
        self.tokenized_vocab = tokenized_vocab
        
        self.dataset_name = f"EPIC-Kitchen-100 Dataset {self.split}"
        self.image_path = "/data/EPIC-KITCHENS-100/images/"
        annotation_file = f"/data/EPIC-KITCHENS-100/annotations/EPIC_100_{self.split}_128_64.csv"
        self.df = loadDatasetDescription(annotation_file)
        self.vocab_encoder = json.load(open("src/dataset/epic_vocab.json"))
        self.updateDF()
        
        self.mean = mean
        self.std = std
                
        with Pool(processes=8) as pool:
            self.video_list = pool.starmap(createVideoRecord, [(row, self.image_path) for _, row in self.df.iterrows()])
        
        
        print(f"Dataset contains a total of {len(self.video_list)} clips")
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Args:
            idx: int
        """
        record: VideoRecord = self.video_list[idx]
        
        fileNames = [os.path.join(record.path, f"frame_{str(i).rjust(10, '0')}.jpg") for i in range(record.start_frame, record.end_frame+1)]
        
        images = []
        for fileName in fileNames:
            img = Image.open(fileName).convert('RGB')
            img = img.resize((64, 64), resample=Image.BICUBIC)
            tensor = F.pil_to_tensor(img)
            images.append(tensor)
            
        tensor = torch.stack(images)
        
        images = tensor.permute(1, 0, 2, 3)
        label = torch.tensor(record.label)
        
        tokens = torch.zeros(MAX_TOKEN_LENGTH, dtype=torch.long)
        temp = torch.tensor([el for x in record.activity.split() for el in self.vocab_encoder[x]], dtype=torch.long)
        tokens[:len(temp)] = temp

        images = images.type(torch.float32)
        images /= 255
        
        images = images.permute(1, 0, 2, 3)
        
        images = F.normalize(images, mean=self.mean, std=self.std)
        
        images = images.permute(1, 0, 2, 3)
        
        return {
            "tensor": images[:, 32:96],
            "tokens": tokens,
            "labels": label
        }
        
    def updateDF(self):
        self.df["length"] = self.df["stop_frame"] - self.df["start_frame"] + 1

        if self.activity_set is None:
            self.activity_set = list(set([re.sub(r'[-.,/]', ' ', x) for x in self.df["activity"]]))

        if self.tokenized_vocab is None:
            self.tokenized_vocab = torch.zeros(len(self.activity_set), MAX_TOKEN_LENGTH, dtype=torch.long)
            for i, item in enumerate(self.activity_set):
                temp = [el for x in item.split() for el in self.vocab_encoder[x]]
                temp = torch.tensor(temp, dtype=torch.long)
                self.tokenized_vocab[i, :len(temp)] = temp 
        
        self.df["label"] = self.df["activity"].apply(lambda x: self.activity_set.index(re.sub(r'[-.,/]', ' ', x)) if re.sub(r'[-.,/]', ' ', x) in self.activity_set else -1)

        sum_error_labels = sum(self.df["label"] < 0)
        print(f"Warning: Number of error labels: {sum_error_labels}")

        # filter error labels to avoid missmatch between train/test dataset and longer clips
        self.df = self.df[(self.df["label"] >= 0) & (127 < self.df["length"])]

def createVideoRecord(row, image_path):
    return VideoRecord(row, image_path)

def loadDatasetDescription(annotation_file):
    with open(annotation_file) as f:
        df = pd.read_csv(f, delimiter=',', encoding='utf-8', header=0)
        df = df[["participant_id", "video_id", "start_frame", "stop_frame", "verb", "noun", "activity"]]
        return df

def get_dataloader(dataset, batch_size, dev):
    """ Get a dataloader from a dataset.
    """
    shuffle = True if dataset.split == "train" else False
    drop_last = True if dataset.split == "train" else False
    dataloader = DataLoader(
        dataset if not dev else Subset(dataset, random.sample(range(len(dataset)), batch_size * 50)),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=True,  # set True when loading data on CPU and training on GPU
    )
    return dataloader

 
def prepare_data(batch_size=32, debug=False):
    """ Prepare data.
    """
    train_dataset = EpicKitchen100("train", mean=[0.3794, 0.3206, 0.2973], std=[0.2150, 0.2096, 0.2016])
    test_dataset = EpicKitchen100("test", mean=[0.3958, 0.3278, 0.3068], std=[0.2193, 0.2103, 0.2025], activity_set=train_dataset.activity_set, vocab_encoder=train_dataset.vocab_encoder, tokenized_vocab=train_dataset.tokenized_vocab)
    train_dl = get_dataloader(train_dataset, batch_size, debug)
    test_dl = get_dataloader(train_dataset, batch_size, True)
    eval_dl = get_dataloader(test_dataset, batch_size, debug)

    return {
        "train_dl": train_dl,
        "test_dl": test_dl,
        "eval_dl": eval_dl,
        "tokenized_vocab": train_dataset.tokenized_vocab
    }       

if __name__ == "__main__":
    dataset = EpicKitchen100("train")
    print(dataset.df.iloc[:5])
    
    it = dataset[5]
    print(it["tensor"].shape)
    print(it["tensor"][0][0])
    print(it["tokens"].shape)
    print(it["tokens"].dtype)
    print(len(dataset))
    print(dataset.tokenized_vocab.shape)
    
    data = prepare_data(32, 600)
    train_dl = data["train_dl"]
    test_dl = data["test_dl"]
    
    batch = next(iter(train_dl))