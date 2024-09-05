from datasets import register
from torch.utils.data import Dataset
from pathlib import Path

@register('ocr_img')
class ocr_dataset(Dataset):
    def __init__(self, path_split, phase='training'):
        self.split_file = path_split
        self.phase = phase
        self.dataset = []
        
        with open(self.split_file, 'r') as f:
            data = f.readlines()    
        for path in data:    
            path_imgs, split = path.split(';')
            path_imgs = path_imgs.strip()
            
            sample = {"img": path_imgs,
                      }
            
            if self.phase in split:
                self.dataset.append(sample)
                
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]
