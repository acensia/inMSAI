from typing import Any
from torchvision.datasets import VOCSegmentation
from torch.utils.data import Dataset
import cv2

class customVOCSeg(VOCSegmentation):
    def __init__(self, root, mode="train", transforms=None) -> None:
        super().__init__(root=root, image_set=mode, download=True, transforms=transforms)
        
    def __getitem__(self, idx: Any) -> Any:
        img = cv2.imread(self.images[idx])
        mask = cv2.imread(self.masks[idx])
        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            #cv2 -> not torch transform
            #albumentation
            
            img = augmented['image']
            mask = augmented['mask']
        
        
        
        return img, mask
    
    # __len__ 부모 클래스에서 정의됨
    
if __name__=="__main__":
    # VOCSegmentation()
    dataset = customVOCSeg()
    for item in dataset:
        img, mask = item
    
    
    pass

