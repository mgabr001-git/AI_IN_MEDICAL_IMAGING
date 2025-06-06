from pathlib import Path
import torch
import numpy as np
import pandas as pd
# import imgaug
# from imgaug.augmentables.bbs import BoundingBox

# import torchvision

from torchvision.tv_tensors import BoundingBoxes
from torchvision.transforms import v2

mu = 0.49
std_mg = 0.082

class CardiacDataset(torch.utils.data.Dataset):

    def __init__(self, path_to_labels_csv, patients, root_path, augs):
        
        self.labels = pd.read_csv(path_to_labels_csv)
        
        self.patients = np.load(patients)
        self.root_path = Path(root_path)
        self.augment = augs
        
    def  __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.patients)
        
    def __getitem__(self, idx):
        """
        Returns an image paired with bbox around the heart
        """
        patient = self.patients[idx]
        # Get data according to index
        data = self.labels[self.labels["name"]==patient]
        
        # Get entries of given patient
        # Extract coordinates
        
        x_min = data["x0"].item()
        y_min = data["y0"].item()
        x_max = x_min + data["w"].item()  # get xmax from width
        y_max = y_min + data["h"].item()  # get ymax from height
        bbox = [x_min, y_min, x_max, y_max]


        # Load file and convert to float32
        file_path = self.root_path/patient  # Create the path to the file
        img = np.load(f"{file_path}.npy").astype(np.float32)
       
        # img = torch.tensor(img)

        
        # Apply imgaug augmentations to image and bounding box
        if self.augment:
            
            bb = BoundingBoxes(
            torch.tensor([bbox]),
            format="XYXY",
            canvas_size=(224, 224)
            )


            ###################IMPORTANT###################
            # Fix for https://discuss.pytorch.org/t/dataloader-workers-generate-the-same-random-augmentations/28830/2
            # https://github.com/pytorch/pytorch/issues/5059
            # random_seed = torch.randint(0, 1000000, (1,)).item()
            # imgaug.seed(random_seed) 
            #####################################################

            # random_seed = torch.randint(0, 1000000, (1,)).item()
            # torch.manual_seed(random_seed) 

            img, aug_bbox  = self.augment(img, bb)
            bbox = aug_bbox[0][0], aug_bbox[0][1], aug_bbox[0][2], aug_bbox[0][3]
        # Normalize the image according to the values computed in Preprocessing

        img = (img - mu) / std_mg

        img = torch.tensor(img).unsqueeze(0)
        img = torch.tensor(img)
        bbox  = torch.tensor(bbox)
        return img, bbox
        
