from pathlib import Path

import torch
import numpy as np

from torchvision.tv_tensors import Mask
# from torchvision.transforms import v2


class CardiacDataset(torch.utils.data.Dataset):
    def __init__(self, root, augment_params):
        self.all_files = self.extract_files(root)
        self.augment_params = augment_params
    
    @staticmethod   # no need for it to acept self, as it does not access any class atributes
    def extract_files(root):
        """
        Extract the paths to all slices given the root path (ends with train or val)
        """
        files = []
        for subject in root.glob("*"):   # Iterate over the subjects
            slice_path = subject/"data"  # Get the slices for current subject
            for slice in slice_path.glob("*.npy"):
                files.append(slice)

        return files
    
    
    @staticmethod
    def change_img_to_label_path(path):
        """
        Replace data with mask to get the masks
        """
        parts = list(path.parts)
        parts[parts.index("data")] = "masks"

        return Path(*parts)

    def augment(self, img, mask):
        """
        Augments slice and segmentation mask in the exact same way
        Note the manual seed initialization
        """
        ###################IMPORTANT###################
        # Fix for https://discuss.pytorch.org/t/dataloader-workers-generate-the-same-random-augmentations/28830/2
        # random_seed = torch.randint(0, 1000000, (1,)).item()
        # imgaug.seed(random_seed)
        #####################################################
        
        # v2 transforms automatically handles this
        img =torch.tensor([img])
        mm =  Mask(torch.tensor([mask]))    # create an instance of Mask. this is necessary for v2 transforms to handle identically both the img and the mask

        aug_img, aug_mask  = self.augment_params(img, mm)
        
        # aug_img = aug_img.unsqueeze(0)
        # aug_mask = aug_mask.unsqueeze(0)
        
        aug_img =torch.tensor(aug_img)
        aug_mask=torch.tensor(aug_mask)

        # aug_img = torch.tensor(np.array(aug_img).astype('float32'))
        # aug_mask = torch.tensor(np.array(aug_mask).astype('float32')) 

        # aug_img = np.array(aug_img).astype('float32')
        # aug_mask = np.array(aug_mask).astype('float32')

        return aug_img, aug_mask

    
    def __len__(self):
        """
        Return the length of the dataset (length of all files)
        """
        return len(self.all_files)
    
    
    def __getitem__(self, idx):
        """
        Given an index return the (augmented) slice and corresponding mask
        Add another dimension for pytorch
        """
        file_path = self.all_files[idx]
        mask_path = self.change_img_to_label_path(file_path)
        slice = np.load(file_path).astype(np.float32)  # Convert to float for torch
        mask = np.load(mask_path)
        
        if self.augment_params:
            slice, mask = self.augment(slice, mask)
        
        # Note that pytorch expects the input of shape BxCxHxW, where B corresponds to the batch size, C to the channels, H to the height and W to Width.
        # As our data is of shape (HxW) we need to manually add the C axis by using expand_dims.
        # The batch dimension is later added by the dataloader

        ### torchvision v2 transforms automatically returns a 3D tensor [1,256,256], 
        # return np.expand_dims(slice, 0), np.expand_dims(mask, 0)     # no need to expand
    
        return slice, mask
        # return np.array(slice), np.array(mask)  









# class CardiacDataset(torch.utils.data.Dataset):
#     def __init__(self, root, augment_params):
#         self.all_files = self.extract_files(root)
#         self.augment_params = augment_params
    
#     @staticmethod
#     def extract_files(root):
#         """
#         Extract the paths to all slices given the root path (ends with train or val)
#         """
#         files = []
#         for subject in root.glob("*"):   # Iterate over the subjects
#             slice_path = subject/"data"  # Get the slices for current subject
#             for slice in slice_path.glob("*"):
#                 files.append(slice)
#         return files
    
    
#     @staticmethod
#     def change_img_to_label_path(path):
#         """
#         Replace data with mask to get the masks
#         """
#         parts = list(path.parts)
#         parts[parts.index("data")] = "masks"
#         return Path(*parts)

#     def augment(self, slice, mask):
#         """
#         Augments slice and segmentation mask in the exact same way
#         Note the manual seed initialization
#         """
#         ###################IMPORTANT###################
#         # Fix for https://discuss.pytorch.org/t/dataloader-workers-generate-the-same-random-augmentations/28830/2
#         random_seed = torch.randint(0, 1000000, (1,))[0].item()
#         imgaug.seed(random_seed)
#         #####################################################
#         mask = SegmentationMapsOnImage(mask, mask.shape)
#         slice_aug, mask_aug = self.augment_params(image=slice, segmentation_maps=mask)
#         mask_aug = mask_aug.get_arr()
#         return slice_aug, mask_aug
    
#     def __len__(self):
#         """
#         Return the length of the dataset (length of all files)
#         """
#         return len(self.all_files)
    
    
#     def __getitem__(self, idx):
#         """
#         Given an index return the (augmented) slice and corresponding mask
#         Add another dimension for pytorch
#         """
#         file_path = self.all_files[idx]
#         mask_path = self.change_img_to_label_path(file_path)
#         slice = np.load(file_path).astype(np.float32)
#         mask = np.load(mask_path)
        
#         if self.augment_params:
#             slice, mask = self.augment(slice, mask)
        
#         return np.expand_dims(slice, 0), np.expand_dims(mask, 0)
        