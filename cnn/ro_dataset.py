# TITLE: Custom PyTorch Dataset
# CONTACT: Gayathri Girish Nair (girishng@tcd.ie)

# IMPORTS
import os
import torch
import rasterio
from glob import glob
from torch.utils.data import Dataset

class RoDataset(Dataset):
    """ A custom dataset for this project. """

    # REQUIRED
    def __init__(self, dir_dst, index_list, transforms=[]):
        """ Constructor.
        Arguments:
        dir_dst {str} -- Path to the directory containing all images.
        index_list {list} -- List of indices to include in this dataset.
        transforms {list} -- List of data transformation functions to apply
                             to each image.
        """
        self.transforms = transforms
        paths = glob(f"{dir_dst}/*.tif")
        self.index_list = index_list
        self._paths = []
        for i in index_list: 
            self._paths.append(paths[i])

    # REQUIRED
    def __len__(self): 
        """ Returns length of this dataset. """
        return len(self._paths)

    # REQUIRED
    def __getitem__(self, index): 
        """ Gets item at given index. 
        
        Arguments:
        index {int} -- Index of requested data in the dataset.
        
        Returns:
        y, x {int, tensor} -- Label and image data.
        """
        # Read raster image an convert to tensor.
        x = rasterio.open(self._paths[index])
        x = torch.tensor(x.read().astype('float32'))
        # Apply transformations.
        for transform in self.transforms:
            x = transform(x)
        # Get label from filename.
        y = int(os.path.basename(self._paths[index]).split("-")[0])
        return y, x