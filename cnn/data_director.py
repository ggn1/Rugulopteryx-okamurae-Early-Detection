# TITLE: Data Director
# CONTACT: Gayathri Girish Nair (girishng@tcd.ie)

# IMPORTS
import math
import json
import torch
import random
import rasterio
import numpy as np
import pandas as pd
from glob import glob
from ro_dataset import RoDataset
from torch.utils.data import DataLoader

class DataDirector:
    """ Data handler. """

    def __init__(self, dir_data, split_pc_trn, split_pc_val,
                 batch_size, shuffle_seed=42):
        """ Constructor. 
        
        Arguments:
        dir_data {str} -- Path to directory containing files "traits.h5"
                          and "index_map_traits.json".
        split_pc_trn {float} -- % of data in the training dataset.
        split_pc_val {float} -- % of data in the validation dataset.
        batch_size {int} -- Data batch size.
        shuffle_seed {int} -- Seed for randomizer when shuffling data.
                              (Default 42)
        """
        # Validate dataset split % values.
        if split_pc_trn <= 0: raise Exception("Train % must be > 0.")
        if split_pc_val <= 0: raise Exception("Train % must be > 0.")
        if split_pc_trn + split_pc_val >= 1: raise Exception(
            "Test % must be > 0. " +
            "So, (train % + validation %) must be < 1.")

        # Set given attributes.
        self.dir_data = dir_data
        
        # Define attributes to be computed.
        self.tensor_min = None
        self.tensor_max = None
        self.dataloader = {"trn": None, "val": None, "tst": None}

        # Compute min and max values per band.
        # (Used for min-max normalization later.)
        self._find_min_max()

        # Initialise dataloaders.
        self._init_dataloaders(batch_size, shuffle_seed, 
                               split_pc_trn, split_pc_val)
        
    def _find_min_max(self):
        """ Finds min and max values per band in all dataset images. """
        print("Finding min and max values per band ...", end=" ")
        file_paths = glob("../data/db/___trn_val_tst/processed/*.tif")
        tensor_min = None
        tensor_max = None
        for p in file_paths:
            img = rasterio.open(p)
            img = torch.tensor(img.read().astype('float32'))
            img_min = torch.min(torch.min(img, dim=1).values, 
                                dim=1).values
            img_max = torch.max(torch.max(img, dim=1).values, 
                                dim=1).values
            if tensor_min is None: tensor_min = img_min
            else: tensor_min = torch.where((img_min < tensor_min),
                                           img_min, tensor_min)
            if tensor_max is None: tensor_max = img_max
            else: tensor_max = torch.where((img_max > tensor_max),
                                           img_max, tensor_max)
        self.tensor_min = tensor_min
        self.tensor_max = tensor_max
        print("Done :)")
    
    def _init_dataloaders(self, batch_size, shuffle_seed, 
                          split_pc_trn, split_pc_val):
        """ Initialized train, validation, and test dataloaders. 
        
        Arguments:
        batch_size {int} -- Data batch size.
        shuffle_seed {int} -- Seed for randomizer when shuffling data.
        split_pc_trn {float} -- % of data in the training dataset.
        split_pc_val {float} -- % of data in the validation dataset.
        """
        print("Initializing DataLoaders ...", end=" ")
        
        # Load image file paths and shuffle them.
        file_path_i = glob(f"{self.dir_data}/*.tif")
        file_path_i = list(range(len(file_path_i)))
        random.Random(shuffle_seed).shuffle(file_path_i)

        # Get data split ranges.
        split_range = {}
        split_range["trn"] = [
            0, math.floor(split_pc_trn * len(file_path_i))
        ]
        split_range["val"] = [
            split_range["trn"][1], 
            split_range["trn"][1]
            + math.floor(split_pc_val * len(file_path_i))
        ]
        split_range["tst"] = [
            split_range["val"][1], len(file_path_i)
        ]

        # Define data transformations.
        data_transformations = [
            self.min_max_norm,
            self.pad_to_shape,
            self.fill_nan
        ]

        # Set data splits.
        for split in ["trn", "val", "tst"]:
            self.dataloader[split] = DataLoader(
                RoDataset(
                    dir_dst=self.dir_data,
                    index_list=file_path_i[split_range[split][0]:
                                           split_range[split][1]],
                    transforms=data_transformations),
                batch_size=batch_size,
                shuffle=False) # Indices were shuffled by _init_index_map.
            
        print("Done :)")

    def pad_to_shape(self, x, target_shape=(64, 64)):
        """ Pads a tensor (C, H, W) to the target spatial shape (H, W).
        
        Pads with zeros only if needed.
        
        Arguments:
            x {torch.Tensor} -- Input tensor of shape (C, H, W).
            target_shape {tuple} -- Target spatial shape (H, W).
        """
        _, h, w = x.shape
        target_h, target_w = target_shape

        pad_h = max(target_h - h, 0)
        pad_w = max(target_w - w, 0)

        if (pad_h > 0) or (pad_w > 0):
            # pad = (pad_left, pad_right, pad_top, pad_bottom)
            # we pad only on right (width) and bottom (height)
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))

        return x

    def fill_nan(self, data):
        """ Fills NA values in data with 0. 
        
        Arguments:
        data {tensor} -- Feature with data to be filled.
        
        Returns:
        {tensor} -- Feature data with NA values filled.
        """
        return torch.nan_to_num(data, nan=0)

    def min_max_norm(self, data, idx=[]):
        """ Normalizes data using min-max normalization. 
        
        Arguments:
        data {tensor} -- Feature with data to be normalized.
        idx {list} -- Indices corresponding to columns in 
                      self.tensor_min and self.tensor_max that
                      that are associated with data that is to
                      be normalized. An empty list means data 
                      = all of yx. (Default [])

        Returns:
        data {tensor} -- Normalized feature data.
        """
        lim_min = self.tensor_min
        lim_min = lim_min.view(lim_min.shape[0], 1, 1)
        lim_max = self.tensor_max
        lim_max = lim_max.view(lim_max.shape[0], 1, 1)
        data_norm = (data - lim_min) / (lim_max - lim_min)
        return data_norm