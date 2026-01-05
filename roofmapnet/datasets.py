from pathlib import Path
import random

import numpy as np
import numpy.linalg as LA
import torch
from skimage import io
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from roofmapnet.config import M


class WireframeDataset(Dataset):
    def __init__(self, rootdir, split, image_dir=None):
        self.rootdir = rootdir
        self.image_dir = Path(image_dir) if image_dir else None
        filelist = list(Path(rootdir).joinpath(split).glob("*.npz"))
        filelist.sort()

        print(f"n{split}:", len(filelist))
        self.split = split
        self.filelist = filelist
        
        # Store image normalization parameters from config
        # This ensures they're available in multiprocessing workers
        self.image_mean = np.array(M.image.mean)
        self.image_stddev = np.array(M.image.stddev)
        self.n_stc_posl = M.n_stc_posl
        self.n_stc_negl = M.n_stc_negl
        self.use_cood = M.use_cood
        self.use_slop = M.use_slop

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        # Determine image path
        if self.image_dir:
            iname = self.image_dir / f"{self.filelist[idx].stem}.png"
        else:
            # Load from same directory as NPZ file
            iname = self.filelist[idx].with_suffix(".png")
        
        image = io.imread(iname).astype(float)[:, :, :3]
        # if "a1" in self.filelist[idx]: # data augmentation for left-right flip, remove for now
        #     image = image[:, ::-1, :]
        image = (image - self.image_mean) / self.image_stddev
        image = np.rollaxis(image, 2).copy()

        # npz["jmap"]: [J, H, W]    Junction heat map
        # npz["joff"]: [J, 2, H, W] Junction offset within each pixel
        # npz["lmap"]: [H, W]       Line heat map with anti-aliasing
        # npz["junc"]: [Na, 3]      Junction coordinates
        # npz["Lpos"]: [M, 2]       Positive lines represented with junction indices
        # npz["Lneg"]: [M, 2]       Negative lines represented with junction indices
        # npz["lpos"]: [Np, 2, 3]   Positive lines represented with junction coordinates
        # npz["lneg"]: [Nn, 2, 3]   Negative lines represented with junction coordinates
        #
        # For junc, lpos, and lneg that stores the junction coordinates, the last
        # dimension is (y, x, t), where t represents the type of that junction.
        with np.load(self.filelist[idx]) as npz:
            target = {
                name: torch.from_numpy(npz[name]).float()
                for name in ["jmap", "joff", "lmap"]
            }
            lpos = np.random.permutation(npz["lpos"])[: self.n_stc_posl]
            lneg = np.random.permutation(npz["lneg"])[: self.n_stc_negl]
            npos, nneg = len(lpos), len(lneg)
            lpre = np.concatenate([lpos, lneg], 0)
            for i in range(len(lpre)):
                if random.random() > 0.5:
                    lpre[i] = lpre[i, ::-1]
            ldir = lpre[:, 0, :2] - lpre[:, 1, :2]
            ldir /= np.clip(LA.norm(ldir, axis=1, keepdims=True), 1e-6, None)
            feat = [
                lpre[:, :, :2].reshape(-1, 4) / 128 * self.use_cood,
                ldir * self.use_slop,
                lpre[:, :, 2],
            ]
            feat = np.concatenate(feat, 1)
            meta = {
                "junc": torch.from_numpy(npz["junc"][:, :2]),
                "jtyp": torch.from_numpy(npz["junc"][:, 2]).byte(),
                "Lpos": self.adjacency_matrix(len(npz["junc"]), npz["Lpos"]),
                "Lneg": self.adjacency_matrix(len(npz["junc"]), npz["Lneg"]),
                "lpre": torch.from_numpy(lpre[:, :, :2]),
                "lpre_label": torch.cat([torch.ones(npos), torch.zeros(nneg)]),
                "lpre_feat": torch.from_numpy(feat),
            }

        return torch.from_numpy(image).float(), meta, target

    def adjacency_matrix(self, n, link):
        mat = torch.zeros(n + 1, n + 1, dtype=torch.uint8)
        link = torch.from_numpy(link)
        if len(link) > 0:
            mat[link[:, 0], link[:, 1]] = 1
            mat[link[:, 1], link[:, 0]] = 1
        return mat


class InferenceDataset(Dataset):
    """Dataset for inference on images without labels.
    
    This dataset loads PNG images from a directory and creates dummy
    meta and target outputs that mimic the shapes expected by the model.
    """
    
    def __init__(self, image_dir):
        """Initialize the InferenceDataset.
        
        Args:
            image_dir: Directory path containing PNG images for inference
        """
        self.image_dir = image_dir
        self.filelist = list(Path(image_dir).glob("*.png"))
        self.filelist.sort()
        
        if len(self.filelist) == 0:
            raise ValueError(f"No PNG images found in {image_dir}")
        
        print(f"Found {len(self.filelist)} images for inference")
        
        # Store config parameters as instance variables for multiprocessing
        self.image_mean = np.array(M.image.mean)
        self.image_stddev = np.array(M.image.stddev)
        self.n_stc_posl = M.n_stc_posl
        self.n_stc_negl = M.n_stc_negl
        self.use_cood = M.use_cood
        self.use_slop = M.use_slop
    
    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, idx):
        # Load and preprocess image
        iname = self.filelist[idx]
        image = io.imread(iname).astype(float)[:, :, :3]
        
        # Center crop to 512x512 if needed
        if image.shape[0] != 512 or image.shape[1] != 512:
            h, w = image.shape[:2]
            
            # If image is smaller than 512, pad it first
            if h < 512 or w < 512:
                pad_h = max(0, 512 - h)
                pad_w = max(0, 512 - w)
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                              mode='constant', constant_values=0)
                h, w = image.shape[:2]
            
            # Calculate crop coordinates for center crop
            top = (h - 512) // 2
            left = (w - 512) // 2
            bottom = top + 512
            right = left + 512
            
            # Crop to 512x512
            image = image[top:bottom, left:right, :]
        
        image = (image - self.image_mean) / self.image_stddev
        image = np.rollaxis(image, 2).copy()
        
        # Get image dimensions
        _, H, W = image.shape
        
        # Create dummy target dictionary
        # These shapes match what the model expects but contain zeros
        target = {
            "jmap": torch.zeros(1, H // 4, W // 4),  # Junction heat map (downsampled)
            "joff": torch.zeros(1, 2, H // 4, W // 4),  # Junction offset
            "lmap": torch.zeros(H // 4, W // 4),  # Line heat map
        }
        
        # Create dummy meta dictionary
        # Using minimal shapes that won't cause issues during inference
        n_dummy_junc = 1
        n_dummy_lines = self.n_stc_posl + self.n_stc_negl
        
        # Create dummy line features
        lpre_feat_dim = 4 * self.use_cood + 2 * self.use_slop + 2
        
        meta = {
            "junc": torch.zeros(n_dummy_junc, 2),  # Junction coordinates
            "jtyp": torch.zeros(n_dummy_junc, dtype=torch.uint8),  # Junction types
            "Lpos": torch.zeros(n_dummy_junc + 1, n_dummy_junc + 1, dtype=torch.uint8),  # Positive line adjacency
            "Lneg": torch.zeros(n_dummy_junc + 1, n_dummy_junc + 1, dtype=torch.uint8),  # Negative line adjacency
            "lpre": torch.zeros(n_dummy_lines, 2, 2),  # Line endpoints
            "lpre_label": torch.zeros(n_dummy_lines),  # Line labels
            "lpre_feat": torch.zeros(n_dummy_lines, lpre_feat_dim),  # Line features
        }
        
        return torch.from_numpy(image).float(), meta, target
    
    def get_image_path(self, idx):
        """Get the file path for a given index.
        
        Args:
            idx: Index of the image
            
        Returns:
            Full path to the image file
        """
        return self.filelist[idx]


def collate(batch):
    return (
        default_collate([b[0] for b in batch]),
        [b[1] for b in batch],
        default_collate([b[2] for b in batch]),
    )
