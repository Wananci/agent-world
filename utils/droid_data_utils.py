import logging
import tensorflow as tf
import tensorflow_datasets as tfds
tf.config.set_visible_devices([], "GPU")
import albumentations as A
import numpy as np
import torch
from torch.cuda.amp import autocast
logger = logging.getLogger(__name__)
import functools
import math
from multiprocessing import Value
from functools import partial
from dataclasses import dataclass
import numpy as np
from PIL import Image
import copy
from torch.utils.data import DataLoader, IterableDataset, get_worker_info, Dataset
from torch.utils.data.distributed import DistributedSampler
from omegaconf import DictConfig
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
import bisect
from itertools import accumulate
import copy
from typing import List
from torchvision import transforms as torchtransforms
import clip
from pdb import set_trace
from scipy.spatial.transform import Rotation as R
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, List, Tuple, Callable, Union

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)

    def forward_traj(self, x):
        n, t, c, h, w = x.size()
        x = x.view(n*t, *x.shape[2:])
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        base_grid = base_grid.unsqueeze(1).repeat(1, t, 1, 1, 1)
        base_grid = base_grid.view(n*t, *base_grid.shape[2:])
        shift = torch.randint(1,
                              2 * self.pad + 1,
                              size=(n*t, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        x = F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
        x = x.view(n, t, *x.shape[1:])
        return x

def preprocess_image(sample, image_processor):
    image = [image_processor(s).unsqueeze(0) for s in sample]
    image = torch.cat(image, dim=0)
    # apply random horizontal flip and color jitter
    return image

def preprocess_text_droid(sample, tokenizer):
    text = tokenizer.tokenize(sample, truncate=True)
    return text

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None
    dataset: Dataset = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

class DroidDataset2(Dataset):
    def __init__(self,
                 image_fn: Callable,
                 text_fn: Callable,
                 dataset_name: str = 'droid', 
                 image_primary_size: int = 200,
                 image_wrist_size: int = 84,
                 pad: bool = True,
                 split: str = 'train',
                 shuffle: bool = True, 
                 window_size: int = 16,
                 transforms: Dict = {},
                 rgb_pad: int = -1,
                 gripper_pad: int = -1,
                 traj_cons: bool = False,
                 data_dir: str = "gs://gresearch/robotics",
                 validation_ratio: float = 0.1,
                 test_ratio: float = 0.1
                 ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.image_fn = image_fn
        self.text_fn = text_fn
        self.dataset_name = dataset_name
        self.image_primary_size = image_primary_size
        self.image_wrist_size = image_wrist_size
        self.pad = pad
        self.split = split
        self.shuffle = shuffle
        self.window_size = window_size
        self.transforms = transforms ## TODO
        self.rgb_pad = rgb_pad
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        if self.rgb_pad != -1:
            self.rgb_shift1 = RandomShiftsAug(rgb_pad)
            self.rgb_shift2 = RandomShiftsAug(rgb_pad)
        self.gripper_pad = gripper_pad
        if self.gripper_pad != -1:
            self.gripper_shift = RandomShiftsAug(gripper_pad)
        self.traj_cons = traj_cons
        self._init_dataset()

    def _init_dataset(self):
        full_dataset = tfds.load(
            self.dataset_name,
            data_dir=self.data_dir,
            split="train"  # Original split is always "train"
        )
        print(f"Load {self.dataset_name} dataset successfully")
        # Get total number of episodes
        total_episodes = sum(1 for _ in full_dataset)
        # Calculate split boundaries
        val_size = int(total_episodes * self.validation_ratio)
        test_size = int(total_episodes * self.test_ratio)
        train_size = total_episodes - val_size - test_size
        # Create deterministic splits using take/skip
        if self.split == "train":
            self.dataset = full_dataset.take(train_size)
        elif self.split == "val":
            self.dataset = full_dataset.skip(train_size).take(val_size)
        elif self.split == "test":
            self.dataset = full_dataset.skip(train_size + val_size).take(test_size)
        else:
            raise ValueError(f"Unknown split: {self.split}")
        # Apply prefetching
        self.dataset = self.dataset.prefetch(tf.data.AUTOTUNE)

        self.all_episodes = list(self.dataset)
        self.episode_list = [i for i in range(0, len(self.dataset))]
        self.num_step_per_episode = [len(episode["steps"]) - self.window_size for episode in self.all_episodes]
        self.num_episode = len(self.episode_list)
        self.accumulated_num_step = list(accumulate(self.num_step_per_episode))
        self.length = self.accumulated_num_step[-1]

    def _get_pad_size(self, sequence: Dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """
        return self.window_size - len(sequence["action"])
    
    @staticmethod
    def _pad_with_repetition(input_tensor: np.ndarray, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a sequence ndarry by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        input_tensor = torch.from_numpy(input_tensor).byte()
        if head:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[0], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((last_repeated, input_tensor))
        else:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((input_tensor, last_repeated))

        return padded
    
    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
            repeats=pad_size,
            dim=0,
        )
        if head:
            padded = torch.vstack((zeros_repeated, input_tensor))
        else:
            padded = torch.vstack((input_tensor, zeros_repeated))

        return padded

    def _pad_sequence(self, seq: Dict, pad_size: int, head: bool=False) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        seq.update({"image_primary_1": self._pad_with_repetition(seq["image_primary_1"], pad_size)})
        seq.update({"image_primary_2": self._pad_with_repetition(seq["image_primary_2"], pad_size)})
        seq.update({"image_wrist": self._pad_with_repetition(seq["image_wrist"], pad_size)})
        seq.update({"action": self._pad_with_repetition(seq["action"], pad_size, head)})
        seq.update({"state": self._pad_with_repetition(seq["state"], pad_size, head)})
        return seq

    def __getitem__(self, idx) -> Any:
        sequence = self._get_sequences(idx, self.window_size)
        head = False
        if self.pad:
            pad_size = self._get_pad_size(sequence)
            sequence = self._pad_sequence(sequence, pad_size, head=head)
        assert self.window_size == len(sequence["action"])

        import copy
        new_list = []
        np_img_1 = copy.deepcopy(sequence["image_primary_1"].numpy())
        for i in range(np_img_1.shape[0]):
            new_list.append(Image.fromarray(np_img_1[i, :, :, :].astype(np.uint8)))
        sequence["image_primary_1"] = new_list

        new_list = []
        np_img_2 = copy.deepcopy(sequence["image_primary_2"].numpy())
        for i in range(np_img_2.shape[0]):
            new_list.append(Image.fromarray(np_img_2[i, :, :, :].astype(np.uint8)))
        sequence["image_primary_2"] = new_list

        new_list = []
        np_gripper = copy.deepcopy(sequence["image_wrist"].numpy())
        for i in range(np_gripper.shape[0]):
            new_list.append(Image.fromarray(np_gripper[i, :, :, :].astype(np.uint8)))
        sequence["image_wrist"] = new_list
        return sequence

    def _get_sequences(self, idx: int, window_size: int) -> Dict:
        episode_id = bisect.bisect_right(self.accumulated_num_step, idx)
        if episode_id - 1 >= 0:
            start_id = idx - self.accumulated_num_step[episode_id - 1]
        else:
            start_id = idx
        num_step_per_episode = self.num_step_per_episode[episode_id]
        end_id = min(start_id + window_size, num_step_per_episode)
        this_episodes = []
        this_episode_steps = list(self.all_episodes[episode_id]["steps"])
        for step_id in range(start_id, end_id):
            data_dict = {}
            data_dict["image_primary_1"] = this_episode_steps[step_id]['observation']['exterior_image_1_left'].numpy()
            data_dict["image_primary_2"] = this_episode_steps[step_id]['observation']['exterior_image_2_left'].numpy()
            data_dict["image_wrist"] = this_episode_steps[step_id]['observation']['wrist_image_left'].numpy()
            
            data_dict["state"] = np.concatenate((this_episode_steps[step_id]['action_dict']['cartesian_position'].numpy(),
                        this_episode_steps[step_id]['action_dict']['gripper_position'].numpy()), axis=0)
            
            data_dict["action"] = this_episode_steps[step_id]["action"]
            this_episodes.append(data_dict)
        keys = ["image_primary_1", "image_primary_2", "image_wrist", "state", "action"]
        episode = {key: np.stack([ep[key] for ep in this_episodes]) for key in keys}
        episode["text"] = self._load_language(this_episode_steps, step_id)

        return episode
    
    def _load_language(self, this_episode_steps, step_id):
        lang1 = this_episode_steps[step_id]['language_instruction'].numpy().decode('utf-8')
        if lang1:
            return lang1

        lang2 = this_episode_steps[step_id]['language_instruction_2'].numpy().decode('utf-8')
        if lang2:
            return lang2
        
        lang3 = this_episode_steps[step_id]['language_instruction_3'].numpy().decode('utf-8')
        if lang3:
            return lang3
        
        return "No language instruction."
    
    def __len__(self):
        return self.length
        
    def collator(self, sample):
        image_primary_1_tensor = torch.stack([self.image_fn(s["image_primary_1"]) for s in sample])
        image_primary_2_tensor = torch.stack([self.image_fn(s["image_primary_2"]) for s in sample])
        image_wrist_tensor = torch.stack([self.image_fn(s["image_wrist"]) for s in sample])
        action_tensor = torch.from_numpy(np.array([np.stack(s["action"]) for s in sample]))
        state_tensor = torch.from_numpy(np.array([np.stack(s["state"]) for s in sample]))
        stacked_language = [s["text"] for s in sample]
        text_tensors = self.text_fn(stacked_language)
        
        if self.rgb_pad != -1:
            bs, seq_len = image_primary_1_tensor.shape[:2]
            if self.traj_cons:
                image_primary_1_tensor = self.rgb_shift1.forward_traj(image_primary_1_tensor)
                image_primary_2_tensor = self.rgb_shift2.forward_traj(image_primary_2_tensor)
            else:
                image_primary_1_tensor = image_primary_1_tensor.view(bs*seq_len, *image_primary_1_tensor.shape[2:])
                image_primary_1_tensor = self.rgb_shift1(image_primary_1_tensor)
                image_primary_1_tensor = image_primary_1_tensor.view(bs, seq_len, *image_primary_1_tensor.shape[1:])

                image_primary_2_tensor = image_primary_2_tensor.view(bs*seq_len, *image_primary_2_tensor.shape[2:])
                image_primary_2_tensor = self.rgb_shift2(image_primary_2_tensor)
                image_primary_2_tensor = image_primary_2_tensor.view(bs, seq_len, *image_primary_2_tensor.shape[1:])
        if self.gripper_pad != -1:
            bs, seq_len = image_wrist_tensor.shape[:2]
            if self.traj_cons:
                image_wrist_tensor = self.gripper_shift.forward_traj(image_wrist_tensor)
            else:
                image_wrist_tensor = image_wrist_tensor.view(bs * seq_len, *image_wrist_tensor.shape[2:])
                image_wrist_tensor = self.gripper_shift(image_wrist_tensor)
                image_wrist_tensor = image_wrist_tensor.view(bs, seq_len, *image_wrist_tensor.shape[1:])
        return image_primary_1_tensor, image_primary_2_tensor, image_wrist_tensor, action_tensor, state_tensor, text_tensors


def get_mydroid_dataset(args, image_processor, tokenizer, epoch=0, split="train", floor=True):
    dataset_name = args.droid_dataset_name
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn = functools.partial(preprocess_image, image_processor=image_processor)
    preprocess_text_fn = functools.partial(preprocess_text_droid, tokenizer=tokenizer)

    droid_dataset = DroidDataset2(
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        dataset_name=dataset_name, 
        image_primary_size=args.image_primary_size,
        image_wrist_size=args.image_wrist_size,
        split=split,
        window_size=args.window_size,
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        traj_cons=args.traj_cons,
        data_dir=args.droid_data_dir,
    )
    round_fn = math.floor if floor else math.ceil
    num_samples = len(droid_dataset)
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  #
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    
    sampler = DistributedSampler(
        droid_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        droid_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=droid_dataset.collator,
        drop_last=True
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=droid_dataset)