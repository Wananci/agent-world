from collections import defaultdict, namedtuple
import logging
import os, json, random
from pathlib import Path
import sys
import time
import PIL.Image as Image
import copy
from collections import deque
from moviepy.editor import ImageSequenceClip
from calvin_agent.models.calvin_base_model import CalvinBaseModel
import time
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_env_state_for_initial_condition,
    get_log_dir,
    print_and_save,
)
import hydra
import numpy as np
from omegaconf import OmegaConf
from termcolor import colored
import torch
from tqdm import tqdm
from calvin_env.envs.play_table_env import get_env
from utils.droid_data_utils import preprocess_image, preprocess_text_droid
import functools
from utils.train_utils import get_cast_dtype, get_autocast
import cv2


os.environ['PYOPENGL_PLATFORM'] = 'egl'
logger = logging.getLogger(__name__)

EP_LEN = 360
NUM_SEQUENCES = 1000

class ModelWrapper(CalvinBaseModel):
    def __init__(self, model, tokenizer, image_processor, cast_dtype, history_len=10, 
                droid_eval_max_steps=360, action_pred_steps=3):
        super().__init__()
        self.model = model
        self.cast_type = cast_dtype
        self.use_diff = False
        self.text_process_fn = functools.partial(preprocess_text_droid, tokenizer=tokenizer)
        self.image_process_fn = functools.partial(preprocess_image, image_processor=image_processor)
        self.action_hist_queue = []
        self.history_len = history_len
        self.droid_eval_max_steps = droid_eval_max_steps
        self.action_pred_steps = action_pred_steps
        self.device = "cuda"
        self.img_queue_1 = deque(maxlen=history_len)
        self.img_queue_2 = deque(maxlen=history_len)
        self.gripper_queue = deque(maxlen=history_len)
        self.state_queue = deque(maxlen=history_len)
        self.mask_queue = deque(maxlen=history_len)
        self.text_queue = deque(maxlen=history_len)
        self.act_queue = deque(maxlen=history_len-1)

    def reset(self):
        self.img_queue_1 = deque(maxlen=self.history_len)
        self.img_queue_2 = deque(maxlen=self.history_len)
        self.gripper_queue = deque(maxlen=self.history_len)
        self.state_queue = deque(maxlen=self.history_len)
        self.mask_queue = deque(maxlen=self.history_len)
        self.text_queue = deque(maxlen=self.history_len)
        self.act_queue = deque(maxlen=self.history_len-1)

    def step(self, obs, goal, timestep):
        image_1 = obs["image_primary_1"]
        image_1 = Image.fromarray(image_1)
        image_x_1 = self.image_process_fn([image_1])
        image_x_1 = image_x_1.unsqueeze(1).to(dtype=self.cast_type)

        image_2 = obs["image_primary_2"]
        image_2 = Image.fromarray(image_2)
        image_x_2 = self.image_process_fn([image_2])
        image_x_2 = image_x_2.unsqueeze(1).to(dtype=self.cast_type)

        gripper = obs["image_wrist"] 
        gripper = Image.fromarray(gripper)
        gripper = self.image_process_fn([gripper])
        gripper = gripper.unsqueeze(1).to(dtype=self.cast_type)

        text_x = self.text_process_fn([goal])
        text_x = text_x.unsqueeze(1)

        state = obs['state']
        state = torch.from_numpy(np.stack([state]))
        state = state.unsqueeze(1).to(dtype=self.cast_type)
        state = torch.cat([state[..., :6], state[..., [-1]]], dim=-1)

        action = obs['action']
        action = torch.from_numpy(np.stack([action]))
        action = action.unsqueeze(1).to(dtype=self.cast_type)
        action = torch.cat([action[..., :6], state[..., [-1]]], dim=-1)

        with torch.no_grad():
            device = 'cuda'
            image_x_1 = image_x_1.to(device)
            image_x_2 = image_x_2.to(device)
            action = action.to(device)
            text_x = text_x.to(device)
            gripper = gripper.to(device)
            state = state.to(device)
            self.img_queue_1.append(image_x_1)  
            self.img_queue_2.append(image_x_2)
            self.gripper_queue.append(gripper)
            self.state_queue.append(state)
            self.act_queue.append(action)
            if len(self.text_queue) == 0 and text_x is not None:  
                self.text_queue.append(text_x)
                for _ in range(self.model.module.sequence_length - 1):
                    self.text_queue.append(text_x)
            image_primary_1 = torch.cat(list(self.img_queue_1), dim=1)
            image_primary_2 = torch.cat(list(self.img_queue_2), dim=1)
            image_wrist = torch.cat(list(self.gripper_queue), dim=1)
            state = torch.cat(list(self.state_queue), dim=1)
            action = torch.cat(list(self.act_queue), dim=1)
            input_text_token = torch.cat(list(self.text_queue), dim=1)
            num_step = image_primary_1.shape[1]
            if num_step < self.history_len:  
                input_image_primary_1 = torch.cat([image_primary_1, image_primary_1[:, -1].repeat(1, self.history_len-num_step, 1, 1, 1)], dim=1)
                input_image_primary_2 = torch.cat([image_primary_2, image_primary_2[:, -1].repeat(1, self.history_len-num_step, 1, 1, 1)], dim=1)
                input_image_wrist = torch.cat([image_wrist, image_wrist[:, -1].repeat(1, self.history_len-num_step, 1, 1, 1)], dim=1)
                input_state = torch.cat([state, state[:, -1].repeat(1, self.history_len-num_step, 1)], dim=1)
                input_action = torch.cat([action, action[:, -1].repeat(1, self.history_len-num_step, 1)], dim=1)
            else:
                input_image_primary_1 = image_primary_1
                input_image_primary_2 = image_primary_2
                input_image_wrist = image_wrist
                input_state = state
                input_action = action
            arm_action, gripper_action, image_pred  = self.model(
                image_primary_1=input_image_primary_1,
                image_primary_2=input_image_primary_2,
                image_wrist=input_image_wrist,
                state=input_state,
                text_token=input_text_token,
                action=input_action,
            )
            action = torch.concat((arm_action[0, :, 0, :], gripper_action[0, :, 0, :] > 0.5), dim=-1)
            # action[:, -1] = (action[:, -1] - 0.5) * 2  # scale to -1 or 1
            action = action.cpu().detach().to(dtype=torch.float16).numpy()
            if num_step < self.history_len:
                action = action[num_step - 1]
            else:
                action = action[-1]
                
        return action, image_pred
    
def eval_one_epoch_droid_ddp(args, model, dataset, image_processor, tokenizer, device_id, eval_log_dir=None, debug=False, future_act_len=-1, reset=False, diverse_inst=False):
    cast_dtype = get_cast_dtype(args.precision)
    hist_len = args.sequence_length
    wrapped_model = ModelWrapper(
                        model, 
                        tokenizer, 
                        image_processor, 
                        cast_dtype, 
                        history_len=hist_len, 
                        calvin_eval_max_steps=EP_LEN,
                        action_pred_steps = args.action_pred_steps)
    droid_dataloader = dataset.dataloader
    autocast = get_autocast(args.precision)
    total_testing_step = len(dataset)
    t = tqdm(
        enumerate(droid_dataloader), 
        disable=args.rank != 0,
        total=total_testing_step, 
        initial=0
    )
    for num_steps, batch in t:
        images_primary_1 = batch[0].to(device_id, dtype=cast_dtype, non_blocking=True)
        images_primary_2 = batch[1].to(device_id, dtype=cast_dtype, non_blocking=True)
        images_wrist = batch[2].to(device_id, dtype=cast_dtype, non_blocking=True)

        # text tokens
        text_tokens = batch[5].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, args.window_size, 1)
        
        # states
        states = batch[4].to(device_id, dtype=cast_dtype, non_blocking=True)
        if args.gripper_width:
            input_states = torch.cat([states[..., :6], states[..., -2:]], dim=-1)
        else:
            input_states = torch.cat([states[..., :6], states[..., [-1]]], dim=-1)
            input_states[..., 6:] = (input_states[..., 6:] + 1) // 2 
        
        # actions
        actions = batch[3].to(device_id, dtype=cast_dtype, non_blocking=True)
        input_image_primary_1 = images_primary_1[:, :args.sequence_length, :]
        input_image_primary_2 = images_primary_2[:, :args.sequence_length, :]

        input_image_wrist = images_wrist[:, :args.sequence_length, :]
        input_text_token = text_tokens[:, :args.sequence_length, :]
        input_state = input_states[:, :args.sequence_length, :]
        input_action = actions[:, :args.sequence_length, :]
        # label action
        label_actions = torch.cat([actions[:, j:args.sequence_length-args.atten_goal+j, :].unsqueeze(-2) for j in range(args.action_pred_steps)], dim=-2) 

        with autocast():  # image_primary, image_wrist, state, language_instruction
            arm_pred_action, gripper_pred_action, image_pred = model(
                image_primary_1=input_image_primary_1,
                image_primary_2=input_image_primary_2,
                image_wrist=input_image_wrist,
                state=input_state,
                text_token=input_text_token,
                action=input_action,
            )
        # TODO: How to evaluate?
        
    


