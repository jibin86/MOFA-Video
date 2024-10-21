import os, io, csv, math, random
import numpy as np
from einops import rearrange
import pandas as pd

import torch
from decord import VideoReader, cpu
import torch.distributed as dist

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from PIL import Image


def pil_image_to_numpy(image):
    """Convert a PIL image to a NumPy array."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)


def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """Convert a NumPy image to a PyTorch tensor."""
    if images.ndim == 3:
        images = images[..., None]
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images.float() / 255


def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)


class WebVid10M(Dataset):
    def __init__(
            self,
            meta_path='/apdcephfs/share_1290939/0_public_datasets/WebVid/metadata/results_2M_train.csv',
            data_dir='/apdcephfs/share_1290939/0_public_datasets/WebVid',
            sample_size=[256, 256], 
            sample_stride=1, 
            sample_n_frames=14,
        ):
        zero_rank_print(f"loading annotations from {meta_path} ...")

        metadata = pd.read_csv(meta_path)
        metadata['caption'] = metadata['name']
        del metadata['name']
        self.metadata = metadata
        self.metadata.dropna(inplace=True)
        self.data_dir = data_dir

        self.length = len(self.metadata)
        print(f"data scale: {self.length}")

        self.sample_stride   = sample_stride
        print(f"sample stride: {self.sample_stride}")
        self.sample_n_frames = sample_n_frames
        
        # sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        print("sample size", sample_size)
        self.sample_size = sample_size

        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.Resize(sample_size),
            # transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    
    def _get_video_path(self, sample):
        rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        return full_video_fp, rel_video_fp
    
    def get_batch(self, index):

        while True:

            index = index % len(self.metadata)
            sample = self.metadata.iloc[index]
            video_path, rel_path = self._get_video_path(sample)

            required_frame_num = self.sample_stride * self.sample_n_frames

            try:
                video_reader = VideoReader(video_path, ctx=cpu(0))
                if len(video_reader) < required_frame_num:
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue
            
            frame_num = len(video_reader)

            ## select a random clip
            random_range = frame_num - required_frame_num
            start_idx = random.randint(0, random_range) if random_range > 0 else 0
            frame_indices = [start_idx + self.sample_stride*i for i in range(self.sample_n_frames)]

            try:
                frames = video_reader.get_batch(frame_indices)
                break
            except:
                print(f"Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]")
                index += 1
                continue

        assert(frames.shape[0] == self.sample_n_frames),f'{len(frames)}, self.video_length={self.sample_n_frames}'

        frames = frames.asnumpy()

        resized_frames = []
        for i in range(frames.shape[0]):
            frame = np.array(Image.fromarray(frames[i]).convert('RGB').resize([self.sample_size[1], self.sample_size[0]]))
            resized_frames.append(frame)
        resized_frames = np.array(resized_frames)

        resized_frames = torch.tensor(resized_frames).permute(0, 3, 1, 2).float()  # [t,h,w,c] -> [t,c,h,w]
    
        return resized_frames, rel_path
        
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        pixel_values, video_name = self.get_batch(idx)

        # pixel_values = self.pixel_transforms(pixel_values)
        pixel_values = pixel_values / 255.
        
        sample = dict(pixel_values=pixel_values, video_name=video_name)
        return sample


import sys
import os
import glob
import re
sys.path.append(os.path.abspath("/home/jibin/Github/AnimateDiff_baseline"))

import os, random
from tqdm import tqdm
import numpy as np
from decord import VideoReader
from einops import rearrange
import torchaudio

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

from animatediff.utils.video_processing import combine_video_audio

class ASyncAudioDataset(Dataset):
    def __init__(
            self,
            video_folder="/mnt/Diffusion/AVSync15/train",
            class_subfolder=None,
            # audio_folder="/mnt/Diffusion/AVSync15/train_audio",
            sample_size=256, 
            # sample_stride=4, 
            sample_n_frames=16, 
            fps=8,
            is_image=False,
            sample_rate=16000,
            # name = "dog barking",
        ):
        # print(f"loading annotations from {csv_path} ...")
        # with open(csv_path, 'r') as csvfile:
            # self.video_list = list(csv.DictReader(csvfile))
        if class_subfolder is not None:
            self.video_list = glob.glob(os.path.join(class_subfolder, '*.mp4'))
        else:
            self.video_list = glob.glob(os.path.join(video_folder, '*/*.mp4'))
            
        [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
        self.labels = [self._extract_label_from_path(path) for path in self.video_list]
        self.length = len(self.video_list)
        print(f"data scale: {self.length}")

        # self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image        = is_image
        self.sample_rate     = sample_rate
        self.fps             = fps
        self.video_duration  = sample_n_frames/fps
        # self.name            = name
        
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            # transforms.RandomCrop(sample_size),
            # transforms.RandomErasing(),
            # transforms.RandomPerspective(distortion_scale=0.3),
            # transforms.ColorJitter(brightness=.5, hue=.1),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    def _extract_label_from_path(self, path):
        label = os.path.basename(os.path.dirname(path))
        return re.sub(r'_', ' ', label)  # replace underscore with space
    
    def get_batch(self, idx):
        video_path = self.video_list[idx]
        video_file = os.path.basename(video_path)
                
        ### Load video (비디오를 고정된 fps로 변경)
        vr = VideoReader(video_path)

        if self.is_image:
            video_length = len(vr)
            batch_index = [random.randint(0, video_length - 1)]
            pixel_values = torch.from_numpy(vr.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
            pixel_values = pixel_values / 255.
            del vr
            pixel_values = pixel_values[0]
            return pixel_values

        # print("len(vr)",len(vr))
        # Get current FPS
        old_fps = vr.get_avg_fps()
        # print(f"{old_fps} video_file: {video_file}")
        
        # Calculate frame factor for sampling
        frame_factor = old_fps / self.fps
        # print("frame_factor",frame_factor)

        if frame_factor < 1:
            raise ValueError(f"Video is less than {self.fps} fps. Video: {video_file}")
        
        num_fps_frames = len(vr)/(frame_factor)
        current_duration = num_fps_frames / self.fps

        start_frame = random.randint(0, int(num_fps_frames)-self.sample_n_frames)
        end_frame = start_frame + self.sample_n_frames

        idx_list = np.arange(start_frame, end_frame) * frame_factor
        idx_list = list(map(round, idx_list))
        # print("idx_list",idx_list)

        pixel_values = torch.from_numpy(vr.get_batch(idx_list).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del vr
        
        # Load audio
        # audio_path = os.path.join(self.audio_folder, video_file.replace('.mp4', '.wav'))
        audio_path = video_path.replace("train/", "train_audio/").replace("test/", "test_audio/").replace('.mp4', '.wav')
        waveform, sr = torchaudio.load(audio_path)

        if self.sample_rate != sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.sample_rate
            ) # waveform.shape torch.Size([2, 64320])

        start_sec = start_frame / self.fps
        end_sec = end_frame / self.fps
        waveform = waveform[:, int(start_sec * self.sample_rate): int(end_sec * self.sample_rate)][0]
        return pixel_values, waveform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        if self.is_image:
            pixel_values = self.get_batch(idx)
            pixel_values = self.pixel_transforms(pixel_values)
            text = self.labels[idx]
            sample = dict(pixel_values=pixel_values, text=text)
            return sample
            
        # print("idx",idx)
        pixel_values, waveform = self.get_batch(idx)
        pixel_values = self.pixel_transforms(pixel_values)
        text = self.labels[idx]
        
        if waveform.shape[0] < 32000:
            raise Exception(f"Empty audio tensor at index: {self.video_list[idx]} ,{waveform.numel()}")
        
        sample = dict(pixel_values=pixel_values, text=text, waveform=waveform)
        return sample
    
if __name__ == "__main__":
    is_image=False
    dataset = ASyncAudioDataset(
        # video_folder="/mnt/Diffusion/AVSync15/train",
        class_subfolder="/mnt/Diffusion/AVSync15/train/dog_barking",
        # audio_folder="/mnt/Diffusion/AVSync15/train_audio",
        sample_size=256,
        # sample_stride=4,
        sample_n_frames=16,
        is_image=is_image,
    )
    print(len(dataset)) # 90
    
    for idx in tqdm(range(len(dataset))):
        video = dataset[idx]

