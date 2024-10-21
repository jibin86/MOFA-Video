
import os
import glob
import re

import os, random
from tqdm import tqdm
import numpy as np
from decord import VideoReader
from einops import rearrange
import torchaudio

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

from PIL import Image

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
        
        self.sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.Resize(sample_size[0]),
            # transforms.CenterCrop(sample_size),
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
        while True:
            video_path = self.video_list[idx]
            text = self.labels[idx]
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

            try:
                if frame_factor < 1:
                    idx += 1
                    print(f"Video is less than {self.fps} fps. Video: {video_file}")
                    continue
                else:
                    pass
            except:
                idx += 1
                print(f"Load video failed! path = {video_path}")
                continue
            
            num_fps_frames = len(vr)/(frame_factor)
            # current_duration = num_fps_frames / self.fps
            # import pdb; pdb.set_trace()
            try:
                if int(num_fps_frames) < self.sample_n_frames:
                    idx += 1
                    continue
                else:
                    pass
            except:
                idx += 1
                print(f"Load video failed! path = {video_path}")
                continue
                
            # start_frame = random.randint(0, int(num_fps_frames)-self.sample_n_frames)
            start_frame = 0
            end_frame = start_frame + self.sample_n_frames

            idx_list = np.arange(start_frame, end_frame) * frame_factor
            idx_list = list(map(round, idx_list))
            # print("idx_list",idx_list)
            
            try:
                # pixel_values = torch.from_numpy(vr.get_batch(idx_list).asnumpy())
                pixel_values = vr.get_batch(idx_list).asnumpy()
                break
            except:
                print(f"Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(idx_list)} / {num_fps_frames}]")
                idx += 1
                continue
        
        assert(pixel_values.shape[0] == self.sample_n_frames),f'{len(pixel_values)}, self.video_length={self.sample_n_frames}'

        resized_frames = []
        for i in range(pixel_values.shape[0]):
            frame = np.array(Image.fromarray(pixel_values[i]).convert('RGB').resize([self.sample_size[1], self.sample_size[0]]))
            resized_frames.append(frame)
        resized_frames = np.array(resized_frames)

        pixel_values = torch.tensor(resized_frames).permute(0, 3, 1, 2).float()  # [t,h,w,c] -> [t,c,h,w]
    


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
        return pixel_values, waveform, video_file, text

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
        pixel_values, waveform, video_name, text = self.get_batch(idx)
        # pixel_values = self.pixel_transforms(pixel_values)
        
        if waveform.shape[0] < 32000:
            raise Exception(f"Empty audio tensor at index: {self.video_list[idx]} ,{waveform.numel()}")
        
        sample = dict(pixel_values=pixel_values, text=text, waveform=waveform, video_name=video_name)
        return sample
    
if __name__ == "__main__":
    is_image=False
    dataset = ASyncAudioDataset(
        # video_folder="/mnt/Diffusion/AVSync15/train",
        # class_subfolder="/mnt/Diffusion/AVSync15/train/dog_barking",
        class_subfolder = "/home/jibin/Github/MOFA-Video/MOFA-Video-My/outputs/sample",
        # audio_folder="/mnt/Diffusion/AVSync15/train_audio",
        sample_size=256,
        # sample_stride=4,
        sample_n_frames=16,
        is_image=is_image,
    )
    print(len(dataset)) # 90
    
    for idx in tqdm(range(len(dataset))):
        video = dataset[idx]



    # for idx in range(10):
    #     video = dataset[idx]
    #     # video_path = f"./dataset_samples/sample_video_{idx}_{video['text']}.mp4"
    #     if is_image:
    #         import torchvision.transforms as T
    #         from PIL import Image
    #         transform = T.ToPILImage()
    #         img = transform(video["pixel_values"].mul_(0.5).add_(0.5))
    #         img.save(f"dataset_samples/img/sample_video_{idx}_{video['text']}_random.png")
    #     else:
    #         video_path = f"./dataset_samples/videos/sample_video_{idx}.mp4"
    #         video = dataset[idx]
    #         print(video["pixel_values"].shape) # torch.Size([16, 3, 256, 256])
    #         video_pix = rearrange(video["pixel_values"].mul_(0.5).add_(0.5), "f c h w -> c f h w").unsqueeze(0)
            
    #         combine_video_audio(video_pix, video["waveform"].unsqueeze(0).unsqueeze(0), video_path, fps=8)

    '''
    # dataset = ASyncAudioDataset(
    #     video_folder="/mnt/Diffusion/AVSync15/train/dog_barking",
    #     audio_folder="/mnt/Diffusion/AVSync15/train_audio/dog_barking",
    #     sample_size=256,
    #     # sample_stride=4,
    #     sample_n_frames=16,
    #     is_image=False,
    # )
    # print(len(dataset)) # 90

    # for idx in range(10):
    #     video_path = f"./dataset_samples/sample_video_{idx}.mp4"
    #     video = dataset[idx]
    #     print(video["pixel_values"].shape) # torch.Size([16, 3, 256, 256])
    #     # combine_video_audio(sample, dataset[idx]["waveform"], video_path)
    #     combine_video_audio(video["pixel_values"], video["waveform"], video_path, fps=8)
        
    #################
    dataset = ASyncDataset(
        video_folder="/mnt/Diffusion/AVSync15/train/dog_barking",
        # audio_folder="/mnt/Diffusion/AVSync15/train_audio/dog_barking",
        sample_size=256,
        sample_stride=4,
        sample_n_frames=16,
        is_image=False,
    )
    print(len(dataset)) # 90

    for idx in range(10):
        video_path = f"./dataset_samples/sample_video_{idx}.mp4"
        video = dataset[idx]
        print(video["pixel_values"].shape) # torch.Size([16, 3, 256, 256])
        # combine_video_audio(sample, dataset[idx]["waveform"], video_path)
        combine_video_audio(video["pixel_values"], video_path=video_path, fps=8)
    '''