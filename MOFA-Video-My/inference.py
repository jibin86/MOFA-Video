import numpy as np
import os
from PIL import Image
from scipy.interpolate import interp1d, PchipInterpolator
import torchvision
from tqdm.auto import tqdm
from typing import Dict

from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import torchvision

from packaging import version

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers import AutoencoderKLTemporalDecoder
from diffusers.utils.import_utils import is_xformers_available

from utils.flow_viz import flow_to_image

import argparse
import datetime

import torch.nn.functional as F
import scipy.io.wavfile as wavfile

from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip

import sys, os
sys.path.append(os.path.abspath("/home/jibin/Github/MOFA-Video"))

from Training.train_utils.unimatch.unimatch.unimatch import UniMatch
from Training.train_utils.unimatch.utils.flow_viz import flow_to_image

# from Training.train_stage1 import *

from utils.dataset import ASyncAudioDataset

def preprocess_size(image1, image2, padding_factor=32):
    '''
        img: [b, c, h, w]
    '''
    transpose_img = False
    # the model is trained with size: width > height
    if image1.size(-2) > image1.size(-1):
        image1 = torch.transpose(image1, -2, -1)
        image2 = torch.transpose(image2, -2, -1)
        transpose_img = True

    # inference_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
    #                 int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]
        
    inference_size = [384, 512]

    assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
    ori_size = image1.shape[-2:]

    # resize before inference
    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                align_corners=True)
        image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                align_corners=True)
    
    return image1, image2, inference_size, ori_size, transpose_img

def postprocess_size(flow_pr, inference_size, ori_size, transpose_img):

    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                align_corners=True)
        flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
        flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

    if transpose_img:
        flow_pr = torch.transpose(flow_pr, -2, -1)
    
    return flow_pr

@torch.no_grad()
def get_optical_flows(unimatch, video_frame):
    '''
        video_frame: [b, t, c, w, h]
    '''

    video_frame = video_frame * 255

    # print(video_frame.dtype)

    flows = []
    for i in range(video_frame.shape[1] - 1):
        image1, image2 = video_frame[:, 0], video_frame[:, i + 1]
        # print(image1.dtype)
        image1_r, image2_r, inference_size, ori_size, transpose_img = preprocess_size(image1, image2)
        # print(image1_r.dtype)
        results_dict_r = unimatch(image1_r, image2_r,
            attn_type='swin',
            attn_splits_list=[2, 8],
            corr_radius_list=[-1, 4],
            prop_radius_list=[-1, 1],
            num_reg_refine=6,
            task='flow',
            pred_bidir_flow=False,
            )
        flow_r = results_dict_r['flow_preds'][-1]  # [b, 2, H, W]
        # print(flow_r.shape)
        flow = postprocess_size(flow_r, inference_size, ori_size, transpose_img)
        flows.append(flow.unsqueeze(1))  # [b, 1, 2, w, h]
    
    flows = torch.cat(flows, dim=1).to(torch.float16)  # [b, t, 2, w, h]
    return flows

def main(
    output_dir: str, # "outputs"
    inference_name: str,
    # inference_data: Dict,

    svd_ckpt: str,
    mofa_ckpt: str,
    unimatch_ckpt: str,

    validation_data: Dict = None,

    ctrl_scale: float = 0.6,
            
    global_seed: int = 1234,
    is_debug: bool = False, # False

    device: str = 'cuda:0',

    enable_xformers_memory_efficient_attention: bool = True,
    allow_tf32: bool = False,
):    
    torch.manual_seed(global_seed)
    
    # Logging folder
    name = f"_{inference_name}"
    
    date_calendar = datetime.datetime.now().strftime("%Y-%m-%d")
    date_time = datetime.datetime.now().strftime("-%H-%M-%S")
    folder_name = "debug" if is_debug else date_calendar+date_time+name

    output_dir = os.path.join(output_dir, folder_name)  
    
    config_save_path = f"{output_dir}"
    os.makedirs(config_save_path, exist_ok=True)

    OmegaConf.save(config, os.path.join(config_save_path, 'config.yaml'))

    generator = torch.Generator(device=device)
    generator.manual_seed(global_seed)

    weight_dtype = torch.float16

    ### 모델 정의
    from models.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
    from pipeline.pipeline import FlowControlNetPipeline
    from models.svdxt_featureflow_forward_controlnet_s2d_fixcmp_norefine import FlowControlNet, CMP_demo

    print('start loading models...')
    # Load scheduler, tokenizer and models.
    feature_extractor = CLIPImageProcessor.from_pretrained(
        svd_ckpt, subfolder="feature_extractor", revision=None
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        svd_ckpt, subfolder="image_encoder", revision=None, variant="fp16"
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        svd_ckpt, subfolder="vae", revision=None, variant="fp16")
    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(
        svd_ckpt,
        subfolder="unet",
        low_cpu_mem_usage=True,
        variant="fp16",
    )

    controlnet = FlowControlNet.from_pretrained(mofa_ckpt)
    
    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)
    
    # Define Unimatch for optical flow prediction
    unimatch = UniMatch(feature_channels=128,
        num_scales=2,
        upsample_factor=4,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        reg_refine=True,
        task='flow').to(device)
    checkpoint = torch.load(unimatch_ckpt)
    unimatch.load_state_dict(checkpoint['model'])
    unimatch.eval()
    unimatch.requires_grad_(False)

    # Move image_encoder and vae to gpu and cast to weight_dtype
    image_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    controlnet.to(device, dtype=weight_dtype)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    pipeline = FlowControlNetPipeline.from_pretrained(
        svd_ckpt,
        unet=unet,
        controlnet=controlnet,
        image_encoder=image_encoder,
        vae=vae,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(device)
    # pipeline.set_progress_bar_config(disable=True)

    print('models loaded.')


    # 데이터셋 로드
    if validation_data is not None:
        valid_dataset = ASyncAudioDataset(**validation_data, is_image=False)
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=True,
        )
        # import pdb; pdb.set_trace()
        
        pbar2 = tqdm(valid_dataloader)
        # for idx, batch in enumerate(pbar2):
        for batch in valid_dataloader:
            # Convert videos to latent space            
            val_pixel_values = batch["pixel_values"].to(pipeline.device)

            # get optical flows via unimatch
            val_flows = get_optical_flows(unimatch, val_pixel_values)  # [b, t-1, 2, h, w]

            val_controlnet_image = val_pixel_values[:, 0:1, :, :, :].repeat(1, val_pixel_values.shape[1], 1, 1, 1)

            pil_val_pixel_values = [Image.fromarray((val_pixel_values[0][i].permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)) for i in range(val_pixel_values.shape[1])]
            
            num_frames = validation_data.sample_n_frames
            sample_size = validation_data.sample_size
            video_frames = pipeline(
                pil_val_pixel_values[0], 
                pil_val_pixel_values[0],
                val_flows,
                height=sample_size,
                width=sample_size,
                num_frames=num_frames,
                decode_chunk_size=8,
                motion_bucket_id=127,
                fps=7,
                noise_aug_strength=0.02,
                controlnet_cond_scale=ctrl_scale, 
                # generator=generator,
            ).frames[0]

            for i in range(num_frames):
                img = video_frames[i]
                video_frames[i] = np.array(img)
            
            viz_flows = []
            for i in range(val_flows.shape[1]):
                temp_flow = val_flows[0][i].permute(1, 2, 0)
                viz_flows.append(flow_to_image(temp_flow))
            viz_flows = [np.uint8(np.ones_like(viz_flows[-1]) * 255)] + viz_flows
            viz_flows = np.stack(viz_flows)  # [t-1, h, w, c]
            
            out_nps = video_frames
            gt_nps = (val_pixel_values[0].permute(0, 2, 3, 1).cpu().numpy()*255).astype(np.uint8)
            ctrl_nps = (val_controlnet_image[0].permute(0, 2, 3, 1).cpu().numpy()*255).astype(np.uint8)
            flow_nps = viz_flows
            total_nps = np.concatenate([ctrl_nps, flow_nps, out_nps, gt_nps], axis=2)

            video_name = batch['video_name'][0].replace('/', '_').replace('-', '_').split('.')[0]
            # import pdb; pdb.set_trace()
            text = batch['text'][0].replace(' ', '_')
            total_path = os.path.join(output_dir,
                f"{text}/{video_name}.mp4",
            )
            os.makedirs(os.path.dirname(total_path), exist_ok=True)
            
            
            # import pdb; pdb.set_trace()
            
            temp_video_path = total_path.replace('.mp4', '_temp.mp4')
            temp_audio_path = total_path.replace('.mp4', '_temp_audio.wav')
            
            torchvision.io.write_video(temp_video_path, total_nps, fps=validation_data.fps, video_codec='h264', options={'crf': '10'})
            wavfile.write(temp_audio_path, validation_data.sample_rate, batch['waveform'].cpu().to(torch.float64).permute(1, 0).numpy())
            
            video_clip = VideoFileClip(temp_video_path)
            audio_clip = AudioFileClip(temp_audio_path)
            final_clip = video_clip.set_audio(audio_clip)
            final_clip.write_videofile(total_path, fps=validation_data.fps, codec='h264', verbose=False)
            os.remove(temp_video_path)
            os.remove(temp_audio_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/inference.yaml")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    
    main(**config)