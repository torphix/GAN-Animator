import os
import yaml
import torch
import librosa
import numpy as np
from tqdm import tqdm
from .utils import (read_video,
                   crop_to_same,
                   process_audio_for_generator,
                   cut_video_sequence,
                   sample_frames,
                   split_audio)
from torchvision import transforms


class DataProcessor:
    '''
    Data should be formatted into seperate audio video folders
    - VideoFlash with *.mp4 extension
    - AudioWAV  with *.wav extension
    If unaligned make sure to pass align flag 
    Outputs under datasets/{DATASET-NAME}/processed:
    - real_video_all
    - real_video_subset
    - real_video_blocks
    - audio_chunks
    - identity_frame
    - audio_generator_input
    '''
    def __init__(self, args):
        self.data_config = yaml.load(open('configs/data.yaml','r'),
                                          Loader=yaml.FullLoader)

        assert self.data_config["name"] in ['lex', 'CREMA-D'], \
            "Dataset names 'lex' or 'CREMA-D' currently supported"

        self.dataset_path = f'data/datasets/{self.data_config["name"]}'
        self.audio_dir = f'{self.dataset_path}/AudioWAV'
        self.video_dir = f'{self.dataset_path}/VideoFlash'
        self.files = [file.split(".")[0] for file in os.listdir(self.audio_dir)]
        
        self.img_transform = transforms.Compose([
            transforms.Resize((self.data_config['video']['img_size'][0], 
                               self.data_config['video']['img_size'][1])),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

        os.makedirs(f'{self.dataset_path}/processed/real_video_all', exist_ok=True)
        os.makedirs(f'{self.dataset_path}/processed/real_video_subset', exist_ok=True)
        os.makedirs(f'{self.dataset_path}/processed/real_video_blocks', exist_ok=True)
        os.makedirs(f'{self.dataset_path}/processed/audio_chunks', exist_ok=True)
        os.makedirs(f'{self.dataset_path}/processed/audio_generator_input', exist_ok=True)

    def process(self):
        for file in tqdm(self.files):
            audio, sr = librosa.load(f'{self.audio_dir}/{file}.wav', 
                                     mono=True,
                                     sr=self.data_config['audio']['sample_rate'])
            # Normalise data
            audio = audio / np.max(audio) 
            video = read_video(f'{self.video_dir}/{file}.mp4') / 255
            video = self.img_transform(video)
            
            cutting_stride = int(self.data_config['audio']['sample_rate'] / 
                                self.data_config['video']['fps'])
            audio_frame_feat_len = int(self.data_config['audio']['sample_rate'] * 
                                    self.data_config['audio']['frame_size'])
            audio_padding = audio_frame_feat_len - cutting_stride
            # Cut audio 1 clip per frame
            audio_generator_input = process_audio_for_generator(
                torch.tensor(audio).view(-1, 1),
                cutting_stride,
                audio_padding,
                audio_frame_feat_len)
            video, audio_generator_input = crop_to_same(video, audio_generator_input)
            # Split audio into 0.2s chunks for sync_discriminator
            audio_chunks = split_audio(
                torch.tensor(audio).view(-1, 1),
                self.data_config['audio']['sample_rate'], 
                self.data_config['audio']['sample_rate'])

            torch.save(video, f'{self.dataset_path}/processed/real_video_all/{file}.pt')
            torch.save(audio_chunks, f'{self.dataset_path}/processed/audio_chunks/{file}.pt')
            torch.save(audio_generator_input, f'{self.dataset_path}/processed/audio_generator_input/{file}.pt')
            
        print(f'Data processing complete, saved to folder {self.dataset_path}/processed')