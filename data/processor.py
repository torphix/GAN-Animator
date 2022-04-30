import os
import yaml
import torch
import librosa
import numpy as np
from tqdm import tqdm

from data.face_alignment.face_alignment import Aligner
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
        
        self.aligner = Aligner(args.dataset_path)
        self.data_config = yaml.load(open('configs/data.yaml','r'),
                                          Loader=yaml.FullLoader)

        assert self.data_config["name"] in ['lex', 'CREMA-D'], \
            "Dataset names 'lex' or 'CREMA-D' currently supported"
        print(f'''
              Processing dataset in folder {args.dataset_path}/{self.data_config["name"]}
              As specified in data_config["name"]
              ''')

        self.dataset_path = f'{args.dataset_path}/{self.data_config["name"]}'
        self.audio_dir = f'{self.dataset_path}/AudioWAV'
        self.video_dir = f'{self.dataset_path}/VideoFlash'
        self.files = [file.split(".")[0] for file in os.listdir(self.video_dir) 
                      if file.endswith("mp4")]
        
        os.makedirs(f'{self.dataset_path}/processed/audio_chunks', exist_ok=True)
        os.makedirs(f'{self.dataset_path}/processed/audio_generator_input', exist_ok=True)
        
    def align(self):
        self.aligner.align_dataset()

    def process(self):
        self.clean_files()
        assert len(os.listdir(self.audio_dir)) == len(os.listdir(self.video_dir)), \
            f'Audio {len(os.listdir(self.audio_dir))} and video {len(os.listdir(self.video_dir))}'
        for file in tqdm(self.files):
            video = read_video(f'{self.video_dir}/{file}.mp4', 
                               self.data_config['video']['fps'])
            # Remove datapoints shorter than 1.2 seconds
            if video.shape[0] < int(self.data_config['video']['fps'] * 1.2):
                os.remove(f'{self.video_dir}/{file}.mp4')
                os.remove(f'{self.audio_dir}/{file}.wav')
                continue
            

            audio, sr = librosa.load(f'{self.audio_dir}/{file}.wav', 
                                     mono=True,
                                     sr=self.data_config['audio']['sample_rate'])
            # Normalise data
            audio = audio / np.max(audio) 
            cutting_stride = int(self.data_config['audio']['sample_rate'] / 
                                self.data_config['video']['fps'])
            audio_frame_feat_len = int(self.data_config['audio']['sample_rate'] * 
                                    self.data_config['audio']['frame_size'])
            audio_padding = audio_frame_feat_len - cutting_stride
            # Cut audio 1 clip per frame 0.2s padded on each side
            audio_generator_input = process_audio_for_generator(
                torch.tensor(audio).view(-1, 1),
                cutting_stride,
                audio_padding,
                audio_frame_feat_len)
            # Split audio into 0.2s chunks for sync_discriminator
            audio_chunks = split_audio(
                torch.tensor(audio).view(-1, 1),
                self.data_config['audio']['sample_rate'], 
                self.data_config['audio']['frame_size'])

            torch.save(audio_chunks, f'{self.dataset_path}/processed/audio_chunks/{file}.pt')
            torch.save(audio_generator_input, f'{self.dataset_path}/processed/audio_generator_input/{file}.pt')
        print(f'Data processing complete, saved to folder {self.dataset_path}/processed')
    
    def clean_files(self):
        [os.remove(f'{self.video_dir}/{file}')
         for file in os.listdir(self.video_dir) 
         if file.endswith('mp4') == False]
        [os.remove(f'{self.audio_dir}/{file}')
         for file in os.listdir(self.audio_dir) 
         if file.endswith('wav') == False]
        audios=[file.split('.')[0] for file in os.listdir(self.audio_dir) if file.endswith('wav')]
        videos=[file.split('.')[0] for file in os.listdir(self.video_dir) if file.endswith('mp4')]
        not_found = [file for file in videos if file not in audios]
        [os.remove(f'{self.video_dir}/{file}.mp4') for file in not_found]
        not_found = [file for file in audios if file not in videos]
        [os.remove(f'{self.audio_dir}/{file}.wav') for file in not_found]
