from multiprocessing.sharedctypes import Value
import os
import torch
from .utils import (pad, read_video,
                   crop_to_same,
                   cut_video_sequence,
                   sample_frames,
                   split_audio)
from torchvision import transforms
from torch.utils.data import Dataset

class GANDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        assert config["name"] in ['lex', 'CREMA-D'], \
            "Dataset names 'lex' or 'CREMA-D' currently supported"
        self.config = config
        self.root_path = f'data/datasets/{config["name"]}/processed'
        self.video_dir = f'data/datasets/{config["name"]}/VideoFlash'
        self.audio_dir = f'data/datasets/{config["name"]}/AudioWAV'
        
        self.files = [file.split(".")[0] for file in os.listdir(self.video_dir)]
        
        self.img_transform = transforms.Compose([
            transforms.Resize((config['video']['img_size'][0], 
                               config['video']['img_size'][1])),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        
        self.use_emotion = config['use_emotion']
        self.emotions = config['emotions']
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio_chunks = torch.load(f'{self.root_path}/audio_chunks/{self.files[idx]}.pt')
        audio_generator_input = torch.load(f'{self.root_path}/audio_generator_input/{self.files[idx]}.pt')
        
        video = read_video(f'{self.video_dir}/{self.files[idx]}.mp4', 
                           self.config['video']['fps']) 
        video = video / 255
        video = self.img_transform(video)
        video, audio_generator_input = crop_to_same(video, audio_generator_input)
        
        if self.use_emotion:
            try:
                emotion = self.emotions[self.files[idx].split("_")[2]]
            except: raise ValueError('Emotion prefix not in filename')
            
        else: emotion = None
        
        datapoint = {
            'file_id':self.files[idx],
            'emotion':emotion,
            # Discriminator inputs
            'real_video_all': video,
            'audio_chunks': audio_chunks,
            # Generator inputs
            'id_frame': video[0],
            'audio_generator_input': audio_generator_input,
        }
        return datapoint


    def collate_fn(self, batch):
        file_id = []
        real_video_all = []
        audio_chunks = []
        id_frames = []
        audio_generator_input = []
        for d in batch:
            file_id.append(d['file_id'])
            real_video_all.append(d['real_video_all'])
            audio_chunks.append(d['audio_chunks'])
            id_frames.append(d['id_frame'])
            audio_generator_input.append(d['audio_generator_input'])
        batch = {
            'file_id':file_id,
            'real_video_all':torch.stack(real_video_all).squeeze(0),
            'audio_chunks':torch.stack(audio_chunks),
            'id_frame':torch.stack(id_frames),
            'audio_generator_input':torch.stack(
                audio_generator_input).squeeze(0).transpose(1,2),
        }
        return batch