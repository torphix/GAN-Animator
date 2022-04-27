import torch
from torch.utils.data import Dataset

class GANDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        assert config["name"] in ['lex', 'CREMA-D'], \
            "Dataset names 'lex' or 'CREMA-D' currently supported"
        self.root_path = f'data/datasets/{config["name"]}/procesed'
        
    def __len__(self):
        return len([file for file in f'{self.root_path}/audio_chunks'])
    
    def __getitem__(self, idx):
        '''
        Audio clip : video frames -> 1:5 ratio , 0.2s : 5x 
        '''
        real_video_all = torch.load(f'{self.root_path}/real_video_all')
        audio_chunks = torch.load(f'{self.root_path}/audio_chunks')
        audio_generator_input = torch.load(f'{self.root_path}/audio_generator_input')
        
        datapoint = {
            # Discriminator inputs
            'real_video_all': real_video_all,
            'audio_chunks': audio_chunks,
            # Generator inputs
            'id_frame': real_video_all[0],
            'audio_generator_input': audio_generator_input,
        }
        return datapoint
