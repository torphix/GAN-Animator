from typing import OrderedDict
import torch
import torch.nn as nn

from data.utils import cut_video_sequence, sample_frames, shuffle_audio

from .utils import calculate_padding
import torch.nn.functional as F


class FrameDiscriminator(nn.Module):
    '''
    To get high quality frame image
    Expects generated / true image (256,256)
    Starting frame for identity (256,256)
    '''
    def __init__(self, config):
        super().__init__() 
        
        self.layers = nn.ModuleList()
        for i in range(len(config['feature_sizes'])-1):
            padding = calculate_padding(config['kernel_size'][i], 
                                        config['stride'])
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(6 if i == 0 else config['feature_sizes'][i],
                              config['feature_sizes'][i+1],
                              config['kernel_size'][i],
                              stride=config['stride'],
                              padding=padding),
                    nn.BatchNorm2d(config['feature_sizes'][i+1]),
                    nn.LeakyReLU(0.2, True)))
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(config['feature_sizes'][-1], 1, (8,7)),
                nn.Sigmoid()))
        # (10, 12)

    def forward(self, x, starting_frame):
        '''
        param: x: [BS, C, H, W] generated video
        param: starting_frame: [1, C, H, W] identity frame
        '''
        starting_frame = starting_frame.expand(x.size(0), -1, -1, -1)
        x = torch.cat((x, starting_frame), dim=1)
        for layer in self.layers:
            x = layer(x)
        return x.view(-1)


class VideoDiscriminator(nn.Module):
    '''To get high quality fluid video generation'''
    def __init__(self, config, img_size, audio_chunk_len):
        super().__init__() 
        
        img_config = config['img_encoder']
        self.img_encoder = nn.ModuleList()
        self.img_rnn = nn.RNN(**img_config['rnn'])
        kernels = list(zip(
            self._calculate_kernels(img_size[0], 
                                    len(img_config['feature_sizes']),
                                    img_config['stride']),
            self._calculate_kernels(img_size[1],
                                    len(img_config['feature_sizes']), 
                                    img_config['stride'])))
        for i in range(len(img_config['feature_sizes'])):
            self.img_encoder.append(
                nn.Sequential(
                    nn.Conv2d(img_config['feature_sizes'][i],
                              img_config['feature_sizes'][i+1]
                              if i != len(img_config['feature_sizes'])-1 
                              else img_config['feature_sizes'][i],
                              kernel_size=(img_config['kernels'][i][0], 
                                           img_config['kernels'][i][1]),
                              stride=img_config['stride']),
                    nn.BatchNorm2d(
                              img_config['feature_sizes'][i+1]
                              if i != len(img_config['feature_sizes'])-1
                              else img_config['feature_sizes'][i]),
                    nn.LeakyReLU()))
        
        
        audio_config = config['audio_encoder']
        self.audio_encoder = nn.ModuleList()
        self.audio_rnn = nn.RNN(**audio_config['rnn'])
        for i in range(len(audio_config['feature_sizes'])):
            self.audio_encoder.append(
                nn.Sequential(
                    nn.Conv1d(audio_config['feature_sizes'][i],
                              audio_config['feature_sizes'][i+1]
                              if i != len(audio_config['feature_sizes'])-1 
                              else audio_config['feature_sizes'][i],
                              kernel_size=int(audio_config['kernels'][i]),
                              stride=audio_config['stride']),
                    nn.BatchNorm1d(
                              audio_config['feature_sizes'][i+1]
                              if i != len(audio_config['feature_sizes'])-1
                              else audio_config['feature_sizes'][i]),
                    nn.LeakyReLU()))
        
        self.linear_classifier = nn.Sequential(
                        nn.Linear(512, 1),
                        nn.Sigmoid())
        
    def _calculate_kernels(self, input_d, n_layers, stride):
        k_sizes = []
        scalar = int(input_d / n_layers)
        for i in range(n_layers):
            if i != n_layers-1:
                target_d = input_d - scalar
                k_sizes.append(int(abs(((target_d - 1) / stride) - input_d)))
                input_d = target_d
            else: # Final layer
                k_sizes.append(input_d)
        return k_sizes
            
    def forward(self, frames, audio):
        '''
        param: video: frames (synthetic or real)
        [N (frames), C, H, W] -> [N (frames), 256] -> RNN -> Prediction
        param: audio: audio (real)
        Audio: [N (frames), L (0.2s*16000=3200), 1] -> [N (frames), 256] -> RNN -> Prediction
        '''
        for layer in self.img_encoder:
            frames = layer(frames)
        frames_z, _ = self.img_rnn(frames.squeeze(-1).squeeze(-1))
        
        audio = audio.squeeze(0)
        for layer in self.audio_encoder:
            audio = layer(audio)
        audio_z, _ = self.audio_rnn(audio.squeeze(-1))
        z = torch.cat((frames_z, audio_z), dim=-1)
        out = self.linear_classifier(z)
        return out
  
  
class SyncDiscriminator(nn.Module):
    '''Syncronization of video and frame'''
    def __init__(self, config, img_size):
        super().__init__() 
        # Video encoder
        self.video_encoder_layers = nn.ModuleList()
        self.video_encoder_layers.append(
            nn.Sequential(
                nn.Conv3d(3, config['video_feature_sizes'][0],
                          kernel_size=(5, 4, 4),
                          stride=(1,2,2)),
                nn.BatchNorm3d(config['video_feature_sizes'][0]),
                nn.LeakyReLU(0.2, inplace=True)))        
        
        for i in range(len(config['video_feature_sizes'])-1):
            self.video_encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(config['video_feature_sizes'][i],
                              config['video_feature_sizes'][i+1],
                              kernel_size=config['video_kernel_sizes'][i],
                              stride=config['video_stride']),
                    nn.BatchNorm2d(config['video_feature_sizes'][i+1]),
                    nn.LeakyReLU(0.2, inplace=True)))
            
        self.height = img_size[0] // 2
        # Only bottom half of image is used
        height, width = img_size[0] // 2, img_size[1]    
        for i in range(len(config['video_feature_sizes'])):
            # output_size = [(W-K + 2P) / S] + 1
            height = int((height-config['video_kernel_sizes'][i] + 2*0) / config['video_stride']) + 1
            width = int((width-config['video_kernel_sizes'][i] + 2*0) / config['video_stride']) + 1
        linear_input = int(height*width*config['video_feature_sizes'][-1])
        # TODO: Fix hardcoded input values
        self.video_linear = nn.Linear(2048, 256)
        
        # Audio encoder
        width = config['audio_length'] 
        self.audio_encoder_layers = nn.ModuleList()
        for i in range(len(config['audio_feature_sizes'])-1):
            # output_size = [(W-K + 2P) / S] + 1
            width = int((width-config['audio_kernel_sizes'][i] + 2*0) / config['audio_stride'][i]) + 1
            self.audio_encoder_layers.append(
                nn.Sequential(
                    nn.Conv1d(1 if i == 0 else config['audio_feature_sizes'][i],
                              config['audio_feature_sizes'][i+1],
                              kernel_size=config['audio_kernel_sizes'][i],
                              stride=config['audio_stride'][i]),
                    nn.BatchNorm1d(config['audio_feature_sizes'][i+1]),
                    nn.LeakyReLU(0.2, inplace=True)))

        linear_input = int(width*config['audio_feature_sizes'][-1])
        # TODO: Fix hardcoded input values
        self.audio_linear = nn.Linear(1280, 256)
        
        self.discriminator = nn.Sequential(
                                    nn.Linear(256, 1),
                                    nn.Sigmoid())
        
        self.param = nn.Parameter(torch.empty(0))
        
    @property    
    def device(self):
        return self.param.device  
    
    def pad(self, frames, audio):
        pad_amount = abs(audio.shape[0] - frames.shape[0])
        if audio.shape[0] > frames.shape[0]:
            pad_frame = torch.zeros((pad_amount, frames.shape[1]), 
                                     device=self.device)
            frames = torch.cat((frames, pad_frame),dim=0)
        elif audio.shape[0] < frames.shape[0]:
            pad_frame = torch.zeros((pad_amount, audio.shape[1]), 
                                     device=self.device)
            audio = torch.cat((audio, pad_frame),dim=0)
        return frames, audio
        
    def forward(self, frames, audio):
        '''
        Recieves chunks of aligned frams and video
        or misaligned chunks and deterines if aligned or not
        Frames: [Num_Chunks, ChunkLen(5), C, H, W]  
        Audio: [Chunks, L (0.2s) * Features, 1]
        Output: [Chunk, 1] Prediction for each frame
        '''
        frames = frames[:, :, :, frames.size(-2)//2:, :]
        assert self.height == frames.size(3), \
            f'Frame height should be {self.height} only half the image is required'
        # 3d conv input: [N, C, T, H, W]
        frames = frames.permute(0, 2, 1, 3, 4).contiguous()
        for i, layer in enumerate(self.video_encoder_layers):
            frames = layer(frames)
            if i == 0: frames = frames.squeeze(2)
        frames = frames.view(frames.size(0), -1)
        frame_emb = self.video_linear(frames)
        
        # Audio
        audio = audio.squeeze(0).float()
        for i, layer in enumerate(self.audio_encoder_layers):
            audio = layer(audio)
        audio = audio.view(audio.size(0), -1)
        audio_emb = self.audio_linear(audio)
        # Pad vid frames to match audio frames
        frame_emb, audio_emb = self.pad(frame_emb, audio_emb)
        # Compute score
        sim_score = (frame_emb - audio_emb)**2
        x = self.discriminator(sim_score)
        return x
    
    
class DiscriminatorsModule(nn.Module):
    def __init__(self, model_config, data_config):
        super().__init__()
        '''Input tensors Outputs Loss for each discriminator'''
        self.data_config = data_config

        self.video_disc = VideoDiscriminator(
            model_config['video_discriminator'],
            data_config['video']['img_size'],
            data_config['audio']['frame_size'] * data_config['audio']['sample_rate'])
        self.sync_disc = SyncDiscriminator(
            model_config['sync_discriminator'],
            data_config['video']['img_size'])
        self.frame_disc = FrameDiscriminator(
            model_config['frame_discriminator'])
                
    def fake_inference(self,
                       fake_video_all, 
                       real_video_all,
                       id_frame,
                       audio_chunks,
                       audio_generator_input,):
        '''When inputs are generated'''
        # Process generator output
        fake_video_subset = sample_frames(
            fake_video_all,
            self.data_config['video']['subset_size'])
        fake_video_blocks = cut_video_sequence(
            fake_video_all,
            self.data_config['video']['img_frames_per_audio_clip'])
        real_video_blocks = cut_video_sequence(
            real_video_all,
            self.data_config['video']['img_frames_per_audio_clip'])
        # Inference
        video_disc_output = self.video_disc(fake_video_all, audio_generator_input)
        frame_disc_output = self.frame_disc(fake_video_subset, id_frame)
        sync_disc_output = self.sync_disc(fake_video_blocks, audio_chunks)
        
        unsync_disc_output = self.sync_disc(real_video_blocks,
                                            shuffle_audio(audio_chunks))
        return OrderedDict({
            'video_disc_output':video_disc_output,
            'frame_disc_output':frame_disc_output,
            'sync_disc_output':sync_disc_output,
            'unsync_disc_output':unsync_disc_output,
        })        

    def real_inference(self,
                       real_video_all,
                       audio_generator_input,
                       audio_chunks,
                       id_frame):
        '''When inputs are real'''        
        # Processing
        real_video_subset = sample_frames(
            real_video_all, 
            self.data_config['video']['subset_size'])
        real_video_blocks = cut_video_sequence(
            real_video_all,
            self.data_config['video']['img_frames_per_audio_clip'])
        # Inference
        video_disc_output = self.video_disc(real_video_all, audio_generator_input)
        frame_disc_output = self.frame_disc(real_video_subset, id_frame)
        sync_disc_output = self.sync_disc(real_video_blocks,
                                          audio_chunks)
        return OrderedDict({
            'video_disc_output':video_disc_output,
            'frame_disc_output':frame_disc_output,
            'sync_disc_output':sync_disc_output,
        })
        
        
        