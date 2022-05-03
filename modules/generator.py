import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import (calculate_output_length, 
                    calculate_padding,
                    is_power2, prime_factors)


class ImageEncoder(nn.Module):
    def __init__(self, config, img_size):
        super().__init__() 
        # Get the dimension which is a power of 2
        if is_power2(max(img_size)):
            stable_dim = max(img_size)
        else:
            stable_dim = min(img_size)
        # TODO: Hard coded should be variable
        final_size_h = int(4 * img_size[0] // stable_dim)+4
        final_size_w = int(4 * img_size[1] // stable_dim)+3
        padding = calculate_padding(config['kernel_size'], 
                                    config['stride'])
        num_layers = int(math.log2(max(img_size)))-2
        
        self.layers = nn.ModuleList()
        hid_d = config['hid_d']
        for i in range(num_layers-1):
            self.layers.append(
                self._make_layer(3 if i == 0 else hid_d,
                                 hid_d * 2,
                                 config['kernel_size'],
                                 config['stride'],
                                 padding//2))
            hid_d = hid_d * 2
        self.layers.append(
                nn.Sequential(
                    nn.Conv2d(hid_d, config['out_d'], 
                              (final_size_h, final_size_w)),
                    nn.Tanh()))
            
    def _make_layer(self, in_d, out_d, kernel_size, stride=1, pad=0):
            return nn.Sequential(
                nn.Conv2d(in_d, out_d, kernel_size, stride, pad),
                nn.BatchNorm2d(out_d),
                nn.ReLU())
            
    def forward(self, x):
        img_latent = []
        for layer in self.layers:
            x = layer(x)
            img_latent.append(x)
        img_latent[-1] = img_latent[-1].squeeze(-1).squeeze(-1)
        return img_latent
    

class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__() 
        hid_d=config['hid_d']
        out_d=config['out_d']
        features = config['audio_length'] * config['sample_rate']
        strides = prime_factors(features)
        kernels = [2 * strides[i] for i in range(len(strides))]
        paddings = [calculate_padding(kernels[i], strides[i], features)
                    for i in range(len(strides))]

        self.layers = nn.ModuleList()
        for i in range(len(strides)):
            features = calculate_output_length(features, 
                                                kernels[i],
                                                stride=strides[i],
                                                padding=paddings[i])
            # Layer 1
            if i == 0:
                self.layers.append(nn.Sequential(
                    nn.Conv1d(1, hid_d, kernels[i], strides[i], paddings[i]),
                    nn.BatchNorm1d(hid_d),
                    nn.ReLU(),    
                ))
            # Intermediate layers
            else:
                self.layers.append(nn.Sequential(
                    nn.Conv1d(hid_d, hid_d*2, kernels[i], strides[i], paddings[i]),
                    nn.BatchNorm1d(hid_d*2),
                    nn.ReLU(),    
                ))
                hid_d = hid_d*2
            # Output layer
        self.layers.append(nn.Sequential(
                    nn.Conv1d(hid_d, out_d, features),
                    nn.BatchNorm1d(out_d),
                    nn.Tanh(),    
                ))
        self.layers = nn.Sequential(
            *self.layers
        )
        self.encoder = nn.GRU(**config['gru'], batch_first=True)
        
    def forward(self, audio):
        '''
        audio: [BS, N, L]
        z: [BS, L, N]
        out: [BS, N, L]
        '''
        z = self.layers(audio)
        z, h_0 = self.encoder(z.transpose(1,2))
        return z.transpose(1,2).squeeze(-1)


class Encoder(nn.Module):
    def __init__(self, config, img_size):
        super().__init__() 
        self.image_encoder = ImageEncoder(config['image_encoder'], img_size)
        self.audio_encoder = AudioEncoder(config['audio_encoder'])
        self.noise_encoder = nn.GRU(**config['noise_generator'])
        self.emotion_encoder = nn.Sequential(
                  nn.Embedding(config['emotion_enc']['n_emos'],
                               config['emotion_enc']['hid_d'],
                               padding_idx=0),
                  nn.Linear(config['emotion_enc']['hid_d'],
                            config['emotion_enc']['hid_d']),
                  nn.ReLU())

        self.param = nn.Parameter(torch.empty(0))
        
    @property    
    def device(self):
        return self.param.device
        
    def forward(self, img, audio, emotion=None):
        '''
        param: img: [3, H, W] identity frame
        param: audio: [N (frames), 1, L]
        '''
        N = audio.shape[0]
        noise = torch.FloatTensor(N, 1, 256).normal_(0, 0.33).to(self.device)
        img = img.expand(audio.size(0), 
                         *img.shape[1:])
        noise_z, h_0 = self.noise_encoder(noise)
        noise_z = F.tanh(noise_z)
        audio_z = self.audio_encoder(audio)
        img_zs = self.image_encoder(img)
        if emotion is not None:
            emotion_emb = self.emotion_encoder(emotion)
            out = torch.cat((img_zs[-1],
                             audio_z,
                             noise_z.squeeze(1)),
                             emotion_emb,
                             dim=-1)
        else: 
            out = torch.cat((img_zs[-1],
                            audio_z,
                            noise_z.squeeze(1)),
                            dim=-1)

        return out, img_zs


class FrameDecoder(nn.Module):
    def __init__(self, config, img_size):
        super().__init__() 
        
        padding_h = calculate_padding(config['kernel_size'][0], 
                                      config['stride'])
        padding_w = calculate_padding(config['kernel_size'][1], 
                                      config['stride'])
        
        num_layers = int(math.log2(max(img_size)))-2
        hid_ds = self.get_hid_ds(num_layers, config['hid_d'])
        
        self.layers = nn.ModuleList()
        self.prelayers = nn.ModuleList()
        
        self.start_layer = self._make_layer(config['in_d'], 
                                            hid_ds[0], 
                                            config['kernel_size'])
        
        for i in range(num_layers-1):            
            self.prelayers.append(
                self._make_layer(
                    # Stacking enc with dec (Unet)
                    hid_ds[i] + hid_ds[i],
                    hid_ds[i],
                    kernel_size=3,pad=1))
            
            self.layers.append(
                self._make_layer(hid_ds[i],
                                 hid_ds[i+1],
                                 stride=config['stride'],
                                 kernel_size=config['kernel_size'],
                                 pad=(padding_h//2, padding_w//2)))
            
        self.end_layer = nn.Sequential(
            nn.ConvTranspose2d(hid_ds[-1], 3, 1),
            nn.Tanh(),
        )
            
    def get_hid_ds(self, num_layers, start_dim):
        hid_ds = []
        for i in range(num_layers):
            hid_ds.append(start_dim)
            start_dim *= 2
        hid_ds.reverse()
        return hid_ds
            
    def _make_layer(self, in_d, out_d, kernel_size, stride=1, pad=0):
            return nn.Sequential(
                nn.ConvTranspose2d(in_d,out_d,
                                   kernel_size, 
                                   stride, pad),
                nn.BatchNorm2d(out_d),
                nn.ReLU())
            
    def forward(self, x, img_latents):
        img_latents.pop() # Last img latent already included in x
        img_latents.reverse() 
        x = self.start_layer(x.view(*x.size(), 1, 1))
        for i, layer in enumerate(self.layers):
            x = torch.cat((x, img_latents[i]), dim=1)
            x = self.prelayers[i](x)
            x = layer(x) 
        x = self.end_layer(x)
        return x
    
    
class Generator(nn.Module):
    def __init__(self, config, img_size):
        super().__init__() 
        self.encoder = Encoder(config['encoder'], img_size)
        self.decoder = FrameDecoder(config['frame_decoder'], img_size)
        
    def forward(self, img, audio_frames):
        '''
        Input Frame: Same image BS times [BS, C, H, W]
        Audio Frames: [BS (# of frames), 0.2s + pad Len, 1]
        '''
        latent, img_latents = self.encoder(img, audio_frames)
        generated_frames = self.decoder(latent, img_latents)
        return generated_frames

    
