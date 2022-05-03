import torch.nn as nn
class VideoDiscriminator(nn.Module):
    '''To get high quality fluid video generation'''
    def __init__(self, config, img_size):
        super().__init__() 
        
        self.img_encoder = nn.ModuleList()
        kernels = list(zip(
            self._calculate_kernels(img_size[0], config['n_layers'], config['stride']),
            self._calculate_kernels(img_size[1], config['n_layers'], config['stride'])))
        print(kernels)
        for i in range(config['n_layers']):
            self.img_encoder.append(
                nn.Sequential(
                    nn.Conv2d(config['feature_sizes'][i],
                              config['feature_sizes'][i+1]
                              if i != config['n_layers']-1 else 
                              config['feature_sizes'][i],
                              (kernels[i][0], kernels[i][1]),
                              stride=config['stride']),
                    nn.LeakyReLU()))
        
    def _calculate_kernels(self, input_d, n_layers, stride):
        k_sizes = []
        scalar = int(input_d / n_layers)
        for i in range(n_layers):
            if i != n_layers-1:
                target_d = input_d - scalar
                k_sizes.append(int(abs(((target_d - 1) / stride) - input_d)))
                input_d = target_d
            else: # Final layer
                k_sizes.append(input_d-2)
        return k_sizes
            
    def forward(self, frames):
        '''
        param: video: frames (synthetic or real)
        [N (frames), C, H, W] -> [N (frames), 256] -> RNN -> Prediction
        param: audio: audio (real)
        Audio: [N (frames), L (0.2s*16000=3200), 1] -> [N (frames), 256] -> RNN -> Prediction
        '''
        for i in self.img_encoder:
            frames = i(frames)
            print(frames.shape)
        
        return frames
  
  
config = {
    'stride':1,
    'n_layers':4,
    'feature_sizes':[3, 2, 3, 4, 256],
}  
dis = VideoDiscriminator(config, (128, 96))

import torch
x = torch.randn((12, 3, 128, 96))
out = dis(x)
print(out.size())

  