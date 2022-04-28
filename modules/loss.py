import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, model_config, device=None):
        super().__init__()
        '''Loss for discriminator & generator'''
        if device is not None:
            self.device = device
        else: 
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            
        self.sync_loss_w = model_config['sync_loss_w']
        self.frame_loss_w = model_config['frame_loss_w']
        self.video_loss_w = model_config['video_loss_w']
        self.recon_loss_w = model_config['recon_loss_w']
        
    def discriminator_loss(self, 
                           fake_outputs,
                           real_outputs):
        '''
        Discriminators want to maximise the likelihood of correctly classifying
        ground truth values as true and generated values as false (1 & 0 respectively)
        '''
        real_targets = torch.cat((
            torch.ones((real_outputs['video_disc_output'].size(0), 1)),
            torch.ones((real_outputs['frame_disc_output'].size(0), 1)),
            torch.ones((real_outputs['sync_disc_output'].size(0), 1)),
        ), dim=0).to(self.device)
        real_predictions = torch.cat((
            real_outputs['video_disc_output'].unsqueeze(1),
            real_outputs['frame_disc_output'].unsqueeze(1),
            real_outputs['sync_disc_output'] ,
        ), dim=0).to(self.device)
        fake_targets = torch.cat((
            torch.zeros((fake_outputs['video_disc_output'].size(0), 1)),
            torch.zeros((fake_outputs['frame_disc_output'].size(0), 1)),
            torch.zeros((fake_outputs['sync_disc_output'].size(0), 1)),
            torch.zeros((fake_outputs['unsync_disc_output'].size(0), 1)),
        ), dim=0).to(self.device)
        fake_predictions = torch.cat((
            fake_outputs['video_disc_output'].unsqueeze(1),
            fake_outputs['frame_disc_output'].unsqueeze(1),
            fake_outputs['sync_disc_output'],
            fake_outputs['unsync_disc_output'],
        ), dim=0).to(self.device)
        
        real_loss = F.binary_cross_entropy(real_predictions, real_targets)
        fake_loss = F.binary_cross_entropy(fake_predictions, fake_targets)
        return real_loss, fake_loss
    
    def generator_loss(self,  
                       real_video_all,
                       fake_video_all,
                       fake_outputs):
        '''
        Reconstruction loss on bottom half of fake and real videos frames
        Generator maximises the likelihood of fake outputs being classed as real (1)
        '''
        targets = torch.cat((
            torch.ones((fake_outputs['video_disc_output'].size(0), 1)),
            torch.ones((fake_outputs['frame_disc_output'].size(0), 1)),
            torch.ones((fake_outputs['sync_disc_output'].size(0), 1)),
        ), dim=0).to(self.device)
        
        predictions = torch.cat((
            fake_outputs['video_disc_output'].unsqueeze(1),
            fake_outputs['frame_disc_output'].unsqueeze(1),
            fake_outputs['sync_disc_output'],
        ), dim=0).to(self.device)
        
        g_loss = F.binary_cross_entropy(predictions, targets)
        
        # Reconstruction loss
        half_size = fake_video_all.size(2) // 2
        fake_video_all = fake_video_all[:,:,half_size:,:]
        real_video_all = real_video_all[:,:,half_size:,:]
        recon_loss = F.l1_loss(fake_video_all, real_video_all)
        recon_loss *= self.recon_loss_w
        return g_loss, recon_loss
    