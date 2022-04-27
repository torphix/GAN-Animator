import os
import yaml
import torch
from modules.loss import Loss
import pytorch_lightning as ptl
from data.data import GANDataset
from collections import OrderedDict
from pytorch_lightning import Trainer
from modules.generator import Generator
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from modules.discriminators import DiscriminatorsModule



class TrainModule(ptl.LightningModule):
    def __init__(self, model_config, data_config):
        self.data_config, self.model_config = data_config, model_config
        self.discriminators = DiscriminatorsModule(model_config, data_config)
        self.generator = Generator(model_config['generator'], data_config['video']['img_size'])
        self.loss = Loss(model_config)

    def train_dataloader(self):
        dataset = GANDataset(self.data_config)
        return DataLoader(dataset, **self.data_config['dataloader'])

    def training_step(self, batch, batch_idx, optimizer_idx):
        fake_video_all = self.generator(batch['id_frame'], batch['audio_generator_input'])
        # Train Discriminator
        if optimizer_idx == 0:    
            fake_out = self.discriminators.fake_inference(
                                fake_video_all, 
                                batch['real_video_all'],
                                batch['id_frame'],
                                batch['audio_chunks'])
            real_out = self.discriminators.real_inference(
                                batch['real_video_all'],
                                batch['audio_chunks'],
                                batch['id_frame'])
            disc_loss = self.loss.discriminator_loss(fake_out, real_out)
            
            tqdm_dict = {"disc_loss": disc_loss}
            output = OrderedDict({"loss": disc_loss,
                                  "progress_bar": tqdm_dict,
                                  "log": tqdm_dict})
            return disc_loss
            
        # Train Generator
        elif optimizer_idx == 1:
            fake_out = self.discriminators.fake_inference(
                                fake_video_all,
                                batch['real_video_all'],
                                batch['id_frame'],
                                batch['audio_chunks'])
            gen_loss = self.loss.generator_loss(batch['real_video_all'],
                                                fake_video_all,
                                                fake_out)
            tqdm_dict = {"gen_loss": gen_loss}
            output = OrderedDict({"loss": gen_loss, 
                                  "progress_bar": tqdm_dict,
                                  "log": tqdm_dict})
            return output
        
    def configure_optimizers(self):
        opt_d = torch.optim.Adam(self.discriminator.parameters(), 
                                 **self.data_config['d_optim'])
        opt_g = torch.optim.Adam(self.generator.parameters(), 
                                 **self.data_config['g_optim'])
        scheduler_d = StepLR(opt_d, **self.data_config['d_scheduler'])
        scheduler_g = StepLR(opt_g, **self.data_config['g_scheduler'])
        return [opt_d, opt_g], [scheduler_d, scheduler_g]            



def train():
    root = os.path.abspath('.')
    module = TrainModule(f'{root}/configs/models.yaml',
                         f'{root}/configs/data.yaml')    
    with open(f'{root}/configs/trainer.yaml', 'r') as f:
        trainer_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        
    if trainer_config['model_dir'] != '' and trainer_config['model_dir'] != None:
        print('Loading Saved Models...')
        module.generator.load_state_dict(torch.load(f"{trainer_config['model_dir']}/generator.pth"))
        module.sync_discriminator.load_state_dict(torch.load(f"{trainer_config['model_dir']}/sync_discriminator.pth"))
        module.frame_discriminator.load_state_dict(torch.load(f"{trainer_config['model_dir']}/frame_discriminator.pth"))
        module.video_discriminator.load_state_dict(torch.load(f"{trainer_config['model_dir']}/video_discriminator.pth"))

    ckpt_path = trainer_config['checkpoint_path']
    trainer_config.pop('checkpoint_path')
    trainer_config.pop('model_dir')
    
    trainer = Trainer(**trainer_config)
    trainer.fit(module, ckpt_path=ckpt_path)