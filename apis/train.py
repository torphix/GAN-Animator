import torch
from modules.loss import Loss
import pytorch_lightning as ptl
from data.data import GANDataset
from collections import OrderedDict
from pytorch_lightning import Trainer
from modules.utils import get_configs
from modules.generator import Generator
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning import loggers as pl_loggers
from modules.discriminators import DiscriminatorsModule


class TrainModule(ptl.LightningModule):
    def __init__(self, model_config, data_config, train_config):
        super().__init__()
        self.data_config, self.model_config, self.train_config = \
            data_config, model_config, train_config
        self.discriminators = DiscriminatorsModule(model_config, data_config)
        self.generator = Generator(model_config['generator'], data_config['video']['img_size'])
        self.loss = Loss(model_config)
        
    def log_values(self, logger, images, scalers):
        # Save subset of images
        for i in range(images.shape[0]):
            if i % 5 == 0: logger.add_image(
                'Fake images', images[i])
        # Save scalers
        for k, v in scalers.items():
            logger.add_scalar(k, v)            
            
        save_image(images, 'output.png')
            

    def train_dataloader(self):
        dataset = GANDataset(self.data_config)
        return DataLoader(dataset, 
                          **self.train_config['train_dataloader'],
                          collate_fn=dataset.collate_fn)

    def training_step(self, batch, batch_idx, optimizer_idx):
        logger = self.logger.experiment
        fake_video_all = self.generator(batch['id_frame'], batch['audio_generator_input'])
        try:
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
                real_loss, fake_loss = self.loss.discriminator_loss(fake_out, real_out)
                disc_loss = real_loss + fake_loss
                
                tqdm_dict = {"disc_loss": disc_loss}
                output = OrderedDict({"loss": disc_loss,
                                    "progress_bar": tqdm_dict,
                                    "log": tqdm_dict})
                
                if batch_idx % 100 == 0:
                    scalers = {
                        'Discriminator Real Loss': real_loss,
                        'Discriminator Fake Loss': fake_loss,
                        'Discriminator Total Loss': disc_loss,
                    }
                    self.log_values(logger, fake_video_all, scalers)
                
                return disc_loss
                
            # Train Generator
            elif optimizer_idx == 1:
                fake_out = self.discriminators.fake_inference(
                                    fake_video_all,
                                    batch['real_video_all'],
                                    batch['id_frame'],
                                    batch['audio_chunks'])
                gen_loss, recon_loss = self.loss.generator_loss(
                                            batch['real_video_all'],
                                            fake_video_all,
                                            fake_out)
                total_loss = gen_loss + recon_loss
                
                tqdm_dict = {"gen_loss": total_loss}
                output = OrderedDict({"loss": total_loss, 
                                    "progress_bar": tqdm_dict,
                                    "log": tqdm_dict})
                
                if batch_idx % 100 == 0:
                    scalers = {
                        'Generator Loss': gen_loss,
                        'Reconstruction Loss': recon_loss,
                        'Total Generator Loss': total_loss,
                    }
                    self.log_values(logger, fake_video_all, scalers)
                
                return output
        except:
            print(batch['file_id'])    
                    
    def configure_optimizers(self):
        opt_d = torch.optim.Adam(self.discriminators.parameters(), 
                                 **self.train_config['d_optim'])
        opt_g = torch.optim.Adam(self.generator.parameters(), 
                                 **self.train_config['g_optim'])
        scheduler_d = StepLR(opt_d, **self.train_config['d_scheduler'])
        scheduler_g = StepLR(opt_g, **self.train_config['g_scheduler'])
        return [opt_d, opt_g], [scheduler_d, scheduler_g]            



def train():
    model_config, data_config, train_config = get_configs()
    trainer_config = train_config['trainer']
    
    module = TrainModule(model_config, data_config, train_config)    
    
    if trainer_config['model_dir'] != '' and trainer_config['model_dir'] != None:
        print('Loading Saved Models...')
        module.generator.load_state_dict(torch.load(f"{trainer_config['model_dir']}/generator.pth"))
        module.sync_discriminator.load_state_dict(torch.load(f"{trainer_config['model_dir']}/sync_discriminator.pth"))
        module.frame_discriminator.load_state_dict(torch.load(f"{trainer_config['model_dir']}/frame_discriminator.pth"))
        module.video_discriminator.load_state_dict(torch.load(f"{trainer_config['model_dir']}/video_discriminator.pth"))

    ckpt_path = trainer_config['checkpoint_path']
    trainer_config.pop('checkpoint_path')
    trainer_config.pop('model_dir')
    
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="tb_logs/")
    trainer = Trainer(**trainer_config, 
                      logger=tb_logger)
    trainer.fit(module, ckpt_path=ckpt_path)
    torch.save(module.generator.state_dict(), f'saved_models/{trainer_config["trainer"]["max_epochs"]}/gen.pth')