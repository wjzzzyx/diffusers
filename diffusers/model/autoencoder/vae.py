from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.model.autoencoder.modules import Encoder, Decoder, NLayerDiscriminator, weights_init
from diffusers.model.autoencoder.distributions import DiagonalGaussianDistribution
from diffusers.model.autoencoder.lpips import LPIPS


class AutoEncoderKL(nn.Module):
    "Variational AutoEncoder."
    def __init__(self, model_config):
        super().__init__()
        self.config = model_config
        self.encoder = Encoder(**model_config)
        self.decoder = Decoder(**model_config)
        self.quant_conv = nn.Conv2d(2 * model_config.z_channels, 2 * model_config.z_channels, 1)
        self.post_quant_conv = nn.Conv2d(model_config.z_channels, model_config.z_channels, 1)
    
    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior
    
    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
    
    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec


class AutoEncoderWithDisc(nn.Module):

    def __init__(
        self,
        model_config,
    ):
        super().__init__()
        self.config = model_config
        self.lw_perceptual = model_config.loss.weight_perceptual
        self.lw_kl = model_config.loss.weight_kl
        self.lw_gan = model_config.loss.weight_gan
        self.gan_start_step = model_config.loss.gan_start_step
        self.ae = AutoEncoderKL(model_config.autoencoder)
        self.discriminator = NLayerDiscriminator(**model_config.discriminator).apply(weights_init)
        self.perceptual_loss = LPIPS().eval()
        self.logvar = nn.Parameter(torch.ones(size=()) * model_config.logvar_init)

    def forward(self, x, optimizer_idx, global_step, sample_posterior=True):
        # TODO make it more efficient by specifying run what parts
        recon, posterior = self.ae(x, sample_posterior)

        if optimizer_idx == 0:    # train generator
            # reconstruction loss
            loss_rec = torch.abs(x - recon)
            # perceptual loss
            if self.lw_perceptual > 0:
                loss_per = self.perceptual_loss(x, recon)
                loss_rec = loss_rec + self.lw_perceptual * loss_per
            # nll loss of Laplacian distribution P(X|Z)?
            loss_nll = loss_rec / torch.exp(self.logvar) + self.logvar
            # No weights are used
            loss_nll_weighted = loss_nll
            loss_nll = torch.sum(loss_nll) / loss_nll.size(0)
            loss_nll_weighted = torch.sum(loss_nll_weighted) / loss_nll_weighted.size(0)
            # prior loss
            loss_kl = posterior.kl()
            loss_kl = torch.sum(loss_kl) / loss_kl.size(0)

            loss = loss_nll_weighted + self.lw_kl * loss_kl
            log_dict = {
                'loss_nll': loss_nll.detach().mean(),
                'loss_rec': loss_rec.detach().mean(),
                'loss_kl': loss_kl.detach().mean(),
                'logvar': self.logvar.detach(),
            }

            # GAN loss
            if global_step >= self.gan_start_step:
                logits_fake = self.discriminator(recon)
                loss_g = -torch.mean(logits_fake)
                if self.training:
                    lw_gan_adap = self.calculate_adaptive_weight(loss_nll, loss_g)
                else:
                    lw_gan_adap = torch.tensor(0.0)

                loss += self.lw_gan * lw_gan_adap * loss_g
                log_dict.update({
                    'loss_g': loss_g.detach().mean(),
                    'd_weight': lw_gan_adap.detach(),
                })
            
            log_dict['loss_total'] = loss.clone().detach().mean(),
            return loss, log_dict
        
        if optimizer_idx == 1:    # train discriminator
            logits_real = self.discriminator(x.detach())
            logits_fake = self.discriminator(recon.detach())
            loss_d = self.hinge_d_loss(logits_real, logits_fake)

            loss = loss_d

            log_dict = {
                'loss_d': loss_d.clone().detach().mean(),
                'logits_real': logits_real.detach().mean(),
                'logits_fake': logits_fake.detach().mean(),
            }
            return loss, log_dict
    
    def calculate_adaptive_weight(self, loss_nll, loss_g):
        grad_nll = torch.autograd.grad(loss_nll, self.ae.decoder.conv_out.weight, retain_graph=True)[0]
        grad_g = torch.autograd.grad(loss_g, self.ae.decoder.conv_out.weight, retain_graph=True)[0]
        d_weight = torch.norm(grad_nll) / (torch.norm(grad_g) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight
    
    def hinge_d_loss(self, logits_real, logits_fake):
        loss_real = torch.mean(F.relu(1. - logits_real))
        loss_fake = torch.mean(F.relu(1. + logits_fake))
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss

    def vanilla_d_loss(self, logits_real, logits_fake):
        d_loss = 0.5 * (
            torch.mean(torch.nn.functional.softplus(-logits_real)) +
            torch.mean(torch.nn.functional.softplus(logits_fake)))
        return d_loss


class PLAutoEncoderWithDisc(pl.LightningModule):

    def __init__(
        self,
        model_config: OmegaConf,
        ckpt_path=None,
    ):
        super().__init__()
        self.config = model_config
        self.model = AutoEncoderWithDisc(model_config)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
        self.automatic_optimization = False
    
    def init_from_ckpt(self, path):
        ckpt = torch.load(path, map_location='cpu')
        missing, unexpected = self.model.load_state_dict(ckpt['state_dict'], strict=False)
        print(f'Restored from checkpoint {path}.')
        print(f'Missing keys: {missing}.')
        print(f'Unexpected keys: {unexpected}.')
    
    def forward(self, x, sample_posterior=True):
        return self.model(x, sample_posterior)

    def training_step(self, batch, batch_idx):
        # Should we train the generator and discriminator at different steps?
        if self.global_step >= self.config.loss.gan_start_step \
            and self.global_step % 2 == 1:    # train discriminator
            optimizer_idx = 1
        else:    # train generator
            optimizer_idx = 0
        optimizer_g, optimizer_d = self.optimizers(use_pl_optimizer=True)
        loss, log_dict = self.model(batch['image'], optimizer_idx, self.global_step, sample_posterior=True)
        log_dict = {f'train/{k}': v for k, v in log_dict.items()}
        
        if optimizer_idx == 0:
            self.toggle_optimizer(optimizer_g)
            optimizer_g.zero_grad()
            self.manual_backward(loss)
            optimizer_g.step()
            self.untoggle_optimizer(optimizer_g)

            self.log('aeloss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        
        else:
            self.toggle_optimizer(optimizer_d)
            optimizer_d.zero_grad()
            self.manual_backward(loss)
            optimizer_d.step()
            self.untoggle_optimizer(optimizer_d)

            self.log('discloss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False)
    
    def validation_step(self, batch, batch_idx):
        loss_ae, log_dict_ae = self.model(batch['image'], 0, self.global_step, sample_posterior=True)
        loss_disc, log_dict_disc = self.model(batch['image'], 1, self.global_step, sample_posterior=True)
        log_dict_ae = {f'val/{k}': v for k, v in log_dict_ae.items()}
        log_dict_disc = {f'val/{k}': v for k, v in log_dict_disc.items()}
        self.log('val/loss_rec', log_dict_ae['val/loss_rec'])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
    
    def configure_optimizers(self,):
        # TODO logvar is not optimized?
        lr = self.config.learning_rate
        optimizer_g = torch.optim.Adam(
            self.model.ae.parameters(), lr=lr, betas=(0.5, 0.9)
        )
        optimizer_d = torch.optim.Adam(
            self.model.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
        )
        return [optimizer_g, optimizer_d], []

    @torch.no_grad()
    def log_images(self, batch, **unused_kwargs):
        image = batch['image']
        rec, posterior = self.model.ae(image, sample_posterior=True)
        sample = self.model.ae.decode(torch.randn_like(posterior.sample()))
        return {
            'inputs': image,
            'reconstructions': rec,
            'samples': sample,
        }
