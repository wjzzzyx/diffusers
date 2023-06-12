import einops
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.model.autoencoder.modules import Encoder, Decoder, NLayerDiscriminator, weights_init
from diffusers.model.autoencoder.lpips import LPIPS
import utils


class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = einops.rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, einops.rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = einops.rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class AutoEncoderVQ(nn.Module):

    def __init__(self, model_config):
        super().__init__()
        self.encoder = Encoder(**model_config)
        self.decoder = Decoder(**model_config)
        self.pre_quant_conv = nn.Conv2d(model_config.z_channels, model_config.z_channels, 1)
        self.quantize = VectorQuantizer(model_config.n_embed, model_config.z_channels, beta=0.25)
        self.post_quant_conv = nn.Conv2d(model_config.z_channels, model_config.z_channels, 1)
    
    def forward(self, x):
        quant, loss_vq, (_, _, ind) = self.encode(x)
        dec = self.decode(quant)
        return dec, loss_vq, ind

    def encode(self, x):
        h = self.encoder(x)
        h = self.pre_quant_conv(h)
        quant, loss_vq, info = self.quantize(h)
        return quant, loss_vq, info
    
    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec


class AutoEncoderVQWithDisc(nn.Module):

    def __init__(self, model_config):
        super().__init__()
        self.config = model_config
        self.loss_recon_type = model_config.loss.loss_recon_type
        self.lw_perceptual = model_config.loss.weight_perceptual
        self.lw_vq = model_config.loss.weight_vq
        self.lw_gan = model_config.loss.weight_gan
        self.gan_start_step = model_config.loss.gan_start_step
        self.ae = AutoEncoderVQ(model_config.autoencoder)
        self.discriminator = NLayerDiscriminator(**model_config.discriminator).apply(weights_init)
        self.perceptual_loss = LPIPS().eval()
    
    def forward(self, x, optimizer_idx, global_step):
        # TODO move the loss computation to one place
        recon, loss_vq, ind = self.ae(x)
        
        if optimizer_idx == 0:
            # reconstruction loss
            if self.loss_recon_type == 'l1':
                loss_rec = torch.abs(x - recon)
            elif self.loss_recon_type == 'l2':
                loss_rec = torch.pow(x - recon, 2)
            else:
                raise ValueError(f'Unsupported reconstruction loss type {self.loss_recon_type}.')
            # perceptual loss
            if self.lw_perceptual > 0:
                loss_per = self.perceptual_loss(x, recon)
                loss_rec = loss_rec + self.lw_perceptual * loss_per
            # nll loss
            loss_nll = torch.mean(loss_rec)

            loss = loss_nll + self.lw_vq * loss_vq.mean()
            log_dict = {
                'loss_nll': loss_nll.detach().mean(),
                'loss_quant': loss_vq.detach().mean(),
                'loss_rec': loss_rec.detach().mean(),
            }

            if global_step >= self.gan_start_step:
                logits_fake = self.discriminator(recon)
                loss_g = -torch.mean(logits_fake)
                if self.training:
                    lw_gan_adap = self.calculate_adaptive_weight(loss_nll, loss_g)
                else:
                    lw_gan_adap = torch.tensor(1.0)
                
                loss += self.lw_gan * lw_gan_adap * loss_g
                log_dict.update({
                    'loss_g': loss_g.detach().mean(),
                    'd_weight': lw_gan_adap.detach(),
                })
            
            log_dict['loss_total'] = loss.clone().detach().mean()
            return loss, log_dict
        
        if optimizer_idx == 1:
            logits_real = self.discriminator(x)
            logits_fake = self.discriminator(recon.detach())
            loss_d = self.hinge_d_loss(logits_real, logits_fake)
            log_dict = {
                'loss_d': loss_d.clone().detach().mean(),
                'logits_real': logits_real.detach().mean(),
                'logits_fake': logits_fake.detach().mean(),
            }
            return loss_d, log_dict
    
    def calculate_adaptive_weight(self, loss_nll, loss_g):
        grads_nll = torch.autograd.grad(loss_nll, self.ae.decoder.conv_out.weight, retain_graph=True)[0]
        grads_g = torch.autograd.grad(loss_g, self.ae.decoder.conv_out.weight, retain_graph=True)[0]
        d_weight = torch.norm(grads_nll) / (torch.norm(grads_g) + 1e-4)
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


class PLVQVAE(pl.LightningModule):
    """Pytorch lightning module for a VQVAE with disctiminative training."""
    def __init__(self, model_config, ckpt_path=None, scheduler_config=None):
        # TODO EMA?
        super().__init__()
        self.config = model_config
        self.scheduler_config = scheduler_config
        self.model = AutoEncoderVQWithDisc(model_config)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
        self.automatic_optimization = False
    
    def init_from_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        missing, unexpected = self.load_state_dict(ckpt['state_dict'], strict=False)
        # TODO logging system
        print(f'Restoring from checkpoint {ckpt_path}.')
        print(f'Missing keys: {missing}')
        print(f'Unexpected keys: {unexpected}')
    
    def training_step(self, batch, batch_idx):
        if self.global_step >= self.config.loss.gan_start_step \
            and self.global_step % 2 == 1:    # train discriminator
            optimizer_idx = 1
        else:    # train generator
            optimizer_idx = 0
        optimizer_g, optimizer_d = self.optimizers(use_pl_optimizer=True)
        loss, logdict = self.model(batch['image'], optimizer_idx, self.global_step)
        logdict = {f'train/{k}': v for k, v in logdict.items()}

        if optimizer_idx == 0:
            self.toggle_optimizer(optimizer_g)
            optimizer_g.zero_grad()
            self.manual_backward(loss)
            optimizer_g.step()
            self.untoggle_optimizer(optimizer_g)

            self.log_dict(logdict, prog_bar=False, logger=True, on_step=True, on_epoch=True, batch_size=batch['image'].size(0))
        
        else:
            self.toggle_optimizer(optimizer_d)
            optimizer_d.zero_grad()
            self.manual_backward(loss)
            optimizer_d.step()
            self.untoggle_optimizer(optimizer_d)

            self.log_dict(logdict, prog_bar=False, logger=True, on_step=True, on_epoch=True, batch_size=batch['image'].size(0))
        
    def validation_step(self, batch, batch_idx):
        # TODO EMA validation?
        loss_ae, logdict_ae = self.model(batch['image'], 0, self.global_step)
        loss_disc, logdict_disc = self.model(batch['image'], 1, self.global_step)
        log_dict_ae = {f'val/{k}': v for k, v in log_dict_ae.items()}
        log_dict_disc = {f'val/{k}': v for k, v in log_dict_disc.items()}
        self.log_dict(logdict_ae, batch_size=batch['image'].size(0))
        self.log_dict(logdict_disc, batch_size=batch['image'].size(0))
    
    def configure_optimizers(self,):
        lr = self.config.learning_rate
        optimizer_g = torch.optim.Adam(
            self.model.ae.parameters(), lr=lr, betas=(0.5, 0.9)
        )
        optimizer_d = torch.optim.Adam(
            self.model.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
        )

        if self.scheduler_config is not None:
            scheduler_g = utils.get_obj_from_str(self.scheduler_config.target)(
                optimizer_g, **self.scheduler_config.get('params', dict())
            )
            scheduler_d = utils.get_obj_from_str(self.scheduler_config.target)(
                optimizer_d, **self.scheduler_config.get('params', dict())
            )
            
            return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]

        return [optimizer_g, optimizer_d], []
    
    @torch.no_grad()
    def log_images(self, batch, **unused_kwargs):
        # TODO sample?
        image = batch['image']
        rec, _, _ = self.model.ae(image)
        return {
            'inputs': image,
            'reconstructions': rec,
        }