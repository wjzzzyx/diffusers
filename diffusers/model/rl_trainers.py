import collections
import contextlib
import itertools
import numpy as np
import os
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm

import torch_utils
import utils
from diffusers.model.lora import Trainer


class PerPromptStatTracker:
    def __init__(self, buffer_size, min_count):
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.stats = {}
    
    def update(self, prompts, rewards):
        world_size = dist.get_world_size()
        gathered_rewards = [torch.zeros_like(rewards) for _ in range(world_size)]
        dist.all_gather(gathered_rewards, rewards)
        rewards = torch.cat(gathered_rewards)
        gathered_prompts = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_prompts, prompts)
        prompts = list(itertools.chain(*gathered_prompts))
        prompts = np.array(prompts)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards)
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = collections.deque(maxlen=self.buffer_size)
            self.stats[prompt].append(prompt_rewards)

            if len(self.stats[prompt]) < self.min_count:
                mean = np.mean(rewards)
                std = np.std(rewards) + 1e-6
            else:
                mean = np.mean(self.stats[prompt])
                std = np.std(self.stats[prompt]) + 1e-6
            advantages[prompts == prompt] = (prompt_rewards - mean) / std
        
        advantages = torch.as_tensor(advantages)
        advantages = advantages.view(world_size, -1)[dist.get_rank()].cuda()
        return advantages
    
    def get_stats(self):
        return {
            k: {"mean": np.mean(v), "std": np.std(v), "count": len(v)}
            for k, v in self.stats.items()
        }


class TrainerDDPO(Trainer):
    def __init__(self, model_config, loss_config, optimizer_config, device):
        super().__init__(model_config, loss_config, optimizer_config, device)
        self.reward_fn = utils.get_obj_from_str(loss_config.target)
        self.rewards_tracker = PerPromptStatTracker(16, 16)
        self.mini_batch_size = 8

        self.loss_meters = {
            "loss_ppo": torch_utils.RunningStatistic(device),
            "approx_kl": torch_utils.RunningStatistic(device),
            "clip_frac": torch_utils.RunningStatistic(device),
            "reward": torch_utils.RunningStatistic(device),
            "reward_std": torch_utils.RunningStatistic(device),
        }
    
    def train_step(self, batch, global_step, epoch, batch_idx, logdir):
        batch_size = len(batch["text"])
        height = batch["height"][0]
        width = batch["width"][0]
        mini_batch_size = self.mini_batch_size
        # TODO same prompt on same device? no
        self.unet.eval()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            all_text_embs = list()
            all_latents = list()
            all_next_latents = list()
            all_logprobs = list()
            all_rewards = list()
            all_images = list()
            for i in tqdm(range(0, batch_size, mini_batch_size), desc="Sampling", disable=dist.get_rank() != 0):
                prompts = batch["text"][i:i+mini_batch_size]
                text_embs = self.clip(prompts)
                neg_prompts = ["" for _ in range(mini_batch_size)]
                # sample images
                noise = torch.randn((mini_batch_size, 4, height // 8, width // 8), device=self.device, generator=None)
                latents, log_probs = self.sampler.sample(
                    self.denoiser, noise, denoiser_args={"cond_prompt": text_embs}, generator=None, return_logprob=True
                )
                images = latents[-1] / self.vae.scale_factor
                images = self.vae.decode(images)
                images = (images.clamp(-1, 1) + 1) / 2
                rewards = self.reward_fn(images, prompts)
                
                all_text_embs.append(text_embs)
                all_latents.append(torch.stack([noise, *latents[:-1]], dim=1))
                all_next_latents.append(torch.stack(latents, dim=1))
                all_logprobs.append(torch.stack(log_probs, dim=1))
                all_rewards.append(rewards)
                all_images.append(images)

            all_text_embs = torch.cat(all_text_embs, dim=0)    # (batch_size, seq, dim)
            all_latents = torch.cat(all_latents, dim=0)    # (batch_size, num_timesteps, c, h, w)
            all_next_latents = torch.cat(all_next_latents, dim=0)    # (batch_size, num_timesteps, c, h, w)
            all_logprobs = torch.cat(all_logprobs, dim=0)    # (batch_size, num_timesteps)
            all_rewards = torch.cat(all_rewards, dim=0)    # (batch_size,)
            all_timesteps = self.sampler.timesteps.cuda().repeat(batch_size, 1)
            all_images = torch.cat(all_images, dim=0)
            
            # gather all rewards and calculate advantages
            all_advantages = self.rewards_tracker(batch["text"], all_rewards)

            self.loss_meters["reward"].update(all_rewards.mean(), batch_size)
            self.loss_meters["reward_std"].update(all_rewards.std(), batch_size)
        
        if global_step % 1 == 0 and dist.get_rank() == 0:
            dirname = os.path.join(logdir, "log_images", "train")
            os.makedirs(dirname, exist_ok=True)
            log_image_dict = {
                "samples": all_images,
                "texts": batch["text"],
                "fpath": batch["fpath"],
                "rewards": all_rewards,
            }
            self.log_image(dirname, global_step, epoch, log_image_dict)
        
        # training
        # TODO iterate over sample how many times?
        perm = torch.randperm(batch_size, device=self.device)
        all_text_embs = all_text_embs[perm]
        all_latents = all_latents[perm]
        all_next_latents = all_next_latents[perm]
        all_logprobs = all_logprobs[perm]
        all_advantages = all_advantages[perm]
        all_timesteps = all_timesteps[perm]

        perm = torch.stack([
            torch.randperm(self.sampler.num_inference_timesteps, device=self.device)
            for _ in range(batch_size)
        ])
        all_timesteps = all_timesteps[torch.arange(batch_size, device=self.device)[:, None], perm]
        all_latents = all_latents[torch.arange(batch_size, device=self.device)[:, None], perm]
        all_next_latents = all_next_latents[torch.arange(batch_size, device=self.device)[:, None], perm]
        all_logprobs = all_logprobs[torch.arange(batch_size, device=self.device)[:, None], perm]

        mini_batch_size = 4
        grad_acc = 2
        self.unet.train()
        for i in tqdm(range(0, batch_size, mini_batch_size), desc="Training", disable=dist.get_rank() != 0):
            timesteps = all_timesteps[i:i+mini_batch_size]
            text_embs = all_text_embs[i:i+mini_batch_size]    # (mini_batch_size, seq, emb)
            latents = all_latents[i:i+mini_batch_size]    # (mini_batch_size, timesteps, c, h, w)
            next_latents = all_next_latents[i:i+mini_batch_size]
            advantages = all_advantages[i:i+mini_batch_size]    # (mini_batch_size,)
            log_probs_old = all_logprobs[i:i+mini_batch_size]    # (mini_batch_size, num_timesteps)
            advantages = torch.clamp(advantages, -self.loss_config.adv_clip_max, self.loss_config.adv_clip_max)
            
            # TODO grad accumulate
            # TODO cfg
            for j in tqdm(range(self.sampler.num_inference_timesteps), disable=dist.get_rank() != 0):
                should_sync = (i // mini_batch_size + 1) % grad_acc == 0 and j == self.sampler.num_inference_timesteps - 1
                if should_sync:
                    context = contextlib.nullcontext()
                else:
                    context = self.unet.no_sync() 
                with context:
                    with torch.autocast("cuda", torch.bfloat16):
                        denoised = self.denoiser(latents[:, j], timesteps[:, j], cond_prompt=text_embs)
                    _, log_probs = self.sampler.step(denoised, latents[:, j], timesteps[:, j], pred_xtm1=next_latents[:, j], generator=None)
                    ratio = torch.exp(log_probs - log_probs_old[:, j])
                    unclipped_part = - advantages * ratio
                    clipped_part = - advantages * torch.clamp(ratio, 1 - self.loss_config.ppo_clip, 1 + self.loss_config.ppo_clip)
                    loss_ppo = torch.maximum(unclipped_part, clipped_part)
                    loss_ppo = loss_ppo.mean()
                    loss_ppo.backward()
                if should_sync:
                    params = itertools.chain(*[x["params"] for x in self.optimizer.param_groups])
                    nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.lr_scheduler.step()

                approx_kl = 0.5 * torch.mean((log_probs - log_probs_old[:, j]) ** 2)
                clip_frac = torch.mean((torch.abs(ratio - 1) > self.loss_config.ppo_clip).float())
                self.loss_meters["approx_kl"].update(approx_kl, mini_batch_size)
                self.loss_meters["clip_frac"].update(clip_frac, mini_batch_size)
                self.loss_meters["loss_ppo"].update(loss_ppo, mini_batch_size)
    
    @torch.no_grad()
    def log_step(self, logdir, global_step, epoch, batch_idx):
        logdict = dict()
        for key in self.loss_meters.keys():
            val = self.loss_meters[key].compute()
            logdict[key] = val.item()
            self.loss_meters[key].reset()
        logdict["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        stats = self.rewards_tracker.get_stats()
        for k in stats:
            mean = stats[k]["mean"]
            std = stats[k]["std"]
            logdict[f"rewards/{k}_mean"] = mean
            logdict[f"rewards/{k}_std"] = std
        return logdict

    def log_image(self, logdir, global_step, epoch, log_image_dict):
        images = (log_image_dict["samples"].permute(0, 2, 3, 1) * 255).type(torch.uint8).cpu().numpy()
        prompts = log_image_dict["prompts"]
        rewards = log_image_dict["rewards"]
        for i, (image, prompt, reward) in enumerate(zip(images, prompts, rewards)):
            pil = Image.fromarray(image)
            new_pil = Image.new(pil.mode, (512, 512 + 20), "white")
            new_pil.paste(pil, (0, 0))
            draw = ImageDraw.Draw(new_pil)
            text_position = (10, pil.height)  # 10 pixels padding from bottom of original image
            caption = f"{prompt:.50} | {reward:.2f}"
            draw.text(text_position, caption, fill=(0, 0, 0))
            new_pil.save(os.path.join(logdir, f"gs{global_step}_{i}.png"))

