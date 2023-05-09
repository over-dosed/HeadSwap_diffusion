import einops
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.models.diffusion.ddpm_hsd_CLIP import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config, default
from ldm.models.diffusion.ddim_hsd import DDIMSampler

from modules.Face_feature import FaceFeatureExtractor


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM_HSD(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, 
                 l_id_weight, retina_path, arcface_model_path, first_stage_cuda, cond_stage_cuda, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

        # set first stage cuda device
        self.first_stage_cuda = first_stage_cuda
        self.cond_stage_cuda = cond_stage_cuda

        # face feature extractor
        self.face_feature_extractor = FaceFeatureExtractor(retina_path, arcface_model_path, output_size=(112, 112), device = 'cuda')

        # ID_loss weight
        self.l_id_weight = l_id_weight
        

    # @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        # control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        return x, dict(c_crossattn=[c], c_concat=[control])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps
    
    # @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        self.first_stage_model.cuda(self.first_stage_cuda)
        z = z.cuda(self.first_stage_cuda)

        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if self.first_stage_key=='inpaint':
            decode_result = self.first_stage_model.decode(z[:,:4,:,:])
        else:
            decode_result = self.first_stage_model.decode(z)

        decode_result = decode_result.cuda(0)
        return decode_result

    @torch.no_grad()
    def encode_first_stage(self, x):
        self.first_stage_model.cuda(self.first_stage_cuda)
        x = x.cuda(self.first_stage_cuda)
        encode_result = self.first_stage_model.encode(x)

        # do some function that get_first_stage_encoding() usually do
        if isinstance(encode_result, DiagonalGaussianDistribution):
            encode_result = encode_result.sample()

        encode_result = encode_result.cuda(0)
        return encode_result
    
    def get_first_stage_encoding(self, encoder_posterior):
        return self.scale_factor * encoder_posterior
    
    def shared_step(self, batch, **kwargs):
        # print('start a step')
        x, c = self.get_input(batch, self.first_stage_key)
        id_gt = batch['target'].detach().clone()
        id_gt = id_gt.to(memory_format=torch.contiguous_format).float()
        del batch
        torch.cuda.empty_cache()
        # global loss
        loss = self(x, c, id_gt = id_gt.to(self.device))
        # print('end a step')

        return loss
    
    def p_losses(self, x_start, cond, t, id_gt, noise=None):
        if self.first_stage_key == 'inpaint':
            # x_start=x_start[:,:4,:,:]
            noise = default(noise, lambda: torch.randn_like(x_start[:,:4,:,:]))
            x_noisy = self.q_sample(x_start=x_start[:,:4,:,:], t=t, noise=noise)
            x_noisy = torch.cat((x_noisy,x_start[:,4:,:,:]),dim=1)
        else:
            noise = default(noise, lambda: torch.randn_like(x_start))
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        # get eps loss
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        # get id loss
        if self.l_id_weight == 0.0:
            id_loss = 0.0
        else:
            predicted_x0 = self.predict_start_from_noise(x_noisy[:,:4,:,:], t, model_output) # predicted x0
            recon_x0 = self.decode_first_stage(predicted_x0) # decode x0, get reconstructed x0: (B, 3, H, W), -1~1
            
            recon_x0_feature = self.face_feature_extractor((recon_x0 + 1.0) * 127.5)
            id_gt_feature = self.face_feature_extractor((id_gt + 1.0) * 127.5)
            id_loss = 1.0 - F.cosine_similarity(id_gt_feature, recon_x0_feature, dim=1) # get id loss
            id_loss = id_loss.mean()
            del predicted_x0
            del recon_x0
            torch.cuda.empty_cache()

        loss_dict.update({f'{prefix}/loss_id': id_loss})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        # loss = self.l_id_weight * id_loss # test
        loss = self.l_simple_weight * loss.mean() + self.l_id_weight * id_loss
        # loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def on_train_batch_end(self, batch_end_output, batch, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def get_learned_conditioning(self, c, cond_key = None):
        self.cond_stage_model.cuda(self.cond_stage_cuda)

        if self.cond_stage_forward is None:
            if cond_key == 'image' or cond_key == 'image_clip_id':
                (xc_id, xc_gloabl) = c
                xc_id = xc_id.cuda(self.cond_stage_cuda)
                xc_gloabl = xc_gloabl.cuda(self.cond_stage_cuda)
                c = self.cond_stage_model(xc_gloabl, xc_id)
                c = c.cuda(0)
                return c
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):              
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    @torch.no_grad()
    def get_unconditional_conditioning(self, N, cond_key, condition_clip = None):
        
        if condition_clip is None:
            zero_condition_clip = torch.zeros(N, 3, 224, 224).float().to(self.device)
        else:
            zero_condition_clip = condition_clip

        if cond_key == 'image' or cond_key == 'image_clip_id':
            zero_condition_id = torch.zeros(N, 512).float().to(self.device)
            c = self.get_learned_conditioning((zero_condition_id, zero_condition_clip), cond_key=cond_key)
        else:
            c = self.get_learned_conditioning(zero_condition_clip, cond_key=cond_key)
        
        if cond_key == 'image_clip_id':
            c = self.proj_out(c)
        return c
    
    def un_norm_clip(self, x):
                    x[:,0,:,:] = x[:,0,:,:] * 0.26862954 + 0.48145466
                    x[:,1,:,:] = x[:,1,:,:] * 0.26130258 + 0.4578275
                    x[:,2,:,:] = x[:,2,:,:] * 0.27577711 + 0.40821073
                    return x

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=True, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=2.0 , unconditional_guidance_id=False,
                   use_ema_scope=True, plot_no_condition = True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0
        # log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)
        log["conditioning"] = self.un_norm_clip(batch['source_global'])

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta, rest=z[:N,4:,:,:])
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if plot_no_condition:
            # zero_condtion has the same shape, dtype and devices
            zero_condition = torch.zeros_like(c, device= c.device)
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [zero_condition]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta, rest=z[:N,4:,:,:])
            x_samples = self.decode_first_stage(samples)
            log["s_nocondition"] = x_samples

        if unconditional_guidance_scale > 1.0:
            if unconditional_guidance_id:
                uc_cross = self.get_unconditional_conditioning(N, self.cond_stage_key, condition_clip = batch['source_global'])
            else:
                uc_cross = self.get_unconditional_conditioning(N, self.cond_stage_key)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             rest=z[:N,4:,:,:]
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            # log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
            log["samples_cfg_scale"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        if not self.control_locked:
            params = list(self.control_model.parameters())
        else:
            params = list()

        # # for train v3.5.1
        # params += list(self.cond_stage_model.mapper.parameters())
        # params += list(self.cond_stage_model.final_ln.parameters())
        # params += list(self.cond_stage_model.id_residual_ST1.parameters())
        # params += list(self.cond_stage_model.id_residual_ST2.parameters())
        # params += list(self.cond_stage_model.id_residual_conv.parameters())
        # params += list(self.proj_out.parameters())

        # # for train v3.5.2
        # params += list(self.cond_stage_model.final_ln.parameters())
        # params += list(self.cond_stage_model.id_residual_ST1.parameters())
        # params += list(self.cond_stage_model.id_residual_ST2.parameters())
        # params += list(self.cond_stage_model.id_residual_conv.parameters())
        # params += list(self.proj_out.parameters())

        # for train v3.6.1
        params += list(self.cond_stage_model.mapper.parameters())
        params += list(self.cond_stage_model.proj_in.parameters())
        params += list(self.cond_stage_model.id_residual_block.parameters())
        params += list(self.cond_stage_model.final_ln.parameters())

        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
