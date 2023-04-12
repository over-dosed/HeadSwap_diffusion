import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel, CLIPVisionModel

import open_clip
from ldm.util import default, count_params

from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential, AttentionBlock
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from .xf import LayerNorm, Transformer


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):

    def encode(self, x):
        return x


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class', ucg_rate=0.1):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate

    def forward(self, batch, key=None, disable_dropout=False):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        if self.ucg_rate > 0. and not disable_dropout:
            mask = 1. - torch.bernoulli(torch.ones_like(c) * self.ucg_rate)
            c = mask * c + (1-mask) * torch.ones_like(c)*(self.n_classes-1)
            c = c.long()
        c = self.embedding(c)
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = self.n_classes - 1  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc}
        return uc


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class FrozenT5Embedder(AbstractEncoder):
    """Uses the T5 transformer encoder for text"""
    def __init__(self, version="google/t5-v1_1-large", device="cuda", max_length=77, freeze=True):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length   # TODO: typical value?
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77,
                 freeze=True, layer="last", layer_idx=None):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)


class FrozenCLIPT5Encoder(AbstractEncoder):
    def __init__(self, clip_version="openai/clip-vit-large-patch14", t5_version="google/t5-v1_1-xl", device="cuda",
                 clip_max_length=77, t5_max_length=77):
        super().__init__()
        self.clip_encoder = FrozenCLIPEmbedder(clip_version, device, max_length=clip_max_length)
        self.t5_encoder = FrozenT5Embedder(t5_version, device, max_length=t5_max_length)
        print(f"{self.clip_encoder.__class__.__name__} has {count_params(self.clip_encoder)*1.e-6:.2f} M parameters, "
              f"{self.t5_encoder.__class__.__name__} comes with {count_params(self.t5_encoder)*1.e-6:.2f} M params.")

    def encode(self, text):
        return self(text)

    def forward(self, text):
        clip_z = self.clip_encoder.encode(text)
        t5_z = self.t5_encoder.encode(text)
        return [clip_z, t5_z]
    
class FrozenCLIPImageEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14"):
        super().__init__()
        self.transformer = CLIPVisionModel.from_pretrained(version)
        self.final_ln = LayerNorm(1024)
        self.mapper = Transformer(
                1,
                1024,
                5,
                1,
            )

        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False
        for param in self.mapper.parameters():
            param.requires_grad = True
        for param in self.final_ln.parameters():
            param.requires_grad = True

    def forward(self, image):
        outputs = self.transformer(pixel_values=image)
        z = outputs.pooler_output
        z = z.unsqueeze(1)
        z = self.mapper(z)
        z = self.final_ln(z)
        return z

    def encode(self, image):
        return self(image)
    
class FrozenCLIPImageEmbedder_id(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14"):
        super().__init__()
        self.transformer = CLIPVisionModel.from_pretrained(version)
        self.final_ln = LayerNorm(1024)
        self.mapper = Transformer(
                1,
                1024,
                5,
                8,
            )

        # global residual (add to id feature)
        self.id_residual_ST1 = SpatialTransformer(1024, 8, 128, depth=1, context_dim=1024, 
                               disable_self_attn=False, use_linear=True, 
                               use_checkpoint=False)
        self.id_residual_ST2 = SpatialTransformer(1024, 8, 128, depth=1, context_dim=1024, 
                               disable_self_attn=False, use_linear=True, 
                               use_checkpoint=False)
        self.id_residual_conv = zero_module(conv_nd(2, 1024, 1024, 1, padding=0))

        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.mapper.parameters():
            param.requires_grad = False
        for param in self.final_ln.parameters():
            param.requires_grad = True
        for param in self.id_residual_ST1.parameters():
            param.requires_grad = True
        for param in self.id_residual_ST2.parameters():
            param.requires_grad = True
        for param in self.id_residual_conv.parameters():
            param.requires_grad = True

    def forward(self, image, id_feature):

        with torch.no_grad():
            outputs = self.transformer(pixel_values=image)
            z = outputs.pooler_output
            B, _ = z.shape

            z = z.unsqueeze(1)
        z = self.mapper(z)  ## z : (B, 1, 1024)

        z = z.view(B, -1, 1, 1)

        id_feature = id_feature.unsqueeze(1) # id_feature: (B, 1, 1024)

        id_residual = self.id_residual_ST1(z, id_feature) # id_residual: (B, 1024, 1, 1)
        id_residual = self.id_residual_ST2(id_residual, id_feature) # id_residual: (B, 1024, 1, 1)
        id_residual = self.id_residual_conv(id_residual) # id_residual: (B, 1024, 1, 1)

        z =  z + id_residual

        z = z.view(B, 1, -1) # z: (B, 1, 1024)
        z = self.final_ln(z)
        
        return z

    def encode(self, image, id_feature):
        return self(image, id_feature)
    
class FrozenCLIPImageEmbedder_Full(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14"):
        super().__init__()
        self.transformer = CLIPVisionModel.from_pretrained(version)
        self.final_ln = LayerNorm(1024)
        self.mapper = Transformer(
                257,
                1024,
                5,
                8,
            )

        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.transformer.parameters():
            param.requires_grad = False

        # for train v1.3
        for param in self.mapper.parameters():
            param.requires_grad = True
        for param in self.final_ln.parameters():
            param.requires_grad = True

    def forward(self, image):
        outputs = self.transformer(pixel_values=image)
        z = outputs.last_hidden_state
        z = self.mapper(z)
        z = self.final_ln(z)
        return z

    def encode(self, image):
        return self(image)

class FrozenCLIPImageEmbedder_id_full(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14"):
        super().__init__()
        self.transformer = CLIPVisionModel.from_pretrained(version)
        self.final_ln = LayerNorm(1024)
        self.mapper = Transformer(
                257,
                1024,
                5,
                8,
            )

        # global residual (add to id feature)
        self.proj_in = nn.Linear(512, 1024)
        self.id_residual_block = id_residual()


        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.mapper.parameters():
            param.requires_grad = False
        for param in self.proj_in.parameters():
            param.requires_grad = True
        for param in self.final_ln.parameters():
            param.requires_grad = True
        for param in self.id_residual_block.parameters():
            param.requires_grad = True

    def forward(self, image, id_feature):

        with torch.no_grad():
            outputs = self.transformer(pixel_values=image)
            z = outputs.last_hidden_state # z : (B, 257, 1024)
            z = self.mapper(z)  # z : (B, 257, 1024)

        id_feature = self.proj_in(id_feature) # id_feature: (B, 1024)
        id_residual = self.id_residual_block(z, id_feature) # id_residual: (B, 257, 1024)

        z =  z + id_residual # z: (B, 257, 1024)
        z = self.final_ln(z)
        
        return z

    def encode(self, image, id_feature):
        return self(image, id_feature)


class id_residual(nn.Module):

    def __init__(self, spatial_depth = 2):
        super(id_residual, self).__init__()

        self.spatial_depth = spatial_depth

        self.id_residual_ST = SpatialTransformer(1024, 8, 128, depth=spatial_depth, 
                                                 context_dim=[1024]*spatial_depth, disable_self_attn=False, 
                                                 use_linear=False, use_checkpoint=False)
        
        self.id_residual_linear = nn.Linear(1, 257)

        self.id_residual_mapper = AttentionBlock(1024, num_heads=8)

    def forward(self, CLIP_output, id_feature):
        # CLIP_output : (B, 257, 1024)
        # id_feature : (B, 1024)

        id_feature = id_feature.unsqueeze(-1).unsqueeze(-1) # (B, 1024, 1, 1)
        CLIP_output = [CLIP_output] * self.spatial_depth
        
        id_residual = self.id_residual_ST(id_feature, CLIP_output) # (B, 1024, 1, 1)

        id_residual = id_residual.squeeze(-1) # (B, 1024, 1)
        id_residual = self.id_residual_linear(id_residual) # (B, 1024, 257)

        id_residual = self.id_residual_mapper(id_residual) # (B, 1024, 257)
        id_residual = id_residual.permute(0, 2, 1) # (B, 257, 1024)

        return id_residual