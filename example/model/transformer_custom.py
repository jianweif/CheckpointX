from collections import OrderedDict
import math
from typing import Callable, Optional, Sequence, Tuple
from torch.utils.checkpoint import checkpoint
import torch
from torch import nn
from .transformer import LayerNorm, LayerScale, PatchDropout, AttentionalPooler

def forward_layers(function_list, *args):
    if len(function_list) == 0:
        raise Exception("cannot forward 0 layers")
    x = args
    # debug_print("at normal forward begin")
    for i, layer in enumerate(function_list):
        if isinstance(x, torch.Tensor) or isinstance(x, dict):
            x = (x,)
        if i == 0:
            x = layer(*x)
        else:
            x = layer(*x)
        # debug_print("after normal forward {}".format(i))
        # todo: handle tuple or list here
    return x

class ResidualAttentionBlockPart1(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(
            q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask
        )[0]

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None

        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        return x

    def to_sequence(self, tensor_dict):
        # for simplicity
        assert "k_x" not in tensor_dict and "v_x" not in tensor_dict
        seq = [OpWrapperFusedWithCopyToDict(self.ln_1, key="x", copy_to_key="orig_x"), AttnWrapper(self.attn, key="x")]
        if not isinstance(self.ls_1, nn.Identity):
            seq.append(SingleInputOpWrapper(self.ls_1, input_key="x"))
        seq.append(AddWrapper(key1="x", key2="orig_x", dst_key="x"))
        return seq, forward_layers(tensor_dict)

    def to_sequence_v2(self, x):
        # for simplicity
        seq = [OpCopy1In2Out(self.ln_1), Attn2In2Out(self.attn)]
        if not isinstance(self.ls_1, nn.Identity):
            seq.append(Op2In2Out(self.ls_1))
        seq.append(OpAdd2In1Out())
        return seq, forward_layers(seq, x)

class ResidualAttentionBlockPart2(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
    ):
        super().__init__()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None
    ):
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x

    def to_sequence(self, tensor_dict):
        # for simplicity
        seq = [OpWrapperFusedWithCopyToDict(self.ln_2, key="x", copy_to_key="orig_x"), ]
        for op in self.mlp:
            seq.append(SingleInputOpWrapper(op, input_key="x"))
        if not isinstance(self.ls_2, nn.Identity):
            seq.append(SingleInputOpWrapper(self.ls_2, input_key="x"))
        seq.append(AddWrapper(key1="x", key2="orig_x", dst_key="x"))
        return seq, nn.Sequential(*seq)(tensor_dict)

    def to_sequence_v2(self, x):
        # for simplicity
        seq = [OpCopy1In2Out(self.ln_2)]
        for op in self.mlp:
            seq.append(Op2In2Out(op))
        if not isinstance(self.ls_2, nn.Identity):
            seq.append(Op2In2Out(self.ls_2))
        seq.append(OpAdd2In1Out())
        return seq, forward_layers(seq, x)

def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)

class SimpleLoss(nn.Module):
    def __init__(self):
        super(SimpleLoss, self).__init__()

    def forward(self, inputs, attn_mask=None):
        return torch.mean(inputs)

class CustomTransformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            add_loss=True,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = []
        for _ in range(layers):
            self.resblocks.append(ResidualAttentionBlockPart1(width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer))
            self.resblocks.append(ResidualAttentionBlockPart2(width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer))
        if add_loss:
            self.resblocks.append(SimpleLoss())
        self.resblocks = nn.ModuleList(self.resblocks)

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(*self.shape)

class Permute(nn.Module):
    def __init__(self, *perm):
        super(Permute, self).__init__()
        self.perm = perm

    def forward(self, x):
        return x.permute(*self.perm)

class PosEmbeddingAdd(nn.Module):
    def __init__(self, class_emb, pos_emb):
        super(PosEmbeddingAdd, self).__init__()
        self.class_emb = class_emb
        self.pos_emb = pos_emb

    def forward(self, x):
        x = torch.cat([_expand_token(self.class_emb, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.pos_emb.to(x.dtype)
        return x

class GlobalPool(nn.Module):
    def __init__(self, pool_type):
        super(GlobalPool, self).__init__()
        self.pool_type = pool_type

    def _global_pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled

    def forward(self, x):
        return self._global_pool(x)

class Proj(nn.Module):
    def __init__(self, proj):
        super(Proj, self).__init__()
        self.proj = proj

    def forward(self, x):
        return x @ self.proj

class Op2In2Out(nn.Module):
    def __init__(self, op):
        super(Op2In2Out, self).__init__()
        self.op = op

    def forward(self, x, y):
        return self.op(x), y

class OpAdd2In1Out(nn.Module):
    def forward(self, x, y):
        return x + y

class OpCopy1In2Out(nn.Module):
    def __init__(self, op):
        super(OpCopy1In2Out, self).__init__()
        self.op = op
    def forward(self, x):
        return self.op(x), x

class OpWrapperFusedWithToDict(nn.Module):
    def __init__(self, op, key):
        super(OpWrapperFusedWithToDict, self).__init__()
        self.op = op
        self.key = key

    def forward(self, tensor):
        return {self.key: self.op(tensor)}

class OpWrapperFusedWithToTensor(nn.Module):
    def __init__(self, op, key):
        super(OpWrapperFusedWithToTensor, self).__init__()
        self.op = op
        self.key = key

    def forward(self, tensor_dict):
        return self.op(tensor_dict[self.key])

class OpWrapperFusedWithCopyToDict(nn.Module):
    def __init__(self, op, key, copy_to_key):
        super(OpWrapperFusedWithCopyToDict, self).__init__()
        self.op = op
        self.key = key
        self.copy_to_key = copy_to_key

    def forward(self, tensor_dict):
        tensor_dict[self.copy_to_key] = tensor_dict[self.key]
        tensor_dict[self.key] = self.op(tensor_dict[self.key])
        return tensor_dict

class Attn2In2Out(nn.Module):
    def __init__(self, attn):
        super(Attn2In2Out, self).__init__()
        self.attn = attn

    def forward(self, x, y):
        x = self.attn(
            x, x, x, need_weights=False, attn_mask=None
        )[0]
        return x, y

class AttnWrapper(nn.Module):
    def __init__(self, attn, key='x'):
        super(AttnWrapper, self).__init__()
        self.attn = attn
        self.key = key

    def forward(self, tensor_dict):
        tensor_dict[self.key] = self.attn(
            tensor_dict[self.key], tensor_dict[self.key], tensor_dict[self.key], need_weights=False, attn_mask=None
        )[0]
        return tensor_dict

class SingleInputOpWrapper(nn.Module):
    def __init__(self, op, input_key):
        super(SingleInputOpWrapper, self).__init__()
        self.op = op
        self.input_key = input_key

    def forward(self, dict_inp):
        dict_inp[self.input_key] = self.op(dict_inp[self.input_key])
        return dict_inp

class AddWrapper(nn.Module):
    def __init__(self, key1, key2, dst_key):
        super(AddWrapper, self).__init__()
        self.key1 = key1
        self.key2 = key2
        self.dst_key = dst_key

    def forward(self, dict_inp):
        dict_inp[self.dst_key] = dict_inp[self.key1] + dict_inp[self.key2]
        # todo: an ugly temporary workaround
        if self.key1 != self.dst_key:
            del dict_inp[self.key1]
        if self.key2 != self.dst_key:
            del dict_inp[self.key2]
        return dict_inp

class CustomVisionTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            attentional_pool: bool = False,
            attn_pooler_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.,
            no_ln_pre: bool = False,
            pos_embed_type: str = 'learnable',
            pool_type: str = 'tok',
            final_ln_after_pool: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
    ):
        super().__init__()
        assert pool_type in ('tok', 'avg', 'none')
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = (image_size, image_size)
        patch_height, patch_width = self.patch_size = (patch_size, patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = final_ln_after_pool  # currently ignored w/ attn pool enabled
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if pos_embed_type == 'learnable':
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        elif pos_embed_type == 'sin_cos_2d':
            # fixed sin-cos embedding
            assert self.grid_size[0] == self.grid_size[1], \
                'currently sin cos 2d pos embedding only supports square input'
            self.positional_embedding = nn.Parameter(
                torch.zeros(self.grid_size[0] * self.grid_size[1] + 1, width), requires_grad=False)
            # todo: skip it for now
            raise NotImplementedError
            pos_embed_type = get_2d_sincos_pos_embed(width, self.grid_size[0], cls_token=True)
            self.positional_embedding.data.copy_(torch.from_numpy(pos_embed_type).float())
        else:
            raise ValueError

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)
        self.transformer = CustomTransformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            add_loss=False
        )

        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attn_pool_type = attentional_pool
                self.pool_type = 'none'
                if attentional_pool in ('parallel', 'cascade'):
                    self.attn_pool = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=attn_pooler_queries,
                    )
                    self.attn_pool_contrastive = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=1,
                    )
                else:
                    assert False
            else:
                self.attn_pool_type = ''
                self.pool_type = pool_type
                self.attn_pool = AttentionalPooler(
                    output_dim,
                    width,
                    n_head=attn_pooler_heads,
                    n_queries=attn_pooler_queries,
                )
                self.attn_pool_contrastive = None
            pool_dim = output_dim
        else:
            self.attn_pool = None
            pool_dim = width
            self.pool_type = pool_type

        self.ln_post = norm_layer(pool_dim)
        self.proj = nn.Parameter(scale * torch.randn(pool_dim, output_dim))

        self.simple_loss = SimpleLoss()
        self.init_parameters()

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.positional_embedding,
                    self.ln_pre,
                ],
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    self.ln_post,
                ],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

    def init_parameters(self):
        # FIXME OpenAI CLIP did not define an init for the VisualTransformer
        # TODO experiment if default PyTorch init, below, or alternate init is best.

        # nn.init.normal_(self.class_embedding, std=self.scale)
        # nn.init.normal_(self.positional_embedding, std=self.scale)
        #
        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        #
        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.scale)
        pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens

        loss = self.simple_loss(pooled)
        return loss

    def get_sequential(self, x):
        seq = []
        seq.append(self.conv1)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        seq.append(Reshape(x.shape[0], x.shape[1], -1))
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        seq.append(Permute(0, 2, 1))
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        seq.append(PosEmbeddingAdd(self.class_embedding, self.positional_embedding))
        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        seq.append(self.patch_dropout)
        x = self.patch_dropout(x)
        seq.append(self.ln_pre)
        x = self.ln_pre(x)
        seq.append(Permute(1, 0, 2))
        x = x.permute(1, 0, 2)  # NLD -> LND
        resblocks = self.transformer.resblocks
        seq = seq + list(resblocks) # todo: check
        x = self.transformer(x)
        seq.append(Permute(1, 0, 2))
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
            raise NotImplementedError
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
            raise NotImplementedError
        else:
            seq.append(self.ln_post)
            x = self.ln_post(x)
            seq.append(GlobalPool(self.pool_type))
            pooled, tokens = self._global_pool(x)

        if self.proj is not None:
            seq.append(Proj(self.proj))
            pooled = pooled @ self.proj

        seq.append(self.simple_loss)

        return seq

    def get_sequential_v2(self, x):
        seq = []
        seq.append(self.conv1)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        seq.append(Reshape(x.shape[0], x.shape[1], -1))
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        seq.append(Permute(0, 2, 1))
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        seq.append(PosEmbeddingAdd(self.class_embedding, self.positional_embedding))
        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        seq.append(self.patch_dropout)
        x = self.patch_dropout(x)
        seq.append(self.ln_pre)
        x = self.ln_pre(x)
        # seq.append(Permute(1, 0, 2))
        seq.append(OpWrapperFusedWithToDict(Permute(1, 0, 2), key="x"))
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = {"x": x}
        for r in self.transformer.resblocks:
            subseq, x = r.to_sequence(x)
            seq += subseq
        # resblocks = self.transformer.resblocks
        # seq = seq + list(resblocks) # todo: modify here
        # x = self.transformer(x)



        # seq.append(Permute(1, 0, 2))
        seq.append(OpWrapperFusedWithToTensor(Permute(1, 0, 2), key="x"))
        x = x["x"].permute(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
            raise NotImplementedError
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
            raise NotImplementedError
        else:
            seq.append(self.ln_post)
            x = self.ln_post(x)
            seq.append(GlobalPool(self.pool_type))
            pooled, tokens = self._global_pool(x)

        if self.proj is not None:
            seq.append(Proj(self.proj))
            pooled = pooled @ self.proj

        seq.append(self.simple_loss)

        return seq

    def get_sequential_v3(self, x):
        seq = []
        seq.append(self.conv1)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        seq.append(Reshape(x.shape[0], x.shape[1], -1))
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        seq.append(Permute(0, 2, 1))
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        seq.append(PosEmbeddingAdd(self.class_embedding, self.positional_embedding))
        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        seq.append(self.patch_dropout)
        x = self.patch_dropout(x)
        seq.append(self.ln_pre)
        x = self.ln_pre(x)
        seq.append(Permute(1, 0, 2))
        x = x.permute(1, 0, 2)  # NLD -> LND
        for r in self.transformer.resblocks:
            subseq, x = r.to_sequence_v2(x)
            seq += subseq
        # resblocks = self.transformer.resblocks
        # seq = seq + list(resblocks) # todo: modify here
        # x = self.transformer(x)



        seq.append(Permute(1, 0, 2))
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
            raise NotImplementedError
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
            raise NotImplementedError
        else:
            seq.append(self.ln_post)
            x = self.ln_post(x)
            seq.append(GlobalPool(self.pool_type))
            pooled, tokens = self._global_pool(x)

        if self.proj is not None:
            seq.append(Proj(self.proj))
            pooled = pooled @ self.proj

        seq.append(self.simple_loss)

        return seq