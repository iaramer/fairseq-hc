import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerModel, TransformerEncoder, TransformerDecoder
from fairseq.modules.transformer_layer import TransformerEncoderLayerBase, TransformerDecoderLayerBase
from torch.nn import LayerNorm
import torch.nn.functional as F

from .transformer_decoder import TransformerDecoderBase
from .transformer_encoder import TransformerEncoderBase
from .hc_multihead_attention import HCMultiheadAttention
from .hc_v2 import HyperCubeBlock
from fairseq.models.transformer.transformer_base import TransformerModelBase
from fairseq.models.transformer.transformer_config import (
    TransformerConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from fairseq.models.transformer.transformer_legacy import base_architecture
from fairseq.dataclass.utils import gen_parser_from_dataclass


class HCTransformerEncoderLayer(TransformerEncoderLayerBase):
    def __init__(self, args):
        self.args = args
        super().__init__(args)
        # self.embed_dim = args.encoder_embed_dim
        self.self_attn = HCMultiheadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            xformers_att_config=args.encoder.xformers_att_config,
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(activation=args.activation_fn)
        self.activation_dropout = args.activation_dropout

        self.fc1 = HyperCubeBlock(self.embed_dim, args.encoder_ffn_embed_dim) 
        self.fc2 = HyperCubeBlock(args.encoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_padding_mask, attn_mask=None):
        residual = x
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn


class HCTransformerDecoderLayer(TransformerDecoderLayerBase):
    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        super().__init__(args, no_encoder_attn)
        self.self_attn = HCMultiheadAttention(
            self.embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not args.cross_self_attention,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            xformers_att_config=args.decoder.xformers_att_config,
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = HCMultiheadAttention(
                self.embed_dim,
                args.decoder_attention_heads,
                kdim=args.encoder.embed_dim,
                vdim=args.encoder.embed_dim,
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
                xformers_att_config=args.encoder.xformers_att_config,
            )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = HyperCubeBlock(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = HyperCubeBlock(args.decoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)


class HCTransformerEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.layers = nn.ModuleList([])
        self.layers.extend([
            HCTransformerEncoderLayer(args) for _ in range(args.encoder_layers)
        ])


class HCTransformerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.layers = nn.ModuleList([])
        self.layers.extend([
            HCTransformerDecoderLayer(args) for _ in range(args.decoder_layers)
        ])


@register_model('hc_transformer')
class HCTransformerModel(TransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return HCTransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return HCTransformerDecoder(args, tgt_dict, embed_tokens)


@register_model_architecture('hc_transformer', 'hc_transformer_wmt_en_de_small')
def hc_transformer_wmt_en_de(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)

    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.merge_src_tgt_embed = getattr(args, "merge_src_tgt_embed", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)


@register_model_architecture("hc_transformer", "hc_transformer_wmt_en_de")
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1728)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 8000)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 27)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1728)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 8000)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 27)
    args.dropout = getattr(args, "dropout", 0.3)
    hc_transformer_wmt_en_de(args)






# DEBUG 1

@register_model("hc_transformer_debug_1")
class TransformerModelHC_debug1(TransformerModelBase):
    def __init__(self, args, encoder, decoder):
        cfg = TransformerConfig.from_namespace(args)
        super().__init__(cfg, encoder, decoder)
        self.args = args

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        # do not set defaults so that settings defaults from various architectures still works
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=True, with_prefix=""
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            args.share_decoder_input_output_embed = True

        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing

        if not args.share_all_embeddings:
            args.min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
        cfg = TransformerConfig.from_namespace(args)
        return super().build_model(cfg, task)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        return super().build_embedding(
            TransformerConfig.from_namespace(args), dictionary, embed_dim, path
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        # return super().build_encoder(
        #     TransformerConfig.from_namespace(args), src_dict, embed_tokens
        # )
        cfg = TransformerConfig.from_namespace(args)
        return TransformerEncoderBase(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        # return super().build_decoder(
        #     TransformerConfig.from_namespace(args), tgt_dict, embed_tokens
        # )
        cfg = TransformerConfig.from_namespace(args)
        return TransformerDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )
    

@register_model_architecture('hc_transformer_debug_1', 'hc_transformer_debug_1_wmt_en_de')
def hc_transformer_debug_1_wmt_en_de(args):
    hc_transformer_wmt_en_de(args)