import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

class AttentionPooling(nn.Module):
    """ Adaptation of AttentionPool2D of https://github.com/openai/clip/blob/main/clip/model.py"""
    def __init__(self, embedding_dim, num_heads):
        super(AttentionPooling, self).__init__()
        self.attn = torch.nn.MultiheadAttention(embedding_dim, num_heads, batch_first=False)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        output, _ = self.attn(x[:1], x, x)
        return output.squeeze(0)

class ResNetFrameFeatureExtractur(nn.Module):
    """ResNet backbone"""
    def __init__(self, name: str, output_dim:int):
        super(ResNetFrameFeatureExtractur, self).__init__()
        backbone = getattr(torchvision.models, name)(weights='ResNet50_Weights.DEFAULT')
        self.body = IntermediateLayerGetter(backbone, return_layers={'layer4': "0"})
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        self.fc = nn.Linear(num_channels, output_dim)
        
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1, 3, 4)
        B, L, C, H, W = x.shape
        
        x = x.reshape(B * L, C, H, W)
        
        x = self.body(x)["0"]
        x = self.avgpool(x)
        x = x.reshape(B, L, -1)
        x = self.fc(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class PositionwiseFeedForward(nn.Module):
    """ Fully connected layer.

    This code is adapted from
    https://github.com/OpenNMT/OpenNMT-py/blob/4815f07fcd482af9a1fe1d3b620d144197178bc5/onmt/modules/position_ffn.py#L18
    """
    def __init__(self, dim, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(dim, d_ff)
        self.w_2 = nn.Linear(d_ff, dim)
        self.layernorm = nn.LayerNorm(dim, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: (B, L, dim)
        Returns:
            (B, L, dim)
        """
        inter = self.dropout_1(self.relu(self.w_1(self.layernorm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

class ObjectVideoCrossModalEncoderLayer(nn.Module):
    """ Cross-modal interaction module. Use object and video as each other's query or (key, value).
    """
    def __init__(self, dim, dropout):
        super(ObjectVideoCrossModalEncoderLayer, self).__init__()
        self.layernorm_o1 = nn.LayerNorm(dim, eps=1e-6)
        self.layernorm_o2 = nn.LayerNorm(dim, eps=1e-6)
        self.layernorm_v1 = nn.LayerNorm(dim, eps=1e-6)
        self.layernorm_v2 = nn.LayerNorm(dim, eps=1e-6)
        self.o2v = nn.MultiheadAttention(embed_dim=dim, num_heads=8, dropout=dropout, batch_first=True)
        self.v2o = nn.MultiheadAttention(embed_dim=dim, num_heads=8, dropout=dropout, batch_first=True)
        self.fc_o = PositionwiseFeedForward(dim=dim, d_ff=4*dim, dropout=dropout)
        self.fc_v = PositionwiseFeedForward(dim=dim, d_ff=4*dim, dropout=dropout)

    def forward(self, video_feature, object_feature):

        o2v, _ = self.o2v(
            query=object_feature,
            key=video_feature,
            value=video_feature
        )
        o2v = self.layernorm_v1(o2v + video_feature)
        o2v = self.layernorm_v2(self.fc_v(o2v) + o2v)

        v2o, _ = self.v2o(
            query=video_feature,
            key=object_feature,
            value=object_feature
        )
        v2o = self.layernorm_o1(v2o + object_feature)
        v2o = self.layernorm_o2(self.fc_o(v2o) + v2o)
        

        return o2v, v2o
    
class ObjectVideoCrossModalEncoder(nn.Module):
    def __init__(self, dim, n_layers, dropout):
        super(ObjectVideoCrossModalEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [ObjectVideoCrossModalEncoderLayer(dim, dropout) for _ in range(n_layers)]
        )

    def forward(self, video_feature, object_feature):
        for layer in self.layers:
            video_feature, object_feature = layer(
                video_feature, object_feature
            )
        return video_feature, object_feature

class QueryVideoCrossModalEncoderLayer(nn.Module):
    """ Cross-modal interaction module. Use query and video as each other's query or (key, value).
    """
    def __init__(self, dim, dropout):
        super(QueryVideoCrossModalEncoderLayer, self).__init__()
        self.layernorm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(dim, eps=1e-6)
        self.q2v = nn.MultiheadAttention(embed_dim=dim, num_heads=8, dropout=dropout, batch_first=True)
        self.v2q = nn.MultiheadAttention(embed_dim=dim, num_heads=8, dropout=dropout, batch_first=True)
        self.fc_q = PositionwiseFeedForward(dim=dim, d_ff=4*dim, dropout=dropout)
        self.fc_v = PositionwiseFeedForward(dim=dim, d_ff=4*dim, dropout=dropout)

    def forward(self, query_feature, video_feature):

        q2v, _ = self.q2v(
            query=query_feature,
            key=video_feature,
            value=video_feature
        )
        q2v = self.fc_q(q2v + query_feature)

        v2q, _ = self.v2q(
            query=video_feature,
            key=query_feature,
            value=query_feature
        )
        v2q = self.fc_v(v2q + video_feature)
        
        query_feature = self.layernorm1(q2v)
        video_feature = self.layernorm2(v2q)

        return q2v, v2q

class QueryVideoCrossModalEncoder(nn.Module):
    def __init__(self, dim, n_layers, dropout):
        super(QueryVideoCrossModalEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [QueryVideoCrossModalEncoderLayer(dim, dropout) for _ in range(n_layers)]
        )

    def forward(self, query_feature, video_feature):
        for layer in self.layers:
            query_feature, video_feature = layer(
                query_feature, video_feature
            )
        return query_feature, video_feature