import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model

from model.CHARBlocks import QueryVideoCrossModalEncoder, ObjectVideoCrossModalEncoder, ResNetFrameFeatureExtractur, AttentionPooling

class CHAR(nn.Module):
    def __init__(self, 
            video_length=128,
            sentence_length=4,
            object_count=16,
            vocab_length=1596,
            transformer_width=512,
            transformer_layers=6,
            transfromer_nheads=8,
            object_embeddgin_dim=512,
            dropout=0.1
        ):
        super(CHAR, self).__init__()
        self.vocab_length = vocab_length
        ### build network
        encoder_layers = nn.TransformerEncoderLayer(transformer_width, transfromer_nheads, transformer_width, dropout, batch_first=True)
        
        # Object Section
        self.fc_o = nn.Linear(object_embeddgin_dim, transformer_width)
        self.object_encoding = nn.Parameter(torch.empty(object_count, transformer_width))
        self.object_encoder = nn.TransformerEncoder(encoder_layers, transformer_layers)
        self.object_ln = nn.LayerNorm(transformer_width)
        self.object_pool = AttentionPooling(transformer_width, transfromer_nheads)
        
        # Query Text Section
        self.query_embedding = nn.Embedding(vocab_length, transformer_width)
        self.query_encoding = nn.Parameter(torch.empty(sentence_length, transformer_width))
        self.query_encoder = nn.TransformerEncoder(encoder_layers, transformer_layers)
        self.query_ln = nn.LayerNorm(transformer_width)
        self.query_pool = AttentionPooling(transformer_width, transfromer_nheads)

        # Vision Section
        # self.frame_extractor = FrameFeatureExtractor(transformer_width)
        self.frame_extractor = ResNetFrameFeatureExtractur("resnet50", transformer_width)
        self.frame_encoding = nn.Parameter(torch.empty(video_length, transformer_width))
        self.frame_encoder = nn.TransformerEncoder(encoder_layers, transformer_layers)
        self.frame_ln = nn.LayerNorm(transformer_width)
        self.frame_pool = AttentionPooling(transformer_width, transfromer_nheads)
        
        # cross-modal section
        self.cross_modal_vid_txt_encoder = QueryVideoCrossModalEncoder(transformer_width, 2, dropout)
        self.cross_modal_vid_obj_encoder = ObjectVideoCrossModalEncoder(transformer_width, transformer_layers, dropout)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # single GPU assumed
        self.use_gpu = False
        self.device = None
        self.gpu_device = torch.device("cuda:0")
        self.cpu_device = torch.device("cpu")
        self.cpu_mode()
        
        self.init_weights()
        
    def init_weights(self) -> None:
        nn.init.normal_(self.query_embedding.weight, std=0.02)
        nn.init.normal_(self.query_encoding, std=0.01)
        nn.init.normal_(self.frame_encoding, std=0.01)
        nn.init.normal_(self.object_encoding, std=0.01)
        
        proj_std = (self.frame_encoding.shape[1] ** -0.5) * ((2 * self.frame_encoding.shape[1]) ** -0.5)
        attn_std = self.frame_encoding.shape[1] ** -0.5
        fc_std = (2 * self.frame_encoding.shape[1]) ** -0.5
        for encoder in [self.query_encoder, self.frame_encoder, self.object_encoder]:
            for layer in encoder.layers:
                nn.init.normal_(layer.self_attn.in_proj_weight, std=attn_std)
                nn.init.normal_(layer.self_attn.out_proj.weight, std=proj_std)
                nn.init.normal_(layer.linear1.weight, std=fc_std)
                nn.init.normal_(layer.linear2.weight, std=proj_std)
                
        for pooling in [self.query_pool, self.frame_pool, self.object_pool]:
            if pooling.attn is not None:
                std = pooling.attn.out_proj.in_features ** -0.5
                nn.init.normal_(pooling.attn.in_proj_weight, std=std)
                nn.init.normal_(pooling.attn.out_proj.weight, std=std)
            
    def pooling(self, x, dim):
        return torch.max(x, dim=dim)[0]
    
    def encode_object(self, object_features):
        object_features = self.fc_o(object_features)
        B, N, O, D = object_features.shape
        object_features += self.object_encoding.type(object_features.dtype)
        object_features = torch.reshape(object_features, (B*N, O, D))
        object_features = self.object_encoder(object_features)
        object_features = self.object_ln(object_features)
        object_features = self.object_pool(object_features)
        object_features = torch.reshape(object_features, (B, N, D))
        
        return object_features
    
    def encode_video(self, clip):
        frame_features = self.frame_extractor(clip)
        frame_features = frame_features + self.frame_encoding.type(frame_features.dtype)
        frame_features = self.frame_encoder(frame_features)
        frame_features = self.frame_ln(frame_features)
        return frame_features
        
    def encode_text(self, text):
        words_feature = self.query_embedding(text)
        words_feature = words_feature + self.query_encoding.type(words_feature.dtype)
        words_feature = self.query_encoder(words_feature)
        words_feature = self.query_ln(words_feature)
        return words_feature
    
    def cross_modality(self, frame_features, words_features, object_features):
        frame_features, _ = self.cross_modal_vid_obj_encoder(frame_features, object_features)
        
        video_features = torch.empty((frame_features.shape[0], words_features.shape[0], frame_features.shape[2]), device=frame_features.device)
        sentence_features = torch.empty((frame_features.shape[0], words_features.shape[0], frame_features.shape[2]), device=words_features.device)
        
        for i, _single_vid_frame_features in enumerate(frame_features):
            _single_vid_frame_features = _single_vid_frame_features.unsqueeze(0).repeat(words_features.shape[0], 1, 1)
            
            _words_feature, _single_vid_features = self.cross_modal_vid_txt_encoder(words_features, _single_vid_frame_features)
            
            video_features[i] = self.frame_pool(_single_vid_features)
            sentence_features[i] = self.query_pool(_words_feature)
            
        return video_features, sentence_features

    def forward(self, images, texts, labels, pred_det):
        """

        Returns:
            loss: single item tensor
        """
        frame_features = self.encode_video(images)
        words_features = self.encode_text(texts)
        object_features = self.encode_object(pred_det)

        # cross modality video <- object + video <-> text
        video_features, sentence_features = self.cross_modality(frame_features, words_features, object_features)
        
        # Normalize IMPORTANT!!!
        video_features = video_features / video_features.norm(dim=2, keepdim=True)
        sentence_features = sentence_features / sentence_features.norm(dim=2, keepdim=True)

        # cosine similarity as logits
        video_text_similarity = F.cosine_similarity(video_features, sentence_features, dim=-1)
        video_text_loss = F.cross_entropy(video_text_similarity, labels)
        
        return video_text_loss, video_text_similarity
    
    def forward_eval(self, images, words_features, object_features):
        frame_features = self.encode_video(images)
        object_features = self.encode_object(object_features)
        
        video_features, sentence_features = self.cross_modality(frame_features, words_features, object_features)        
        return video_features, sentence_features

    def load_checkpoint(self, exp_folder_path, suffix):
        load_model(self, os.path.join(exp_folder_path, "model_{}.safetensors".format(suffix)))
        print("== Checkpoint ({}) is loaded from {}".format(suffix, exp_folder_path))

    def save_checkpoint(self, exp_folder_path, suffix):
        save_model(self, os.path.join(exp_folder_path, "model_{}.safetensors".format(suffix)))
        print("== Checkpoint ({}) is saved to {}".format(suffix, exp_folder_path))

    def cpu_mode(self):
        self.use_gpu = False
        self.to(self.cpu_device)
        self.device = self.cpu_device

    def gpu_mode(self):
        self.use_gpu = True
        self.to(self.gpu_device)
        self.device = self.gpu_device

    def train_mode(self):
        self.train()

    def eval_mode(self):
        self.eval()