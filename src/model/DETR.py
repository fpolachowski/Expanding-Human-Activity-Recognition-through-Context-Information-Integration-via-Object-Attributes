import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .DETR_backbone import build_backbone
from .DETR_Transformer import Transformer
from safetensors.torch import save_model, load_model

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, 
                 backbone_name,
                 train_backbone,
                 dilation,
                 position_embedding_name,
                 transformer_width,
                 transformer_dropout,
                 transformer_nheads,
                 transformer_width_ffn,
                 transformer_nencodelayers,
                 transformer_ndecodelayers,
                 transformer_prenorm,
                 num_classes, 
                 num_queries, 
                 aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = Transformer(
            d_model=transformer_width,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_width_ffn,
            num_encoder_layers=transformer_nencodelayers,
            num_decoder_layers=transformer_ndecodelayers,
            normalize_before=transformer_prenorm,
            return_intermediate_dec=True,
        )
        self.class_embed = nn.Linear(transformer_width, num_classes + 1)
        self.bbox_embed = MLP(transformer_width, transformer_width, 4, 3)
        self.query_embed = nn.Embedding(num_queries, transformer_width)
        self.backbone = build_backbone(transformer_width, position_embedding_name, train_backbone, backbone_name, dilation)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, transformer_width, kernel_size=1)
        self.aux_loss = aux_loss
        
        
        self.gpu_device = torch.device("cuda:0")
        self.cpu_device = torch.device("cpu")

    def forward(self, samples: torch.Tensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples: batched images, of shape [batch_size x 3 x H x W]

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        features, pos = self.backbone(samples)

        hs = self.transformer(self.input_proj(features), self.query_embed.weight, pos)[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out
    
    def predict(self, samples: torch.Tensor):
        features, pos = self.backbone(samples)

        hs = self.transformer(self.input_proj(features), self.query_embed.weight, pos)[0]
        
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        
        return hs[-1], out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        
    def load_checkpoint(self, exp_folder_path, suffix):
        load_model(self, os.path.join(exp_folder_path, "model_{}.safetensors".format(suffix)))
        # self.load_state_dict(torch.load(os.path.join(exp_folder_path, "model_{}.pt".format(suffix))))
        # self.optimizer.load_state_dict(torch.load(os.path.join(exp_folder_path, "optimizer_{}.pt".format(suffix))))
        # self.scheduler.load_state_dict(torch.load(os.path.join(exp_folder_path, "scheduler_{}.pt".format(suffix))))
        print("== Checkpoint ({}) is loaded from {}".format(suffix, exp_folder_path))

    def save_checkpoint(self, exp_folder_path, suffix):
        save_model(self, os.path.join(exp_folder_path, "model_{}.safetensors".format(suffix)))
        # torch.save(self.state_dict(), os.path.join(exp_folder_path, "model_{}.pt".format(suffix)))
        # torch.save(self.optimizer.state_dict(), os.path.join(exp_folder_path, "optimizer_{}.pt".format(suffix)))
        # torch.save(self.scheduler.state_dict(), os.path.join(exp_folder_path, "scheduler_{}.pt".format(suffix)))
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