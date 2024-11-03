import torch
import random
import numpy as np
import torch.nn.functional as F

from dataset.EpicKitchen55 import prepare_data
from model.CHAR import CHAR
from utils.util import n_params, calculate_cosine_eval_accuracy, calculate_top_N_cosine_eval_accuracy
from utils.evaluator import Evaluator
from model.DETR import DETR

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

TEST_TRAIN = True
TEST_EVAL = True

LR = 0.001

def predict_object_information(detector, images):
    # B, C, N, H, W -> B, N, C, H, W
    detection_images = images.permute(0, 2, 1, 3, 4)
    B, N, C, H, W = detection_images.shape
    detection_images = torch.reshape(detection_images, (B*N, C, H, W))
    object_features, _ = detector.predict(detection_images)
    return torch.reshape(object_features, (B, N, 16, 512)) 

def debug():
    model = CHAR(
        video_length=64,
        sentence_length=4, # max sentence length = 8
        vocab_length=1596,
        vocab_size=421,
        transformer_width=256,
        transformer_layers=6,
        dropout=0.1
    )
    
    detector = DETR(
        backbone_name="resnet50",
        train_backbone=True,
        dilation=True,
        position_embedding_name="sine",
        transformer_width=512,
        transformer_dropout=0.1,
        transformer_nheads=8,
        transformer_width_ffn=2048,
        transformer_nencodelayers=6,
        transformer_ndecodelayers=6,
        transformer_prenorm=True,
        num_classes=290,
        num_queries=16, # max class count = 13
        aux_loss=False
    )
    detector.load_checkpoint("./src/model", "detr")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    evaluator = Evaluator(3)
    
    print(f"Model has {n_params(model)} parameters.")
    use_gpu = torch.cuda.is_available()
    print(f"Using GPU: {use_gpu}.\n")
    if use_gpu:
        model.gpu_mode()
        detector.gpu_mode()
        device = torch.device("cuda:0")
    else:
        model.cpu_mode()
        detector.cpu_mode()
        device = torch.device("cpu")
    
    # prepare data
    data = prepare_data(batch_size=2)
    train_dl = data["train_dl"]
    eval_dl = data["eval_dl"]
    test_dl = data["test_dl"]
    tokenized_vocab = data["tokenized_vocab"].to(device)
    
    if TEST_TRAIN:
        print("Training Section")
        model.train_mode()
    
        batch = next(iter(train_dl))
        optimizer.zero_grad()
        images, labels = batch["tensor"].to(device), batch["labels"].to(device)
        pred_det = predict_object_information(detector, images)
        video_text_loss, video_text_similarity = model(images, tokenized_vocab, labels, pred_det)
            
            
        pos_labels = F.one_hot(labels, model.vocab_length)
        neg_labels = torch.ones(pos_labels.shape, device=pos_labels.device) - pos_labels

        video_text_pos_sim = pos_labels * video_text_similarity
        video_text_neg_sim = neg_labels * video_text_similarity
        
        max_video_text_neg_sim, _ = video_text_neg_sim.max(dim=1)
        max_video_text_pos_sim, _ = video_text_pos_sim.max(dim=1)
        
        prediction_diff = (max_video_text_pos_sim - max_video_text_neg_sim).mean()        
                    
        video_text_pos_sim = torch.sum(video_text_pos_sim) / torch.sum(pos_labels) # custom mean as num of pos and neg sample is hugh difference
        video_text_neg_sim = torch.sum(video_text_neg_sim) / torch.sum(neg_labels) # very similar to mean but more accurate
        
        video_text_pos_sim_loss = (1 - video_text_pos_sim)
        video_text_neg_sim_loss = video_text_neg_sim
        
        loss = video_text_loss + video_text_pos_sim_loss + max_video_text_neg_sim.mean()
        loss.backward()
        optimizer.step()
        
        print(loss, video_text_loss, video_text_pos_sim_loss, max_video_text_neg_sim.mean(), prediction_diff)
        print(video_text_pos_sim, video_text_neg_sim, max_video_text_neg_sim.mean())
        
        model.eval_mode()
        with torch.no_grad():
            encoded_vocab = model.encode_text(tokenized_vocab)
            
            batch = next(iter(test_dl))
            images, labels = batch["tensor"].to(device), batch["labels"].to(device)
            pred_det = predict_object_information(detector, images)
            video_feature, sentence_feature = model.forward_eval(images, encoded_vocab, pred_det)
            
            evaluator.run_evaluation("accuracy", calculate_cosine_eval_accuracy, (video_feature, sentence_feature, labels))
            evaluator.run_evaluation("top_5_acc", calculate_top_N_cosine_eval_accuracy, (video_feature, sentence_feature, labels, 5))
            
            best_epoch, accuracy = evaluator.step("accuracy", 1, check_for_best_epoch=False)
            _, top_5_acc = evaluator.step("top_5_acc", 1, check_for_best_epoch=False)
            print(best_epoch, accuracy)
            print(top_5_acc)
        
        
    if TEST_EVAL:
        print("Evaluation Section:")
        model.eval_mode()
        with torch.no_grad():
            encoded_vocab = model.encode_text(tokenized_vocab)
            
            batch = next(iter(eval_dl))
            images, labels = batch["tensor"].to(device), batch["labels"].to(device)
            pred_det = predict_object_information(detector, images)
            video_feature, sentence_feature = model.forward_eval(images, encoded_vocab, pred_det)
            evaluator.run_evaluation("accuracy", calculate_cosine_eval_accuracy, (video_feature, sentence_feature, labels))
            evaluator.run_evaluation("top_5_acc", calculate_top_N_cosine_eval_accuracy, (video_feature, sentence_feature, labels, 5))
            
            best_epoch, accuracy = evaluator.step("accuracy", 1)
            _, top_5_acc = evaluator.step("top_5_acc", 1)
            print(best_epoch, accuracy)
            print(top_5_acc)
            
    model.save_checkpoint("./experiments/debug", "best_var")
    detector.save_checkpoint("./experiments/debug", "best_detr")
            
            

        
if __name__ == '__main__':
    debug()