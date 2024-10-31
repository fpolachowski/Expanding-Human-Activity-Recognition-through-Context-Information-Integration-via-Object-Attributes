import torch
import random
import os
import numpy as np
import torch.nn.functional as F

from dataset.EpicKitchen55 import prepare_data
from model.CHAR import CHAR
from utils.util import n_params, safe_create_folder, calculate_cosine_eval_accuracy, calculate_top_N_cosine_eval_accuracy 
from utils.evaluator import Evaluator
from utils.config import TrainingConfig
from model.DETR import DETR

@torch.no_grad()
def predict_object_information(detector, images):
    # B, C, N, H, W -> B, N, C, H, W
    detection_images = images.permute(0, 2, 1, 3, 4)
    B, N, C, H, W = detection_images.shape
    detection_images = torch.reshape(detection_images, (B*N, C, H, W))
    object_features, _ = detector.predict(detection_images)
    return torch.reshape(object_features, (B, N, 16, 512)) 


def train(options: TrainingConfig):
    from utils.WandbLogger import WandbLogger
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    logger = WandbLogger(name="New_CHAR_Training", options=options)
    
    # update experiment folder name
    options.experiment_folder = os.path.join(options.experiment_folder, f"{logger.run.name}")
    
    # disable torch features for speedup
    torch.autograd.set_detect_anomaly(mode=False)
    torch.autograd.profiler.profile(enabled=False)
    
    # enable cuDNN auto-tuner
    torch.backends.cudnn.benchmark = True

    model = CHAR(
        video_length=64,
        sentence_length=4, # max sentence length = 8
        vocab_length=1596,
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
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=options.scheduler_steps, gamma=options.scheduler_decay)
    
    evaluator = Evaluator(options.early_stoping_step)
    
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
    data = prepare_data(batch_size=options.batch_size)
    train_dl = data["train_dl"]
    eval_dl = data["eval_dl"]
    test_dl = data["test_dl"]
    tokenized_vocab = data["tokenized_vocab"].to(device)
    
    for epoch in range(1, options.epochs + 1):
        model.train_mode()
        print("Training epoch {} with lr {} (0/{})".format(epoch, optimizer.param_groups[0]["lr"], len(train_dl)))
        for i, batch in enumerate(train_dl):
            images, labels = batch["tensor"].to(device), batch["labels"].to(device)
            pred_det = predict_object_information(detector, images)
            video_text_loss, video_text_similarity = model(images, tokenized_vocab, labels, pred_det)
            
            
            pos_labels = F.one_hot(labels, model.vocab_length)
            neg_labels = torch.ones(pos_labels.shape, device=pos_labels.device) - pos_labels

            video_text_pos_sim = pos_labels * video_text_similarity
            video_text_neg_sim = neg_labels * video_text_similarity
            
            max_video_text_neg_sim, _ = video_text_neg_sim.max(dim=1)
                        
            video_text_pos_sim = torch.sum(video_text_pos_sim) / torch.sum(pos_labels) # custom mean as num of pos and neg sample is hugh difference
            video_text_neg_sim = torch.sum(video_text_neg_sim) / torch.sum(neg_labels) # very similar to mean but more accurate
            
            video_text_pos_sim_loss = (1 - video_text_pos_sim)
            video_text_neg_sim_loss = video_text_neg_sim
            
            loss = video_text_loss # + video_text_pos_sim_loss + max_video_text_neg_sim.mean()
            
            
            
            logger.log({"loss": loss.item(), "video_text_pos_sim_loss":video_text_pos_sim_loss.item(), "pos_text_sim": video_text_pos_sim.item(), "neg_text_sim": video_text_neg_sim.item(), "max_neg_sim": max_video_text_neg_sim.mean().item()})
            loss /= options.loss_accumulation_step # for loss accumulation
            loss.backward()
            
            # accumulate loss over options.loss_accumulation_step steps
            if i % options.loss_accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
            # run test on training data; no best epoch logging
            if i % (len(train_dl) // 3) == len(train_dl) // 3 - 1:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    print("Running test in epoch {} with lr {} (0/{})".format(epoch, optimizer.param_groups[0]["lr"], len(test_dl)))
                    
                    model.eval_mode()
                    encoded_vocab = model.encode_text(tokenized_vocab)
                    for batch in test_dl:
                        images, labels = batch["tensor"].to(device), batch["labels"].to(device)
                        pred_det = predict_object_information(detector, images)
                        video_feature, sentence_feature = model.forward_eval(images, encoded_vocab, pred_det)
                        evaluator.run_evaluation("train accuracy", calculate_cosine_eval_accuracy, (video_feature, sentence_feature, labels))
                        evaluator.run_evaluation("train top_5_acc", calculate_top_N_cosine_eval_accuracy, (video_feature, sentence_feature, labels, 5))
                    
                    _, accuracy = evaluator.step("train accuracy", len(test_dl), False)
                    _, top_5_acc = evaluator.step("train top_5_acc", len(test_dl), False)
                    logger.log({"train accuracy": accuracy, "train top_5_acc": top_5_acc})
                    
                    print("Finished test in epoch {} with accuracy {}".format(epoch, accuracy))
                    
                    model.train_mode()
                    evaluator.reset()
                
            
        # step learning rate scheduler
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        
        # run eval dataset
        with torch.no_grad():
            print("Running evaluation in epoch {} with lr {} (0/{})".format(epoch, optimizer.param_groups[0]["lr"], len(eval_dl)))
            
            model.eval_mode()
            encoded_vocab = model.encode_text(tokenized_vocab)
            for batch in eval_dl:
                images, labels = batch["tensor"].to(device), batch["labels"].to(device)
                pred_det = predict_object_information(detector, images)
                video_feature, sentence_feature = model.forward_eval(images, encoded_vocab, pred_det)
                evaluator.run_evaluation("accuracy", calculate_cosine_eval_accuracy, (video_feature, sentence_feature, labels))
                evaluator.run_evaluation("top_5_acc", calculate_top_N_cosine_eval_accuracy, (video_feature, sentence_feature, labels, 5))
                
            
            best_epoch, accuracy = evaluator.step("accuracy", len(eval_dl))
            _, top_5_acc = evaluator.step("top_5_acc", len(eval_dl))
            logger.log({"accuracy": accuracy, "top_5_acc": top_5_acc})
            
            print("Finished test in epoch {} with accuracy {} / top 5 accuracy {}".format(epoch, accuracy, top_5_acc))

            evaluator.reset()

        # save every N epochs
        # if not best_epoch and epoch % options.model_save_interval == 1: # since we start with 1
        #     model.save_checkpoint(options.experiment_folder, f"{epoch}")
        # save best epoch
        if best_epoch:
            safe_create_folder(options.experiment_folder)
            model.save_checkpoint(options.experiment_folder, "best_var")
            detector.save_checkpoint(options.experiment_folder, "best_detr")
            
        # epoch cleanup section
        stop_early = evaluator.check_early_stoping(epoch)
        if stop_early and epoch > options.epochs*(1/2):
            print("Stoping experiment early due to stop critereon reached!")
            return
            
if __name__ == '__main__':
    options = TrainingConfig(
        learning_rate=1e-4,
        epochs=25,
        batch_size=1,
        scheduler_steps=[18, 22],
        loss_accumulation_step=5,
        scheduler_decay=0.1,
        model_save_interval=5,
        early_stoping_step=5,
        experiment_folder="experiments",
        debug_flag = False)
    train(options)