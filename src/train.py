from train_model import train
from utils.config import TrainingConfig

options = TrainingConfig(
    learning_rate=1e-4,
    epochs=5,
    batch_size=3,
    scheduler_steps=[3, 4],
    loss_accumulation_step=10,
    scheduler_decay=0.1,
    model_save_interval=5,
    early_stoping_step=3,
    experiment_folder="experiments",
    debug_flag = False,
    files=["src/train_model.py", "src/model/CHAR.py", "src/model/CHARBlocks.py", "src/model/DETR.py", "src/dataset/EpicKitchen55.py"],
)

if __name__ == "__main__":
    train(options)