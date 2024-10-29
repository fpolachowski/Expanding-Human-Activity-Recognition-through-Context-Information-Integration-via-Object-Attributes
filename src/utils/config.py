from dataclasses import dataclass

@dataclass
class TrainingConfig():
    learning_rate: float
    epochs: int
    batch_size: int
    early_stoping_step: int
    loss_accumulation_step: int
    scheduler_steps: list[int]
    scheduler_decay: float
    model_save_interval: int
    experiment_folder: str
    debug_flag: bool
    files: list[str]