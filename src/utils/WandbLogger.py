import wandb
from .config import TrainingConfig

class WandbLogger():
    def __init__(self, name, options: TrainingConfig) -> None:
        self.options = options
        wandb.login()
        self.run = wandb.init(
            # Set the project where this run will be logged
            project=name,
            # Track hyperparameters and run metadata
            config={
                "epochs": options.epochs,
                "learning_rate": options.learning_rate,
                "scheduler_steps": options.scheduler_steps,
                "scheduler_decay": options.scheduler_decay,
                "batch_size": options.batch_size,
                "early_stoping_step": options.early_stoping_step
            },
            settings=wandb.Settings(disable_git=True)
        )
        
        for f in options.files:
            wandb.save(f)
        
        
            
    def log(self, value_dict: dict[str, float]):
        for name in value_dict:
            self.run.log({name: value_dict[name]})
        