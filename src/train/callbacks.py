import wandb

from datetime import datetime
from mltu.torch.callbacks import Callback


class WandbLogger(Callback):
    """W&B Logger callback for MLTU training."""

    def __init__(
            self,
            project: str = "default_project",
            name: str = None,
            config: dict = None,
            log_model: bool = False,
    ):
        """Initialize the W&B logger.

        Args:
            project (str): W&B project name.
            name (str, optional): Run name. If None, a timestamp is used.
            config (dict, optional): Config dictionary to log hyperparameters.
            log_model (bool): Whether to log model artifacts.
        """
        super(WandbLogger, self).__init__()
        self.project = project
        self.name = name or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.config = config or {}
        self.log_model = log_model
        self.run = None

    def on_train_begin(self, logs=None):
        self.run = wandb.init(
            project=self.project,
            name=self.name,
            config=self.config,
            reinit=True
        )
        wandb.watch(self.model.model, log="all", log_freq=500)

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        wandb.log({"epoch": epoch, **logs})

        # Optionally log learning rate
        for param_group in self.model.optimizer.param_groups:
            wandb.log({"learning_rate": param_group["lr"]}, step=epoch)

    def on_train_end(self, logs=None):
        wandb.finish()
