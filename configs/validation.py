from .args import TrainingArgs, ModelArgs, CollatorArgs


def validate_args(*args):
    for arg in args:
        if isinstance(arg, TrainingArgs):
            if arg.dataset not in [
                "cdminix/libritts-aligned",
                "cdminix/libritts-r-aligned",
            ]:
                raise ValueError(f"dataset {arg.dataset} not supported")
            if arg.lr_schedule not in ["linear_with_warmup"]:
                raise ValueError(f"lr_schedule {arg.lr_schedule} not supported")
            if arg.wandb_mode not in ["online", "offline"]:
                raise ValueError(f"wandb_mode {arg.wandb_mode} not supported")
            if arg.wandb_mode == "online":
                if arg.wandb_project is None:
                    raise ValueError("wandb_project must be specified")
            if arg.push_to_hub:
                if arg.hub_repo is None:
                    raise ValueError("hub_repo must be specified")
        if isinstance(arg, ModelArgs):
            if arg.hidden_dim % 2 != 0:
                raise ValueError("hidden_dim should be divisible by 2")
        if isinstance(arg, CollatorArgs):
            if arg.name not in ["default", "algo"]:
                raise ValueError(f"collator {arg.name} not supported")
