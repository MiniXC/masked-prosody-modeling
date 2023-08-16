from pathlib import Path
import os
import tempfile
import shutil

import wandb
from huggingface_hub import Repository, create_repo, HfApi
import huggingface_hub
from rich.prompt import Prompt
from rich.console import Console

console = Console()


def wandb_update_config(*args):
    if len(args) == 1 and isinstance(args[0], dict):
        for k, v in args[0].items():
            v = v.__dict__
            v = {f"{k}/{_k}": _v for _k, _v in v.items()}
            wandb.config.update(v)
    for arg in args:
        wandb.config.update(arg)


def wandb_init(wandb_name, wandb_project, wandb_dir, wandb_mode):
    os.environ["WANDB_SILENT"] = "true"
    console.rule("Weights & Biases")
    if wandb_mode == "offline":
        console.print(
            f"logging in [dark_orange][b]OFFLINE[/b][/dark_orange] mode to [magenta][b]{wandb_dir}[/b][/magenta] directory"
        )
    if wandb_mode == "online":
        wandb_dir = Path(wandb_dir)
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb_last_input = wandb_dir / "last_input.txt"
        if wandb_last_input.exists():
            last_project, last_name = open(wandb_last_input, "r").read().split()
        else:
            last_project, last_name = (None, None)
        if wandb_project is None:
            wandb_project = Prompt.ask("wandb project", default=last_project)
        if wandb_name is None:
            wandb_name = Prompt.ask("wandb name", default=last_name)
        console.print(
            f"logging in [green][b]ONLINE[/b][/green] mode to [magenta][b]{wandb_project}[/b][/magenta] project"
        )
    console.print(f"run name: [magenta][b]{wandb_name}[/b][/magenta]")
    wandb.init(name=wandb_name, project=wandb_project, dir=wandb_dir, mode=wandb_mode)
    os.environ["WANDB_SILENT"] = "false"


def push_to_hub(repo_name, checkpoint_dir, commit_message="update model"):
    hub_token = os.getenv("HUGGING_FACE_HUB_TOKEN", default=None)
    if hub_token is None:
        console.log("no hub token found")
        raise ValueError("$HUGGING_FACE_HUB_TOKEN is not set")
    try:
        create_repo(repo_name, token=hub_token)
    except huggingface_hub.utils._errors.HfHubHTTPError:
        console.print(f"[magenta]{repo_name}[/magenta] already exists")
    repo_name = HfApi().get_full_repo_name(repo_name)
    temp_dict = {}
    console.print(
        f"pushing [magenta]{checkpoint_dir}[/magenta] to [magenta]{repo_name}[/magenta]"
    )
    try:
        repo = Repository(checkpoint_dir, clone_from=repo_name, token=hub_token)
        repo.git_pull()
    except EnvironmentError:
        for file in Path(checkpoint_dir).glob("*"):
            temp_copy = tempfile.NamedTemporaryFile(delete=False)
            temp_copy.write(file.read_bytes())
            temp_copy.close()
            temp_dict[file.name] = temp_copy.name
        shutil.rmtree(checkpoint_dir)
        repo = Repository(checkpoint_dir, clone_from=repo_name, token=hub_token)
        repo.git_pull()
        for name, path in temp_dict.items():
            # copy file to repo
            shutil.copy(path, checkpoint_dir / name)
            # add file to git
    repo.git_add(".")  # add all files
    git_head_commit_url = repo.push_to_hub(
        commit_message=commit_message, blocking=True, auto_lfs_prune=True
    )
    return git_head_commit_url
