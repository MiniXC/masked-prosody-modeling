import os
import sys
from collections import deque
from pathlib import Path

sys.path.append(".")  # add root of project to path

# torch & hf
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup, HfArgumentParser
from datasets import load_dataset

# logging & etc
from torchinfo import summary
from torchview import draw_graph
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm.auto import tqdm
import yaml
from rich.console import Console

# plotting
import matplotlib.pyplot as plt
from figures.plotting import plot_first_batch

console = Console()

# local imports
from configs.args import TrainingArgs, ModelArgs, CollatorArgs
from configs.validation import validate_args
from util.remote import wandb_update_config, wandb_init, push_to_hub
from model.masked_prosody_model import MaskedProsodyModel
from collators import get_collator


def print_and_draw_model():
    dummy_input = model.dummy_input
    # repeat dummy input to match batch size (regardless of how many dimensions)
    dummy_input = dummy_input.repeat(
        (training_args.batch_size,) + (1,) * (len(dummy_input.shape) - 1)
    )
    console_print(f"[green]input shape[/green]: {dummy_input.shape}")
    model_summary = summary(
        model,
        input_data=dummy_input,
        verbose=0,
        col_names=[
            "input_size",
            "output_size",
            "num_params",
        ],
    )
    console_print(model_summary)
    if accelerator.is_main_process:
        draw_graph(
            model,
            input_data=dummy_input,
            save_graph=True,
            directory="figures/",
            filename="model",
            expand_nested=True,
        )


def console_print(*args, **kwargs):
    if accelerator.is_main_process:
        console.print(*args, **kwargs)


def console_rule(*args, **kwargs):
    if accelerator.is_main_process:
        console.rule(*args, **kwargs)


def wandb_log(prefix, log_dict, round_n=3, print_log=True):
    if accelerator.is_main_process:
        log_dict = {f"{prefix}/{k}": v for k, v in log_dict.items()}
        wandb.log(log_dict, step=global_step)
        if print_log:
            log_dict = {k: round(v, round_n) for k, v in log_dict.items()}
            console.log(log_dict)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def save_checkpoint():
    accelerator.wait_for_everyone()
    checkpoint_name = training_args.run_name
    checkpoint_path = (
        Path(training_args.checkpoint_path) / checkpoint_name / f"step_{global_step}"
    )
    # model
    model.save_model(checkpoint_path, accelerator, onnx=training_args.save_onnx)
    if accelerator.is_main_process:
        # training args
        with open(checkpoint_path / "training_args.yml", "w") as f:
            f.write(yaml.dump(training_args.__dict__, Dumper=yaml.Dumper))
        # collator args
        with open(checkpoint_path / "collator_args.yml", "w") as f:
            f.write(yaml.dump(collator_args.__dict__, Dumper=yaml.Dumper))
        if training_args.push_to_hub:
            push_to_hub(
                training_args.hub_repo,
                checkpoint_path,
                commit_message=f"step {global_step}",
            )
    accelerator.wait_for_everyone()


def train_epoch(epoch):
    global global_step
    model.train()
    losses = deque(maxlen=training_args.log_every_n_steps)
    step = 0
    console_rule(f"Epoch {epoch}")
    last_loss = None
    for batch in train_dl:
        with accelerator.accumulate(model):
            y = model(batch["image"])
            loss = torch.nn.functional.cross_entropy(y, batch["target"])
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(
                model.parameters(), training_args.gradient_clip_val
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        losses.append(loss.detach())
        if (
            step > 0
            and step % training_args.log_every_n_steps == 0
            and accelerator.is_main_process
        ):
            last_loss = torch.mean(torch.tensor(losses)).item()
            wandb_log("train", {"loss": last_loss}, print_log=False)
        if (
            training_args.do_save
            and global_step > 0
            and global_step % training_args.save_every_n_steps == 0
        ):
            save_checkpoint()
        if training_args.n_steps is not None and global_step >= training_args.n_steps:
            return
        if (
            training_args.eval_every_n_steps is not None
            and global_step > 0
            and global_step % training_args.eval_every_n_steps == 0
            and accelerator.is_main_process
        ):
            if training_args.do_full_eval:
                evaluate()
            else:
                evaluate_loss_only()
            console_rule(f"Epoch {epoch}")
        step += 1
        global_step += 1
        if accelerator.is_main_process:
            pbar.update(1)
            if last_loss is not None:
                pbar.set_postfix({"loss": f"{last_loss:.3f}"})


def evaluate():
    model.eval()
    y_true = []
    y_pred = []
    losses = []
    console_rule("Evaluation")
    for batch in val_dl:
        y = model(batch["image"])
        loss = torch.nn.functional.cross_entropy(y, batch["target"])
        losses.append(loss.detach())
        y_true.append(batch["target"].cpu().numpy())
        y_pred.append(y.argmax(-1).cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    wandb_log("val", {"loss": torch.mean(torch.tensor(losses)).item()})
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro")
    wandb_log("val", {"acc": acc, "f1": f1, "precision": precision, "recall": recall})


def evaluate_loss_only():
    model.eval()
    losses = []
    console_rule("Evaluation")
    for batch in val_dl:
        y = model(batch["image"])
        loss = torch.nn.functional.cross_entropy(y, batch["target"])
        losses.append(loss.detach())
    wandb_log("val", {"loss": torch.mean(torch.tensor(losses)).item()})


def main():
    global accelerator, training_args, model_args, collator_args, train_dl, val_dl, optimizer, scheduler, model, global_step, pbar

    global_step = 0

    parser = HfArgumentParser([TrainingArgs, ModelArgs, CollatorArgs])

    accelerator = Accelerator()

    # parse args
    if len(sys.argv) > 1 and sys.argv[1].endswith(".yml"):
        with open(sys.argv[1], "r") as f:
            args_dict = yaml.load(f, Loader=yaml.Loader)
        # additonally parse args from command line
        (
            training_args,
            model_args,
            collator_args,
        ) = parser.parse_args_into_dataclasses(sys.argv[2:])
        # update args from yaml
        for k, v in args_dict.items():
            if hasattr(training_args, k):
                setattr(training_args, k, v)
            if hasattr(model_args, k):
                setattr(model_args, k, v)
            if hasattr(collator_args, k):
                setattr(collator_args, k, v)
        if len(sys.argv) > 2:
            console_print(
                f"[yellow]WARNING[/yellow]: yaml args will be override command line args"
            )
    else:
        (
            training_args,
            model_args,
            collator_args,
        ) = parser.parse_args_into_dataclasses()

    # check if run name is specified
    if training_args.run_name is None:
        raise ValueError("run_name must be specified")
    if (
        training_args.do_save
        and (Path(training_args.checkpoint_path) / training_args.run_name).exists()
    ):
        raise ValueError(f"run_name {training_args.run_name} already exists")

    # wandb
    if accelerator.is_main_process:
        wandb_name, wandb_project, wandb_dir, wandb_mode = (
            training_args.run_name,
            training_args.wandb_project,
            training_args.wandb_dir,
            training_args.wandb_mode,
        )
        wandb_init(wandb_name, wandb_project, wandb_dir, wandb_mode)
        wandb.run.log_code()

    # log args
    console_rule("Arguments")
    console_print(training_args)
    console_print(model_args)
    console_print(collator_args)
    if accelerator.is_main_process:
        wandb_update_config(
            {
                "training": training_args,
                "model": model_args,
            }
        )
    validate_args(training_args, model_args, collator_args)

    # Distribution Information
    console_rule("Distribution Information")
    console_print(f"[green]accelerator[/green]: {accelerator}")
    console_print(f"[green]n_procs[/green]: {accelerator.num_processes}")
    console_print(f"[green]process_index[/green]: {accelerator.process_index}")

    # model
    model = MaskedProsodyModel(model_args)
    console_rule("Model")
    print_and_draw_model()

    # dataset
    console_rule("Dataset")

    console_print(f"[green]dataset[/green]: {training_args.dataset}")
    console_print(f"[green]train_split[/green]: {training_args.train_split}")
    console_print(f"[green]val_split[/green]: {training_args.val_split}")

    train_ds = load_dataset(training_args.dataset, split=training_args.train_split)
    val_ds = load_dataset(training_args.dataset, split=training_args.val_split)

    console_print(f"[green]train[/green]: {len(train_ds)}")
    console_print(f"[green]val[/green]: {len(val_ds)}")

    # collator
    collator = get_collator(collator_args)

    # plot first batch
    if accelerator.is_main_process:
        first_batch = collator([train_ds[i] for i in range(training_args.batch_size)])
        print(first_batch)
        plot_first_batch(first_batch, training_args)
        raise
        plt.savefig("figures/first_batch.png")

    # dataloader
    train_dl = DataLoader(
        train_ds,
        batch_size=training_args.batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=training_args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.lr)

    # scheduler
    if training_args.lr_schedule == "linear_with_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.lr_warmup_steps,
            num_training_steps=training_args.n_steps,
        )
    else:
        raise NotImplementedError(f"{training_args.lr_schedule} not implemented")

    # accelerator
    model, optimizer, train_dl, val_dl, scheduler = accelerator.prepare(
        model, optimizer, train_dl, val_dl, scheduler
    )

    # evaluation
    if training_args.eval_only:
        console_rule("Evaluation")
        seed_everything(training_args.seed)
        evaluate()
        return

    # training
    console_rule("Training")
    seed_everything(training_args.seed)
    pbar_total = training_args.n_steps
    training_args.n_epochs = training_args.n_steps // len(train_dl) + 1
    console_print(f"[green]n_epochs[/green]: {training_args.n_epochs}")
    console_print(
        f"[green]effective_batch_size[/green]: {training_args.batch_size*accelerator.num_processes}"
    )
    pbar = tqdm(total=pbar_total, desc="step")
    for i in range(training_args.n_epochs):
        train_epoch(i)
    console_rule("Evaluation")
    seed_everything(training_args.seed)
    evaluate()

    # save final model
    console_rule("Saving")
    if training_args.do_save:
        save_checkpoint()

    # wandb sync reminder
    if accelerator.is_main_process and training_args.wandb_mode == "offline":
        console_rule("Weights & Biases")
        console_print(
            f"use \n[magenta]wandb sync {Path(wandb.run.dir).parent}[/magenta]\nto sync offline run"
        )


if __name__ == "__main__":
    main()
