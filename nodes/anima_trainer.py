import os
import subprocess
import sys
import time

import folder_paths
from comfy_api.latest import io

TRAINER_SCRIPT = "anima_train_network.py"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUSTOM_NODES_DIR = os.path.dirname(PACK_DIR)


def find_trainer_dir(hint: str) -> str:
    if hint.strip():
        path = os.path.abspath(os.path.expanduser(hint.strip()))
        if os.path.isfile(os.path.join(path, TRAINER_SCRIPT)):
            return path
        raise ValueError(f"{TRAINER_SCRIPT} not found in: {path}")

    for name in sorted(os.listdir(CUSTOM_NODES_DIR)):
        candidate = os.path.join(CUSTOM_NODES_DIR, name, "backend", "sd-scripts")
        if os.path.isfile(os.path.join(candidate, TRAINER_SCRIPT)):
            return candidate
    raise ValueError("sd-scripts backend not found: point sd_scripts_dir at the folder containing " + TRAINER_SCRIPT)


def list_subsets(dataset_dir: str):
    root = os.path.abspath(os.path.expanduser(dataset_dir.strip()))
    if not root:
        raise ValueError("dataset_dir is empty")
    if not os.path.isdir(root):
        raise ValueError(f"dataset folder does not exist: {root}")

    def has_images(path):
        return any(n.lower().endswith(IMAGE_EXTENSIONS) for n in os.listdir(path))

    subsets = []
    if has_images(root):
        subsets.append((root, 1))
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isdir(path) and has_images(path):
            head = name.split("_")[0]
            repeats = int(head) if head.isdigit() and int(head) > 0 else 1
            subsets.append((path, repeats))

    if not subsets:
        raise ValueError(f"no images found under: {root}")
    return subsets


def model_path(folder: str, name: str) -> str:
    path = folder_paths.get_full_path(folder, name)
    if path is None:
        raise ValueError(f"{folder} model not found: {name}")
    return os.path.abspath(path).replace("\\", "/")


def write_dataset_config(path, subsets, resolution, bucket_step, max_bucket_reso):
    lines = [
        "[general]",
        f"resolution = {resolution}",
        "enable_bucket = true",
        f"bucket_reso_steps = {bucket_step}",
        "min_bucket_reso = 256",
        f"max_bucket_reso = {max_bucket_reso}",
        "",
    ]
    for image_dir, repeats in subsets:
        lines += [
            "[[datasets]]",
            "batch_size = 1",
            "",
            "  [[datasets.subsets]]",
            '  image_dir = "{}"'.format(image_dir.replace("\\", "/")),
            f"  num_repeats = {repeats}",
            '  caption_extension = ".txt"',
            "",
        ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_train_config(path, values, output_name):
    lines = [
        'pretrained_model_name_or_path = "{}"'.format(values["dit"]),
        'vae = "{}"'.format(values["vae"]),
        'qwen3 = "{}"'.format(values["text_encoder"]),
        "",
        'output_dir = "{}"'.format(values["output_dir"].replace("\\", "/")),
        'output_name = "{}"'.format(output_name),
        'logging_dir = "{}"'.format((values["output_dir"] + "/logs").replace("\\", "/")),
        "",
        'network_module = "networks.lora_anima"',
        f'network_dim = {values["network_dim"]}',
        f'network_alpha = {values["network_alpha"]}',
        "network_train_unet_only = true",
        "",
        'optimizer_type = "{}"'.format(values["optimizer"]),
        f'unet_lr = {values["unet_lr"]}',
        'lr_scheduler = "{}"'.format(values["lr_scheduler"]),
        f'lr_warmup_steps = {values["lr_warmup_steps"]}',
        "",
        "train_batch_size = 1",
        "max_data_loader_n_workers = 0",
        f'max_train_epochs = {values["max_train_epochs"]}',
        'mixed_precision = "{}"'.format(values["mixed_precision"]),
        "gradient_checkpointing = {}".format("true" if values["gradient_checkpointing"] else "false"),
        "cache_latents = true",
        "cache_text_encoder_outputs = true",
        f'save_every_n_epochs = {values["save_every_n_epochs"]}',
        'save_precision = "fp16"',
        f'seed = {values["seed"]}',
        'timestep_sampling = "shift"',
        "discrete_flow_shift = 3.0",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def launch(trainer_dir, config_path, dataset_config_path, log_path):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP

    log = open(log_path, "w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [sys.executable, TRAINER_SCRIPT, "--config_file", config_path, "--dataset_config", dataset_config_path],
            cwd=trainer_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=creationflags,
        )
    finally:
        log.close()
    return process.pid


class AnimaLoraTrainer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MyPack_AnimaLoraTrainer",
            display_name="Anima LoRA Trainer",
            search_aliases=["train lora", "anima training", "lora trainer", "start training"],
            category="my_pack/training",
            description="Write an sd-scripts config for an Anima LoRA and optionally start training in the background.",
            not_idempotent=True,
            inputs=[
                io.String.Input("dataset_dir", default="", placeholder="D:/AnimaTrain/prepared"),
                io.Combo.Input("dit", options=folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("vae", options=folder_paths.get_filename_list("vae")),
                io.Combo.Input("text_encoder", options=folder_paths.get_filename_list("text_encoders")),
                io.String.Input("output_dir", default="", placeholder="D:/AnimaTrain/output"),
                io.Int.Input("network_dim", default=16, min=4, max=128, step=4),
                io.Int.Input("network_alpha", default=16, min=1, max=128, step=1),
                io.Float.Input("unet_lr", default=1e-4, min=1e-7, max=1e-2, step=1e-5),
                io.Int.Input("max_train_epochs", default=40, min=1, max=1000, step=1),
                io.Int.Input("resolution", default=1024, min=256, max=2048, step=64),
                io.Int.Input("bucket_step", default=64, min=8, max=256, step=8),
                io.Int.Input("max_bucket_reso", default=2048, min=256, max=4096, step=64),
                io.Combo.Input("optimizer", options=["AdamW8bit", "AdamW", "Lion", "Prodigy"], default="AdamW8bit"),
                io.Combo.Input("lr_scheduler", options=["cosine", "constant", "linear"], default="cosine"),
                io.Int.Input("lr_warmup_steps", default=50, min=0, max=10000, step=10),
                io.Combo.Input("mixed_precision", options=["bf16", "fp16"], default="bf16"),
                io.Boolean.Input("gradient_checkpointing", default=False,
                                 tooltip="Trade speed for VRAM. Off is faster when the GPU has room."),
                io.Int.Input("save_every_n_epochs", default=5, min=1, max=100, step=1),
                io.Int.Input("seed", default=42, min=0, max=2**31 - 1, step=1),
                io.String.Input("sd_scripts_dir", default="",
                                tooltip="Folder with anima_train_network.py. Empty: auto-detect under custom_nodes."),
                io.Boolean.Input("start", default=False,
                                 tooltip="Off only writes the configs. On starts training in the background."),
            ],
            outputs=[
                io.String.Output(display_name="config_path"),
                io.String.Output(display_name="log_path"),
                io.String.Output(display_name="summary"),
            ],
        )

    @classmethod
    def execute(cls, dataset_dir, dit, vae, text_encoder, output_dir, network_dim, network_alpha, unet_lr,
                max_train_epochs, resolution, bucket_step, max_bucket_reso, optimizer, lr_scheduler,
                lr_warmup_steps, mixed_precision, gradient_checkpointing, save_every_n_epochs, seed,
                sd_scripts_dir, start) -> io.NodeOutput:
        subsets = list_subsets(dataset_dir)
        image_count = sum(repeats * len([n for n in os.listdir(path) if n.lower().endswith(IMAGE_EXTENSIONS)])
                          for path, repeats in subsets)

        output_dir = output_dir.strip()
        if not output_dir:
            raise ValueError("output_dir is empty")
        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        config_dir = os.path.join(output_dir, "configs")
        os.makedirs(config_dir, exist_ok=True)

        values = {
            "dit": model_path("diffusion_models", dit),
            "vae": model_path("vae", vae),
            "text_encoder": model_path("text_encoders", text_encoder),
            "output_dir": output_dir,
            "network_dim": network_dim,
            "network_alpha": network_alpha,
            "unet_lr": unet_lr,
            "max_train_epochs": max_train_epochs,
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "lr_warmup_steps": lr_warmup_steps,
            "mixed_precision": mixed_precision,
            "gradient_checkpointing": gradient_checkpointing,
            "save_every_n_epochs": save_every_n_epochs,
            "seed": seed,
        }

        output_name = "lora_" + time.strftime("%Y%m%d_%H%M%S")
        config_path = os.path.join(config_dir, "train.toml")
        dataset_config_path = os.path.join(config_dir, "dataset.toml")
        write_train_config(config_path, values, output_name)
        write_dataset_config(dataset_config_path, subsets, resolution, bucket_step, max_bucket_reso)

        lines = [
            f"dataset: {image_count} images in {len(subsets)} subset(s)",
            f"lora: dim {network_dim} / alpha {network_alpha} -> "
            + os.path.join(output_dir, output_name + ".safetensors").replace("\\", "/"),
            f"steps: {image_count * max_train_epochs} ({max_train_epochs} epochs)",
            f"config: {config_path}",
        ]

        log_path = ""
        if start:
            trainer_dir = find_trainer_dir(sd_scripts_dir)
            log_path = os.path.join(output_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")
            pid = launch(trainer_dir, config_path, dataset_config_path, log_path)
            lines.append(f"started: pid {pid}, log {log_path}")
        else:
            lines.append("dry run: configs written, training not started")

        return io.NodeOutput(config_path, log_path, "\n".join(lines))
