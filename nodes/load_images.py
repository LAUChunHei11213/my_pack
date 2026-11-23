import os

import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths
import node_helpers
from comfy_api.latest import io

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def resolve_folder(folder: str) -> str:
    folder = folder.strip()
    if not folder:
        raise ValueError("folder is empty: set a path, or a folder name relative to the ComfyUI input directory.")

    if not os.path.isabs(folder):
        folder = os.path.join(folder_paths.get_input_directory(), folder)

    path = os.path.abspath(os.path.expanduser(folder))
    if not os.path.isdir(path):
        raise ValueError(f"folder does not exist: {path}")
    return path


def list_images(root: str, recursive: bool) -> list[str]:
    if not recursive:
        names = sorted(os.listdir(root))
        return [
            os.path.join(root, name)
            for name in names
            if name.lower().endswith(IMAGE_EXTENSIONS) and os.path.isfile(os.path.join(root, name))
        ]

    paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        paths.extend(
            os.path.join(dirpath, name)
            for name in sorted(filenames)
            if name.lower().endswith(IMAGE_EXTENSIONS)
        )
    return paths


def read_caption(image_path: str) -> str:
    caption_path = os.path.splitext(image_path)[0] + ".txt"
    if not os.path.isfile(caption_path):
        return ""
    with open(caption_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_image(image_path: str, max_side: int = 0) -> torch.Tensor:
    image = node_helpers.pillow(Image.open, image_path)
    image = ImageOps.exif_transpose(image)
    if max_side and max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.LANCZOS)
    if image.mode == "I":
        image = image.point(lambda i: i * (1 / 255))
    pixels = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(pixels)[None]


class LoadImagesFromFolder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MyPack_LoadImagesFromFolder",
            display_name="Load Images From Folder",
            search_aliases=["dataset", "batch load", "lora dataset", "training images"],
            category="my_pack/io",
            description="Load every image in a folder as a list, with the .txt caption stored next to each image.",
            inputs=[
                io.String.Input(
                    "folder",
                    default="",
                    placeholder="D:/datasets/my_lora or a folder under ComfyUI/input",
                ),
                io.Boolean.Input("recursive", default=False, tooltip="Also load images from subfolders."),
                io.Int.Input("max_images", default=0, min=0, step=1, tooltip="0 loads every image."),
                io.Int.Input("max_side", default=0, min=0, max=8192, step=64,
                             tooltip="Downscale on load so a huge source never sits in memory at full size. 0 keeps the original."),
            ],
            outputs=[
                io.Image.Output(display_name="images", is_output_list=True),
                io.String.Output(display_name="captions", is_output_list=True),
                io.String.Output(display_name="filenames", is_output_list=True),
                io.Int.Output(display_name="count"),
            ],
        )

    @classmethod
    def execute(cls, folder, recursive, max_images, max_side=0) -> io.NodeOutput:
        root = resolve_folder(folder)
        paths = list_images(root, recursive)
        if max_images > 0:
            paths = paths[:max_images]
        if not paths:
            raise ValueError(f"no images ({', '.join(IMAGE_EXTENSIONS)}) found in: {root}")

        images = [load_image(path, max_side) for path in paths]
        captions = [read_caption(path) for path in paths]
        filenames = [os.path.relpath(path, root).replace("\\", "/") for path in paths]
        return io.NodeOutput(images, captions, filenames, len(images))
