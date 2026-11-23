import os

import numpy as np
from PIL import Image
from comfy_api.latest import io


def to_pil(image):
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"expected one image per item, got shape {tuple(image.shape)}")
    pixels = image[0].clamp(0, 1).cpu().numpy()
    pixels = (pixels * 255.0).round().astype(np.uint8)
    if pixels.shape[-1] == 4:
        pixels = pixels[..., :3]
    return Image.fromarray(pixels)


def clean_stem(name):
    stem = os.path.splitext(os.path.basename(name.replace("\\", "/")))[0]
    stem = "".join(c for c in stem if c.isalnum() or c in " -_.")
    return stem.strip(" .")


class SaveImagesWithCaptions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MyPack_SaveImagesWithCaptions",
            display_name="Save Images With Captions",
            search_aliases=["save dataset", "export dataset", "save training set", "dump captions"],
            category="my_pack/preprocess",
            description="Write images as PNG with a .txt caption next to each one, ready for a kohya-style trainer.",
            is_input_list=True,
            inputs=[
                io.String.Input("folder", default="", placeholder="D:/AnimaTrain/firefly_train_1024"),
                io.Image.Input("images"),
                io.String.Input("captions", optional=True, tooltip="Caption per image; empty ones are skipped."),
                io.String.Input("filenames", optional=True,
                                tooltip="Names to reuse, e.g. the filenames output of Load Images From Folder."),
            ],
            outputs=[
                io.String.Output(display_name="paths", is_output_list=True),
                io.Int.Output(display_name="count"),
            ],
        )

    @classmethod
    def execute(cls, folder, images, captions=None, filenames=None) -> io.NodeOutput:
        # is_input_list wraps every input in a list, including widget values.
        folder = folder[0].strip()
        if not folder:
            raise ValueError("folder is empty: set the destination folder for the exported dataset.")
        folder = os.path.abspath(os.path.expanduser(folder))
        os.makedirs(folder, exist_ok=True)

        filenames = filenames or []
        captions = captions or []

        paths = []
        used = set()
        for index, image in enumerate(images):
            stem = clean_stem(filenames[index]) if index < len(filenames) else ""
            stem = stem or f"{index + 1:05d}"

            name = stem
            suffix = 2
            while name.lower() in used:
                name = f"{stem}_{suffix}"
                suffix += 1
            used.add(name.lower())

            image_path = os.path.join(folder, name + ".png")
            to_pil(image).save(image_path)

            caption = captions[index].strip() if index < len(captions) and captions[index] else ""
            if caption:
                with open(os.path.join(folder, name + ".txt"), "w", encoding="utf-8") as f:
                    f.write(caption)

            paths.append(image_path)

        return io.NodeOutput(paths, len(paths))
