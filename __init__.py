from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .nodes.anima_trainer import AnimaLoraTrainer
from .nodes.load_images import LoadImagesFromFolder
from .nodes.resize_for_training import ResizeImagesForTraining
from .nodes.save_dataset import SaveImagesWithCaptions


class MyPackExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [LoadImagesFromFolder, ResizeImagesForTraining, SaveImagesWithCaptions, AnimaLoraTrainer]


async def comfy_entrypoint() -> MyPackExtension:
    return MyPackExtension()
