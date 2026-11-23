from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .nodes.load_images import LoadImagesFromFolder


class MyPackExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [LoadImagesFromFolder]


async def comfy_entrypoint() -> MyPackExtension:
    return MyPackExtension()
