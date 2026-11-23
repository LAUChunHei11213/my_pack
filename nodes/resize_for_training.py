import math

import torch
import torch.nn.functional as F
from comfy_api.latest import io


def plan_bucket(width, height, base_resolution, step, max_side, allow_upscale):
    """Target size for one image: keep the aspect ratio, hold the area near base_resolution²,
    then snap both sides to a multiple of step so the trainer can bucket it."""
    scale = math.sqrt((base_resolution * base_resolution) / (width * height))
    if not allow_upscale:
        scale = min(scale, 1.0)

    width *= scale
    height *= scale
    if max(width, height) > max_side:
        shrink = max_side / max(width, height)
        width *= shrink
        height *= shrink

    if allow_upscale:
        return max(step, round(width / step) * step), max(step, round(height / step) * step)
    # Rounding up would grow the source again, so floor to stay at or below its own size.
    return max(step, int(width // step) * step), max(step, int(height // step) * step)


def resample(image, width, height):
    x = image.movedim(-1, 1)
    if width * height < x.shape[2] * x.shape[3]:
        x = F.interpolate(x, size=(height, width), mode="area")
    else:
        x = F.interpolate(x, size=(height, width), mode="bicubic", align_corners=False)
    return x.movedim(1, -1)


def fit_image(image, width, height, fit):
    if fit == "center_crop":
        src_h, src_w = image.shape[1], image.shape[2]
        scale = max(width / src_w, height / src_h)
        scaled_w, scaled_h = round(src_w * scale), round(src_h * scale)
        image = resample(image, max(width, scaled_w), max(height, scaled_h))
        top = (image.shape[1] - height) // 2
        left = (image.shape[2] - width) // 2
        return image[:, top:top + height, left:left + width]
    return resample(image, width, height)


class ResizeImagesForTraining(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MyPack_ResizeImagesForTraining",
            display_name="Resize Images For Training",
            search_aliases=["bucket", "training resize", "dataset resize", "preprocess"],
            category="my_pack/preprocess",
            description="Scale every image to a training bucket around base_resolution, keeping the aspect ratio.",
            is_input_list=True,
            inputs=[
                io.Image.Input("images", tooltip="Images to resize, usually straight from Load Images From Folder."),
                io.Int.Input("base_resolution", default=1024, min=64, max=4096, step=64,
                             tooltip="Bucket area is base_resolution², e.g. 1024 gives ~1024x1024 worth of pixels."),
                io.Int.Input("bucket_step", default=64, min=8, max=256, step=8,
                             tooltip="Bucket granularity. Anima / kohya use 64."),
                io.Int.Input("max_side", default=2048, min=256, max=8192, step=64,
                             tooltip="Hard cap for the long side, matching the trainer's max bucket resolution."),
                io.Combo.Input("fit", options=["snap", "center_crop"], default="snap",
                               tooltip="snap squeezes the image onto the bucket (distortion below one step); "
                                       "center_crop resizes to cover the bucket and crops the overflow."),
                io.Boolean.Input("allow_upscale", default=True,
                                 tooltip="Let images smaller than base_resolution grow to their bucket."),
            ],
            outputs=[
                io.Image.Output(display_name="images", is_output_list=True),
                io.String.Output(display_name="sizes", is_output_list=True, tooltip="Bucket size per image, WxH."),
            ],
        )

    @classmethod
    def execute(cls, images, base_resolution, bucket_step, max_side, fit, allow_upscale) -> io.NodeOutput:
        # is_input_list wraps every input in a list, including widget values.
        base_resolution = base_resolution[0]
        bucket_step = bucket_step[0]
        max_side = max_side[0]
        fit = fit[0]
        allow_upscale = allow_upscale[0]

        resized = []
        sizes = []
        for image in images:
            items = image.split(1, dim=0) if image.shape[0] > 1 else (image,)
            for item in items:
                height, width = item.shape[1], item.shape[2]
                bucket_w, bucket_h = plan_bucket(width, height, base_resolution, bucket_step, max_side, allow_upscale)
                resized.append(fit_image(item, bucket_w, bucket_h, fit))
                sizes.append(f"{bucket_w}x{bucket_h}")
        return io.NodeOutput(resized, sizes)
