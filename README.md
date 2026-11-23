# my_pack

ComfyUI custom nodes for building a LoRA training data pipeline in the graph itself.
中文说明在下方。

## Nodes

| Node | Category | What it does |
| --- | --- | --- |
| **Load Images From Folder** | `my_pack/io` | Reads every image in a folder, pairs each one with its `.txt` caption, returns lists. |
| **Resize Images For Training** | `my_pack/preprocess` | Scales each image to a training bucket around `base_resolution`, keeping the aspect ratio. |
| **Save Images With Captions** | `my_pack/preprocess` | Writes PNG + `.txt` per image into a folder, ready for a kohya-style trainer. |

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/LAUChunHei11213/my_pack.git
```

Restart ComfyUI. No extra dependencies: the pack uses the torch, numpy and Pillow that ComfyUI already ships.

## A working pipeline

```
Load Images From Folder  ->  Resize Images For Training  ->  Save Images With Captions
```

Settings that match Anima base v1.0 (1024² DiT trained with kohya-style buckets):

| Node | Setting | Value |
| --- | --- | --- |
| Load Images From Folder | `max_side` | `2048` (downscale while decoding) |
| Resize Images For Training | `base_resolution` / `bucket_step` / `max_side` | `1024` / `64` / `2048` |
| Resize Images For Training | `fit` | `snap` (or `center_crop` to cut wide shots) |
| Save Images With Captions | `folder` | a new folder, e.g. `D:/AnimaTrain/firefly_train_1024` |
| Anima trainer | `resolution` / `enable_bucket` | `1024,1024` / on, `min_bucket_reso=256`, `max_bucket_reso=2048`, `bucket_reso_steps=64` |

Then point `train_data_dir` at the folder written by the save node.

Why the downscale matters: a 10240×5760 source is 59 MP of pixels that a 1024 model never sees. Loading it at full
size costs ~700 MB of float32 for that single image, and the trainer would only throw the detail away. Downscaling
once, with `area`/LANCZOS quality, keeps the dataset at ~1 MP per image and makes the run cheap.

## Load Images From Folder

Inputs:

- `folder` — absolute path (`D:/datasets/my_lora`), or a path relative to `ComfyUI/input`.
- `recursive` — also walk subfolders.
- `max_images` — `0` loads everything; otherwise take the first N images in name order.
- `max_side` — downscale while decoding when a side is larger than this. `0` keeps the original size.

Outputs: `images` (native resolution, RGB, `[1, H, W, 3]` in `0..1`), `captions` (the `.txt` with the same stem,
empty when missing), `filenames` (path relative to `folder`), `count`.

Images are not resized or cropped unless `max_side` is set, and files are read in sorted order, so re-running the
same folder produces the same list.

## Resize Images For Training

`base_resolution` is an **area** target, not a side length: the image is scaled so that `width * height` lands near
`base_resolution²`, the aspect ratio is kept, and both sides are then rounded to a multiple of `bucket_step`. That
is exactly how kohya picks buckets, so the trainer will not resize your images again.

- `fit = snap` — resize straight onto the bucket. Two sides can shift by less than one step (~2–4% squeeze), nothing
  is thrown away.
- `fit = center_crop` — resize to cover the bucket and crop the overflow. Keeps pixel geometry exact, loses edges;
  useful for wide shots where the subject sits in the middle.
- `allow_upscale = false` — never enlarge sources that are already below `base_resolution`.
- `max_side` — hard cap on the long side, matching the trainer's `max_bucket_reso`.

Outputs: `images` and `sizes` (`WxH` per image) so the bucket assignment can be checked at a glance.

## Save Images With Captions

Inputs: `folder`, `images`, and the optional `captions` / `filenames` lists (feed the loader's outputs straight in).

Writes `<name>.png` plus `<name>.txt` when the caption is not empty, reuses the original file names, appends `_2`,
`_3` on collisions, and falls back to `00001` style names when no `filenames` are connected. Outputs `paths` and
`count`.

## 中文

ComfyUI 自定义节点包，把 LoRA 训练的数据准备流程直接搭在工作流里。

**链路**：`Load Images From Folder` → `Resize Images For Training` → `Save Images With Captions`，最后把
`train_data_dir` 指向保存节点输出的文件夹。

**Anima base v1.0 的推荐值**：`base_resolution=1024`、`bucket_step=64`、`max_side=2048`、`fit=snap`；
加载节点把 `max_side` 设成 2048，解码时就把大图降下来；训练器里 `resolution=1024,1024`、开 `enable_bucket`、
`min_bucket_reso=256`、`max_bucket_reso=2048`、`bucket_reso_steps=64`。

**Resize Images For Training** 的 `base_resolution` 是**面积**目标（`base²` 像素），不是边长：先按面积等比缩放，
再把宽高对齐到 `bucket_step` 的整数倍，和 kohya 的分桶算法一致，所以训练器不会再缩一次。`fit=snap` 允许
不到一个 step 的轻微拉伸、不丢内容；`fit=center_crop` 保持几何精确但会裁掉边缘，适合主体居中的宽图。

**为什么要先缩**：10240×5760 是 59MP，1024 的模型根本用不到这些像素，但按原尺寸加载单张就要 700MB 以上
float32 显存/内存。加载时降采样一次，全库压到每张约 1MP，训练才跑得轻。

## Roadmap

1. Load Images From Folder ← done
2. Resize / bucket ← done
3. Save with captions ← done
4. Caption editing and trigger-word templating
5. Export directly for the Anima trainer config

## License

MIT
