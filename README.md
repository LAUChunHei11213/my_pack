# my_pack

ComfyUI custom nodes for building a LoRA training data pipeline in the graph itself.
中文说明在下方。

## Nodes

| Node | What it does |
| --- | --- |
| **Load Images From Folder** (`my_pack/io`) | Reads every image in a folder, pairs each one with its `.txt` caption, returns lists. |

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/LAUChunHei11213/my_pack.git
```

Restart ComfyUI. The node shows up under **my_pack / io** in the Add Node menu, or search "Load Images From Folder".

No extra dependencies are needed: the pack uses the torch, numpy and Pillow that ComfyUI already ships.

## Load Images From Folder

Inputs:

- `folder` — absolute path (`D:/datasets/my_lora`), or a path relative to `ComfyUI/input`.
- `recursive` — also walk subfolders.
- `max_images` — `0` loads everything; otherwise take the first N images in name order.

Outputs:

- `images` — list of images at their native resolution, RGB, `[1, H, W, 3]` in `0..1`.
- `captions` — the content of `image_name.txt` next to each image, empty string when there is no caption file.
- `filenames` — path relative to `folder`, so subfolders stay visible.
- `count` — number of loaded images.

Images are **not** resized or cropped here: mixed resolutions go through untouched and bucketing/resizing stays a
separate step, which keeps the dataset intact and the node cheap. Files are read in sorted order, so a re-run of the
same folder produces the same list.

## Roadmap

Rough order for the LoRA training pipeline this pack is growing into:

1. Load Images From Folder ← done
2. Caption loading / editing
3. Resolution bucketing and resizing for training
4. Trigger word / caption templating
5. Export the prepared dataset for a training backend

## 中文

ComfyUI 自定义节点包，目标是把 LoRA 训练的数据准备流程直接搭在工作流里。

**安装**：把仓库克隆到 `ComfyUI/custom_nodes/`，重启 ComfyUI，节点出现在 **my_pack / io** 分类下。

**Load Images From Folder**：

- `folder`：绝对路径（如 `D:/datasets/my_lora`），或相对 `ComfyUI/input` 的路径
- `recursive`：是否递归子目录
- `max_images`：`0` 表示全部，否则按文件名顺序取前 N 张
- 输出：`images`（原始分辨率的图片列表）、`captions`（同名 `.txt` 的内容，没有则为空）、`filenames`（相对 folder 的路径）、`count`（数量）

节点不做缩放和裁剪，图片保持原尺寸往下传，尺寸统一交给后续的 resize / bucket 步骤处理；文件按名字排序读取，同一文件夹每次运行结果一致。

**路线图**：加载文件夹（已完成）→ 标注读写与编辑 → 训练用分辨率分桶 → 触发词/标注模板 → 导出数据集。

## License

MIT
