# 🚀 ComfyUI-LaoLi-lineup (老李显存排队)

**[中文说明](#中文说明) | [English](#english)**

---

<a name="中文说明"></a>
## 中文说明

**ComfyUI 的显存交通指挥。用于显存有限时使用大体积模型卡顿/爆显存，特别是 Flux/Qwen 等大模型挂载 ControlNet 时的爆显存、假死问题。**

### 🧐 解决了什么痛点？
即使你拥有 16GB 甚至 24GB 显存的显卡，在以下极端场景中也经常会遇到 `CUDA out of memory` 或 `invalid argument` 报错，甚至导致电脑卡死：
1.  **大模型 + ControlNet：** 比如运行 Flux.1 Dev 或 QwenImage 时，一开 ControlNet 就长时间卡顿/炸显存。
2.  **多重叠加：** 同时加载多个 ControlNet、IPAdapter 或大量 LoRA，导致显存碎片化严重。
3.  **超高清生图：** 进行高分辨率放大或大 Batch Size 生成时，显存无法及时释放。
4.  **显存不足 (8G/12G)：** 小显存显卡强行运行 SDXL/Flux 量化模型时不稳定。

ComfyUI 自带的 `--lowvram` 参数有时会失效或导致其他节点报错，本插件提供了更强力的解决方案。

### 💡 核心原理：老李排队策略 (Lineup)
本插件实施了**“一步一清”**的严格策略。它会自动识别模型中的每一个计算层（Block），并在每一层计算开始前，强制执行 **CUDA 流同步** 和 **显存软清理**。

*   **🛡️ 极致防崩：** 就像一个有洁癖的管理员，每算一层就打扫一次显存，确保 ControlNet 等“大块头”随时有连续的显存空间可用。
*   **🧠 零配置全自动：** 不需要手动设置层数，代码自动适配 Flux1/2, SDXL, Qwen, Wan2.2, SD1.5 等各种架构。
*   **🤝 原生兼容：** 不暴力搬运内存（这容易导致报错），而是利用 ComfyUI 原生机制进行强制管理，稳定第一。

### ⚠️ 兼容性重要提示 (必读)
本插件仅支持 **ComfyUI 标准模型类型 (`MODEL`)**。
*   **✅ 支持：** `Load Checkpoint`, `UNET Loader`, `Load Diffusion Model` 等原生节点加载的模型（包括 Flux, SDXL, Pony, QwenImage 等）。
*   **❌ 不支持：** 使用非标准封装类型的插件，例如 **`ComfyUI-WanVideoWrapper`** (其使用的是 `WANVIDEOMODEL` 类型)。这些插件通常自带了显存管理机制，无法与本插件串联。

### 💻 硬件要求
由于本插件会将大量显存数据临时暂存到系统内存（RAM）中，因此对**内存容量**有一定要求：
*   **✅ 推荐：64GB 或以上** 系统内存（最佳稳定性，从容应对数据交换）。
*   **⚠️ 最低：32GB** 系统内存（可以尝试，但在处理极复杂工作流时可能会出现内存不足）。

### 📦 安装方法

1.  进入你的 ComfyUI 插件目录：
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  克隆本项目：
    ```bash
    git clone https://github.com/Laolilzp/ComfyUI-LaoLi-lineup.git
    ```
3.  重启 ComfyUI。
<img width="785" height="274" alt="image" src="https://github.com/user-attachments/assets/eaeed326-e390-4a2c-b743-bb7f590e2abc" />

### 🔧 使用说明

1.  **添加节点：** 在菜单中找到 `LaoLi Nodes` -> `Optimization` -> `老李_LaoLi🚀 Lineup (显存排队)`。
2.  **连接节点：** 必须串联在 **大模型加载** 之后，**任何使用模型的操作** 之前。
    *   `model` 输入：连大模型 (Load Checkpoint / UNET Loader)。
    *   `model` 输出：连到 KSampler、ControlNet、LoRA 或 IPAdapter。
3.  **完成！** 连上即生效，无需任何参数配置。

### ⚠️ 性能提示
**稳定压倒一切。**
由于插件强制 GPU 在每一步计算前进行同步和清理，阻断了部分并行加速，生成速度可能会下降 **5% - 15%**。
*   **推荐场景：** 当你的工作流因为爆显存而**完全跑不通**，或者频繁报错时，请务必试用本插件。
*   **不推荐场景：** 如果你的显存非常充足，且运行非常稳定，则不需要使用本插件。不建议GGUF类型的模型使用该节点。
   
---

<a name="english"></a>
## English

**The "VRAM Traffic Controller" for ComfyUI. Solves OOM crashes and system freezes when running massive models (Flux/Qwen/Wan2.2) with ControlNet on limited VRAM.**

### 🧐 The Problem
Even with a high-end GPU (e.g., 16GB or 24GB VRAM), you might encounter **OOM (Out of Memory)** errors, `CUDA error: invalid argument`, or severe system lag in the following scenarios:
1.  **Large Model + ControlNet:** The base model (e.g., Flux.1 Dev, QwenImage) fits, but enabling ControlNet causes immediate stuttering or crashes.
2.  **Multiple Stack:** Using multiple ControlNets, IPAdapters, or heavy LoRAs simultaneously, causing severe memory fragmentation.
3.  **High Resolution:** High-res upscaling or large batch sizes where VRAM isn't released fast enough.
4.  **Low VRAM (8G/12G):** Trying to run quantized SDXL/Flux models on cards with limited memory.

The native `--lowvram` flag can sometimes fail or conflict with other nodes. This plugin provides a more robust solution.

### 💡 The Solution: LaoLi Lineup Strategy
This node implements a strict **"Step-by-Step Cleaning"** strategy. It automatically detects every computation block in your model and forces a **Stream Synchronization + Soft Cache Cleanup** *before* every single calculation step.

*   **🛡️ Ultimate Crash Protection:** Acts like a strict memory janitor. It ensures the GPU is clean before the next layer loads, ensuring large models like ControlNet always have continuous space available.
*   **🧠 Zero Config:** Automatically detects model architectures, including **Flux1/2, SDXL, Qwen, Wan2.2, SD1.5**, etc. No manual setup required.
*   **🤝 Native Compatibility:** Does not move memory tensors manually (which causes bugs). Instead, it leverages ComfyUI's native memory manager to enforce cleanliness.

### ⚠️ Compatibility Note (Important)
This node only supports the **Standard ComfyUI Model Type (`MODEL`)**.
*   **✅ Supported:** Models loaded via `Load Checkpoint`, `UNET Loader`, etc. (Flux, SDXL, Pony, QwenImage).
*   **❌ Not Supported:** Custom wrappers like **`ComfyUI-WanVideoWrapper`** (which uses the `WANVIDEOMODEL` type). These suites typically have their own memory management and cannot be connected to this node.

### 💻 Hardware Requirements
Since this strategy offloads VRAM data to your System RAM during processing, adequate memory is crucial:
*   **✅ Recommended: 64GB+** System RAM (Best performance and stability).
*   **⚠️ Minimum: 32GB** System RAM (You can try, but it might struggle with extremely large/complex workflows).

### 📦 Installation

1.  Navigate to your ComfyUI custom nodes directory:
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  Clone this repository:
    ```bash
    git clone https://github.com/Laolilzp/ComfyUI-LaoLi-lineup.git
    ```
3.  Restart ComfyUI.
<img width="785" height="274" alt="image" src="https://github.com/user-attachments/assets/64274757-95e4-4b32-a946-838fef99dbcf" />

### 🔧 Usage

1.  **Add Node:** Right-click -> `LaoLi Nodes` -> `Optimization` -> `老李_LaoLi🚀 Lineup (显存排队)`.
2.  **Connect:** Place it strictly **after** loading the model and **before** any sampling/lora/controlnet operations.
    *   `Model Input` -> Connect to `Load Checkpoint` / `UNET Loader`.
    *   `Model Output` -> Connect to `KSampler`, `Load LoRA`, `IPAdapter` or `Apply ControlNet`.
3.  **Done!** It runs automatically without any parameters.

### ⚠️ Performance Note
**Stability > Speed.**
Since this node forces the GPU to synchronize and clean memory at every single step, preventing some parallel acceleration, generation speed may decrease by **5% - 15%**.
*   **Recommended:** Try to use this when your workflow **cannot run** due to OOM or crashes.
*   **Not Recommended:** If your VRAM is sufficient and workflows run stably, you do not need this node.

---

### 📄 License

MIT License



