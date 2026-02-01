# 🚀 ComfyUI-LaoLi-lineup (显存排队独立版本) 

**[中文说明](#中文说明) | [English](#english)**

---

<a name="中文说明"></a>
## 中文说明

**ComfyUI 的显存交通指挥官。 智能版：平时隐身不降速，危急时刻显神威。**

专治显存有限（12G/16G/24G）时运行大体积模型（Flux/Qwen/Wan2.2）挂载 ControlNet 导致的**卡顿、爆显存 (OOM)、CUDA Error** 问题。

### 🧐 解决了什么痛点？
即使拥有 4070Ti Super (16G) 或 4090 (24G)，在以下场景中显存依然捉襟见肘：
1.  **大模型 + ControlNet：** Flux.1 Dev 或 QwenImage 本体已占满大部分显存，开启 ControlNet 瞬间显存溢出。
2.  **多重叠加：** 同时加载多个 ControlNet、IPAdapter 或大量 LoRA，导致显存碎片化。
3.  **高负荷任务：** 高分辨率放大、视频生成 (Wan2.2) 或大 Batch Size。
4.  **硬盘狂转/系统假死：** 显存溢出导致系统使用虚拟内存（硬盘），电脑卡死，生成极慢。

### 💡 核心原理：智能阈值 (Smart Threshold)
本插件引入了**智能监控**：
1.  **实时监控：** 在模型计算的每一层（Block）之前，毫秒级检测当前显存占用率。
2.  **按需介入：**
    *   **显存充足 (<85%)：** 插件**休眠**，不做任何操作，让模型全速运行（零性能损耗）。
    *   **显存告急 (>85%)：** 插件**唤醒**，强制执行同步与清理，防止 OOM。

### 💻 硬件要求
*   **显存 (VRAM):** 8GB - 24GB 均适用。
*   **内存 (RAM):** 推荐 **64GB**，最低 **32GB**。
    *   *原理：显存不够时，数据会暂存在内存中。如果内存也不够，会溢出到硬盘（虚拟内存），导致极度卡顿。*

<img width="633" height="218" alt="image" src="https://github.com/user-attachments/assets/9f0391c9-c0a6-4c20-b674-a5a4fff058dd" />


### 🎛️ 参数说明 (重要)

*   **`vram_threshold` (显存阈值):**
    *   **0.85 (默认/推荐)**: 当显存占用超过 **85%** 时触发清理。预留 15% 空间是为了应对 ControlNet 突然的显存尖峰。
    *   **1.0**: 相当于关闭智能监控，始终不清理（除非配合 cleaning_interval）。

*   **`cleaning_interval` (清理间隔):**
    *   **1 (默认/推荐)**: **最稳模式**。只要显存超过阈值，**每一层**都进行清理。
    *   **⚠️ 警告：** 在 16G 显存跑大模型时，**切勿将此数值设得过大（如 10）**。这会导致显卡试图硬吃 10 层数据，瞬间挤爆显存和内存，引发**硬盘 100% 占用**和系统假死。

*   **`strict_mode` (严格模式):**
    *   **True (默认/防崩)**: 执行 `同步 (Synchronize)` + `清理`。防止 `CUDA error: invalid argument`。
    *   **False (极速)**: 仅执行 `清理`。速度更快，但在某些极端环境下可能导致报错。建议先尝试 False，若报错则改回 True。

### ❓ 常见问题 (Troubleshooting)

**Q: 为什么我的 C 盘（硬盘）活动时间 100%，电脑非常卡？**
*   **A:** 你可能把 `cleaning_interval` 设得太大了（比如 10）。
*   **原因:** 数据积压太多，显存装不下 -> 溢出到内存 -> 内存装不下 -> 溢出到硬盘（虚拟内存交换）。硬盘读写速度远慢于内存，导致卡死。
*   **解法:** 把 `cleaning_interval` 改回 **1**。

**Q: 这个插件支持 GGUF 模型吗？**
*   **A:** **完美支持。** 特别是当你用小显存（12G）跑 GGUF+ControlNet 时，此插件效果显著。

**Q: 支持 WanVideoWrapper 吗？**
*   **A:** **不支持。** 仅支持 ComfyUI 标准模型接口（`MODEL` 类型），如 Load Checkpoint, UNET Loader, GGUF Loader 等。

### 📦 安装与使用
1.  `cd ComfyUI/custom_nodes/`
2.  `git clone https://github.com/Laolilzp/ComfyUI-LaoLi-lineup.git`
3.  重启 ComfyUI。
4.  **连接：** 模型/LoRA加载器 -> **老李_LaoLi🚀 Lineup** -> KSampler/ControlNet。

---

<a name="english"></a>
## English

**The "VRAM Traffic Controller" for ComfyUI.  Smart Edition: Invisible when safe, lifesaving when critical.**

Solves **OOM (Out of Memory), CUDA Errors, and System Freezes** when running massive models (Flux/Qwen/Wan2.2) with ControlNet on limited VRAM (12G/16G/24G).

### 🧐 The Problem
Even with a high-end GPU (e.g., 16GB or 24GB VRAM), you might encounter issues in the following scenarios:
1.  **Large Model + ControlNet:** The base model (Flux.1 Dev / QwenImage) fits, but enabling ControlNet immediately causes VRAM overflow.
2.  **Multiple Stack:** Using multiple ControlNets, IPAdapters, or heavy LoRAs simultaneously, causing severe memory fragmentation.
3.  **High Load:** High-res upscaling, video generation (Wan2.2), or large batch sizes.
4.  **Disk Thrashing/Freeze:** VRAM overflow forces the system to use virtual memory (Disk), causing the PC to freeze and generation to crawl.

### 💡  Core Logic: Smart Threshold
Instead of cleaning memory blindly at every step,  uses **Active Monitoring**:
1.  **Monitor:** Checks VRAM usage ratio before every model layer computation.
2.  **React:**
    *   **Safe Zone (<85%):** The node **sleeps**. Zero performance impact.
    *   **Danger Zone (>85%):** The node **wakes up**, forcing synchronization and cleanup to prevent crashes.

### 💻 Hardware Requirements
*   **VRAM:** 8GB - 24GB (Effective on all).
*   **RAM:** **64GB Recommended**, 32GB Minimum.
    *   *Reason: When VRAM is full, data swaps to System RAM. If RAM is also full, it spills to the Disk (PageFile), causing severe lag.*
 
<img width="633" height="218" alt="image" src="https://github.com/user-attachments/assets/dcf4d682-160e-4775-9369-cfcb78a9cfbf" />


### 🎛️ Parameters

*   **`vram_threshold`:**
    *   **0.85 (Default/Recommended)**: Triggers cleanup when VRAM usage exceeds **85%**. The 15% buffer allows room for sudden spikes from ControlNet.
    *   **1.0**: Disables smart monitoring (always cleans if used with interval).

*   **`cleaning_interval`:**
    *   **1 (Default/Recommended)**: **Most Stable**. Checks and cleans (if needed) at *every* layer.
    *   **⚠️ WARNING:** Do NOT set this value too high (e.g., 10) on 16GB cards with large models. This will cause massive spillover to your System RAM and Disk, resulting in **100% Disk Usage** and system freeze.

*   **`strict_mode`:**
    *   **True (Anti-Crash)**: Forces GPU Synchronization + Cleanup. Prevents `invalid argument` errors.
    *   **False (Speed)**: Only frees memory without stopping the GPU pipeline. Faster, but try `True` if you crash.

### ❓ Troubleshooting

**Q: Why is my SSD/Disk usage at 100% and the system lagging?**
*   **A:** You likely set `cleaning_interval` too high (e.g., > 5).
*   **Fix:** Set `cleaning_interval` back to **1**. This ensures data flows smoothly instead of flooding your RAM and Disk.

**Q: Does it work with GGUF?**
*   **A:** **Yes.** It is highly effective for low-VRAM cards running GGUF + ControlNet.

**Q: Compatibility?**
*   **A:** Supports standard ComfyUI `MODEL` types (Flux, SDXL, SD1.5, Qwen, Pony). Does **NOT** support custom wrappers like `WanVideoWrapper` (which use non-standard types).

### 📦 Installation
1.  `cd ComfyUI/custom_nodes/`
2.  `git clone https://github.com/Laolilzp/ComfyUI-LaoLi-lineup.git`
3.  Restart ComfyUI.
4.  **Connect:** Load Checkpoint/LoRA -> **LaoLi Lineup** -> KSampler/ControlNet.

---

### 📄 License
MIT License


