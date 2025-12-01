# ComfyUI-LaoLi-lineup
ComfyUI 的显存交通指挥。用于显存有限时使用大体积模型卡顿/爆显存，特别是 Flux/Qwen 等大模型挂载 ControlNet 时的爆显存、假死问题。本插件实施了“一步一清”的严格策略。它会自动识别模型中的每一个计算层（Block），并在每一层计算开始前，强制执行 CUDA 流同步 和 显存软清理。
