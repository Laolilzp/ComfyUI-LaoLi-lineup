import torch
import comfy.model_management as mm
from comfy.model_patcher import ModelPatcher

class LaoLi_Lineup_Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                # 1. 显存阈值: 超过此比例(0.85)触发清理
                "vram_threshold": ("FLOAT", {
                    "default": 0.85, 
                    "min": 0.1, 
                    "max": 1.0, 
                    "step": 0.05,
                    "display": "number"
                }),
                # 2. 清理间隔: 每 N 层清理一次
                "cleaning_interval": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 10, 
                    "step": 1,
                    "display": "number"
                }),
                # 3. 严格模式: True=同步+清理, False=仅清理
                "strict_mode": ("BOOLEAN", {
                    "default": True
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("optimized_model",)
    FUNCTION = "apply_lineup"
    CATEGORY = "LaoLi Nodes/Optimization"
    # 这里是鼠标悬停在节点标题上时会显示的说明
    DESCRIPTION = "老李显存排队 V9:\n- vram_threshold: 显存占用超过此比例(默认0.85)时触发清理。\n- cleaning_interval: 清理频率(默认1，即每层都判断)。\n- strict_mode: 开启防崩(同步+清理)，关闭提速(仅清理)。"

    def apply_lineup(self, model, vram_threshold, cleaning_interval, strict_mode):
        # 1. 安全检查
        if not isinstance(model, ModelPatcher):
            return (model,)

        try:
            new_model = model.clone()
            
            # 2. 获取当前使用的 GPU 设备
            device = mm.get_torch_device()
            
            # 只有在 GPU 模式下才启用显存监控
            total_vram = 0
            if device.type == 'cuda':
                # 获取当前设备的总显存
                total_vram = torch.cuda.get_device_properties(device).total_memory
            
            # --- 定义智能钩子 ---
            def smart_hook(module, input):
                # 如果不是 CUDA 设备，直接跳过
                if total_vram == 0: 
                    return None

                # A. 显存监控
                # memory_reserved 是 PyTorch 向系统申请的显存，memory_allocated 是实际占用的
                # 我们使用 reserved 来判断是否接近物理极限
                current_reserved = torch.cuda.memory_reserved(device)
                usage_ratio = current_reserved / total_vram

                # B. 阈值判断 (默认 > 85%)
                if usage_ratio >= vram_threshold:
                    if strict_mode:
                        torch.cuda.synchronize() # 强制 GPU 停机等待 (防崩关键)
                    mm.soft_empty_cache()       # 释放未锁定显存
                
                return None

            # 3. 挂载逻辑
            blocks = self._find_blocks(new_model.model)
            mounted_count = 0
            
            for i, block in enumerate(blocks):
                # 遵守间隔设定 (通常设为1，即每层都监控)
                if i % cleaning_interval == 0:
                    block.register_forward_pre_hook(smart_hook)
                    mounted_count += 1

            # 控制台输出确认信息
            print(f"🚀 [老李 Lineup V9] 启动 | 设备: {device} | 阈值: {int(vram_threshold*100)}% | 模式: {'严格(同步)' if strict_mode else '极速(异步)'}")
            
            return (new_model,)

        except Exception as e:
            print(f"❌ [LaoLi_Lineup Error] {e}")
            return (model,)

    def _find_blocks(self, module):
        """递归查找模型中的计算层"""
        blocks = []
        target_names = [
            'transformer_blocks', 'double_blocks', 'single_blocks', 
            'blocks', 'input_blocks', 'middle_block', 'output_blocks'
        ]
        
        # 优先查找底层 diffusion_model
        root = getattr(module, 'diffusion_model', module)

        # 浅层搜索
        for name in target_names:
            attr = getattr(root, name, None)
            if isinstance(attr, (list, torch.nn.ModuleList)):
                blocks.extend(attr)
        
        # 深层搜索 (防止漏网之鱼)
        if not blocks:
            for name, child in root.named_children():
                if any(t in name for t in target_names):
                    if isinstance(child, (list, torch.nn.ModuleList)):
                        blocks.extend(child)
        
        return blocks

# 节点注册
NODE_CLASS_MAPPINGS = {
    "LaoLi_Lineup": LaoLi_Lineup_Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LaoLi_Lineup": "老李_LaoLi🚀 Lineup (显存排队)"
}