import torch
import comfy.model_management as mm
from comfy.model_patcher import ModelPatcher

class LaoLi_Lineup_Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("optimized_model",)
    FUNCTION = "apply_lineup"
    CATEGORY = "LaoLi Nodes/Optimization"
    DESCRIPTION = "老李显存排队：强制模型一步一清"

    def apply_lineup(self, model):
        # 仅处理有效模型
        if not isinstance(model, ModelPatcher):
            return (model,)

        try:
            new_model = model.clone()

            # 核心钩子：强制同步流并软清理显存
            def strict_clean_hook(module, input):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                mm.soft_empty_cache()
                return None

            # 挂载钩子到所有计算层
            blocks = self._find_blocks(new_model.model)
            for block in blocks:
                block.register_forward_pre_hook(strict_clean_hook)

            return (new_model,)
        except Exception as e:
            print(f"[LaoLi_Lineup Error] {e}")
            return (model,)

    def _find_blocks(self, module):
        blocks = []
        target_names = [
            'transformer_blocks', 'double_blocks', 'single_blocks', 
            'blocks', 'input_blocks', 'middle_block', 'output_blocks'
        ]
        
        # 优先查找底层扩散模型
        root = getattr(module, 'diffusion_model', module)

        # 浅层搜索
        for name in target_names:
            attr = getattr(root, name, None)
            if isinstance(attr, (list, torch.nn.ModuleList)):
                blocks.extend(attr)
        
        # 深层搜索 (兜底)
        if not blocks:
            for name, child in root.named_children():
                if any(t in name for t in target_names):
                    if isinstance(child, (list, torch.nn.ModuleList)):
                        blocks.extend(child)
        
        return blocks

NODE_CLASS_MAPPINGS = {
    "LaoLi_Lineup": LaoLi_Lineup_Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LaoLi_Lineup": "老李_LaoLi🚀 Lineup (显存排队)"
}