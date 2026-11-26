import torch


def masked_action_l1_loss(prediction, target, action_loss_mask):
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError('动作预测和标签必须同为 [batch,chunk,7]')
    if prediction.shape[-1] != 7 or action_loss_mask.shape != prediction.shape[:2]:
        raise ValueError('动作块的每一步必须对应一个监督权重')
    weights=action_loss_mask.to(device=prediction.device,dtype=torch.float32)
    if not torch.isfinite(weights).all() or (weights<0).any() or (weights>1).any():
        raise ValueError('监督权重必须为有限的 [0,1] 数值')
    count=weights.sum()
    if count<=0:
        raise ValueError('当前批次没有用于监督的动作')
    # 在float32中归一化；被屏蔽动作保留历史输入，但回归梯度为零。
    error=(prediction.float()-target.float()).abs()
    return (error*weights.unsqueeze(-1)).sum()/(count*prediction.shape[-1])
