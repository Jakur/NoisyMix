import torch
import torch.nn as nn
import torch.nn.functional as F

def freeze(model):
    for name, param in model.named_parameters():
        param.requires_grad_(False)

def unfreeze(model):
    for name, param in model.named_parameters():
        param.requires_grad_(True)

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss
    
def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student+1e-5)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, reduction='sum')
        * (temperature**2)
        / target.shape[0]
    )
    # print(tckd_loss)
    # if len(target.size()) > 1:
    #     label = torch.max(target, dim=1, keepdim=True)[1]
    # else:
    #     label = target.view(len(target), 1)

    # # N*class
    # N, c = logits_student.shape
    
    # mask = torch.ones_like(logits_student).scatter_(1, label, 0).bool()
    # logits_student = logits_student[mask].reshape(N, -1)
    # logits_teacher = logits_teacher[mask].reshape(N, -1)
    
    # pred_teacher_part2 = F.softmax(
    #     logits_teacher / temperature, dim=1
    # )
    
    # log_pred_student_part2 = F.log_softmax(
    #     logits_student / temperature, dim=1
    # )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    # print(pred_teacher_part2)
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    # print(log_pred_student_part2)
    # print('\n\n')
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='sum')
        * (temperature**2)
        / target.shape[0]
    )
    # print(nckd_loss)
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt