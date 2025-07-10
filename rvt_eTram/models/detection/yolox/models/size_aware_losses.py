#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Size-aware Loss Functions for Small Object Detection Enhancement

import torch
import torch.nn as nn
import torch.nn.functional as F


class SizeAwareIOULoss(nn.Module):
    """
    Size-aware IoU Loss that gives higher weights to smaller objects.
    This is designed to improve small object detection performance.
    """
    
    def __init__(self, reduction="none", loss_type="iou", 
                 size_aware_weight=2.0, small_threshold=32*32, 
                 medium_threshold=96*96, weight_type="exponential"):
        """
        Args:
            reduction: Loss reduction method ("none", "mean", "sum")
            loss_type: Type of IoU loss ("iou", "giou")
            size_aware_weight: Maximum weight multiplier for smallest objects
            small_threshold: Area threshold for small objects (default: 32x32 = 1024)
            medium_threshold: Area threshold for medium objects (default: 96x96 = 9216)
            weight_type: Type of weighting function ("exponential", "step", "linear")
        """
        super(SizeAwareIOULoss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type
        self.size_aware_weight = size_aware_weight
        self.small_threshold = small_threshold
        self.medium_threshold = medium_threshold
        self.weight_type = weight_type
        
    def compute_size_weights(self, target_boxes):
        """
        Compute size-aware weights based on target bounding box areas.
        
        Args:
            target_boxes: [N, 4] tensor in format [cx, cy, w, h]
            
        Returns:
            weights: [N] tensor with size-aware weights
        """
        # Calculate areas (w * h)
        areas = target_boxes[:, 2] * target_boxes[:, 3]
        
        if self.weight_type == "exponential":
            # Exponential decay: smaller objects get exponentially higher weights
            # weight = size_aware_weight * exp(-area / small_threshold)
            weights = self.size_aware_weight * torch.exp(-areas / self.small_threshold)
            # Clamp to reasonable range [1.0, size_aware_weight]
            weights = torch.clamp(weights, min=1.0, max=self.size_aware_weight)
            
        elif self.weight_type == "step":
            # Step function: discrete weights based on size categories
            weights = torch.ones_like(areas)
            small_mask = areas <= self.small_threshold
            medium_mask = (areas > self.small_threshold) & (areas <= self.medium_threshold)
            
            weights[small_mask] = self.size_aware_weight  # Highest weight for small objects
            weights[medium_mask] = (self.size_aware_weight + 1.0) / 2.0  # Medium weight for medium objects
            # Large objects keep weight = 1.0
            
        elif self.weight_type == "linear":
            # Linear interpolation: smooth transition between weights
            # Normalize areas to [0, 1] range
            normalized_areas = torch.clamp(areas / self.medium_threshold, min=0.0, max=1.0)
            # Linear interpolation: weight = size_aware_weight * (1 - normalized_area)
            weights = self.size_aware_weight * (1.0 - normalized_areas) + 1.0
            weights = torch.clamp(weights, min=1.0, max=self.size_aware_weight)
            
        else:
            raise ValueError(f"Unknown weight_type: {self.weight_type}")
            
        return weights

    def forward(self, pred, target):
        """
        Forward pass with size-aware weighting.
        
        Args:
            pred: [N, 4] predicted boxes in format [cx, cy, w, h]
            target: [N, 4] target boxes in format [cx, cy, w, h]
            
        Returns:
            loss: Size-aware IoU loss
        """
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        
        # Calculate size-aware weights based on target boxes
        size_weights = self.compute_size_weights(target)
        
        # Convert to corner coordinates for IoU calculation
        pred_tl = pred[:, :2] - pred[:, 2:] / 2  # top-left
        pred_br = pred[:, :2] + pred[:, 2:] / 2  # bottom-right
        target_tl = target[:, :2] - target[:, 2:] / 2
        target_br = target[:, :2] + target[:, 2:] / 2
        
        # Intersection coordinates
        tl = torch.max(pred_tl, target_tl)
        br = torch.min(pred_br, target_br)

        # Calculate areas
        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        # Calculate intersection
        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            # Convex hull coordinates
            c_tl = torch.min(pred_tl, target_tl)
            c_br = torch.max(pred_br, target_br)
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        else:
            raise NotImplementedError

        # Apply size-aware weights
        weighted_loss = loss * size_weights

        if self.reduction == "mean":
            weighted_loss = weighted_loss.mean()
        elif self.reduction == "sum":
            weighted_loss = weighted_loss.sum()

        return weighted_loss


class SizeAwareFocalLoss(nn.Module):
    """
    Size-aware Focal Loss for classification that emphasizes hard examples
    and gives higher weights to smaller objects.
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, size_aware_weight=2.0, 
                 small_threshold=32*32, weight_type="exponential"):
        """
        Args:
            alpha: Weighting factor for rare class (typically 0.25)
            gamma: Focusing parameter (typically 2.0)
            size_aware_weight: Maximum weight multiplier for smallest objects
            small_threshold: Area threshold for small objects
            weight_type: Type of weighting function ("exponential", "step", "linear")
        """
        super(SizeAwareFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_aware_weight = size_aware_weight
        self.small_threshold = small_threshold
        self.weight_type = weight_type
        
    def compute_size_weights(self, target_boxes):
        """Compute size-aware weights (same as SizeAwareIOULoss)"""
        areas = target_boxes[:, 2] * target_boxes[:, 3]
        
        if self.weight_type == "exponential":
            weights = self.size_aware_weight * torch.exp(-areas / self.small_threshold)
            weights = torch.clamp(weights, min=1.0, max=self.size_aware_weight)
        elif self.weight_type == "step":
            weights = torch.ones_like(areas)
            small_mask = areas <= self.small_threshold
            weights[small_mask] = self.size_aware_weight
        elif self.weight_type == "linear":
            normalized_areas = torch.clamp(areas / (self.small_threshold * 3), min=0.0, max=1.0)
            weights = self.size_aware_weight * (1.0 - normalized_areas) + 1.0
        else:
            raise ValueError(f"Unknown weight_type: {self.weight_type}")
            
        return weights

    def forward(self, pred, target, target_boxes):
        """
        Forward pass with size-aware focal loss.
        
        Args:
            pred: [N, num_classes] predicted logits
            target: [N] target class indices
            target_boxes: [N, 4] target boxes for size weight calculation
            
        Returns:
            loss: Size-aware focal loss
        """
        # Convert to probabilities
        pred_prob = torch.sigmoid(pred)
        
        # Calculate size-aware weights
        size_weights = self.compute_size_weights(target_boxes)
        
        # One-hot encode targets
        target_one_hot = F.one_hot(target, num_classes=pred.size(1)).float()
        
        # Focal loss calculation
        pt = pred_prob * target_one_hot + (1 - pred_prob) * (1 - target_one_hot)
        alpha_t = self.alpha * target_one_hot + (1 - self.alpha) * (1 - target_one_hot)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        
        # Binary cross entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target_one_hot, reduction='none')
        
        # Apply focal weighting
        focal_loss = focal_weight * bce_loss
        
        # Apply size-aware weighting
        size_weighted_loss = focal_loss * size_weights.unsqueeze(1)
        
        return size_weighted_loss.mean()


class SizeAwareLossWrapper(nn.Module):
    """
    Wrapper that combines size-aware IoU loss, focal loss, and objectness loss.
    This is designed to be a drop-in replacement for standard YOLOX losses.
    """
    
    def __init__(self, use_size_aware=True, size_aware_weight=2.0, 
                 small_threshold=32*32, medium_threshold=96*96, 
                 weight_type="exponential", loss_type="iou"):
        """
        Args:
            use_size_aware: Whether to use size-aware weighting
            size_aware_weight: Maximum weight multiplier for smallest objects
            small_threshold: Area threshold for small objects
            medium_threshold: Area threshold for medium objects
            weight_type: Type of weighting function
            loss_type: Type of IoU loss ("iou", "giou")
        """
        super(SizeAwareLossWrapper, self).__init__()
        self.use_size_aware = use_size_aware
        
        if use_size_aware:
            self.iou_loss = SizeAwareIOULoss(
                reduction="none", 
                loss_type=loss_type,
                size_aware_weight=size_aware_weight,
                small_threshold=small_threshold,
                medium_threshold=medium_threshold,
                weight_type=weight_type
            )
            self.focal_loss = SizeAwareFocalLoss(
                alpha=0.25,
                gamma=2.0,
                size_aware_weight=size_aware_weight,
                small_threshold=small_threshold,
                weight_type=weight_type
            )
        else:
            # Use standard losses for comparison
            from .losses import IOUloss
            self.iou_loss = IOUloss(reduction="none", loss_type=loss_type)
            self.focal_loss = nn.CrossEntropyLoss(reduction='none')
            
    def forward(self, pred_reg, pred_cls, pred_obj, target_reg, target_cls, target_boxes):
        """
        Forward pass combining all losses with size-aware weighting.
        
        Args:
            pred_reg: [N, 4] predicted regression
            pred_cls: [N, num_classes] predicted classification
            pred_obj: [N, 1] predicted objectness
            target_reg: [N, 4] target regression
            target_cls: [N] target classification
            target_boxes: [N, 4] target boxes for size calculation
            
        Returns:
            dict: Dictionary containing individual and total losses
        """
        # IoU/Regression loss
        iou_loss = self.iou_loss(pred_reg, target_reg)
        
        # Classification loss
        if self.use_size_aware:
            cls_loss = self.focal_loss(pred_cls, target_cls, target_boxes)
        else:
            cls_loss = F.cross_entropy(pred_cls, target_cls)
            
        # Objectness loss (standard binary cross entropy)
        obj_target = torch.ones_like(pred_obj)  # All positive samples
        obj_loss = F.binary_cross_entropy_with_logits(pred_obj, obj_target)
        
        # Combine losses
        total_loss = iou_loss.mean() + cls_loss + obj_loss
        
        return {
            'iou_loss': iou_loss.mean(),
            'cls_loss': cls_loss,
            'obj_loss': obj_loss,
            'total_loss': total_loss
        }