#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility for building and visualising an NxN confusion-matrix during YOLOX
validation *without* wandb.  After every validation epoch you can call
`ConfusionMeter.save_png()` to dump a coloured matrix image (PNG) to
`/home/oeoiewt/eTraM/rvt_eTram/`.

Example
-------
from utils.evaluation.confusion import ConfusionMeter

cm = ConfusionMeter(num_classes=8, iou_thres=0.5)
...
cm.update(pred_boxes, pred_cls, gt_boxes, gt_cls)   # in validation_step
...
cm.save_png("/home/oeoiewt/eTraM/rvt_eTram/confusion_matrix.png",
            class_names=["cls0","cls1", ...])
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt

# reuse IoU helper from yolox
from models.detection.yolox.utils.boxes import bboxes_iou


class ConfusionMeter:
    """Accumulate an NxN confusion matrix for object detection."""

    def __init__(
        self,
        num_classes: int,
        iou_thres: float = 0.1,
        conf_thres: float = 0.01,
    ) -> None:
        """
        Parameters
        ----------
        num_classes : int
            Number of object classes (N).
        iou_thres : float
            GT-prediction IoU ≥ `iou_thres` is considered a match.
        conf_thres : float
            Predictions with (obj_conf x cls_conf) below this are ignored.
        """
        self.N = num_classes
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self._mat = np.zeros((self.N, self.N), dtype=np.int64)

    # --------------------------------------------------------------------- #
    #                    Accumulation (called every batch)                  #
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def update(
        self,
        pred_boxes: List[torch.Tensor],
        pred_labels: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        pred_scores: Optional[List[torch.Tensor]] = None,
    ) -> None:
        """
        Add a batch of predictions and GT targets to the confusion matrix.

        Arguments
        ---------
        pred_boxes   : list[P_i,4] - xyxy format (same units as GT)
        pred_labels  : list[P_i]   - class indices 0…N-1
        gt_boxes     : list[G_i,4] - xyxy format
        gt_labels    : list[G_i]   - class indices
        pred_scores  : list[P_i]   - total confidence (optional).  Rows whose
                                     score < `conf_thres` are filtered out.
        """
        batch_size = len(gt_boxes)
        for b in range(batch_size):
            boxes_p = pred_boxes[b]
            cls_p = pred_labels[b].long()
            if pred_scores is not None:
                keep = pred_scores[b] >= self.conf_thres
                boxes_p = boxes_p[keep]
                cls_p = cls_p[keep]

            boxes_g = gt_boxes[b]
            cls_g = gt_labels[b].long()

            # skip trivial cases
            if boxes_p.numel() == 0 or boxes_g.numel() == 0:
                continue

            # IoU matrix shape (G,P)
            ious = bboxes_iou(boxes_g, boxes_p, xyxy=True)
            best_iou, best_p = ious.max(dim=1)

            taken_pred = torch.zeros(boxes_p.size(0), dtype=torch.bool)

            for gi, (iou, pi) in enumerate(zip(best_iou, best_p)):
                if iou >= self.iou_thres and not taken_pred[pi]:
                    taken_pred[pi] = True
                    gt_c = int(cls_g[gi])
                    pred_c = int(cls_p[pi])
                    self._mat[gt_c, pred_c] += 1
                # unmatched GTs → FN (ignored for confusion‑matrix visualisation)

    # --------------------------------------------------------------------- #
    #                              Utilities                                #
    # --------------------------------------------------------------------- #
    def matrix(self, reset: bool = False) -> np.ndarray:
        """Return current matrix.  `reset=True` clears after reading."""
        mat = self._mat.copy()
        if reset:
            self._mat[...] = 0
        return mat

    def reset(self) -> None:
        """Clear stored counts."""
        self._mat[...] = 0

    # --------------------------------------------------------------------- #
    #                           Visualisation                               #
    # --------------------------------------------------------------------- #
    def save_png(
        self,
        save_path: str | Path,
        class_names: Optional[List[str]] = None,
        figsize: tuple[int, int] = (8, 6),
        cmap: str = "Blues",
        normalize: bool = True,
        reset: bool = True,
    ) -> None:
        """
        Plot and save the confusion matrix as a PNG image.

        Parameters
        ----------
        save_path : str or pathlib.Path
            Destination file (e.g. /home/.../confusion_matrix.png)
        class_names : list[str], optional
            Labels for axes.  If None, will use '0'…'N-1'.
        figsize : (w,h)
            Matplotlib figure size in inches.
        cmap : str
            Colour-map for imshow.
        normalize : bool
            If True, each row is scaled to % so the diagonal shows recall.
        reset : bool
            Reset internal counts after saving.
        """
        mat = self.matrix(reset=reset)
        if normalize:
            with np.errstate(all="ignore"):
                mat = mat / mat.sum(axis=1, keepdims=True)
                mat[np.isnan(mat)] = 0

        cls = class_names or [str(i) for i in range(self.N)]
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(mat, interpolation="nearest", cmap=cmap)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Tick labels
        ax.set(
            xticks=np.arange(self.N),
            yticks=np.arange(self.N),
            xticklabels=cls,
            yticklabels=cls,
            xlabel="Predicted",
            ylabel="Ground Truth",
            title="Confusion Matrix",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Annotate cells
        fmt = ".2f" if normalize else "d"
        thresh = mat.max() / 2.0
        for i in range(self.N):
            for j in range(self.N):
                val = mat[i, j]
                ax.text(
                    j,
                    i,
                    format(val, fmt),
                    ha="center",
                    va="center",
                    color="white" if val > thresh else "black",
                    fontsize=9,
                )

        fig.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
