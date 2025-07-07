#!/usr/bin/env python3
"""
Detailed metrics calculation for class-wise mAP, AP50, AP75, AP95
Extended from Prophesee evaluation to provide per-class metrics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pycocotools.coco import COCO
try:
    from detectron2.evaluation.fast_eval_api import COCOeval_opt as COCOeval
except ImportError:
    from pycocotools.cocoeval import COCOeval

from .prophesee.metrics.coco_eval import _to_coco_format, _match_times


class DetailedMetricsCalculator:
    """Calculate detailed per-class metrics for object detection."""
    
    def __init__(self):
        # eTraM 8-class system
        self.class_names = [
            "Car",           # 0 - Large object
            "Truck",         # 1 - Large object  
            "Motorcycle",    # 2 - Small object ⭐
            "Bicycle",       # 3 - Small object ⭐
            "Pedestrian",    # 4 - Very small object ⭐⭐
            "Bus",           # 5 - Large object
            "Static",        # 6 - Static object
            "Other"          # 7 - Other
        ]
        
        self.small_object_classes = [2, 3, 4]  # Motorcycle, Bicycle, Pedestrian
        
    def evaluate_detailed_detection(self, 
                                  gt_boxes_list: List[np.ndarray], 
                                  dt_boxes_list: List[np.ndarray],
                                  height: int = 384, 
                                  width: int = 640,
                                  time_tol: int = 50000) -> Dict:
        """
        Compute detailed detection metrics including per-class AP at different IoU thresholds.
        
        Args:
            gt_boxes_list: List of ground truth boxes (numpy structured arrays)
            dt_boxes_list: List of detection boxes (numpy structured arrays)
            height: Image height
            width: Image width
            time_tol: Time tolerance for matching detections to ground truth
            
        Returns:
            Dictionary containing overall and per-class metrics
        """
        
        # Match ground truth and detections by timestamp
        flattened_gt = []
        flattened_dt = []
        
        for gt_boxes, dt_boxes in zip(gt_boxes_list, dt_boxes_list):
            if len(gt_boxes) == 0:
                continue
                
            # Ensure temporal ordering
            if len(gt_boxes) > 1:
                assert np.all(gt_boxes['t'][1:] >= gt_boxes['t'][:-1])
            if len(dt_boxes) > 1:
                assert np.all(dt_boxes['t'][1:] >= dt_boxes['t'][:-1])

            all_ts = np.unique(gt_boxes['t'])
            gt_win, dt_win = _match_times(all_ts, gt_boxes, dt_boxes, time_tol)
            flattened_gt.extend(gt_win)
            flattened_dt.extend(dt_win)
        
        if len(flattened_gt) == 0:
            return self._empty_results()
            
        return self._compute_coco_metrics(flattened_gt, flattened_dt, height, width)
    
    def _compute_coco_metrics(self, 
                            flattened_gt: List[np.ndarray], 
                            flattened_dt: List[np.ndarray],
                            height: int, 
                            width: int) -> Dict:
        """Compute COCO-style metrics with per-class breakdown."""
        
        # Create COCO categories for all 8 classes
        categories = []
        for i, name in enumerate(self.class_names):
            categories.append({
                'id': i + 1,
                'name': name,
                'supercategory': 'vehicle' if i != 4 else 'person'
            })
        
        # Convert to COCO format
        dataset, results = _to_coco_format(flattened_gt, flattened_dt, categories, height, width)
        
        if len(results) == 0:
            return self._empty_results()
        
        # Create COCO objects
        coco = COCO()
        coco.dataset = dataset
        coco.createIndex()
        
        coco_dets = coco.loadRes(results)
        coco_eval = COCOeval(coco, coco_dets, 'bbox')
        
        # Evaluate
        coco_eval.evaluate()
        coco_eval.accumulate()
        
        # Extract detailed metrics
        return self._extract_detailed_metrics(coco_eval)
    
    def _extract_detailed_metrics(self, coco_eval: COCOeval) -> Dict:
        """Extract detailed metrics from COCO evaluation."""
        
        # Overall metrics
        overall_metrics = self._compute_overall_metrics(coco_eval)
        
        # Per-class metrics
        class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            class_id = i + 1  # COCO uses 1-based indexing
            
            # Filter evaluation for this class only
            class_ap_metrics = self._compute_class_metrics(coco_eval, class_id)
            class_metrics[f"{i}_{class_name}"] = class_ap_metrics
        
        # Small object analysis
        small_object_analysis = self._analyze_small_objects(class_metrics)
        
        return {
            "overall_metrics": overall_metrics,
            "class_metrics": class_metrics,
            "small_object_analysis": small_object_analysis,
            "evaluation_summary": self._create_summary(overall_metrics, class_metrics)
        }
    
    def _compute_overall_metrics(self, coco_eval: COCOeval) -> Dict:
        """Compute overall mAP metrics."""
        
        # COCO evaluation provides metrics at different IoU thresholds
        # coco_eval.stats contains: [mAP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]
        stats = coco_eval.stats
        
        return {
            "mAP": float(stats[0]) if not np.isnan(stats[0]) else 0.0,      # mAP @ IoU=0.50:0.95
            "AP50": float(stats[1]) if not np.isnan(stats[1]) else 0.0,     # mAP @ IoU=0.50
            "AP75": float(stats[2]) if not np.isnan(stats[2]) else 0.0,     # mAP @ IoU=0.75
            "AP95": self._compute_ap95(coco_eval),                          # AP @ IoU=0.95
            "APs": float(stats[3]) if not np.isnan(stats[3]) else 0.0,      # AP for small objects
            "APm": float(stats[4]) if not np.isnan(stats[4]) else 0.0,      # AP for medium objects  
            "APl": float(stats[5]) if not np.isnan(stats[5]) else 0.0,      # AP for large objects
        }
    
    def _compute_class_metrics(self, coco_eval: COCOeval, class_id: int) -> Dict:
        """Compute metrics for a specific class."""
        
        # Get precision array: [IoU, Recall, Category, Area, MaxDets]
        precision = coco_eval.eval['precision']
        
        if precision.size == 0:
            return {"mAP": 0.0, "AP50": 0.0, "AP75": 0.0, "AP95": 0.0, "count": 0}
        
        # Filter for this class (category dimension is 2, 0-based indexing)
        class_idx = class_id - 1  # Convert to 0-based for indexing
        if class_idx >= precision.shape[2]:
            return {"mAP": 0.0, "AP50": 0.0, "AP75": 0.0, "AP95": 0.0, "count": 0}
        
        class_precision = precision[:, :, class_idx, 0, 2]  # All IoU, all recall, this class, area=all, maxDets=100
        
        # Compute AP at different IoU thresholds
        ap_values = {}
        
        # mAP (average over IoU 0.5:0.95, step 0.05)
        valid_ious = class_precision[class_precision >= 0]  # Remove -1 values
        ap_values["mAP"] = float(np.mean(valid_ious)) if len(valid_ious) > 0 else 0.0
        
        # AP50 (IoU = 0.5, which is index 0)
        ap50_precision = class_precision[0, :]  # IoU=0.5
        ap_values["AP50"] = float(np.mean(ap50_precision[ap50_precision >= 0])) if np.any(ap50_precision >= 0) else 0.0
        
        # AP75 (IoU = 0.75, which is index 5)  
        if class_precision.shape[0] > 5:
            ap75_precision = class_precision[5, :]  # IoU=0.75
            ap_values["AP75"] = float(np.mean(ap75_precision[ap75_precision >= 0])) if np.any(ap75_precision >= 0) else 0.0
        else:
            ap_values["AP75"] = 0.0
        
        # AP95 (IoU = 0.95, which is index 9)
        if class_precision.shape[0] > 9:
            ap95_precision = class_precision[9, :]  # IoU=0.95  
            ap_values["AP95"] = float(np.mean(ap95_precision[ap95_precision >= 0])) if np.any(ap95_precision >= 0) else 0.0
        else:
            ap_values["AP95"] = 0.0
        
        # Count detections for this class
        class_count = self._count_class_detections(coco_eval, class_id)
        ap_values["count"] = class_count
        
        return ap_values
    
    def _compute_ap95(self, coco_eval: COCOeval) -> float:
        """Compute AP at IoU=0.95 specifically."""
        precision = coco_eval.eval['precision']
        
        if precision.size == 0:
            return 0.0
        
        # IoU=0.95 is at index 9 (0.5 + 9*0.05 = 0.95)
        if precision.shape[0] > 9:
            ap95_precision = precision[9, :, :, 0, 2]  # IoU=0.95, all recall, all classes
            valid_values = ap95_precision[ap95_precision >= 0]
            return float(np.mean(valid_values)) if len(valid_values) > 0 else 0.0
        return 0.0
    
    def _count_class_detections(self, coco_eval: COCOeval, class_id: int) -> int:
        """Count number of detections for a specific class."""
        try:
            # Access ground truth annotations
            gt_anns = [ann for ann in coco_eval.cocoGt.anns.values() if ann['category_id'] == class_id]
            return len(gt_anns)
        except:
            return 0
    
    def _analyze_small_objects(self, class_metrics: Dict) -> Dict:
        """Analyze performance on small object classes."""
        
        small_class_names = ["2_Motorcycle", "3_Bicycle", "4_Pedestrian"]
        small_object_maps = []
        
        for class_name in small_class_names:
            if class_name in class_metrics:
                small_object_maps.append(class_metrics[class_name]["mAP"])
        
        avg_small_map = np.mean(small_object_maps) if small_object_maps else 0.0
        
        return {
            "small_classes": small_class_names,
            "individual_maps": {name: class_metrics.get(name, {}).get("mAP", 0.0) for name in small_class_names},
            "avg_small_mAP": float(avg_small_map),
            "small_object_count": sum(class_metrics.get(name, {}).get("count", 0) for name in small_class_names)
        }
    
    def _create_summary(self, overall_metrics: Dict, class_metrics: Dict) -> Dict:
        """Create a summary of the evaluation."""
        
        # Count classes with non-zero performance
        active_classes = sum(1 for metrics in class_metrics.values() if metrics.get("mAP", 0) > 0)
        
        # Find best and worst performing classes
        class_performances = [(name, metrics.get("mAP", 0)) for name, metrics in class_metrics.items()]
        class_performances.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "total_classes": len(self.class_names),
            "active_classes": active_classes,
            "best_class": class_performances[0] if class_performances else ("None", 0.0),
            "worst_class": class_performances[-1] if class_performances else ("None", 0.0),
            "overall_mAP": overall_metrics.get("mAP", 0.0)
        }
    
    def _empty_results(self) -> Dict:
        """Return empty results structure when no data is available."""
        
        empty_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            empty_class_metrics[f"{i}_{class_name}"] = {
                "mAP": 0.0, "AP50": 0.0, "AP75": 0.0, "AP95": 0.0, "count": 0
            }
        
        return {
            "overall_metrics": {"mAP": 0.0, "AP50": 0.0, "AP75": 0.0, "AP95": 0.0, "APs": 0.0, "APm": 0.0, "APl": 0.0},
            "class_metrics": empty_class_metrics,
            "small_object_analysis": {
                "small_classes": ["2_Motorcycle", "3_Bicycle", "4_Pedestrian"],
                "individual_maps": {"2_Motorcycle": 0.0, "3_Bicycle": 0.0, "4_Pedestrian": 0.0},
                "avg_small_mAP": 0.0,
                "small_object_count": 0
            },
            "evaluation_summary": {
                "total_classes": 8,
                "active_classes": 0,
                "best_class": ("None", 0.0),
                "worst_class": ("None", 0.0),
                "overall_mAP": 0.0
            }
        }