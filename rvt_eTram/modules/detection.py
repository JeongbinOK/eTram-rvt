from typing import Any, Optional, Tuple, Union, Dict
from warnings import warn

import numpy as np
import pytorch_lightning as pl
import torch
import torch as th
import torch.distributed as dist
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT

from data.genx_utils.labels import ObjectLabels
from data.utils.types import DataType, LstmStates, ObjDetOutput, DatasetSamplingMode
from models.detection.yolox.utils.boxes import postprocess
from models.detection.yolox_extension.models.detector import YoloXDetector
from utils.evaluation.prophesee.evaluator import PropheseeEvaluator
from utils.evaluation.prophesee.io.box_loading import to_prophesee
from utils.padding import InputPadderFromShape
from .utils.detection import (
    BackboneFeatureSelector,
    EventReprSelector,
    RNNStates,
    Mode,
    mode_2_string,
    merge_mixed_batches,
)

from utils.evaluation.confusion import ConfusionMeter
from utils.evaluation.detailed_metrics import DetailedMetricsCalculator
from utils.evaluation.experiment_logger import ExperimentLogger


class Module(pl.LightningModule):
    def __init__(self, full_config: DictConfig):
        super().__init__()

        self.full_config = full_config

        self.mdl_config = full_config.model
        in_res_hw = tuple(self.mdl_config.backbone.in_res_hw)
        self.input_padder = InputPadderFromShape(desired_hw=in_res_hw)

        self.mdl = YoloXDetector(self.mdl_config)

        self.mode_2_rnn_states: Dict[Mode, RNNStates] = {
            Mode.TRAIN: RNNStates(),
            Mode.VAL: RNNStates(),
            Mode.TEST: RNNStates(),
        }
        self.cm = ConfusionMeter(
            num_classes=self.mdl_config.head.num_classes, iou_thres=0.5
        )
        
        # Initialize detailed metrics calculator and experiment logger
        self.detailed_metrics = DetailedMetricsCalculator()
        self.experiment_logger = None  # Will be initialized when needed

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_name = self.full_config.dataset.name
        self.mode_2_hw: Dict[Mode, Optional[Tuple[int, int]]] = {}
        self.mode_2_batch_size: Dict[Mode, Optional[int]] = {}
        self.mode_2_psee_evaluator: Dict[Mode, Optional[PropheseeEvaluator]] = {}
        self.mode_2_sampling_mode: Dict[Mode, DatasetSamplingMode] = {}

        self.started_training = True

        dataset_train_sampling = self.full_config.dataset.train.sampling
        dataset_eval_sampling = self.full_config.dataset.eval.sampling
        assert dataset_train_sampling in iter(DatasetSamplingMode)
        assert dataset_eval_sampling in (
            DatasetSamplingMode.STREAM,
            DatasetSamplingMode.RANDOM,
        )
        if stage == "fit":  # train + val
            self.train_config = self.full_config.training
            self.train_metrics_config = self.full_config.logging.train.metrics

            if self.train_metrics_config.compute:
                self.mode_2_psee_evaluator[Mode.TRAIN] = PropheseeEvaluator(
                    dataset=dataset_name,
                    downsample_by_2=self.full_config.dataset.downsample_by_factor_2,
                )
            self.mode_2_psee_evaluator[Mode.VAL] = PropheseeEvaluator(
                dataset=dataset_name,
                downsample_by_2=self.full_config.dataset.downsample_by_factor_2,
            )
            self.mode_2_sampling_mode[Mode.TRAIN] = dataset_train_sampling
            self.mode_2_sampling_mode[Mode.VAL] = dataset_eval_sampling

            for mode in (Mode.TRAIN, Mode.VAL):
                self.mode_2_hw[mode] = None
                self.mode_2_batch_size[mode] = None
            self.started_training = False
        elif stage == "validate":
            mode = Mode.VAL
            self.mode_2_psee_evaluator[mode] = PropheseeEvaluator(
                dataset=dataset_name,
                downsample_by_2=self.full_config.dataset.downsample_by_factor_2,
            )
            self.mode_2_sampling_mode[Mode.VAL] = dataset_eval_sampling
            self.mode_2_hw[mode] = None
            self.mode_2_batch_size[mode] = None
        elif stage == "test":
            mode = Mode.TEST
            self.mode_2_psee_evaluator[mode] = PropheseeEvaluator(
                dataset=dataset_name,
                downsample_by_2=self.full_config.dataset.downsample_by_factor_2,
            )
            self.mode_2_sampling_mode[Mode.TEST] = dataset_eval_sampling
            self.mode_2_hw[mode] = None
            self.mode_2_batch_size[mode] = None
        else:
            raise NotImplementedError

    def forward(
        self,
        event_tensor: th.Tensor,
        previous_states: Optional[LstmStates] = None,
        retrieve_detections: bool = True,
        targets=None,
    ) -> Tuple[Union[th.Tensor, None], Union[Dict[str, th.Tensor], None], LstmStates]:
        return self.mdl(
            x=event_tensor,
            previous_states=previous_states,
            retrieve_detections=retrieve_detections,
            targets=targets,
        )

    def get_worker_id_from_batch(self, batch: Any) -> int:
        return batch["worker_id"]

    def get_data_from_batch(self, batch: Any):
        return batch["data"]

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        batch = merge_mixed_batches(batch)
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)

        mode = Mode.TRAIN
        self.started_training = True
        step = self.trainer.global_step
        ev_tensor_sequence = data[DataType.EV_REPR]
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]
        token_mask_sequence = data.get(DataType.TOKEN_MASK, None)

        self.mode_2_rnn_states[mode].reset(
            worker_id=worker_id, indices_or_bool_tensor=is_first_sample
        )

        sequence_len = len(ev_tensor_sequence)
        assert sequence_len > 0
        batch_size = len(sparse_obj_labels[0])
        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = batch_size
        else:
            assert self.mode_2_batch_size[mode] == batch_size

        prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
        backbone_feature_selector = BackboneFeatureSelector()
        ev_repr_selector = EventReprSelector()
        obj_labels = list()
        for tidx in range(sequence_len):
            ev_tensors = ev_tensor_sequence[tidx]
            ev_tensors = ev_tensors.to(dtype=self.dtype)
            ev_tensors = self.input_padder.pad_tensor_ev_repr(ev_tensors)
            if token_mask_sequence is not None:
                token_masks = self.input_padder.pad_token_mask(
                    token_mask=token_mask_sequence[tidx]
                )
            else:
                token_masks = None

            if self.mode_2_hw[mode] is None:
                self.mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])
            else:
                assert self.mode_2_hw[mode] == ev_tensors.shape[-2:]

            backbone_features, states = self.mdl.forward_backbone(
                x=ev_tensors, previous_states=prev_states, token_mask=token_masks
            )
            prev_states = states

            current_labels, valid_batch_indices = sparse_obj_labels[
                tidx
            ].get_valid_labels_and_batch_indices()
            # Store backbone features that correspond to the available labels.
            if len(current_labels) > 0:
                backbone_feature_selector.add_backbone_features(
                    backbone_features=backbone_features,
                    selected_indices=valid_batch_indices,
                )
                obj_labels.extend(current_labels)
                ev_repr_selector.add_event_representations(
                    event_representations=ev_tensors,
                    selected_indices=valid_batch_indices,
                )

        self.mode_2_rnn_states[mode].save_states_and_detach(
            worker_id=worker_id, states=prev_states
        )
        assert len(obj_labels) > 0
        # Batch the backbone features and labels to parallelize the detection code.
        selected_backbone_features = (
            backbone_feature_selector.get_batched_backbone_features()
        )
        labels_yolox = ObjectLabels.get_labels_as_batched_tensor(
            obj_label_list=obj_labels, format_="yolox"
        )
        labels_yolox = labels_yolox.to(dtype=self.dtype)

        predictions, losses = self.mdl.forward_detect(
            backbone_features=selected_backbone_features, targets=labels_yolox
        )

        if self.mode_2_sampling_mode[mode] in (
            DatasetSamplingMode.MIXED,
            DatasetSamplingMode.RANDOM,
        ):
            # We only want to evaluate the last batch_size samples if we use random sampling (or mixed).
            # This is because otherwise we would mostly evaluate the init phase of the sequence.
            predictions = predictions[-batch_size:]
            obj_labels = obj_labels[-batch_size:]

        pred_processed = postprocess(
            prediction=predictions,
            num_classes=self.mdl_config.head.num_classes,
            conf_thre=self.mdl_config.postprocess.confidence_threshold,
            nms_thre=self.mdl_config.postprocess.nms_threshold,
        )

        loaded_labels_proph, yolox_preds_proph = to_prophesee(
            obj_labels, pred_processed
        )

        assert losses is not None
        assert "loss" in losses

        # For visualization, we only use the last batch_size items.
        output = {
            ObjDetOutput.LABELS_PROPH: loaded_labels_proph[-batch_size:],
            ObjDetOutput.PRED_PROPH: yolox_preds_proph[-batch_size:],
            ObjDetOutput.EV_REPR: ev_repr_selector.get_event_representations_as_list(
                start_idx=-batch_size
            ),
            ObjDetOutput.SKIP_VIZ: False,
            "loss": losses["loss"],
        }

        # Logging
        prefix = f"{mode_2_string[mode]}/"
        log_dict = {f"{prefix}{k}": v for k, v in losses.items()}
        self.log_dict(
            log_dict, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True
        )

        if mode in self.mode_2_psee_evaluator:
            self.mode_2_psee_evaluator[mode].add_labels(loaded_labels_proph)
            self.mode_2_psee_evaluator[mode].add_predictions(yolox_preds_proph)
            if (
                self.train_metrics_config.detection_metrics_every_n_steps is not None
                and step > 0
                and step % self.train_metrics_config.detection_metrics_every_n_steps
                == 0
            ):
                self.run_psee_evaluator(mode=mode)

        return output

    def _val_test_step_impl(self, batch: Any, mode: Mode) -> Optional[STEP_OUTPUT]:
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)

        assert mode in (Mode.VAL, Mode.TEST)
        ev_tensor_sequence = data[DataType.EV_REPR]
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]

        self.mode_2_rnn_states[mode].reset(
            worker_id=worker_id, indices_or_bool_tensor=is_first_sample
        )

        sequence_len = len(ev_tensor_sequence)
        assert sequence_len > 0
        batch_size = len(sparse_obj_labels[0])
        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = batch_size
        else:
            assert self.mode_2_batch_size[mode] == batch_size

        prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
        backbone_feature_selector = BackboneFeatureSelector()
        ev_repr_selector = EventReprSelector()
        obj_labels = list()
        for tidx in range(sequence_len):
            collect_predictions = (tidx == sequence_len - 1) or (
                self.mode_2_sampling_mode[mode] == DatasetSamplingMode.STREAM
            )
            ev_tensors = ev_tensor_sequence[tidx]
            ev_tensors = ev_tensors.to(dtype=self.dtype)
            ev_tensors = self.input_padder.pad_tensor_ev_repr(ev_tensors)
            if self.mode_2_hw[mode] is None:
                self.mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])
            else:
                assert self.mode_2_hw[mode] == ev_tensors.shape[-2:]

            backbone_features, states = self.mdl.forward_backbone(
                x=ev_tensors, previous_states=prev_states
            )
            prev_states = states

            if collect_predictions:
                current_labels, valid_batch_indices = sparse_obj_labels[
                    tidx
                ].get_valid_labels_and_batch_indices()
                # Store backbone features that correspond to the available labels.
                if len(current_labels) > 0:
                    backbone_feature_selector.add_backbone_features(
                        backbone_features=backbone_features,
                        selected_indices=valid_batch_indices,
                    )

                    obj_labels.extend(current_labels)
                    ev_repr_selector.add_event_representations(
                        event_representations=ev_tensors,
                        selected_indices=valid_batch_indices,
                    )
        self.mode_2_rnn_states[mode].save_states_and_detach(
            worker_id=worker_id, states=prev_states
        )
        if len(obj_labels) == 0:
            return {ObjDetOutput.SKIP_VIZ: True}
        selected_backbone_features = (
            backbone_feature_selector.get_batched_backbone_features()
        )
        predictions, _ = self.mdl.forward_detect(
            backbone_features=selected_backbone_features
        )

        pred_processed = postprocess(
            prediction=predictions,
            num_classes=self.mdl_config.head.num_classes,
            conf_thre=self.mdl_config.postprocess.confidence_threshold,
            nms_thre=self.mdl_config.postprocess.nms_threshold,
        )

        loaded_labels_proph, yolox_preds_proph = to_prophesee(
            obj_labels, pred_processed
        )

        # For visualization, we only use the last item (per batch).
        output = {
            ObjDetOutput.LABELS_PROPH: loaded_labels_proph[-1],
            ObjDetOutput.PRED_PROPH: yolox_preds_proph[-1],
            ObjDetOutput.EV_REPR: ev_repr_selector.get_event_representations_as_list(
                start_idx=-1
            )[0],
            ObjDetOutput.SKIP_VIZ: False,
        }

        if self.started_training:
            self.mode_2_psee_evaluator[mode].add_labels(loaded_labels_proph)
            self.mode_2_psee_evaluator[mode].add_predictions(yolox_preds_proph)

        if mode == Mode.VAL:
            # YOLOX postprocess 결과에서 boxes/labels 준비
            pred_boxes_list, pred_labels_list = [], []
            for det in pred_processed:  # det == None ↔ 예측 없음
                if det is None:
                    pred_boxes_list.append(torch.empty((0, 4), device=self.device))
                    pred_labels_list.append(
                        torch.empty((0,), dtype=torch.long, device=self.device)
                    )
                else:
                    pred_boxes_list.append(det[:, :4])  # xyxy
                    pred_labels_list.append(det[:, 6].long())  # class_id
            # ---------------- GT for confusion matrix -----------------
            gt_boxes_list, gt_labels_list = [], []
            for lbl_np in loaded_labels_proph:  # list length = batch
                if lbl_np.size == 0:  # no GT in this frame
                    gt_boxes_list.append(torch.empty((0, 4), device=self.device))
                    gt_labels_list.append(
                        torch.empty((0,), dtype=torch.long, device=self.device)
                    )
                else:
                    # lbl_np fields: x, y, w, h, class_id
                    # convert to float32 / int64 explicitly to avoid dtype issues
                    x1 = torch.as_tensor(
                        lbl_np["x"].astype(np.float32), device=self.device
                    )
                    y1 = torch.as_tensor(
                        lbl_np["y"].astype(np.float32), device=self.device
                    )
                    x2 = x1 + torch.as_tensor(
                        lbl_np["w"].astype(np.float32), device=self.device
                    )
                    y2 = y1 + torch.as_tensor(
                        lbl_np["h"].astype(np.float32), device=self.device
                    )
                    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
                    cls_ids = torch.as_tensor(
                        lbl_np["class_id"].astype(np.int64), device=self.device
                    )
                    gt_boxes_list.append(boxes_xyxy)
                    gt_labels_list.append(cls_ids)

            self.cm.update(
                pred_boxes_list, pred_labels_list, gt_boxes_list, gt_labels_list
            )
        # modules/detection.py _val_test_step_impl 끝부근 임시 삽입
        if self.global_step == 0:  # 첫 batch 한번만
            print("GT uniq :", {int(x) for g in gt_labels_list for x in g.tolist()})
            print("PR uniq :", {int(x) for p in pred_labels_list for x in p.tolist()})

        return output

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self._val_test_step_impl(batch=batch, mode=Mode.VAL)

    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self._val_test_step_impl(batch=batch, mode=Mode.TEST)

    def run_psee_evaluator(self, mode: Mode):
        psee_evaluator = self.mode_2_psee_evaluator[mode]
        batch_size = self.mode_2_batch_size[mode]
        hw_tuple = self.mode_2_hw[mode]
        if psee_evaluator is None:
            warn(f"psee_evaluator is None in {mode=}", UserWarning, stacklevel=2)
            return
        assert batch_size is not None
        assert hw_tuple is not None
        if psee_evaluator.has_data():
            metrics = psee_evaluator.evaluate_buffer(
                img_height=hw_tuple[0], img_width=hw_tuple[1]
            )
            assert metrics is not None

            prefix = f"{mode_2_string[mode]}/"
            step = self.trainer.global_step
            log_dict = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    value = torch.tensor(v)
                elif isinstance(v, np.ndarray):
                    value = torch.from_numpy(v)
                elif isinstance(v, torch.Tensor):
                    value = v
                else:
                    raise NotImplementedError
                assert (
                    value.ndim == 0
                ), f"tensor must be a scalar.\n{v=}\n{type(v)=}\n{value=}\n{type(value)=}"
                # put them on the current device to avoid this error: https://github.com/Lightning-AI/lightning/discussions/2529
                log_dict[f"{prefix}{k}"] = value.to(self.device)
            # Somehow self.log does not work when we eval during the training epoch.
            self.log_dict(
                log_dict,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )
            if dist.is_available() and dist.is_initialized():
                # We now have to manually sync (average the metrics) across processes in case of distributed training.
                # NOTE: This is necessary to ensure that we have the same numbers for the checkpoint metric (metadata)
                # and wandb metric:
                # - checkpoint callback is using the self.log function which uses global sync (avg across ranks)
                # - wandb uses log_metrics that we reduce manually to global rank 0
                dist.barrier()
                for k, v in log_dict.items():
                    dist.reduce(log_dict[k], dst=0, op=dist.ReduceOp.SUM)
                    if dist.get_rank() == 0:
                        log_dict[k] /= dist.get_world_size()
            if self.trainer.is_global_zero:
                # For some reason we need to increase the step by 2 to enable consistent logging in wandb here.
                # I might not understand wandb login correctly. This works reasonably well for now.
                add_hack = 2
                self.logger.log_metrics(metrics=log_dict, step=step + add_hack)

            psee_evaluator.reset_buffer()
        else:
            warn(f"psee_evaluator has not data in {mode=}", UserWarning, stacklevel=2)

    def on_train_epoch_end(self) -> None:
        mode = Mode.TRAIN
        if (
            mode in self.mode_2_psee_evaluator
            and self.train_metrics_config.detection_metrics_every_n_steps is None
            and self.mode_2_hw[mode] is not None
        ):
            # For some reason PL calls this function when resuming.
            # We don't know yet the value of train_height_width, so we skip this
            self.run_psee_evaluator(mode=mode)

    def on_validation_epoch_end(self) -> None:
        mode = Mode.VAL
        # Run Prophesee evaluator if we accumulated data during training
        if self.started_training:
            assert self.mode_2_psee_evaluator[mode].has_data()
            self.run_psee_evaluator(mode=mode)

        # -------- Confusion‑matrix saving --------
        epoch_idx = self.current_epoch
        step_idx = self.trainer.global_step

        cm_filename = f"/home/oeoiewt/eTraM/rvt_eTram/confM/confusion_matrix_e{epoch_idx:03d}_s{step_idx:07d}.png"

        # Save exactly **once** (reset=True) so counts start fresh next epoch
        self.cm.save_png(
            cm_filename,
            class_names=[str(i) for i in range(self.mdl_config.head.num_classes)],
            reset=True,
            normalize=False,  # <-- show raw counts instead of row‑normalized %
        )

        # Keep a copy with a fixed name for quick inspection
        import shutil

        latest_path = "/home/oeoiewt/eTraM/rvt_eTram/confM/confusion_matrix_latest.png"
        shutil.copyfile(cm_filename, latest_path)
        
        # -------- Detailed metrics calculation and logging --------
        self._calculate_and_save_detailed_metrics(epoch_idx, step_idx)

    def on_test_epoch_end(self) -> None:
        mode = Mode.TEST
        assert self.mode_2_psee_evaluator[mode].has_data()
        self.run_psee_evaluator(mode=mode)

    def configure_optimizers(self) -> Any:
        lr = self.train_config.learning_rate
        weight_decay = self.train_config.weight_decay
        optimizer = th.optim.AdamW(
            self.mdl.parameters(), lr=lr, weight_decay=weight_decay
        )

        scheduler_params = self.train_config.lr_scheduler
        if not scheduler_params.use:
            return optimizer

        total_steps = scheduler_params.total_steps
        assert total_steps is not None
        assert total_steps > 0
        # Here we interpret the final lr as max_lr/final_div_factor.
        # Note that Pytorch OneCycleLR interprets it as initial_lr/final_div_factor:
        final_div_factor_pytorch = (
            scheduler_params.final_div_factor / scheduler_params.div_factor
        )
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            div_factor=scheduler_params.div_factor,
            final_div_factor=final_div_factor_pytorch,
            total_steps=total_steps,
            pct_start=scheduler_params.pct_start,
            cycle_momentum=False,
            anneal_strategy="linear",
        )
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": True,
            "name": "learning_rate",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
    def _calculate_and_save_detailed_metrics(self, epoch_idx: int, step_idx: int) -> None:
        """Calculate detailed per-class metrics and save experiment results."""
        
        # Get validation data from Prophesee evaluator
        mode = Mode.VAL
        psee_evaluator = self.mode_2_psee_evaluator[mode]
        
        if psee_evaluator is None or not psee_evaluator.has_data():
            print(f"⚠️ No validation data available for detailed metrics calculation")
            return
        
        try:
            # Extract ground truth and prediction data from evaluator
            gt_boxes_list = psee_evaluator.gt_boxes_list
            dt_boxes_list = psee_evaluator.dt_boxes_list
            
            if not gt_boxes_list or not dt_boxes_list:
                print(f"⚠️ Empty boxes lists in evaluator")
                return
            
            # Calculate detailed metrics
            hw_tuple = self.mode_2_hw[mode]
            if hw_tuple is None:
                height, width = 384, 640  # Default values
            else:
                height, width = hw_tuple
            
            detailed_results = self.detailed_metrics.evaluate_detailed_detection(
                gt_boxes_list=gt_boxes_list,
                dt_boxes_list=dt_boxes_list,
                height=height,
                width=width
            )
            
            # Create experiment ID
            experiment_id = f"4scale_fpn_e{epoch_idx:03d}_s{step_idx:07d}"
            
            # Prepare experiment information
            model_modifications = {
                "architecture_changes": [
                    "Added P1 features (stride 4) to backbone output",
                    "Extended FPN from 3-scale to 4-scale (strides: 4,8,16,32)", 
                    "Modified YOLOPAFPN to support 4 input stages",
                    "Updated detection head to process 4 feature scales"
                ],
                "config_changes": [
                    "fpn.in_stages: [2,3,4] → [1,2,3,4]",
                    f"training.max_steps: {self.trainer.max_steps if hasattr(self.trainer, 'max_steps') else 'N/A'}",
                    f"dataset: {self.full_config.dataset.name} with {self.full_config.dataset.path}"
                ],
                "baseline_vs_modified": "4-scale FPN vs original 3-scale FPN",
                "key_features": [
                    "High-resolution P1 features for small object detection",
                    "4-scale feature pyramid network",
                    "Enhanced small object detection capability"
                ]
            }
            
            training_config = {
                "max_steps": getattr(self.trainer, 'max_steps', 'N/A'),
                "max_epochs": getattr(self.trainer, 'max_epochs', 'N/A'),
                "current_epoch": epoch_idx,
                "current_step": step_idx,
                "batch_size": {
                    "train": self.full_config.batch_size.train,
                    "eval": self.full_config.batch_size.eval
                },
                "learning_rate": self.full_config.training.learning_rate,
                "dataset_name": self.full_config.dataset.name,
                "dataset_path": self.full_config.dataset.path
            }
            
            # Initialize experiment logger if needed
            if self.experiment_logger is None:
                experiments_dir = "/home/oeoiewt/eTraM/rvt_eTram/experiments"
                self.experiment_logger = ExperimentLogger(experiments_dir)
            
            # Save detailed results
            result_file = self.experiment_logger.save_experiment_results(
                experiment_id=experiment_id,
                model_modifications=model_modifications,
                training_config=training_config,
                metrics_data=detailed_results,
                additional_info={
                    "confusion_matrix_file": f"confusion_matrix_e{epoch_idx:03d}_s{step_idx:07d}.png",
                    "checkpoint_info": {
                        "epoch": epoch_idx,
                        "step": step_idx,
                        "model_state": "validation_checkpoint"
                    }
                }
            )
            
            # Log summary to console
            overall_map = detailed_results.get("overall_metrics", {}).get("mAP", 0.0)
            small_obj_map = detailed_results.get("small_object_analysis", {}).get("avg_small_mAP", 0.0)
            
            print(f"📊 Detailed metrics saved: {result_file.name}")
            print(f"   Overall mAP: {overall_map:.4f}")
            print(f"   Small object avg mAP: {small_obj_map:.4f}")
            print(f"   Active classes: {detailed_results.get('evaluation_summary', {}).get('active_classes', 0)}/8")
            
        except Exception as e:
            print(f"❌ Error calculating detailed metrics: {e}")
            import traceback
            traceback.print_exc()
