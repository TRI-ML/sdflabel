# Copyright 2019 Toyota Research Institute. All rights reserved.
"""3D evaluation format is based on KITTI3D annotation style

Consumes a list of predictions and a list of ground truths. They must follow the format specified here:
https://github.com/NVIDIA/DIGITS/blob/v4.0.0-rc.3/digits/extensions/data/objectDetection/README.md
Each annotation for ground truth and detection follows the format:

anno (dic):
    'id': str
    'name': list[str]
    'truncated': 1d Ndarray
    'occluded': 1d Ndarray
    'alpha': 1d Ndarray
    'bbox': 2d Ndarray
    'dimensions': 2d Ndarray
    'location': 2d Ndarray
    'rotation_y': 1d Ndarray

Important note: KITTI require data in camera coordiate, this eval code also work in Lidar and vehicle coordinate.

See tests/test_detection_3d_camera_frame.py `get_label_anno` function for specifics on how to populate object.
"""
import math
from enum import IntEnum

import numba
import numpy as np
from scipy.spatial.distance import cdist

from pipelines.constants import KITTI_CLASS_NAMES, KITTI_OVERLAP_THRESHOLDS, NU_OVERLAP_THRESHOLDS
from pipelines.rotate_iou import (d3_box_overlap_kernel, image_box_overlap, rotate_iou_gpu_eval)


class Metrics(IntEnum):
    """
    Types of metrics supported by the evaluation class.
    BBOX_2D_AP: Evaluate Average Precision for 2D bbox on image.
    BEV_3D_AP: Evaluate Average Precision for BEV(Birds eye view) boxes.
    BBOX_3D_KITTI_AP: Evaluate Average Precision for 3D bbox based on IOU association.
    BBOX_3D_NU_AP: Evaluate Average Precision for 3D bbox based on distance based association.
    """

    BBOX_2D_AP = 0
    BEV_3D_AP = 1
    BBOX_3D_KITTI_AP = 2
    BBOX_3D_NU_AP = 3


class CoordinateFrame(IntEnum):
    """
    Type of coordinateFrame of ground truth and detections.
    Camera frame assume the x(left) y(down) z(front)
    Lidar frame and vehicle frame are flexible on x, y but need z(up).
    """

    LIDAR = 0
    VEHICLE = 1
    CAMERA = 2


class Detection3DEvaluator:
    """Class for performing 3D detection evaluation. Ground truths and detections are assumed to be in the KITTI format.
    To be noted, the data do not need to be transformed in camera frame.

    Parameters
    ----------
    filter_data_fn: fn
        To use this evaluator for other datasets, you must provide a data filtering function which matches the API
        of the example KITTI `clean_kitti_data` or `generic_clean_data` functions below.

        Must match the APIs EXACTLY to work.

    id_to_name: dict, default: KITTI_CLASS_NAMES
        Mapping of class ID -> class name. Defaults to KITTI ontology, see above for example.

    per_class_iou_overlap_thresholds: np.ndarray, default: KITTI_OVERLAP_THRESHOLDS
        Numpy array of shape (num_metrics, num_thresholds, num_difficulties, num_classes), corresponding to IoU
        thresholds for positive detections. If None, default to KITTI settings. See above for example.

    per_class_dist_thresholds: np.ndarray, default: NU_OVERLAP_THRESH
        Numpy array of shape [num_metrics, num_thresholds, num_difficulties, num_classes], corresponding
        to distance thresholds for positive detections. If None, default to NuScence settings. See above for example.

    coordinate_frame: CoordinateFrame.Enum.val
        Type of Data's coordinate frame, as specified in CoordinateFrame Enum.

    compute_angular_metrics: bool, default: True
        Compute orientation Error. Must provide 'rotation_y' as a field in predictions and ground truths.
        Compute orientation similarity. Must provide 'alpha' as a field in predictions and ground truths.

    compute_nuscenes: bool, default: False
        Compute NuScence detection metric. If set as True, will compute NuScence 3D metric
        otherwise, will compute just KITTI 3D metrics.

    sample_points: int, default: 41
        Number of points at which we compute precision and recall (includes 0 recall)

    sampling_frequency: int, default: 1
        Set the spacing of points along precision curve used to compute mAP
    """

    def __init__(
        self,
        filter_data_fn,
        id_to_name=KITTI_CLASS_NAMES,
        per_class_iou_overlap_thresholds=KITTI_OVERLAP_THRESHOLDS,
        per_class_dist_thresholds=NU_OVERLAP_THRESHOLDS,
        coordinate_frame=CoordinateFrame.LIDAR,
        compute_angular_metrics=True,
        compute_nuscenes=True,
        sample_points=41,
        sampling_frequency=1
    ):
        self.filter_data_fn = filter_data_fn
        self.sample_points = sample_points
        self.compute_angular_metrics = compute_angular_metrics
        self.coordinate_frame = coordinate_frame
        self.compute_nuscenes = compute_nuscenes
        self.sampling_frequency = sampling_frequency

        # Default to the KITTI ontology if ID to name it not provided.
        self.id_to_name = id_to_name
        self.name_to_id = {v: n for n, v in self.id_to_name.items()}
        self.overlap_thresholds = per_class_iou_overlap_thresholds
        self.dist_thresholds = per_class_dist_thresholds

    def evaluate_detection_3d(self, gt_annos, dt_annos, classes_for_eval=None, difficulties=(0, )):
        """ Evaluate 3D detection using KITTI style evaluation protocol.
        Computes 2D BBOX AP, 3D BBOX AP, BEV AP, aos (average orientation similarity) and aoe (average orientation
        error).

        Parameters
        ----------
        gt_annos: list
            List of ground truth annotations for bounding boxes. For details on format, please examine
            `validate_anno_format`

        dt_annos: list
            List of predicted bounding boxes. For details on format, please examine
            `validate_anno_format`

        classes_for_eval: list or tuple, default: None
            List or tuple of class names for which to compute metrics. If None,
            evaluate on all classes.

        difficulties: tuple, default: (0,)
            Difficulties to evaluate (0 = easy, 1 = medium, 2 = hard). Used in filter_data_fn to filter boxes
        """
        assert max(difficulties) <= self.overlap_thresholds.shape[2], \
            "difficuty index shall be smaller than {} but get {}.".format(self.overlap_thresholds.shape[2],
                                                                          max(difficulties))

        if self.compute_nuscenes:
            assert max(difficulties) <= self.dist_thresholds.shape[2], \
                "difficuty index shall be smaller than {} but get {}.".format(self.dist_thresholds.shape[2],
                                                                              max(difficulties))

        self.validate_anno_format(gt_annos, dt_annos)

        # Convert classes for evaluation to IDs
        classes_for_eval_ids = []
        assert isinstance(classes_for_eval, (list, tuple)), "Please list of class names for evaluation"
        for curcls in classes_for_eval:
            try:
                classes_for_eval_ids.append(self.name_to_id[curcls])
            except:
                raise KeyError("{} is not a valid class to evaluate in the given ontology".format(curcls))
        classes_for_eval = classes_for_eval_ids

        # Make sure every annotation provides an alpha if we want to compute aoe
        if self.compute_angular_metrics:
            for anno in dt_annos:
                assert 'rotation_y' in anno
                assert 'alpha' in anno

        dist_thresholds, overlap_thresholds = None, None
        if self.compute_nuscenes:
            dist_thresholds = self.dist_thresholds[:, :, :, classes_for_eval]

        overlap_thresholds = self.overlap_thresholds[:, :, :, classes_for_eval]

        mAPbbox, mAPbev, mAP3d, mAPaoe_iou, mAPaoe_dist, mAPaos_iou, mAPaos_dist, mAPnu3d, bbox_2d_pr_curves, \
        bev_pr_curves, \
        bbox_3d_kitti_pr_curves, \
        bbox_3d_nu_pr_curves = self.do_eval(
            gt_annos, dt_annos, classes_for_eval, difficulties, overlap_thresholds, dist_thresholds)

        formatted_result = ""
        for k, difficulty in enumerate(difficulties):
            formatted_result += "============================\n"
            formatted_result += "Difficuty Level {}:\n".format(difficulty)
            formatted_result += "============================\n"

            for j, curcls in enumerate(classes_for_eval):
                if self.compute_nuscenes:
                    # mAP threshold array: [metric, num_min_distance, num_diff, class]
                    # mAP result: [num_class, num_diff, num_min_distance]
                    for i in range(dist_thresholds.shape[1]):
                        formatted_result += "{} AP: \n".format(self.id_to_name[curcls])
                        formatted_result += "NuScenes 3D   @ {:.2f}: {:.4f}\n".format(
                            dist_thresholds[Metrics.BBOX_3D_NU_AP, i, k, j], mAPnu3d[j, k, i]
                        )

                        if self.compute_angular_metrics:
                            formatted_result += "AOE_dist  @ {:.2f}: {:.2f}\n".format(
                                dist_thresholds[Metrics.BBOX_3D_NU_AP, i, k, j], mAPaoe_dist[j, k, i]
                            )
                else:
                    # mAP threshold array: [metric, num_minoverlap, num_diff, class]
                    # mAP result: [num_class, num_diff, num_minoverlap]
                    for i in range(overlap_thresholds.shape[1]):
                        formatted_result += "{} AP: \n".format(self.id_to_name[curcls])
                        formatted_result += "Bbox @ {:.2f}: {:.4f}\n".format(
                            overlap_thresholds[Metrics.BBOX_2D_AP, i, k, j], mAPbbox[j, k, i]
                        )
                        formatted_result += "BEV  @ {:.2f}: {:.4f}\n".format(
                            overlap_thresholds[Metrics.BEV_3D_AP, i, k, j], mAPbev[j, k, i]
                        )
                        formatted_result += "3D   @ {:.2f}: {:.4f}\n".format(
                            overlap_thresholds[Metrics.BBOX_3D_KITTI_AP, i, k, j], mAP3d[j, k, i]
                        )

                        if self.compute_angular_metrics:
                            formatted_result += "AOE_iou  @ {:.2f}: {:.2f}\n".format(
                                overlap_thresholds[Metrics.BBOX_3D_KITTI_AP, i, k, j], mAPaoe_iou[j, k, i]
                            )
                            formatted_result += "AOS_iou  @ {:.2f}: {:.2f}\n".format(
                                overlap_thresholds[Metrics.BBOX_3D_KITTI_AP, i, k, j], mAPaos_iou[j, k, i]
                            )

        result_dict = {}
        for metric_name, metric in zip([
            "Box2DAP", "BevAP", "Box3DAP", "AoeAP_iou", "AoeAP_dist", "AosAP_iou", "AosAP_dist", "Box3DAP_Nu"
        ], [mAPbbox, mAPbev, mAP3d, mAPaoe_iou, mAPaoe_dist, mAPaos_iou, mAPaos_dist, mAPnu3d]):
            if metric is not None:
                result_dict[metric_name] = metric

        for metric_name, metric in zip([
            "bbox_2d_pre_curves", "bev_pre_curves", "bbox_3d_kitti_pre_curves", "bbox_3d_nu_pre_curves"
        ], [bbox_2d_pr_curves, bev_pr_curves, bbox_3d_kitti_pr_curves, bbox_3d_nu_pr_curves]):
            if metric is not None:
                result_dict[metric_name] = metric

        return formatted_result, result_dict

    def validate_anno_format(self, gt_annos, dt_annos):
        """Verify that the format/dimensions for the annotations are correct.
        Keys correspond to defintions here:
        https://github.com/NVIDIA/DIGITS/blob/v4.0.0-rc.3/digits/extensions/data/objectDetection/README.md
        """
        necessary_keys = ['name', 'alpha', 'bbox', 'dimensions', 'location', 'rotation_y', 'score']
        for i, (gt_anno, dt_anno) in enumerate(zip(gt_annos, dt_annos)):
            for key in necessary_keys:
                assert key in gt_anno, "{} not present in GT {}".format(key, i)
                assert key in dt_anno, "{} not present in prediction {}".format(key, i)
                if key in ['bbox', 'dimensions', 'location']:
                    # make sure these fields are 2D numpy array
                    assert len(gt_anno[key].shape) == 2
                    assert len(dt_anno[key].shape) == 2

    def do_eval(self, gt_annos, dt_annos, classes_for_eval, difficulties, overlap_thresholds, dist_thresholds):
        """Wrapper for performing evaluation for each metric on given classes.

        Parameters:
        ----------
        gt_annos: list
            List of ground truth annotations for objects.The box is assumed to be sensor or vehicle
            coordinate. The location is assumed to be the center of the 3D box. Dimension follows wlh(dgp) format.
            Please see test case for more details of annotation.

        dt_annos: list
            List of predicted bounding boxes. Format is the same as gt_annos.

        classes_for_eval: list or tuple, default: None
            List or tuple of class names for which to compute metrics. If None,
            evaluate on all classes.

        difficulties: tuple, default: (0,1,2)
            Difficulties to evaluate (0 = easy, 1 = medium, 2 = hard). Definitions per
            class are defined in difficulty enum.

        metric: Metrics.Enum.val
            Type of evaluation, as specified in Metrics Enum.

        overlap_thresholds: np.ndarray
            Array of shape [num_metrics, num_thresholds, num_difficulties, num_classes].

        dist_thresholds: nd.ndarray
            Array of shape [num_metrics, num_thresholds, num_difficulties, num_classes].

        Returns:
        --------
        list of metric fields (numpy.ndarray)
            mAP_bbox: 2D bbox mAP with shape [num_class, num_diff, num_overlap_thresholds]
            mAP_bev: Birds eye view mAP with shape [num_class, num_diff, num_overlap_thresholds]
            mAP_3d: 3D box mAP using 3D IOU threshold with shape [num_class, num_diff, num_overlap_thresholds]
            mAP_aoe_iou: Angular orientation error with shape [num_class, num_diff, num_overlap_thresholds]
            mAP_aoe_dist: Angular orientation error with shape [num_class, num_diff, num_dist_thresholds]
            mAP_aos_iou: Angular orientation similarity with shape [num_class, num_diff, num_overlap_thresholds]
            mAP_aos_dist: Angular orientation similarity with shape [num_class, num_diff, num_dist_thresholds]
            mAPnu_3d: 3D NuScence mAP using box 3D centroid distance with shape [num_class, num_diff,
            num_distance_threshold]
            bbox_2d_pr_curves: Precision/Recall/Errors(IOU error)
            bev_pr_curves:  Precision/Recall/Errors(IOU error)
            bbox_3d_kitti_pr_curves: Precision/Recall/Errors(IOU error)
            bbox_3d_nu_pr_curves: Precision/Recall/Errors(Centroid distance error)
        """

        # Compute 2D BBox metrics
        bbox_2d_pr_curves = self.eval_metric(
            gt_annos, dt_annos, classes_for_eval, difficulties, Metrics.BBOX_2D_AP, overlap_thresholds, dist_thresholds
        )
        mAP_bbox = self.get_mAP(bbox_2d_pr_curves["precision"], bbox_2d_pr_curves["recall"])

        # Compute BEV and 3D BBox
        bev_pr_curves = self.eval_metric(
            gt_annos, dt_annos, classes_for_eval, difficulties, Metrics.BEV_3D_AP, overlap_thresholds, dist_thresholds
        )

        mAP_bev = self.get_mAP(bev_pr_curves["precision"], bev_pr_curves["recall"])

        mAP_3d, mAPnu_3d, mAP_aos_iou, mAP_aos_dist, mAP_aoe_iou, mAP_aoe_dist, bbox_3d_kitti_pr_curves, \
        bbox_3d_nu_pr_curves = None, None, None, None, None, None, None, None

        if self.compute_nuscenes:
            bbox_3d_nu_pr_curves = self.eval_metric(
                gt_annos, dt_annos, classes_for_eval, difficulties, Metrics.BBOX_3D_NU_AP, overlap_thresholds,
                dist_thresholds, self.compute_angular_metrics
            )

            mAPnu_3d = self.get_mAP(bbox_3d_nu_pr_curves["precision"], bbox_3d_nu_pr_curves["recall"])
            if self.compute_angular_metrics:
                mAP_aoe_dist = self.get_mAP(bbox_3d_nu_pr_curves["orientation_aoe"], bbox_3d_nu_pr_curves["recall"])
                mAP_aos_dist = self.get_mAP(bbox_3d_nu_pr_curves["orientation_aos"], bbox_3d_nu_pr_curves["recall"])
        else:
            bbox_3d_kitti_pr_curves = self.eval_metric(
                gt_annos, dt_annos, classes_for_eval, difficulties, Metrics.BBOX_3D_KITTI_AP, overlap_thresholds,
                dist_thresholds, self.compute_angular_metrics
            )
            mAP_3d = self.get_mAP(bbox_3d_kitti_pr_curves["precision"], bbox_3d_kitti_pr_curves["recall"])
            if self.compute_angular_metrics:
                mAP_aoe_iou = self.get_mAP(
                    bbox_3d_kitti_pr_curves["orientation_aoe"], bbox_3d_kitti_pr_curves["recall"]
                )
                mAP_aos_iou = self.get_mAP(
                    bbox_3d_kitti_pr_curves["orientation_aos"], bbox_3d_kitti_pr_curves["recall"]
                )

        return mAP_bbox, mAP_bev, mAP_3d, mAP_aoe_iou, mAP_aoe_dist, mAP_aos_iou, mAP_aos_dist, mAPnu_3d, \
               bbox_2d_pr_curves, bev_pr_curves, bbox_3d_kitti_pr_curves, bbox_3d_nu_pr_curves

    def eval_metric(
        self,
        gt_annos,
        dt_annos,
        classes_for_eval,
        difficulties,
        metric,
        overlap_thresholds,
        dist_thresholds,
        compute_angular_metrics=False,
        num_shards=50
    ):
        """KITTI Style Eval. Supports 2D/BEV/3D/aoe evaluation.

        Parameters:
        ----------
        gt_annos: list
            List of ground truth annotations for bounding boxes.

        dt_annos: list
            List of predicted bounding boxes

        classes_for_eval: list or tuple, default: None
            List or tuple of class names for which to compute metrics. If None,
            evaluate on all classes.

        difficulties: tuple, default: (0,1,2)
            Difficulties to evaluate (0 = easy, 1 = medium, 2 = hard). Definitions per
            class are defined in difficulty enum.

        metric: Metrics.Enum.val
            Type of evaluation, as specified in Metrics Enum.

        overlap_thresholds: np.ndarray
            Array of shape [num_metrics, num_thresholds, num_difficulties, num_classes].

        dist_thresholds: nd.ndarray
            Array of shape [num_metrics, num_thresholds, num_difficulties, num_classes].

        compute_angular_metrics: bool
            Compute angular orientation similarity and angular orientation error.

        num_shards: int, default: 50
            Number of shards for IoU computations

        Returns:
        --------
        dict of np.array:
            Set of curves for recall, precision, and aoe, organized as follows:
                recall: shape [num_classes, num_difficulties, num_overlap_thresholds, num_sample_points]
                precision: [num_classes, num_difficulties, num_overlap_thresholds, num_sample_points]
                aoe: [num_classes, num_difficulties, num_overlap_thresholds, num_sample_points]
                aos: [num_classes, num_difficulties, num_overlap_thresholds, num_sample_points]
                tp_mean_error: average location error when nuscenes otherwise average iou error.
                tp_mean_confidence_error: average confidence error of all true positives detections.

        """
        assert len(gt_annos) == len(dt_annos), "Must provide a prediction for every ground truth sample"
        num_ground_truths = len(gt_annos)
        shards = self.get_shards(num_ground_truths, num_shards)

        overlaps, overlaps_by_shard, total_gt_num, total_dt_num = self.calculate_match_degree_sharded(
            gt_annos, dt_annos, metric, num_shards
        )

        all_thresholds = -1.0 * dist_thresholds[metric, :, :, :] if metric == Metrics.BBOX_3D_NU_AP else \
            overlap_thresholds[metric, :, :, :]

        num_minoverlap = len(all_thresholds)
        num_classes = len(classes_for_eval)
        num_difficulties = len(difficulties)

        precision = np.zeros([num_classes, num_difficulties, num_minoverlap, self.sample_points])
        recall = np.zeros([num_classes, num_difficulties, num_minoverlap, self.sample_points])
        orientation_aoe = np.zeros([num_classes, num_difficulties, num_minoverlap, self.sample_points])
        orientation_aos = np.zeros([num_classes, num_difficulties, num_minoverlap, self.sample_points])
        tp_mean_error = np.zeros([num_classes, num_difficulties, num_minoverlap, self.sample_points])
        tp_mean_confidence_error = np.zeros([num_classes, num_difficulties, num_minoverlap, self.sample_points])

        for m, current_class in enumerate(classes_for_eval):
            for l, difficulty in enumerate(difficulties):
                gt_data_list, dt_data_list, ignored_gts, ignored_dets, dontcares, ignores_per_sample, \
                total_num_valid_gt = self.prepare_data(
                    gt_annos, dt_annos, current_class, difficulty
                )
                for k, min_overlap in enumerate(all_thresholds[:, l, m]):
                    thresholds_list = []
                    for i in range(len(gt_annos)):
                        results = compute_statistics(
                            overlaps[i],
                            gt_data_list[i],
                            dt_data_list[i],
                            ignored_gts[i],
                            ignored_dets[i],
                            dontcares[i],
                            metric,
                            min_overlap=min_overlap,
                            thresh=0.0,
                            compute_fp=False
                        )
                        thresholds_list += results['thresholds'].tolist()
                    thresholds = np.array(
                        get_thresholds(np.array(thresholds_list), total_num_valid_gt, self.sample_points)
                    )
                    # TODO: Refactor hard coded numbers and strings
                    # [num_threshold, num_fields], fields: tp, fp, fn, aoe, aos, iou/dist error, -log(Probability)
                    pr = np.zeros([len(thresholds), 7])

                    idx = 0
                    for j, num_samples_per_shard in enumerate(shards):
                        gt_datas_part = np.concatenate(gt_data_list[idx:idx + num_samples_per_shard], 0)
                        dt_datas_part = np.concatenate(dt_data_list[idx:idx + num_samples_per_shard], 0)
                        dc_datas_part = np.concatenate(dontcares[idx:idx + num_samples_per_shard], 0)
                        ignored_dets_part = np.concatenate(ignored_dets[idx:idx + num_samples_per_shard], 0)
                        ignored_gts_part = np.concatenate(ignored_gts[idx:idx + num_samples_per_shard], 0)
                        fused_compute_statistics(
                            overlaps_by_shard[j],
                            pr,
                            total_gt_num[idx:idx + num_samples_per_shard],
                            total_dt_num[idx:idx + num_samples_per_shard],
                            ignores_per_sample[idx:idx + num_samples_per_shard],
                            gt_datas_part,
                            dt_datas_part,
                            dc_datas_part,
                            ignored_gts_part,
                            ignored_dets_part,
                            metric,
                            min_overlap=min_overlap,
                            thresholds=thresholds,
                            compute_angular_metrics=compute_angular_metrics
                        )
                        idx += num_samples_per_shard

                    for i in range(len(thresholds)):
                        recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                        precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
                        tp_mean_error[m, l, k, i] = pr[i, 5] / pr[i, 0]
                        tp_mean_confidence_error[m, l, k, i] = pr[i, 6] / pr[i, 0]
                        if metric != Metrics.BBOX_3D_NU_AP:
                            tp_mean_error[m, l, k, i] = abs(1.0 - tp_mean_error[m, l, k, i])
                        if compute_angular_metrics:
                            orientation_aoe[m, l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
                            orientation_aos[m, l, k, i] = pr[i, 4] / (pr[i, 0] + pr[i, 1])
        return {
            "recall": recall,
            "precision": precision,
            "orientation_aoe": orientation_aoe,
            "orientation_aos": orientation_aos,
            "tp_mean_error": tp_mean_error,
            "tp_mean_confidence_error": tp_mean_confidence_error
        }

    def calculate_match_degree_sharded(self, gt_annos, dt_annos, metric, num_shards):
        """Fast iou/distance algorithm. It calculate the overlap/distance of groud truth and detection.
        This function can be used independently to do result analysis.

        Assumes (Z-Forward, Y-Down, X-Right) -
        Must be used in CAMERA coordinate system. See here for more details:
        http://www.cvlibs.net/datasets/kitti/setup.php

        Parameters
        ----------
        gt_annos: list
            List of ground truth annotations for bounding boxes.

        dt_annos: list
            List of predicted bounding boxes

        metric: Metrics.Enum.val
            Type of evaluation, as specified in Metrics Enum.

        num_shards: int
            IoU can be computed faster per sample

        Returns
        -------
        overlaps: list[np.ndarray]
            List of IoUs between predictions/GTs -> N * K, where N is the number of
            predictions per image, K is number of GTs.

        overlaps_by_shard: list[list[[np.ndarray]]]
            Same as above, but by shard

        total_gt_num: int
            Number of ground truths

        total_dt_num: int
            Number of predictions
        """
        assert len(gt_annos) == len(dt_annos)
        total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
        total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)

        overlaps_by_shard = []
        sample_idx = 0
        num_ground_truths = len(gt_annos)
        shards = self.get_shards(num_ground_truths, num_shards)

        for num_samples_per_shard in shards:
            gt_annos_part = gt_annos[sample_idx:sample_idx + num_samples_per_shard]
            dt_annos_part = dt_annos[sample_idx:sample_idx + num_samples_per_shard]

            if metric == Metrics.BBOX_2D_AP:
                gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)
                dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)
                shard_match = image_box_overlap(dt_boxes, gt_boxes)

            elif metric == Metrics.BEV_3D_AP and self.coordinate_frame == CoordinateFrame.CAMERA:
                loc = np.concatenate([a["location"][:, [0, 2]] for a in gt_annos_part], 0)
                dims = np.concatenate([a["dimensions"][:, [0, 2]] for a in gt_annos_part], 0)
                rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
                gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
                loc = np.concatenate([a["location"][:, [0, 2]] for a in dt_annos_part], 0)
                dims = np.concatenate([a["dimensions"][:, [0, 2]] for a in dt_annos_part], 0)
                rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
                dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
                shard_match = self.bev_box_overlap(dt_boxes, gt_boxes).astype(np.float64)

            elif metric == Metrics.BEV_3D_AP and not (self.coordinate_frame == CoordinateFrame.CAMERA):
                loc = np.concatenate([a["location"][:, [0, 1]] for a in gt_annos_part], 0)
                dims = np.concatenate([a["dimensions"][:, [0, 1]] for a in gt_annos_part], 0)
                rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
                gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
                loc = np.concatenate([a["location"][:, [0, 1]] for a in dt_annos_part], 0)
                dims = np.concatenate([a["dimensions"][:, [0, 1]] for a in dt_annos_part], 0)
                rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
                dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
                shard_match = self.bev_box_overlap(dt_boxes, gt_boxes).astype(np.float64)

                #print('BEV IOU:', shard_match)

            elif metric == Metrics.BBOX_3D_KITTI_AP:
                loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
                dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
                rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
                gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
                loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
                dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
                rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
                dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
                shard_match = self.box_3d_overlap(dt_boxes, gt_boxes).astype(np.float64)

                #print('3D IOU:', shard_match)

            elif metric == Metrics.BBOX_3D_NU_AP and not (self.coordinate_frame == CoordinateFrame.CAMERA):
                # https://github.com/nutonomy/nuscenes-devkit/blob/f3594b967cbf42396da5c6cb08bd714437b53111/
                # python-sdk/nuscenes/eval/detection/utils.py#L71
                loc_gt = np.concatenate([a["location"][:, [0, 1]] for a in gt_annos_part], 0)
                loc_det = np.concatenate([a["location"][:, [0, 1]] for a in dt_annos_part], 0)
                shard_match = -1 * cdist(loc_det, loc_gt)

            elif metric == Metrics.BBOX_3D_NU_AP and self.coordinate_frame == CoordinateFrame.CAMERA:
                # https://github.com/nutonomy/nuscenes-devkit/blob/f3594b967cbf42396da5c6cb08bd714437b53111/
                # python-sdk/nuscenes/eval/detection/utils.py#L71
                loc_gt = np.concatenate([a["location"][:, [0, 2]] for a in gt_annos_part], 0)
                loc_det = np.concatenate([a["location"][:, [0, 2]] for a in dt_annos_part], 0)
                shard_match = -1 * cdist(loc_det, loc_gt)

            else:
                raise ValueError("Unknown metric")

            # On each shard, we compute an IoU between all N predicted boxes and K GT boxes.
            # Shard overlap is a (N X K) array
            overlaps_by_shard.append(shard_match)
            sample_idx += num_samples_per_shard

        # Flatten into unsharded list
        overlaps = []
        sample_idx = 0
        for j, num_samples_per_shard in enumerate(shards):
            gt_num_idx, dt_num_idx = 0, 0
            for i in range(num_samples_per_shard):
                gt_box_num = total_gt_num[sample_idx + i]
                dt_box_num = total_dt_num[sample_idx + i]
                overlaps.append(
                    overlaps_by_shard[j][dt_num_idx:dt_num_idx + dt_box_num, gt_num_idx:gt_num_idx + gt_box_num, ]
                )
                gt_num_idx += gt_box_num
                dt_num_idx += dt_box_num
            sample_idx += num_samples_per_shard
        return overlaps, overlaps_by_shard, total_gt_num, total_dt_num

    def get_shards(self, num, num_shards):
        """Shard number into evenly sized parts. `Remaining` values are put into the last shard.

        Parameters
        ----------
        num: int
            Number to shard

        num_shards: int
            Number of shards

        Returns
        -------
        List of length (num_shards or num_shards +1), depending on whether num is perfectly divisible by num_shards
        """
        assert num_shards > 0, "Invalid number of shards"
        num_per_shard = num // num_shards
        remaining_num = num % num_shards
        full_shards = num_shards * (num_per_shard > 0)
        if remaining_num == 0:
            return [num_per_shard] * full_shards
        else:
            return [num_per_shard] * full_shards + [remaining_num]

    def bev_box_overlap(self, boxes, qboxes, criterion=-1):
        """Compute overlap in BEV"""
        riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
        return riou

    def box_3d_overlap(self, boxes, qboxes, criterion=-1):
        """Compute 3D box IoU"""
        # For scale cuboid: use x, y to calculate bev iou, for kitti, use x, z to calculate bev iou
        rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]], qboxes[:, [0, 2, 3, 5, 6]], 2) if self.coordinate_frame\
            == CoordinateFrame.CAMERA else rotate_iou_gpu_eval(boxes[:, [0, 1, 3, 4, 6]], qboxes[:, [0, 1, 3, 4, 6]], 2)
        d3_box_overlap_kernel(boxes, qboxes, rinc, criterion, self.coordinate_frame == CoordinateFrame.CAMERA)
        return rinc

    def prepare_data(self, gt_annos, dt_annos, current_class, difficulty):
        """Wrapper function for cleaning data before computing metrics.
        """
        gt_list = []
        dt_list = []
        ignores_per_sample = []
        ignored_gts, ignored_dets, dontcares = [], [], []
        total_num_valid_gt = 0

        for gt_anno, dt_anno in zip(gt_annos, dt_annos):
            num_valid_gt, ignored_gt, ignored_det, ignored_bboxes = self.filter_data_fn(
                gt_anno, dt_anno, current_class, difficulty, self.id_to_name, self.coordinate_frame
            )
            ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
            ignored_dets.append(np.array(ignored_det, dtype=np.int64))

            if len(ignored_bboxes) == 0:
                ignored_bboxes = np.zeros((0, 4)).astype(np.float64)
            else:
                ignored_bboxes = np.stack(ignored_bboxes, 0).astype(np.float64)

            ignores_per_sample.append(ignored_bboxes.shape[0])
            dontcares.append(ignored_bboxes)
            total_num_valid_gt += num_valid_gt
            gt_list.append(
                np.concatenate([
                    gt_anno["bbox"], gt_anno["rotation_y"][..., np.newaxis], gt_anno["alpha"][..., np.newaxis]
                ], 1)
            )

            dt_list.append(
                np.concatenate([
                    dt_anno["bbox"], dt_anno["rotation_y"][..., np.newaxis], dt_anno["alpha"][..., np.newaxis],
                    dt_anno["score"][..., np.newaxis]
                ], 1)
            )

        ignores_per_sample = np.stack(ignores_per_sample, axis=0)
        return gt_list, dt_list, ignored_gts, ignored_dets, dontcares, ignores_per_sample, total_num_valid_gt

    def get_mAP(self, precision, recall):
        """ Get mAP from precision. Sample evenly along the recall range, and interpolate the precision
        based on AP from section 6 from https://research.mapillary.com/img/publications/MonoDIS.pdf

        Parameters
        ----------
        precision: np.ndarray
            Numpy array of precision curves at different recalls, of shape
            [num_classes, num_difficulties, num_overlap_thresholds,self.sample_points]

        recall: np.ndarray
            Numpy array of recall values corresponding to each precision, of shape
            [num_classes, num_difficulties, num_overlap_thresholds,self.sample_points]

        Returns
        -------
        ap: np.ndarray
            Numpy array of mean AP evaluated at different points along PR curve.
            Shape [num_classes, num_difficulties, num_overlap_thresholds]
        """
        precisions = []
        # Don't count recall at 0
        recall_spacing = [1. / (self.sample_points - 1) * i for i in range(1, self.sample_points)]

        for r in recall_spacing:
            precisions_above_recall = (recall >= r) * precision
            precisions.append(precisions_above_recall.max(axis=3))

        ap = 100.0 * sum(precisions) / (self.sample_points - 1)
        return ap


@numba.jit(nopython=True)
def angle_diff(x, y, period):
    """
    Get the smallest angle difference between 2 angles: the angle from y to x.
    :param x: To angle.
    :param y: From angle.
    :param period: Periodicity in radians for assessing angle difference.
    :return: <float>. Signed smallest between-angle difference in range (-pi, pi).
    """

    # calculate angle difference, modulo to [0, 2*pi]
    diff = (x - y + period / 2) % period - period / 2
    if diff > np.pi:
        diff = diff - (2 * np.pi)  # shift (pi, 2*pi] to (-pi, 0]

    return diff


def difficulty_by_distance(
    gt_anno,
    dt_anno,
    current_class,
    difficulty,
    id_to_name,
    coordinate_frame=CoordinateFrame.LIDAR,
    max_depth=(30, 80, 150),
    min_height=20
):
    """function for binning data. Follows the same API as `clean_kitti_data`,
    but filters on depth, rather than custom occlusion/truncation parameters of KITTI dataset.

    We filter with the following heuristics:
        If a ground truth matches the current class AND within the max depth,
        we count it as a valid gt (append 0 in `ignored_gt` list).

        If a ground truth matches the current class but NOT within max depth,
        we ignore it (append 1 in `ignored_gt` list)

        If a ground truth doesn't belong to the current class, we ignore it (append -1 in `ignored_gt`)

        If a prediction matches the current class AND is above the minimum height threshold, we count it
        as a valid detection (append 0 in `ignored_dt`)

        If a prediction matches the current class AND it is too small, we ignore it (append 1 in `ignored_dt`)

        If a prediction doesn't belong to the class, we ignore it (append -1 in `ignored_dt`)

    Parameters
    ----------
    gt_anno: dict
        KITTI format ground truth. Please refer to note at the top for details on format.

    dt_anno: dict
        KITTI format prediction.  Please refer to note at the top for details on format.

    current_class: int
        Class ID, as int

    difficulty: int
        Difficulty: easy=0, moderate=1, difficult=2

    id_to_name: dict
        Mapping from class ID (int) to string name

    coordinate_frame: Enum
        Coordinate frame of data.

    max_depth: list default=(30, 80, 150), unit in meters
        depths list to filter data

    min_height: int default=20, unit in pixel
        minimum height to filter data

    Returns
    -------
    num_valid_gt: int
        Number of valid ground truths

    ignored_gt: list[int]
        List of length num GTs. Populated as described above.

    ignored_dt: list[int]
        List of length num detections. Populated as described above.

    ignored_bboxes: list[np.ndarray]
        List of np.ndarray corresponding to boxes that are to be ignored
    """

    ignored_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = id_to_name[current_class].lower()
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0

    for i in range(num_gt):
        bbox = gt_anno["bbox"][i]
        gt_name = gt_anno["name"][i].lower()

        # Filter by depth as a proxy for difficulty
        ignore_for_depth = False

        distance = gt_anno["location"][i][2] if coordinate_frame == CoordinateFrame.CAMERA else \
            math.sqrt(gt_anno["location"][i][0]**2 + gt_anno["location"][i][1]**2)
        if distance > max_depth[difficulty]:
            ignore_for_depth = True

        if gt_name == current_cls_name and not ignore_for_depth:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif gt_name == current_cls_name and ignore_for_depth:
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)

    for i in range(num_dt):
        height = abs(dt_anno["bbox"][i, 3] - dt_anno["bbox"][i, 1])

        # If a box is too small, ignore it in evaluation
        if height < min_height:
            ignored_dt.append(1)
        elif dt_anno["name"][i].lower() == current_cls_name:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, ignored_bboxes


def clean_kitti_data(gt_anno, dt_anno, current_class, difficulty, id_to_name, coordinate_frame=CoordinateFrame.CAMERA):
    """Function for filtering KITTI data by difficulty and class.

    We filter with the following heuristics:
        If a ground truth matches the current class AND it falls below the difficulty
        threshold, we count it as a valid gt (append 0 in `ignored_gt` list).

        If a ground truth matches the current class but NOT the difficulty, OR it matches
        a class that is semantically too close to penalize (i.e. Van <-> Car),
        we ignore it (append 1 in `ignored_gt` list)

        If a ground truth doesn't belong to the current class, we ignore it (append -1 in `ignored_gt`)

        If a ground truth corresponds to a "DontCare" box, we append that box to the `ignored_bboxes` list.

        If a prediction matches the current class AND is above the minimum height threshold, we count it
        as a valid detection (append 0 in `ignored_dt`)

        If a prediction matches the current class AND it is too small, we ignore it (append 1 in `ignored_dt`)

        If a prediction doesn't belong to the class, we ignore it (append -1 in `ignored_dt`)

    Parameters
    ----------
    gt_anno: dict
        KITTI format ground truth. Please refer to note at the top for details on format.

    dt_anno: dict
        KITTI format prediction.  Please refer to note at the top for details on format.

    current_class: int
        Class ID, as int

    difficulty: int
        Difficulty: easy=0, moderate=1, difficult=2

    id_to_name: dict
        Mapping from class ID (int) to string name

    coordinate_frame: Enum
        Coordinate frame of data. Not used in kitti difficulty.

    Returns
    -------
    num_valid_gt: int
        Number of valid ground truths

    ignored_gt: list[int]
        List of length num GTs. Populated as described above.

    ignored_dt: list[int]
        List of length num detections. Populated as described above.

    ignored_bboxes: list[np.ndarray]
        List of np.ndarray corresponding to boxes that are to be ignored
    """
    MAX_OCCLUSION = (0, 1, 2)
    MAX_TRUNCATION = (0.15, 0.3, 0.5)
    MIN_HEIGHT = (40, 25, 25)
    ignored_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = id_to_name[current_class].lower()
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0

    for i in range(num_gt):
        bbox = gt_anno["bbox"][i]
        gt_name = gt_anno["name"][i].lower()
        height = bbox[3] - bbox[1]
        valid_class = -1

        # For KITTI, Van does not penalize car detections and person sitting does not penalize pedestrian
        if (gt_name == current_cls_name):
            valid_class = 1
        elif (current_cls_name == "Pedestrian".lower() and "Person_sitting".lower() == gt_name):
            valid_class = 0
        elif (current_cls_name == "Car".lower() and "Van".lower() == gt_name):
            valid_class = 0
        else:
            valid_class = -1

        # Filter by occlusion/truncation
        ignore_for_truncation_occlusion = False
        if ((gt_anno["occluded"][i] > MAX_OCCLUSION[difficulty])
            or (gt_anno["truncated"][i] > MAX_TRUNCATION[difficulty]) or (height <= MIN_HEIGHT[difficulty])):
            ignore_for_truncation_occlusion = True

        if valid_class == 1 and not ignore_for_truncation_occlusion:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif (valid_class == 0 or (ignore_for_truncation_occlusion and (valid_class == 1))):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)

        # Track boxes are in "dontcare" areas
        if gt_name == "dontcare":
            ignored_bboxes.append(bbox)

    for i in range(num_dt):
        if (dt_anno["name"][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        height = abs(dt_anno["bbox"][i, 3] - dt_anno["bbox"][i, 1])

        # If a box is too small, ignore it
        if height < MIN_HEIGHT[difficulty]:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, ignored_bboxes


# ------- JIT utils for computing TP/FP curves -------


@numba.jit(nopython=True, fastmath=True)
def get_thresholds(scores, num_gt, num_sample_pts=41):
    """Get thresholds from a set of scores, up to num sample points

    Parameters
    ----------
    score: np.ndarray
        Numpy array of scores for predictions

    num_gt: int
        Number of ground truths

    num_sample_pts: int, default: 41
        Max number of thresholds on PR curve

    Returns
    -------
    threshold: np.ndarray
        Array of length 41, containing recall thresholds
    """
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall)) and (i < (len(scores) - 1))):
            continue
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds


def compute_statistics(
    overlaps,
    gt_datas,
    dt_datas,
    ignored_gt,
    ignored_det,
    ignored_bboxes,
    metric,
    min_overlap,
    thresh=0.0,
    compute_fp=False,
    compute_angular_metrics=False
):
    """Wrapper function of compute_statistics_jit

    Parameters
    ----------
    overlaps: np.ndarray
    IoUs between predictions/GTs -> N * K, where N is the number of
    predictions per image, K is number of GTs.

    gt_datas: np.ndarray
    Column names as [gt_anno["bbox"], gt_anno["rotation_y"], gt_anno["alpha"]]

    dt_datas: np.ndarray
    Column names as [dt_anno["bbox"], dt_anno["rotation_y"], dt_anno["alpha"], dt_anno["score"]

    ignored_gt: np.ndarray
    -1 for ignored, 0 for valid, 1 for ignore due to occlution or too small or truncation.

    ignored_det: np.ndarray
    -1 for ignored, 0 for valid, 1 for ignore due to too small

    ignored_bboxes: np.ndarray
    Boxes with "dontcare" label

    metric: IntEnum
    Metric type in Metrics enum.

    min_overlap: float
    Minimum overlap threshold

    thresh: float
    Detection score threshold.

    compute_fp: bool
    Boolean for compute false positve detections.

    compute_angular_metrics: bool
    Boolean whether or not compute angular metrics, AOS and AOE.
    """
    results = {}
    tp, fp, fn, error_yaw, similarity, thresholds, match_degree, confidence_error = \
        compute_statistics_jit(overlaps,
        gt_datas,
        dt_datas,
        ignored_gt,
        ignored_det,
        ignored_bboxes,
        metric,
        min_overlap,
        thresh,
        compute_fp,
        compute_angular_metrics)

    results['thresholds'] = thresholds

    return results


@numba.jit(nopython=True, fastmath=True)
def compute_statistics_jit(
    overlaps,
    gt_datas,
    dt_datas,
    ignored_gt,
    ignored_det,
    ignored_bboxes,
    metric,
    min_overlap,
    thresh=0.0,
    compute_fp=False,
    compute_angular_metrics=False
):
    """Compute TP/FP statistics.
    Modified from https://github.com/sshaoehuai/PointRCNN/blob/master/tools/kitti_object_eval_python/eval.py
    """
    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]
    dt_yaws = dt_datas[:, 4]
    gt_yaws = gt_datas[:, 4]
    dt_alphas = dt_datas[:, 5]
    gt_alphas = gt_datas[:, 5]
    dt_bboxes = dt_datas[:, :4]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True

    NO_DETECTION = -10000000
    tp, fp, fn, error_yaw, similarity, match_degree, confidence_error = 0, 0, 0, 0, 0, 0, 0
    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0
    delta_yaw = np.zeros((gt_size, ))
    delta_alpha = np.zeros((gt_size, ))
    delta_idx = 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = -100000
        assigned_ignored_det = False

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]

            # Not hit during TP/FP computation
            if (not compute_fp and (overlap > min_overlap) and dt_score > valid_detection):
                assert not compute_fp, "For sanity, compute_fp shoudl be False if we are here"
                det_idx = j
                valid_detection = dt_score
            elif (
                compute_fp and (overlap > min_overlap) and (overlap > max_overlap or assigned_ignored_det)
                and ignored_det[j] == 0
            ):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False

            elif (compute_fp and (overlap > min_overlap) and (valid_detection == NO_DETECTION) and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        # No matched prediction found, valid GT
        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1

        # Matched prediction, but NO valid GT or matched prediction is too small so we ignore it (NOT BECAUSE THE
        # CLASS IS WRONG)
        elif ((valid_detection != NO_DETECTION) and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True

        # Matched prediction
        elif valid_detection != NO_DETECTION:
            tp += 1
            match_degree += abs(max_overlap)
            confidence_error += -math.log(dt_scores[det_idx])
            # Build a big list of all thresholds associated to true positives
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1

            if compute_angular_metrics:
                delta_yaw[delta_idx] = abs(angle_diff(float(gt_yaws[i]), float(dt_yaws[det_idx]), 2 * np.pi))
                delta_alpha[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta_idx += 1

            assigned_detection[det_idx] = True

    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1 or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0
        if metric == Metrics.BBOX_2D_AP:
            overlaps_dt_dc = image_box_overlap(dt_bboxes, ignored_bboxes, 0)
            for i in range(ignored_bboxes.shape[0]):
                for j in range(det_size):
                    if (assigned_detection[j]):
                        continue
                    if (ignored_det[j] == -1 or ignored_det[j] == 1):
                        continue
                    if (ignored_threshold[j]):
                        continue
                    if overlaps_dt_dc[j, i] > min_overlap:
                        assigned_detection[j] = True
                        nstuff += 1
        fp -= nstuff
        if compute_angular_metrics:
            tmp_yaw = np.zeros((fp + delta_idx, ))
            tmp_alpha = np.zeros((fp + delta_idx, ))
            for i in range(delta_idx):
                tmp_yaw[i + fp] = delta_yaw[i]
                tmp_alpha[i + fp] = (1.0 + np.cos(delta_alpha[i])) / 2.0

            if tp > 0 or fp > 0:
                error_yaw = np.sum(tmp_yaw)
                similarity = np.sum(tmp_alpha)
            else:
                error_yaw = -1
                similarity = -1

    return tp, fp, fn, error_yaw, similarity, thresholds[:thresh_idx], match_degree, confidence_error


@numba.jit(nopython=True, fastmath=True)
def fused_compute_statistics(
    overlaps,
    pr,
    gt_nums,
    dt_nums,
    dc_nums,
    gt_datas,
    dt_datas,
    dontcares,
    ignored_gts,
    ignored_dets,
    metric,
    min_overlap,
    thresholds,
    compute_angular_metrics=False,
):
    """Compute TP/FP statistics.
    Taken from https://github.com/sshaoehuai/PointRCNN/blob/master/tools/kitti_object_eval_python/eval.py
    without changes to avoid introducing errors"""

    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            # The key line that determines the ordering of the IoU matrix
            overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:gt_num + gt_nums[i]]

            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
            tp, fp, fn, error_yaw, similarity, _, match_degree, confidence_error = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_angular_metrics=compute_angular_metrics
            )
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            pr[t, 5] += match_degree
            pr[t, 6] += confidence_error
            if error_yaw != -1:
                pr[t, 3] += error_yaw
            if similarity != -1:
                pr[t, 4] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]
