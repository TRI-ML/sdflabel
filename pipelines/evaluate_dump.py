import os
import glob
import numpy as np
import pickle
from collections import OrderedDict

import utils.data as config
from pipelines.detection_3d import Detection3DEvaluator, clean_kitti_data, CoordinateFrame


def evaluate(cfgp):
    """
    Evaluate generated dump
    Args:
        cfgp: Configuration parser
    """
    # Get input folder
    path_autolabels = config.read_cfg_string(cfgp, 'output', 'labels', default='')

    # KITTI benchmark
    gt_annotations, pred_annotations = OrderedDict(), OrderedDict()

    # Loop through autolabeling estimations
    for f in sorted(glob.glob(os.path.join(path_autolabels, '*.pkl'))):
        anno = pickle.load(open(f, "rb"))

        if 'skipped_frames' in f:
            continue

        base = os.path.basename(f)
        frame_id = int(base.split('.')[0])

        # Parse stuff
        gt = anno[0]
        estimations = anno[1]

        if 'name' not in estimations:
            estimations['name'] = []
            estimations['location'] = np.zeros((0, 3))
            estimations['dimensions'] = np.zeros((0, 3))
            estimations['bbox'] = np.zeros((0, 4))
            estimations['rotation_y'] = np.zeros((0, ))
            estimations['alpha'] = np.zeros((0, ))
            estimations['score'] = np.zeros((0, ))

        gt_annotations[frame_id] = gt
        pred_annotations[frame_id] = estimations

    # Evaluate KITTI metrics
    evaluator = Detection3DEvaluator(clean_kitti_data, compute_nuscenes=False, coordinate_frame=CoordinateFrame.CAMERA)
    formatted_result, result_dict = evaluator.evaluate_detection_3d(
        list(gt_annotations.values()), list(pred_annotations.values()), ['Car'], difficulties=[0, 1]
    )
    print(formatted_result)

    # Evaluate nuScenes metric
    evaluator = Detection3DEvaluator(clean_kitti_data, compute_nuscenes=True, coordinate_frame=CoordinateFrame.CAMERA)
    formatted_result, result_dict = evaluator.evaluate_detection_3d(
        list(gt_annotations.values()), list(pred_annotations.values()), ['Car'], difficulties=[0, 1]
    )

    print(formatted_result)
