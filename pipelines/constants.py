# Copyright 2019 Toyota Research Institute.  All rights reserved.
import numpy as np

# Default ontology for KITTI
KITTI_CLASS_NAMES = {
    0: "Cyclist",
    1: "Van",
    2: "Car",
    3: "Truck",
    4: "Pedestrian",
    5: "Person_sitting",
    6: "Tram",
}

# KITTI use IOU threshold for gt matching, here the threshold level array is with shape [num_difficulties, num_Classes].
# KITTI uses three levels of difficulties by default.
KITTI_OVERLAP_MODERATE = np.array([[0.5, 0.7, 0.7, 0.5, 0.5, 0.7, 0.5], [0.5, 0.7, 0.7, 0.5, 0.5, 0.7, 0.5],
                                   [0.5, 0.7, 0.7, 0.5, 0.5, 0.7, 0.5]])
KITTI_OVERLAP_EASY_2D = np.array([[0.5, 0.7, 0.7, 0.5, 0.5, 0.5, 0.5], [0.5, 0.7, 0.7, 0.5, 0.5, 0.5, 0.5],
                                [0.5, 0.7, 0.7, 0.5, 0.5, 0.5, 0.5]])
KITTI_OVERLAP_EASY_BEV = np.array([[0.25, 0.5, 0.5, 0.25, 0.25, 0.5, 0.5], [0.25, 0.5, 0.5, 0.25, 0.25, 0.5, 0.5],
                                   [0.25, 0.5, 0.5, 0.25, 0.25, 0.5, 0.5]])
KITTI_OVERLAP_EASY_3D = np.array([[0.25, 0.5, 0.5, 0.25, 0.25, 0.5, 0.5], [0.25, 0.5, 0.5, 0.25, 0.25, 0.5, 0.5],
                                [0.25, 0.5, 0.5, 0.25, 0.25, 0.5, 0.5]])

# Create threshold array for two levels of threshold for each metric class. [2, 3, 7] -> [Thresholds, Difficulties,
# Classes]
KITTI_OVERLAPs_2D = np.stack([KITTI_OVERLAP_MODERATE, KITTI_OVERLAP_EASY_2D], axis=0)
KITTI_OVERLAPs_BEV = np.stack([KITTI_OVERLAP_MODERATE, KITTI_OVERLAP_EASY_BEV], axis=0)
KITTI_OVERLAPs_3D = np.stack([KITTI_OVERLAP_MODERATE, KITTI_OVERLAP_EASY_3D], axis=0)

# Create threshold array by combining subarrays for each metric. [4, 2, 3, 7] -> [Metric_types, Thresholds,
# Difficulties, Classes]
KITTI_OVERLAP_THRESHOLDS = np.stack([KITTI_OVERLAPs_2D, KITTI_OVERLAPs_BEV, KITTI_OVERLAPs_3D, KITTI_OVERLAPs_3D], axis=0)


# NuScenes use distance thresh for gt matching,
# here the threshold level array is with shape [num_difficulties, num_Classes] for each threshold level.
NU_OVERLAP_MODERATE = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])

NU_OVERLAP_EASY = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])

# [2, 3, 7] -> [Thresholds, Difficulties, Classes]
NU_OVERLAP = np.stack([NU_OVERLAP_MODERATE, NU_OVERLAP_EASY], axis=0)
# [4, 2, 3, 7] -> [Metric_types, Thresholds, Difficulties, Classes]
NU_OVERLAP_THRESHOLDS = np.stack([NU_OVERLAP, NU_OVERLAP, NU_OVERLAP, NU_OVERLAP], axis=0)