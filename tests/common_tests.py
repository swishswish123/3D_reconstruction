import numpy as np


def arrays_equal(ground_truth, acual, tolerance=5):
    assert len(ground_truth) == len(acual)

    for gt_row, actual_row in zip(ground_truth, acual):
        for gt_item, actual_item in zip(gt_row, actual_row):
            assert np.isclose(gt_item ,  actual_item, atol = tolerance) # x: actual (+/-tol)mm

