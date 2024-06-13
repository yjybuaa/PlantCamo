import numpy as np
from py_sod_metrics.sod_metrics import Emeasure, Fmeasure, MAE, Smeasure, WeightedFmeasure
import os
import cv2

class CalTotalMetric(object):
    __slots__ = ["cal_mae", "cal_fm", "cal_sm", "cal_em", "cal_wfm"]

    def __init__(self):
        self.cal_mae = MAE()
        self.cal_fm = Fmeasure()
        self.cal_sm = Smeasure()
        self.cal_em = Emeasure()
        self.cal_wfm = WeightedFmeasure()

    def step(self, pred: np.ndarray, gt: np.ndarray):
        assert pred.ndim == gt.ndim and pred.shape == gt.shape
        assert pred.dtype == np.uint8, pred.dtype
        assert gt.dtype == np.uint8, gt.dtype

        self.cal_mae.step(pred, gt)
        self.cal_fm.step(pred, gt)
        self.cal_sm.step(pred, gt)
        self.cal_em.step(pred, gt)
        self.cal_wfm.step(pred, gt)

    def get_results(self, bit_width: int = 3) -> dict:
        fm = self.cal_fm.get_results()["fm"]
        wfm = self.cal_wfm.get_results()["wfm"]
        sm = self.cal_sm.get_results()["sm"]
        em = self.cal_em.get_results()["em"]
        mae = self.cal_mae.get_results()["mae"]
        results = {
            "Smeasure": sm,
            "wFmeasure": wfm,
            "MAE": mae,
            "adpEm": em["adp"],
            "meanEm": em["curve"].mean(),
            "maxEm": em["curve"].max(),
            "adpFm": fm["adp"],
            "meanFm": fm["curve"].mean(),
            "maxFm": fm["curve"].max(),
        }

        def _round_w_zero_padding(_x):
            _x = str(_x.round(bit_width))
            _x += "0" * (bit_width - len(_x.split(".")[-1]))
            return _x

        results = {name: _round_w_zero_padding(metric) for name, metric in results.items()}
        return results


if __name__ == '__main__':
    cal_total_seg_metrics = CalTotalMetric()

    img_path = './results1/PlantCAMO1250/PlantCAMO1250/'
    mask_path = './datasets/PlantCAMO1250/test/gt/'

    img_list = os.listdir(img_path)
    mae_sum = 0
    for img in img_list:
        print('=>{} eval start!'.format(img))
        pred_path = os.path.join(img_path, img)
        gt_path = os.path.join(mask_path, img)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        cal_total_seg_metrics.step(pred, gt)
        print('**{} eval finish!'.format(img))

    fixed_seg_results = cal_total_seg_metrics.get_results()
    print(fixed_seg_results)
    # with open("./results0/PlantCAMO1250/results.txt","a")as f :
    #     f.write(str(fixed_seg_results)+'\n')
