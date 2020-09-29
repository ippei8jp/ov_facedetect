import numpy as np

# ユーザ定義モジュール
from detector import Detector
from utils import cut_rois, resize_input

# 向き検出器クラス
class HeadposeDetector(Detector):
    # 結果クラス
    class Result:
        def __init__(self, outputs):
            self.rawResult = outputs
            
            self.pitch = outputs['angle_p_fc'][0][0]
            self.yaw   = outputs['angle_y_fc'][0][0]
            self.roll  = outputs['angle_r_fc'][0][0]
        
        def get_array(self):
            return np.array(self.rawResult, dtype=np.float64)
    
    def __init__(self, model):
        # 基底クラスのコンストラクタ呼び出し
        super(HeadposeDetector, self).__init__(model)
        
        # パラメータ等チェック
        assert len(model.inputs) == 1, "Expected 1 input blob"
        assert len(model.outputs) == 3, "Expected 3 output blob"
        
        self.input_blob = next(iter(model.inputs))
        self.output_blob = next(iter(model.outputs))
        self.input_shape = model.inputs[self.input_blob].shape
        
        assert np.array_equal([1, 1], model.outputs[self.output_blob].shape), \
            f"Expected model output shape {[1, 1]}, but got {model.outputs[self.output_blob].shape}"

    def preprocess(self, frame, rois):
        assert len(frame.shape) == 4, "Frame shape should be [1, c, h, w]"
        inputs = cut_rois(frame, rois)
        inputs = [resize_input(input, self.input_shape) for input in inputs]
        return inputs

    def enqueue(self, input):
        return super(HeadposeDetector, self).enqueue({self.input_blob: input})

    def start_async(self, frame, rois):
        inputs = self.preprocess(frame, rois)
        for input in inputs:
            self.enqueue(input)

    def get_headposes(self):
        outputs = self.get_outputs()
        results = [HeadposeDetector.Result(out) for out in outputs]
        return results
