import os
import sys
import time
import logging as log

import numpy as np

from openvino.inference_engine import IECore

# ユーザ定義モジュール
from face_detector import FaceDetector
from landmarks_detector import LandmarksDetector
from headpose_detector import HeadposeDetector

class FrameProcessor:
    # 同時に検出できる数
    QUEUE_SIZE = 16
    
    def __init__(self, args):
        # 追加検出処理フラグ
        self.lm_enabe = False
        self.hp_enabe = False
        
        # 推論エンジン
        self.iecore = IECore()

        # 推論に使用するデバイス一覧
        used_devices = set([args.d_fd, args.d_lm, args.d_hp])
        
        # puluginのロード
        start_time = time.time()
        log.info(f"Loading plugins for devices: {used_devices}")
        
        if 'CPU' in used_devices and not len(args.cpu_lib) == 0:
            log.info(f"Using CPU extensions library '{args.cpu_lib}'")
            assert os.path.isfile(cpu_ext), "Failed to open CPU extensions library"
            self.iecore.add_extension(args.cpu_lib, "CPU")
        
        if 'GPU' in used_devices and not len(args.gpu_lib) == 0:
            log.info(f"Using GPU extensions library '{args.gpu_lib}'")
            assert os.path.isfile(gpu_ext), "Failed to open GPU definitions file"
            self.iecore.set_config({"CONFIG_FILE": gpu_ext}, "GPU")
        
        log.info(f"Plugins are loaded.    loading time : {time.time()- start_time:.4f}sec")
        
        for d in used_devices:
            self.iecore.set_config({"PERF_COUNT": "YES" if args.perf_stats else "NO"}, d)
        
        # モデルのロード
        log.info("Loading models")
        if (args.m_fd) :
            log.info("    Face Detect model")
            face_detector_net = self.load_model(args.m_fd)
            self.face_detector = FaceDetector(face_detector_net,
                                              confidence_threshold=args.t_fd,
                                              roi_scale_factor=args.exp_r_fd)
            self.face_detector.deploy(args.d_fd, self.iecore)
        else :
            # 顔検出モデルが指定されていなければエラー
            log.error("--m-fd option is mandatory")
            raise RuntimeError("--m-fd option is mandatory")
        
        if (args.m_lm) :
            log.info("    Face Landmark model")
            self.lm_enabe = True
            landmarks_net = self.load_model(args.m_lm)
            self.landmarks_detector = LandmarksDetector(landmarks_net)
            self.landmarks_detector.deploy(args.d_lm, self.iecore, queue_size=self.QUEUE_SIZE)
        
        if (args.m_hp) :
            log.info("    Head pose model")
            self.hp_enabe = True
            headpose_net = self.load_model(args.m_hp)
            self.headpose_detector = HeadposeDetector(headpose_net)
            self.headpose_detector.deploy(args.d_hp, self.iecore, queue_size=self.QUEUE_SIZE)
        
        log.info("Models are loaded")
    
    # IR(Intermediate Representation ;中間表現)ファイル(.xml & .bin) の読み込み
    def load_model(self, model_path):
        start_time = time.time()                                # ロード時間測定用
        model_path = os.path.abspath(model_path)
        model_description_path = model_path
        model_weights_path = os.path.splitext(model_path)[0] + ".bin"
        log.info(f"    Loading the model from '{model_description_path}'")
        assert os.path.isfile(model_description_path), \
            f"Model description is not found at '{model_description_path}'"
        assert os.path.isfile(model_weights_path), \
            f"Model weights are not found at '{model_weights_path}'"
        
        model = self.iecore.read_network(model=model_description_path, weights=model_weights_path)
        log.info(f"    Model is loaded    loading time : {time.time()- start_time:.4f}sec")
        return model
    
    # フレーム毎の処理
    def process(self, frame):
        assert len(frame.shape) == 3,    "Expected input frame in (H, W, C) format"
        assert frame.shape[2] in [3, 4], "Expected BGR or BGRA input"
        
        orig_image = frame.copy()
        frame = frame.transpose((2, 0, 1)) # HWC to CHW
        frame = np.expand_dims(frame, axis=0)
        
        self.face_detector.clear()
        if self.lm_enabe :
            self.landmarks_detector.clear()
        if self.hp_enabe :
            self.headpose_detector.clear()
        
        # log.info("Face Detect")
        # 認識処理
        self.face_detector.start_async(frame)
        
        # 結果取得
        rois = self.face_detector.get_roi_proposals(frame)
        
        # 認識された顔が多すぎる場合は最大値まで縮小する
        if self.QUEUE_SIZE < len(rois):
            log.warning("Too many faces for processing." \
                    " Will be processed only {self.QUEUE_SIZE} of {len(rois)}.")
            rois = rois[:self.QUEUE_SIZE]
        
        # 特徴点検出
        if self.lm_enabe :
            # log.info("Landmarks")
            # 認識処理
            self.landmarks_detector.start_async(frame, rois)
            # 結果取得
            landmarks = self.landmarks_detector.get_landmarks()
        else :
            landmarks = [None] * len(rois)
        
        # 向き検出
        if self.hp_enabe :
            # log.info("Headpose")
            # 認識処理
            self.headpose_detector.start_async(frame, rois)
            # 結果取得
            headposes = self.headpose_detector.get_headposes()
        else :
            headposes = [None] * len(rois)
        
        return rois, landmarks, headposes
    
    def get_performance_stats(self):
        stats = {'face_detector': self.face_detector.get_performance_stats()}
        if self.lm_enabe :
            stats['landmarks'] = self.landmarks_detector.get_performance_stats()
        if self.hp_enabe :
            stats['headpose'] = self.headpose_detector.get_performance_stats()
        return stats
