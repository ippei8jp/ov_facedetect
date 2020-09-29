import os
import sys
import time
import math
import logging as log

import cv2
import numpy as np

class Visualizer:
    BREAK_KEY_LABELS = "q(Q) or Escape"
    BREAK_KEYS = {ord('q'), ord('Q'), 27}
    
    def __init__(self, args):
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.frame_count = -1
        
        self.crop_size = args.crop
        
        self.frame_timeout = 0 if args.timelapse else 1
    
    def terminete(self) :
        cv2.destroyAllWindows()
    
    def crop_frame(self, frame) :
        if self.crop_size is not None:
            frame = self.center_crop(frame, self.crop_size)
        return frame
    
    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now
    
    def draw_text_with_background(self, frame, text, origin,
                                  font=cv2.FONT_HERSHEY_SIMPLEX, scale=1.0,
                                  color=(0, 0, 0), thickness=1, bgcolor=(255, 255, 255)):
        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(frame,
                      tuple((origin + (0, baseline)).astype(int)),
                      tuple((origin + (text_size[0], -text_size[1])).astype(int)),
                      bgcolor, cv2.FILLED)
        cv2.putText(frame, text,
                    tuple(origin.astype(int)),
                    font, scale, color, thickness)
        return text_size, baseline
    
    def draw_detection_roi(self, frame, roi, label, color1, txtcolor=(0,0,0)):
        cv2.rectangle(frame, tuple(roi.position), tuple(roi.position + roi.size), color1, 2)
        
        if label :
            # Draw label
            text_scale = 0.5
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize("H1", font, text_scale, 1)
            line_height = np.array([0, text_size[0][1]])
            self.draw_text_with_background(frame, label,
                                           roi.position - line_height * 0.5,
                                           font, scale=text_scale,
                                           color=txtcolor, bgcolor=color1)
    
    def draw_detection_landmarks(self, frame, roi, landmarks):
        keypoints = [landmarks.left_eye,
                     landmarks.right_eye,
                     landmarks.nose_tip,
                     landmarks.left_lip_corner,
                     landmarks.right_lip_corner]
        
        for point in keypoints:
            center = roi.position + roi.size * point
            cv2.circle(frame, tuple(center.astype(int)), 2, (0, 255, 255), 2)
    
    def draw_detection_headpose(self, frame, roi, headpose):
        cpoint = roi.position + roi.size / 2
        yaw   = headpose.yaw   * np.pi / 180.0
        pitch = headpose.pitch * np.pi / 180.0
        roll  = headpose.roll  * np.pi / 180.0
        
        yawMatrix = np.matrix([[math.cos(yaw), 0, -math.sin(yaw)], [0, 1, 0], [math.sin(yaw), 0, math.cos(yaw)]])                    
        pitchMatrix = np.matrix([[1, 0, 0],[0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
        rollMatrix = np.matrix([[math.cos(roll), -math.sin(roll), 0],[math.sin(roll), math.cos(roll), 0], [0, 0, 1]])                    
        
        #Rotational Matrix
        R = yawMatrix * pitchMatrix * rollMatrix
        rows=frame.shape[0]
        cols=frame.shape[1]
        
        cameraMatrix=np.zeros((3,3), dtype=np.float32)
        cameraMatrix[0][0]= 950.0
        cameraMatrix[0][2]= cols/2
        cameraMatrix[1][0]= 950.0
        cameraMatrix[1][1]= rows/2
        cameraMatrix[2][1]= 1
        
        xAxis=np.zeros((3,1), dtype=np.float32)
        xAxis[0]=50
        xAxis[1]=0
        xAxis[2]=0
        
        yAxis=np.zeros((3,1), dtype=np.float32)
        yAxis[0]=0
        yAxis[1]=-50
        yAxis[2]=0
        
        zAxis=np.zeros((3,1), dtype=np.float32)
        zAxis[0]=0
        zAxis[1]=0
        zAxis[2]=-50
        
        zAxis1=np.zeros((3,1), dtype=np.float32)
        zAxis1[0]=0
        zAxis1[1]=0
        zAxis1[2]=50
        
        o=np.zeros((3,1), dtype=np.float32)
        o[2]=cameraMatrix[0][0]
        
        xAxis=R*xAxis+o
        yAxis=R*yAxis+o
        zAxis=R*zAxis+o
        zAxis1=R*zAxis1+o
        
        p2x=int((xAxis[0]/xAxis[2]*cameraMatrix[0][0])+cpoint[0])
        p2y=int((xAxis[1]/xAxis[2]*cameraMatrix[1][0])+cpoint[1])
        cv2.line(frame,(cpoint[0],cpoint[1]),(p2x,p2y),(0,0,255),2)
        
        p2x=int((yAxis[0]/yAxis[2]*cameraMatrix[0][0])+cpoint[0])
        p2y=int((yAxis[1]/yAxis[2]*cameraMatrix[1][0])+cpoint[1])
        cv2.line(frame,(cpoint[0],cpoint[1]),(p2x,p2y),(0,255,0),2)
        
        p1x=int((zAxis1[0]/zAxis1[2]*cameraMatrix[0][0])+cpoint[0])
        p1y=int((zAxis1[1]/zAxis1[2]*cameraMatrix[1][0])+cpoint[1])
        
        p2x=int((zAxis[0]/zAxis[2]*cameraMatrix[0][0])+cpoint[0])
        p2y=int((zAxis[1]/zAxis[2]*cameraMatrix[1][0])+cpoint[1])
        
        cv2.line(frame,(p1x,p1y),(p2x,p2y),(255,0,0),2)
        cv2.circle(frame,(p2x,p2y),3,(255,0,0))
        
    def draw_detections(self, frame, rois, landmarks, headposes):
        for roi, landmark, headpose in zip(rois, landmarks, headposes) :
            # Draw face ROI border
            label    = None
            color1   = (  0,   0, 220)
            txtcolor = (255, 255, 255)
            self.draw_detection_roi(frame, roi, label, color1, txtcolor)
            if landmark :
                self.draw_detection_landmarks(frame, roi, landmark)
            if headpose :
                self.draw_detection_headpose(frame, roi, headpose)
    
    def draw_status(self, frame, detections, frame_number):
        origin = np.array([10, 10])
        color = (10, 160, 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        text_size, _ = self.draw_text_with_background(frame,
                                        f"Frame time: {self.frame_time:.3f}s",
                                        origin, font, text_scale, color)
        self.draw_text_with_background(frame,
                                        f"FPS: {self.fps:.1f}",
                                        (origin + (0, text_size[1] * 1.5)), font, text_scale, color)
        
        log.debug(f'Frame: {frame_number}/{self.frame_count}, ' \
                  f'frame time: {self.frame_time:.3f}s, ' \
                  f'fps: {self.fps:.1f}')
    
    def display_interactive_window(self, frame):
        """
        # 中断方法の表示
        color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        text = "Press '%s' key to exit" % (self.BREAK_KEY_LABELS)
        thickness = 2
        text_size = cv2.getTextSize(text, font, text_scale, thickness)
        origin = np.array([frame.shape[-2] - text_size[0][0] - 10, 10])
        line_height = np.array([0, text_size[0][1]]) * 1.5
        cv2.putText(frame, text,
                    tuple(origin.astype(int)), font, text_scale, color, thickness)
        """
        
        cv2.imshow('Face recognition demo', frame)
    
    def should_stop_display(self, fevr=False) :
        if fevr :
            # wait forever
            key = cv2.waitKey(0) & 0xFF
        else :
            key = cv2.waitKey(self.frame_timeout) & 0xFF
        return key in self.BREAK_KEYS
    
    
    def center_crop(self, frame, crop_size):
        fh, fw, fc = frame.shape
        crop_size[0] = min(fw, crop_size[0])
        crop_size[1] = min(fh, crop_size[1])
        return frame[(fh - crop_size[1]) // 2 : (fh + crop_size[1]) // 2,
                     (fw - crop_size[0]) // 2 : (fw + crop_size[0]) // 2,
                     :]
    
    def open_input_stream(self, path):
        log.info(f"Reading input data from {path}")
        p = path
        try:
            # pathは数字(カメラ指定)？
            p = int(path)
        except ValueError:
            # 数字でなければ絶対パスに変換
            p = os.path.abspath(path)
        
        # ファイル/カメラをオープン
        stream = cv2.VideoCapture(p)
        
        # get fps/frame size/total frames
        self.fps         =      stream.get(cv2.CAP_PROP_FPS)
        self.frame_count =  int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_size  = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        # cropping options
        if self.crop_size is not None:
            self.frame_size = tuple(np.minimum(self.frame_size, self.crop_size))
        
        log.info(f"Input stream info: {self.frame_size[0]} x {self.frame_size[1]} @ {self.fps:.2f} FPS")
        
        return stream
    
    def open_output_stream(self, path):
        output_stream = None
        if path :
            forcc = cv2.VideoWriter.fourcc(*'mp4v') if path.endswith('.mp4') else cv2.VideoWriter.fourcc(*'MJPG')
            log.info(f"Writing output to '{path}'")
            output_stream = cv2.VideoWriter(path, forcc, self.fps, self.frame_size)
        return output_stream




