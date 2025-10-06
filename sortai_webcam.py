#!/usr/bin/env python3
"""
Eco-Sort AI - Webcam Detection Script

This script opens your webcam, runs a YOLOv8 model, and detects plastic types
(PET, HDPE, PVC, LDPE, PP, PS).

Usage:
    python sortai_webcam.py --weights yolov8n.pt --source 0

Dependencies:
    pip install ultralytics opencv-python numpy
"""

import argparse
import time
import cv2
import numpy as np
from collections import deque, defaultdict
from ultralytics import YOLO

CLASS_MAP = {0:'PET',1:'HDPE',2:'PVC',3:'LDPE',4:'PP',5:'PS'}

class RollingConfidence:
    def __init__(self, window=5):
        self.window = window
        self.buffers = defaultdict(lambda: deque(maxlen=self.window))
    def add(self, cls, conf):
        self.buffers[cls].append(conf)
    def mean(self, cls):
        b = self.buffers[cls]
        return sum(b)/len(b) if b else 0.0

def process_results(results, conf_thresh=0.4):
    dets = []
    if not hasattr(results, 'boxes'): return dets
    for box in results.boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        if conf < conf_thresh: continue
        label = CLASS_MAP.get(cls, str(cls))
        dets.append({'xyxy':xyxy,'conf':conf,'cls':cls,'label':label})
    return dets

def draw_boxes(frame, dets):
    for d in dets:
        x1,y1,x2,y2 = map(int,d['xyxy'])
        color = (0,255,0)
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame,f"{d['label']} {d['conf']:.2f}",(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    return frame

def main(args):
    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("Camera not found!"); return
    rolling = RollingConfidence(window=6)
    print("Running... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        results = model(frame)[0]
        dets = process_results(results, args.conf_thresh)
        for d in dets: rolling.add(d['cls'], d['conf'])
        for d in dets:
            avg = rolling.mean(d['cls'])
            if avg > 0.5:
                print(f"[ACTUATE] Detected {d['label']} (avg conf {avg:.2f})")
        frame = draw_boxes(frame, dets)
        cv2.imshow("Eco-Sort AI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="yolov8n.pt")
    p.add_argument("--source", type=int, default=0)
    p.add_argument("--conf_thresh", type=float, default=0.4)
    args = p.parse_args()
    main(args)
