#!/usr/bin/env python3

import json, sys
import cv2
import numpy as np

"""
rotation_flow.py:
Given a crop, compute rotation anomaly via:
- Farneback optical flow
- Curl (rotation component)
- PCA orientation
"""

def compute_rotation(flow):
    fx = flow[..., 0]
    fy = flow[..., 1]

    dvy_dx = cv2.Sobel(fy, cv2.CV_32F, 1, 0, ksize=3)
    dvx_dy = cv2.Sobel(fx, cv2.CV_32F, 0, 1, ksize=3)
    curl = dvy_dx - dvx_dy

    return float(np.mean(np.abs(curl)))


def compute_angle(prev, curr):
    diff = cv2.absdiff(prev, curr)
    _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return 0.0

    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    mean, eigen = cv2.PCACompute(pts, mean=None)
    v = eigen[0]
    angle = np.degrees(np.arctan2(v[1], v[0]))
    return float(angle)


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "no image"}))
        return

    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(json.dumps({"error": "invalid image"}))
        return

    prev = img[:, :-1]
    curr = img[:, 1:]

    flow = cv2.calcOpticalFlowFarneback(prev, curr, None,
                                        0.5, 3, 15, 3, 5, 1.1, 0)

    rot_mag = compute_rotation(flow)
    angle = compute_angle(prev, curr)

    # Normalize rotation
    rot_score = min(rot_mag / 3.0, 1.0)

    print(json.dumps({
        "angle": angle,
        "angular_velocity": rot_mag,
        "rotation_score": rot_score
    }))


if __name__ == "__main__":
    main()
