#!/usr/bin/env python3
"""
deform_analyzer.py

Detects deformation focused on tires (or other circular parts).
Usage:
    python3 deform_analyzer.py prev_crop.png curr_crop.png

Outputs JSON to stdout:
{
  "deformation_score": 0.82,
  "method": "tire_roi",                # "tire_roi" or "global_fallback"
  "tire_roi": [x,y,w,h] or null,
  "ssim": 0.63,
  "edge_change": 0.48,
  "keypoint_ratio": 0.12,
  "circle_prev": 18.0,
  "circle_curr": 0.0,
  "kp_prev": 120,
  "kp_curr": 32
}

Dependencies:
    pip install opencv-python numpy scikit-image
"""
import sys, json, math
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read {path}")
    return img

def hough_circles(img):

    bl = cv2.medianBlur(img, 7)
    h, w = img.shape[:2]
    dp = 1.2
    minDist = max(12, min(w, h) // 6)
    minR = max(6, min(w, h) // 40)
    maxR = max(10, min(w, h) // 6)
    circles = cv2.HoughCircles(bl, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                               param1=100, param2=22, minRadius=minR, maxRadius=maxR)
    if circles is None:
        return []
    circles = np.round(circles[0, :]).astype(int)
    
    circles = sorted(circles.tolist(), key=lambda c: c[2], reverse=True)
    return circles

def contour_circular_candidates(img):

    h, w = img.shape[:2]
    area_thresh = (w*h) * 0.001
    half = h // 2
    roi = img[half:, :]
    
    edges = cv2.Canny(roi, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < area_thresh:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue
        circularity = 4 * math.pi * a / (perimeter * perimeter)
        if circularity > 0.5: 
            (x, y, cw, ch) = cv2.boundingRect(c)
            
            candidates.append((x, y+half, cw, ch, circularity))
    
    candidates = sorted(candidates, key=lambda t: (t[4], t[2]*t[3]), reverse=True)
    return candidates

def choose_tire_roi(prev, curr):
    
    pcir = hough_circles(prev)
    if len(pcir) > 0:
        x, y, r = pcir[0]
        
        margin = int(r * 1.4)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(prev.shape[1]-1, x + margin)
        y2 = min(prev.shape[0]-1, y + margin)
        return ("circle", (x1,y1,x2-x1,y2-y1), r, 0.0)


    ccir = hough_circles(curr)
    if len(ccir) > 0:
        x, y, r = ccir[0]
        margin = int(r * 1.4)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(prev.shape[1]-1, x + margin)
        y2 = min(prev.shape[0]-1, y + margin)
    
        return ("circle", (x1,y1,x2-x1,y2-y1), 0.0, r)


    candidates = contour_circular_candidates(prev)
    if len(candidates) > 0:
        x, y, cw, ch, circ = candidates[0]

        pad = int(max(cw, ch) * 0.25)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(prev.shape[1]-1, x + cw + pad)
        y2 = min(prev.shape[0]-1, y + ch + pad)
        return ("contour", (x1,y1,x2-x1,y2-y1), circ, 0.0)


    return (None, None, 0.0, 0.0)

def compute_ssim(a, b):
    try:
        s, _ = ssim(a, b, full=True, data_range=255)
    except Exception:
        s = 1.0
    return float(s)

def edge_change(a, b):
    e1 = cv2.Canny(a, 50, 150)
    e2 = cv2.Canny(b, 50, 150)
    diff = cv2.absdiff(e1, e2)
    changed = np.count_nonzero(diff)
    total = e1.size
    return float(changed) / float(total)

def keypoint_match_ratio(a, b, max_kp=800):
    orb = cv2.ORB_create(nfeatures=max_kp)
    k1, d1 = orb.detectAndCompute(a, None)
    k2, d2 = orb.detectAndCompute(b, None)
    n1 = 0 if d1 is None else len(d1)
    n2 = 0 if d2 is None else len(d2)
    if d1 is None or d2 is None or len(d1) < 4 or len(d2) < 4:
        return 0.0, n1, n2
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    if not matches:
        return 0.0, n1, n2
    matches = sorted(matches, key=lambda m: m.distance)
    good = sum(1 for m in matches if m.distance < 60)
    denom = max(1, min(len(d1), len(d2)))
    return float(good) / float(denom), n1, n2

def crop_and_resize(img, bbox, size):
    x,y,w,h = bbox
    x2 = min(img.shape[1], x+w)
    y2 = min(img.shape[0], y+h)
    if x>=img.shape[1] or y>=img.shape[0] or x2<=x or y2<=y:
        return None
    c = img[y:y2, x:x2]
    if c.size == 0:
        return None
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_LINEAR)

def global_metrics(prev, curr, size=512):
    
    p = cv2.resize(prev, (size,size), interpolation=cv2.INTER_LINEAR)
    c = cv2.resize(curr, (size,size), interpolation=cv2.INTER_LINEAR)
    s = compute_ssim(p,c)
    e = edge_change(p,c)
    kp, n1, n2 = keypoint_match_ratio(p,c)
    return s, e, kp, (n1,n2)

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error":"need prev curr"}))
        return
    prev_p = sys.argv[1]
    curr_p = sys.argv[2]
    prev = read_gray(prev_p)
    curr = read_gray(curr_p)

    
    if prev.shape != curr.shape:
        curr = cv2.resize(curr, (prev.shape[1], prev.shape[0]), interpolation=cv2.INTER_LINEAR)

    method, bbox, circle_prev_r, circle_curr_r = choose_tire_roi(prev, curr)

    metrics = {}
    if method is not None and bbox is not None:
        roi_prev = crop_and_resize(prev, bbox, 256)
        roi_curr = crop_and_resize(curr, bbox, 256)
        if roi_prev is None or roi_curr is None:
    
            s, e, kp, (n1,n2) = global_metrics(prev, curr)
            method = "global_fallback"
            metrics.update({"ssim": s, "edge_change": e, "keypoint_ratio": kp, "kp_prev": n1, "kp_curr": n2, "tire_roi": None})
        else:
            s = compute_ssim(roi_prev, roi_curr)
            e = edge_change(roi_prev, roi_curr)
            kp, n1, n2 = keypoint_match_ratio(roi_prev, roi_curr)
            metrics.update({"ssim": s, "edge_change": e, "keypoint_ratio": kp, "kp_prev": n1, "kp_curr": n2, "tire_roi": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]})
    else:
        s, e, kp, (n1,n2) = global_metrics(prev, curr)
        metrics.update({"ssim": s, "edge_change": e, "keypoint_ratio": kp, "kp_prev": n1, "kp_curr": n2, "tire_roi": None})
        method = "global_fallback"


    circ_prev = float(circle_prev_r if circle_prev_r is not None else 0.0)
    circ_curr = float(circle_curr_r if circle_curr_r is not None else 0.0)


    if circ_prev > 1.0 and circ_curr < 1.0:
        circle_loss = 1.0
    elif circ_prev < 1.0 and circ_curr < 1.0:
        circle_loss = 0.0
    else:
        if circ_prev > 0.0:
            circle_loss = max(0.0, (circ_prev - circ_curr) / circ_prev)
            circle_loss = float(np.clip(circle_loss, 0.0, 1.0))
        else:
            circle_loss = 0.0

    
    kp_loss = 1.0 - float(metrics["keypoint_ratio"])

    
    ssim_loss = 1.0 - float(metrics["ssim"])


    edge_ch = float(metrics["edge_change"])

    
    w_circle = 0.40
    w_ssim = 0.25
    w_kp = 0.20
    w_edge = 0.15

    deformation_score = w_circle * circle_loss + w_ssim * ssim_loss + w_kp * kp_loss + w_edge * edge_ch
    deformation_score = float(np.clip(deformation_score, 0.0, 1.0))

    out = {
        "deformation_score": deformation_score,
        "method": method,
        "tire_roi": metrics.get("tire_roi"),
        "ssim": float(metrics["ssim"]),
        "edge_change": float(metrics["edge_change"]),
        "keypoint_ratio": float(metrics["keypoint_ratio"]),
        "kp_prev": int(metrics.get("kp_prev", 0)),
        "kp_curr": int(metrics.get("kp_curr", 0)),
        "circle_prev": circ_prev,
        "circle_curr": circ_curr,
        "circle_loss": circle_loss,
        "kp_loss": kp_loss,
        "ssim_loss": ssim_loss
    }

    print(json.dumps(out))

if __name__ == "__main__":
    main()
