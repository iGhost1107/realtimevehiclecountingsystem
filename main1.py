import cv2
import yaml
import time
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import sys
import os

# --- 1. CONFIGURATION & SOURCE ---
# Set webcam index (0 is usually the built-in webcam)
WEBCAM_INDEX = 0 

tracker_config = {
    'tracker_type': 'bytetrack',
    'track_high_thresh': 0.5,
    'track_low_thresh': 0.1,
    'new_track_thresh': 0.6,
    'track_buffer': 75,
    'match_thresh': 0.8,
    'gmc_method': 'sparseOptFlow',
    'fuse_score': True,
}
with open('custom_bytetrack.yaml', 'w') as f:
    yaml.dump(tracker_config, f)

# --- 2. GLOBAL VARIABLES & SELECTION ---
drawing_points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_points.append((x, y))
        print(f"✅ Point added: {(x, y)}")

def select_region(source):
    global drawing_points
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError("Error opening webcam")

    success, frame = cap.read()
    if not success:
        raise ValueError("Could not read frame from webcam")

    window_name = "Draw Region (Click 4 points -> ENTER)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("\n--- INSTRUCTIONS ---")
    print("🖱️  LEFT CLICK: Add point")
    print("cx  PRESS 'c' : Clear points")
    print("⏎  PRESS ENTER: Confirm Region & Start")
    print("--------------------\n")

    while True:
        temp_img = frame.copy()
        if len(drawing_points) > 0:
            for pt in drawing_points: cv2.circle(temp_img, pt, 5, (0, 0, 255), -1)
            if len(drawing_points) > 1:
                cv2.polylines(temp_img, [np.array(drawing_points)], isClosed=False, color=(0, 255, 0), thickness=2)
            if len(drawing_points) > 2:
                 cv2.polylines(temp_img, [np.array(drawing_points)], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imshow(window_name, temp_img)
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == 32: # Enter/Space
            if len(drawing_points) < 3:
                print("⚠️  Region needs at least 3 points!")
                continue
            break
        elif key == ord('c'):
            drawing_points.clear()

    cv2.destroyWindow(window_name)
    cap.release()
    return np.array(drawing_points, np.int32)

# Initialize Region Selection
region_poly = select_region(WEBCAM_INDEX)

# Load Model
model = YOLO("best.pt")
class_names = model.names 

# --- 3. METRICS VARIABLES ---
entered_ids = set()
type_counts = defaultdict(int)
start_time = time.time()

# --- 4. LIVE PROCESSING LOOP ---
cap = cv2.VideoCapture(WEBCAM_INDEX)
print("🚀 Webcam Analytics Started...")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    t_process_start = time.time()

    # --- TRACKING ---
    # We use persist=True for ByteTrack to maintain IDs across frames
    results = model.track(frame, persist=True, tracker="custom_bytetrack.yaml", verbose=False)
    
    cv2.polylines(frame, [region_poly], isClosed=True, color=(255, 0, 0), thickness=2)

    current_occupancy = 0
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy() 

        for box, track_id, conf, cls_id in zip(boxes, ids, confidences, classes):
            x1, y1, x2, y2 = box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            # Logic: Is Center Inside?
            is_inside = cv2.pointPolygonTest(region_poly, (cx, cy), False)

            if is_inside >= 0:
                current_occupancy += 1
                
                if track_id not in entered_ids:
                    entered_ids.add(track_id)
                    c_name = class_names[int(cls_id)]
                    type_counts[c_name] += 1
                
                # Visuals
                cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                label = f"#{int(track_id)} {class_names[int(cls_id)]} {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 5), 0, 0.5, (0, 255, 0), 1)
            else:
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

    # --- DASHBOARD ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (400, 200 + (len(type_counts)*30)), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    cv2.putText(frame, f"OCCUPANCY: {current_occupancy}", (15, 50), 0, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, f"Total Unique: {len(entered_ids)}", (15, 90), 0, 0.7, (200, 200, 200), 2)
    
    # Timing
    current_elapsed = time.time() - start_time
    elapsed_minutes = current_elapsed / 60
    flow_rate = len(entered_ids) / elapsed_minutes if elapsed_minutes > 0.05 else 0
    cv2.putText(frame, f"Flow: {flow_rate:.1f} units/min", (15, 125), 0, 0.7, (0, 255, 255), 2)

    # Type Breakdown
    y_offset = 180
    for c_name, count in type_counts.items():
        cv2.putText(frame, f"{c_name.upper()}: {count}", (30, y_offset), 0, 0.7, (255, 200, 100), 2)
        y_offset += 30

    # System Performance
    fps = 1 / (time.time() - t_process_start)
    cv2.putText(frame, f"System FPS: {fps:.1f}", (15, y_offset + 20), 0, 0.5, (0, 0, 255), 1)

    cv2.imshow("Webcam Live Analytics", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()