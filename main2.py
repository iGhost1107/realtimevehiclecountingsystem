import cv2
import yaml
import time
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import sys
import os

# Check if an argument was provided
if len(sys.argv) < 2:
    print("Usage: python3 main2.py <video_name>")
    sys.exit(1)

# Get the argument from the command line
video_name = sys.argv[1]

# --- 2. DEFINE VIDEO & SELECT REGION ---
# Using f-string to inject the variable into the path
video_path = f"/home/jonah-nguyen/BytetrackandYOLOv8/Dataset/{video_name}/{video_name}.mp4"

print(f"Processing: {video_path}")


# --- GLOBAL VARIABLES FOR DRAWING ---
drawing_points = []

def mouse_callback(event, x, y, flags, param):
    """Handles mouse clicks to store coordinates."""
    global drawing_points
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_points.append((x, y))
        print(f"✅ Point added: {(x, y)}")

def select_region(video_path):
    """Opens video, allows drawing ONE region, returns points."""
    global drawing_points
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error reading video file")

    success, frame = cap.read()
    if not success:
        raise ValueError("Could not read first frame")

    window_name = "Draw Region (Click 4 points -> ENTER)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    h, w = frame.shape[:2]
    if w > 1920: cv2.resizeWindow(window_name, 1280, int(h * 1280/w))

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

# --- 1. CONFIGURATION ---
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

# video_path = "/home/jonah-nguyen/BytetrackandYOLOv8/Dataset/MVI_39211/MVI_39211.mp4"

# --- 2. SETUP ---
region_poly = select_region(video_path)
print(f"🔒 Region Locked: {region_poly}")

# Load Standard YOLO
model = YOLO("best.pt")
class_names = model.names 

# --- 3. METRICS VARIABLES ---
entered_ids = set()           # Stores unique IDs that have entered
type_counts = defaultdict(int) # Stores counts per class (e.g., {'car': 5, 'bus': 1})
start_time = time.time()

# --- 4. PROCESSING LOOP ---
cap = cv2.VideoCapture(video_path)
VIDEO_FPS = cap.get(cv2.CAP_PROP_FPS)
if VIDEO_FPS == 0: VIDEO_FPS = 25 

print(f"🚀 Real-Time Analytics Started ({VIDEO_FPS} FPS Target)...")

while cap.isOpened():
    # --- A. SYNC & FRAME DROPPING ---
    elapsed_time = time.time() - start_time
    expected_frame_idx = int(elapsed_time * VIDEO_FPS)
    current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    frames_to_skip = expected_frame_idx - current_frame_idx
    if frames_to_skip > 0:
        for _ in range(frames_to_skip):
            cap.grab()

    # --- B. READ FRAME ---
    success, frame = cap.read()
    if not success: break
    
    t_process_start = time.time()

    # --- C. TRACKING ---
    results = model.track(frame, persist=True, tracker="custom_bytetrack.yaml", verbose=False)
    
    cv2.polylines(frame, [region_poly], isClosed=True, color=(255, 0, 0), thickness=2)

    # --- D. LOGIC ---
    current_occupancy = 0
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy() 

        for box, track_id, conf, cls_id in zip(boxes, ids, confidences, classes):
            x1, y1, x2, y2 = box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            # Check Logic: Is Center Inside?
            is_inside = cv2.pointPolygonTest(region_poly, (cx, cy), False)

            if is_inside >= 0:
                current_occupancy += 1
                
                # --- HISTORICAL COUNTING LOGIC ---
                # Only count if this ID has NEVER been seen inside before
                if track_id not in entered_ids:
                    entered_ids.add(track_id)
                    
                    # Get Class Name and Increment Counter
                    c_name = class_names[int(cls_id)]
                    type_counts[c_name] += 1
                
                # Visuals
                cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Label
                class_name = class_names[int(cls_id)]
                label = f"#{int(track_id)} {class_name} {conf:.2f}"
                t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
                c2 = int(x1) + t_size[0], int(y1) - t_size[1] - 3
                cv2.rectangle(frame, (int(x1), int(y1)), c2, (0, 255, 0), -1, cv2.LINE_AA) 
                cv2.putText(frame, label, (int(x1), int(y1) - 2), 0, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            else:
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

    # --- E. DASHBOARD VISUALIZATION ---
    # Expand overlay height dynamically based on how many car types we have
    base_height = 180
    extra_height = len(type_counts) * 30
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (400, base_height + extra_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    # 1. Standard Metrics
    cv2.putText(frame, f"OCCUPANCY: {current_occupancy}", (15, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.putText(frame, f"Total Unique: {len(entered_ids)}", (15, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    elapsed_minutes = elapsed_time / 60
    flow_rate = len(entered_ids) / elapsed_minutes if elapsed_minutes > 0.1 else 0
    cv2.putText(frame, f"Flow: {flow_rate:.1f} cars/min", (15, 125), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 2. SEPARATOR LINE
    cv2.line(frame, (15, 135), (380, 135), (255, 255, 255), 1)
    cv2.putText(frame, "TYPE BREAKDOWN:", (15, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # 3. Dynamic Type List
    y_offset = 180
    for c_name, count in type_counts.items():
        # Format: "car: 5"
        text = f"{c_name.upper()}: {count}"
        cv2.putText(frame, text, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2)
        y_offset += 30

    # 4. System FPS (Moved to bottom)
    process_time = time.time() - t_process_start
    system_fps = 1 / process_time if process_time > 0 else 0
    cv2.putText(frame, f"FPS: {system_fps:.1f}", (15, y_offset + 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow("Smart Analytics", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()