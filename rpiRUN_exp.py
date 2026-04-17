import math
import time
import cv2
import numpy as np
import vision.utils.box_utils_numpy as box_utils
import onnxruntime as ort
from sort import Sort
import subprocess
from picamera2 import Picamera2

# --- CONFIGURATION ---
#UDP_IP = "10.42.0.255" 
UDP_IP = "10.20.30.255"  
UDP_PORT = 5005
X, Y = 640, 480
threshold = 0.7 
deadzone = 5 
HFOV_degrees = 60.0  

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0: continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs, iou_threshold=iou_threshold, top_k=top_k)
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs: return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

# --- INIT AI & TRACKER ---
print("Initializing ONNX and SORT Tracker...")
tracker = Sort()
onnx_path = "models/onnx/version-RFB-320-perfect.onnx"
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 2
ort_session = ort.InferenceSession(onnx_path, sess_options)
input_name = ort_session.get_inputs()[0].name
locked_target_id = None
focal_length_px = (X / 2.0) / math.tan(math.radians(HFOV_degrees) / 2.0)

# --- INIT FFmpeg PIPE ---
# Optimized for Raspberry Pi -> PC Broadcast

"""
ffmpeg_cmd = [
    'ffmpeg',
    '-y', 
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', f"{X}x{Y}",
    '-r', '15', 
    '-i', '-', 
    '-c:v', 'libx264',
    '-preset', 'ultrafast',
    '-tune', 'zerolatency',
    '-pix_fmt', 'yuv420p',
    '-f', 'mpegts',
    f'udp://{UDP_IP}:{UDP_PORT}?pkt_size=1316&broadcast=1' 
]
"""

ffmpeg_cmd = [
    'ffmpeg',
    '-y', 
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', f"{X}x{Y}",
    '-r', '15', 
    '-i', '-', 
    '-c:v', 'libx264',
    '-preset', 'ultrafast',
    '-tune', 'zerolatency',
    '-pix_fmt', 'yuv420p',
    '-f', 'mpegts',
    f'tcp://0.0.0.0:{UDP_PORT}?listen' 
]
stream_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# --- INIT CAMERA ---
print("Starting Camera (OV9281 detected)...")
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (X, Y)})
picam2.configure(config)
picam2.start()

print(f">>> Streaming to {UDP_IP}:{UDP_PORT} <<<")

try:
    while True:
        t0 = time.perf_counter()
        
        # 1. CAPTURE
        frame = picam2.capture_array()
        if frame is None: continue
        
        # Handle OV9281 Mono vs Color formats
        if len(frame.shape) == 2:
            orig_image = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            orig_image = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            orig_image = frame
        
        # 2. AI PREP (Fixed expand_dims here)
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 240))
        image = (image.astype(np.float32) - 127.0) / 128.0
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0) # FIXED
        
        # 3. INFERENCE & TRACKING
        confidences, boxes = ort_session.run(None, {input_name: image})
        boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
        
        dets = np.concatenate((boxes, probs.reshape(-1, 1)), axis=1) if boxes.shape[0] > 0 else np.empty((0, 5))
        tracked_objects = tracker.update(dets)

        # 4. DRAWING LOGIC
        frame_center_x = X // 2
        deadzone_pixels = int(math.tan(math.radians(deadzone)) * focal_length_px)
        
        # Draw deadzone markers
        cv2.line(orig_image, (frame_center_x - deadzone_pixels, 0), (frame_center_x - deadzone_pixels, Y), (255, 0, 0), 1)
        cv2.line(orig_image, (frame_center_x + deadzone_pixels, 0), (frame_center_x + deadzone_pixels, Y), (255, 0, 0), 1)

        if len(tracked_objects) > 0:
            # Simple Target Locking
            current_ids = [int(obj[4]) for obj in tracked_objects]
            if locked_target_id not in current_ids:
                locked_target_id = current_ids[0]

            for obj in tracked_objects:
                x1, y1, x2, y2, obj_id = [int(i) for i in obj]
                is_target = (obj_id == locked_target_id)
                color = (0, 0, 255) if is_target else (255, 0, 0)
                
                cv2.rectangle(orig_image, (x1, y1), (x2, y2), color, 2)
                
                if is_target:
                    # Logic for Arrow/Targeting
                    target_x = int((x1 + x2) / 2)
                    error_x = target_x - frame_center_x
                    angle_deg = math.degrees(math.atan(error_x / focal_length_px))
                    
                    pivot = (frame_center_x, Y - 40)
                    arrow_col = (0, 255, 0) if abs(angle_deg) < deadzone else (0, 0, 255)
                    visual_angle = 270 + angle_deg
                    end_x = int(pivot[0] + 60 * math.cos(math.radians(visual_angle)))
                    end_y = int(pivot[1] + 60 * math.sin(math.radians(visual_angle)))
                    cv2.arrowedLine(orig_image, pivot, (end_x, end_y), arrow_col, 3)

        # 5. SEND TO STREAM PIPE
        try:
            stream_proc.stdin.write(orig_image.tobytes())
        except Exception as e:
            print(f"Stream Pipe Error: {e}")
            break

        # Display performance stats in console
        fps = 1.0 / (time.perf_counter() - t0)
        print(f"Processing... FPS: {fps:.1f}", end='\r')

except KeyboardInterrupt:
    print("\nStopping Drone Feed...")
finally:
    picam2.stop()
    stream_proc.stdin.close()
    stream_proc.wait()