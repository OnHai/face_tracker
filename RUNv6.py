import math
import time
import cv2
import numpy as np
import vision.utils.box_utils_numpy as box_utils
import onnxruntime as ort
from sort import Sort
import subprocess
from picamera2 import Picamera2

# ============================================================================
# CONFIGURATION - All parameters in one place
# ============================================================================
# Network & Streaming
UDP_IP = "10.20.30.255"
UDP_PORT = 5005
STREAM_PROTOCOL = "tcp"  # "tcp" or "udp"

# Camera Settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_RATE = 15

# AI/Detection Settings
MODEL_PATH = "models/onnx/version-RFB-320-perfect.onnx"
CONFIDENCE_THRESHOLD = 0.7
IOU_THRESHOLD = 0.3
ONNX_THREADS = 2

# Targeting Settings
HFOV_DEGREES = 60.0
DEADZONE_DEGREES = 5

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes, confidences = boxes[0], confidences[0]
    picked_box_probs, picked_labels = [], []
    
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
    
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, [0, 2]] *= width
    picked_box_probs[:, [1, 3]] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


def prepare_frame(frame):
    """Convert camera frame to BGR format"""
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def preprocess_for_inference(image):
    """Prepare image for ONNX model"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 240))
    image = (image.astype(np.float32) - 127.0) / 128.0
    image = np.transpose(image, [2, 0, 1])
    return np.expand_dims(image, axis=0)


# ============================================================================
# INITIALIZATION
# ============================================================================
print("Initializing...")

# AI Model & Tracker
tracker = Sort()
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = ONNX_THREADS
ort_session = ort.InferenceSession(MODEL_PATH, sess_options)
input_name = ort_session.get_inputs()[0].name

# Calculate derived parameters
focal_length_px = (FRAME_WIDTH / 2.0) / math.tan(math.radians(HFOV_DEGREES / 2.0))
deadzone_pixels = int(math.tan(math.radians(DEADZONE_DEGREES)) * focal_length_px)
frame_center_x = FRAME_WIDTH // 2

# FFmpeg Stream
stream_url = f"{STREAM_PROTOCOL}://{'0.0.0.0' if STREAM_PROTOCOL == 'tcp' else UDP_IP}:{UDP_PORT}"
stream_url += "?listen" if STREAM_PROTOCOL == "tcp" else "?pkt_size=1316&broadcast=1"

ffmpeg_cmd = [
    'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24',
    '-s', f"{FRAME_WIDTH}x{FRAME_HEIGHT}", '-r', str(FRAME_RATE), '-i', '-',
    '-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency',
    '-pix_fmt', 'yuv420p', '-f', 'mpegts', stream_url
]
stream_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# Camera
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
picam2.configure(config)
picam2.start()

print(f">>> Streaming to {stream_url} <<<\n")

# ============================================================================
# MAIN LOOP
# ============================================================================
locked_target_id = None

try:
    while True:
        t0 = time.perf_counter()
        
        # Capture & prepare frame
        frame = picam2.capture_array()
        if frame is None: continue
        orig_image = prepare_frame(frame)
        
        # Run inference
        image = preprocess_for_inference(orig_image)
        confidences, boxes = ort_session.run(None, {input_name: image})
        boxes, labels, probs = predict(FRAME_WIDTH, FRAME_HEIGHT, confidences, boxes, CONFIDENCE_THRESHOLD, IOU_THRESHOLD)
        
        # Track objects
        dets = np.concatenate((boxes, probs.reshape(-1, 1)), axis=1) if boxes.shape[0] > 0 else np.empty((0, 5))
        tracked_objects = tracker.update(dets)
        
        # Draw deadzone markers
        cv2.line(orig_image, (frame_center_x - deadzone_pixels, 0), (frame_center_x - deadzone_pixels, FRAME_HEIGHT), (255, 0, 0), 1)
        cv2.line(orig_image, (frame_center_x + deadzone_pixels, 0), (frame_center_x + deadzone_pixels, FRAME_HEIGHT), (255, 0, 0), 1)
        
        # Target locking & drawing
        if len(tracked_objects) > 0:
            current_ids = [int(obj[4]) for obj in tracked_objects]
            if locked_target_id not in current_ids:
                locked_target_id = current_ids[0]
            
            for obj in tracked_objects:
                x1, y1, x2, y2, obj_id = [int(i) for i in obj]
                is_target = (obj_id == locked_target_id)
                color = (0, 0, 255) if is_target else (255, 0, 0)
                cv2.rectangle(orig_image, (x1, y1), (x2, y2), color, 2)
                
                if is_target:
                    target_x = (x1 + x2) // 2
                    error_x = target_x - frame_center_x
                    angle_deg = math.degrees(math.atan(error_x / focal_length_px))
                    
                    pivot = (frame_center_x, FRAME_HEIGHT - 40)
                    arrow_col = (0, 255, 0) if abs(angle_deg) < DEADZONE_DEGREES else (0, 0, 255)
                    visual_angle = 270 + angle_deg
                    end_x = int(pivot[0] + 60 * math.cos(math.radians(visual_angle)))
                    end_y = int(pivot[1] + 60 * math.sin(math.radians(visual_angle)))
                    cv2.arrowedLine(orig_image, pivot, (end_x, end_y), arrow_col, 3)
        
        # Stream frame
        try:
            stream_proc.stdin.write(orig_image.tobytes())
        except Exception as e:
            print(f"Stream error: {e}")
            break
        
        print(f"FPS: {1.0 / (time.perf_counter() - t0):.1f}", end='\r')

except KeyboardInterrupt:
    print("\n\nStopping...")
finally:
    picam2.stop()
    stream_proc.stdin.close()
    stream_proc.wait()