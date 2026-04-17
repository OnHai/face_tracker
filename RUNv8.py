# debugging
RUN_TARGET = "rpi"  # "rpi" or "pc"
STREAM_PROTOCOL = "tcp"  # "tcp" or "udp" or "none"
PC_VIDEO_PATH = r"sim_davu.mp4"


import math
import time
import cv2
import numpy as np
import vision.utils.box_utils_numpy as box_utils
import onnxruntime as ort # type: ignore
from sort import Sort
import subprocess

if RUN_TARGET == "rpi":
    from picamera2 import Picamera2 # type: ignore

# ============================================================================

# Network & Streaming
IP = "0.0.0.0"
PORT = 5005

# Camera Settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_RATE = 20

# Video Quality 
VIDEO_QUALITY = 28  # lower = better quality
VIDEO_PRESET = "ultrafast"  # ultrafast, superfast, veryfast
BITRATE = "500k"  

# Detection Settings
MODEL_PATH = "models/onnx/version-RFB-320-perfect.onnx"
CONFIDENCE_THRESHOLD = 0.7
ONNX_THREADS = 2

IOU_THRESHOLD = 0.1
MAX_AGE = 5
MIN_HITS = 3

# Targeting Settings
HFOV_DEGREES = 60.0
DEADZONE_DEGREES = 5

# Reconnection Settings
AUTO_RECONNECT = True
RECONNECT_DELAY = 2  # seconds

# ============================================================================
# FUNCTIONS
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

def create_stream_process():
    """Create FFmpeg streaming process with low-latency settings"""
    stream_url = f"{STREAM_PROTOCOL}://{'0.0.0.0' if STREAM_PROTOCOL == 'tcp' else IP}:{PORT}"
    stream_url += "?listen=1" if STREAM_PROTOCOL == "tcp" else "?pkt_size=1316&broadcast=1"
    
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
        '-r', str(FRAME_RATE),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', VIDEO_PRESET,
        '-tune', 'zerolatency',
        '-crf', str(VIDEO_QUALITY),
        '-b:v', BITRATE,
        '-maxrate', BITRATE,
        '-bufsize', f"{int(BITRATE[:-1]) * 2}k",
        '-pix_fmt', 'yuv420p',
        '-g', str(FRAME_RATE * 2),  # Keyframe interval
        '-sc_threshold', '0',
        '-f', 'mpegts',
        stream_url
    ]
    
    return subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

# ============================================================================
# INITIALIZATION
# ============================================================================
print("Initializing...")

picam2 = None
cap = None
stream_proc = None

# AI Model & Tracker
tracker = Sort(max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THRESHOLD)
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = ONNX_THREADS
ort_session = ort.InferenceSession(MODEL_PATH, sess_options)
input_name = ort_session.get_inputs()[0].name

# Calculate derived parameters
focal_length_px = (FRAME_WIDTH / 2.0) / math.tan(math.radians(HFOV_DEGREES / 2.0))
deadzone_pixels = int(math.tan(math.radians(DEADZONE_DEGREES)) * focal_length_px)
frame_center_x = FRAME_WIDTH // 2

# Initial stream
if STREAM_PROTOCOL != "none":
    stream_proc = create_stream_process()
    print(f">>> Streaming to {STREAM_PROTOCOL.upper()}:{PORT} <<<")
    print(f">>> Quality: CRF={VIDEO_QUALITY}, Bitrate={BITRATE} <<<\n")


if RUN_TARGET == "rpi":
        # Camera
        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT)}
        )
        picam2.configure(config)
        
        picam2.set_controls({
            "AeExposureMode" : 1,  # short
            "AeEnable": True,      # auto exposure   
            "AeConstraintMode": 0  # Normal constraint mode
        })
        
        picam2.start()
else:
    # Load a prerecorded file when running on the PC.
    print(f">>> PC MODE - Running from local video file: {PC_VIDEO_PATH} <<<")
    cap = cv2.VideoCapture(PC_VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {PC_VIDEO_PATH}")
        exit(1)


# ============================================================================
# MAIN LOOP
# ============================================================================
locked_target_id = None
stream_active = True

try:
    while True:
        t0 = time.perf_counter()
        
        # Capture & prepare frame
        if RUN_TARGET == "rpi":
            frame = picam2.capture_array()
        else:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

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
        
        # Stream frame with reconnection logic
        if STREAM_PROTOCOL != "none" and stream_proc is not None:
            try:
                stream_proc.stdin.write(orig_image.tobytes())
                if not stream_active:
                    print(">>> Stream reconnected <<<")
                    stream_active = True
            except (BrokenPipeError, OSError) as e:
                if stream_active:
                    print(f"\n>>> Stream disconnected, waiting for viewer... <<<")
                    stream_active = False
                
                if AUTO_RECONNECT:
                    # Clean up old process
                    try:
                        stream_proc.stdin.close()
                        stream_proc.terminate()
                        stream_proc.wait(timeout=1)
                    except:
                        pass
                    
                    # Wait and create new stream
                    time.sleep(RECONNECT_DELAY)
                    stream_proc = create_stream_process()
                    continue
                else:
                    break
        else:
            # display locally
            cv2.imshow("RPI Camera Stream", orig_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        
        fps = 1.0 / (time.perf_counter() - t0)
        status = "LIVE" if stream_active else "WAITING"
        print(f"[{status}] FPS: {fps:.1f} | Detections: {len(tracked_objects)}          ", end='\r')

except KeyboardInterrupt:
    print("\n\nStopping...")
finally:
    if picam2 is not None:
        picam2.stop()
    if cap is not None:
        cap.release()
    if stream_proc is not None:
        try:
            stream_proc.stdin.close()
            stream_proc.terminate()
            stream_proc.wait(timeout=2)
        except:
            stream_proc.kill()
