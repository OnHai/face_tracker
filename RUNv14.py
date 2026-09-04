RUN_TARGET = "rpi"
STREAM_PROTOCOL = "tcp"
PC_VIDEO_PATH = r"test_tracking.mp4"
ENABLE_CRAZYFLIE = False

import math
import struct
import time
import cv2
import numpy as np
import subprocess
import logging
import os
import supervision as sv 

if RUN_TARGET == "rpi":
    from picamera2 import Picamera2
    import serial

IP = "0.0.0.0"
PORT = 5005

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_RATE = 10

INFERENCE_W = 320
INFERENCE_H = 240

VIDEO_QUALITY = 20
VIDEO_PRESET = "ultrafast"
BITRATE = "1000k"  

MODEL_PATH = "face_detection_yunet_2026may.onnx"
CONFIDENCE_THRESHOLD = 0.5 

IOU_THRESHOLD = 0.1
MAX_AGE = 15
MIN_HITS = 2

HFOV_DEGREES = 110.0
DEADZONE_DEGREES = 5

AUTO_RECONNECT = True
RECONNECT_DELAY = 2
CAMERA_CONTROL_FILE = "camera_control.txt"
CAMERA_CONTROL_POLL_INTERVAL = 2.0

# CLAHE Configuration
CLAHE_ENABLED = True
CLAHE_FULL_FRAME = False 
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = 8

messenger = [0, 0, 0]  

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('face_tracker.log'),
        logging.StreamHandler() 
    ]
)
logger = logging.getLogger(__name__)

port = '/dev/ttyAMA0' 
baud = 115200

def connect_crazyflie():
    try:
        ser = serial.Serial(port, baud, timeout=1, write_timeout=1)
        print(f"Opened {port} at {baud}")
        return ser
    except Exception as e:
        print(f"Error opening port: {e}")
        return None

stop_requested = False
last_control_check = 0.0
last_control_mtime = None

def apply_camera_controls():
    if auto_exposure:
        picam2.set_controls({
            "AeEnable": True,
            "AeExposureMode": ae_exposure_mode
        })
    else:
        picam2.set_controls({
            "AeEnable": False,
            "ExposureTime": exposure_time_us,
            "AnalogueGain": analogue_gain
        })

def process_camera_control_file(now):
    global auto_exposure, exposure_time_us, analogue_gain
    global ae_exposure_mode, clahe_enabled, clahe_full_frame, clahe_clip_limit, clahe_tile_size
    global last_control_check, last_control_mtime

    if RUN_TARGET != "rpi" or now - last_control_check < CAMERA_CONTROL_POLL_INTERVAL:
        return

    last_control_check = now
    try:
        control_mtime = os.path.getmtime(CAMERA_CONTROL_FILE)
    except OSError:
        return
    if control_mtime == last_control_mtime:
        return

    try:
        with open(CAMERA_CONTROL_FILE, "r", encoding="ascii") as control_file:
            values = {}
            for line in control_file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, separator, value = line.partition("=")
                if not separator or not key.strip() or not value.strip():
                    raise ValueError("incomplete line")
                values[key.strip()] = value.strip()

        new_auto_exposure = values.get("auto_exposure", str(auto_exposure)).lower()
        new_clahe_enabled = values.get("clahe_enabled", str(clahe_enabled)).lower()
        new_clahe_full_frame = values.get("clahe_full_frame", str(clahe_full_frame)).lower()
        
        if new_auto_exposure not in ("true", "false"):
            raise ValueError("auto_exposure must be true or false")
        if new_clahe_enabled not in ("true", "false"):
            raise ValueError("clahe_enabled must be true or false")
        if new_clahe_full_frame not in ("true", "false"):
            raise ValueError("clahe_full_frame must be true or false")

        ae_exposure_mode = int(values.get("ae_exposure_mode", ae_exposure_mode))
        exposure_time_us = int(values.get("exposure_time_us", exposure_time_us))
        analogue_gain = float(values.get("analogue_gain", analogue_gain))
        clahe_clip_limit = float(values.get("clahe_clip_limit", clahe_clip_limit))
        clahe_tile_size = int(values.get("clahe_tile_size", clahe_tile_size))

        auto_exposure = new_auto_exposure == "true"
        clahe_enabled = new_clahe_enabled == "true"
        clahe_full_frame = new_clahe_full_frame == "true"
        
        apply_camera_controls()
        last_control_mtime = control_mtime
    except (OSError, ValueError):
        last_control_mtime = None

def send_angle_to_crazyflie(ser, angle_degx, angle_degy, isize):
    try:
        packet = struct.pack('<cfff', b'S', angle_degx, angle_degy, isize)
        ser.write(packet)
        ser.flush() 
    except serial.SerialTimeoutException:
        logger.warning("Crazyflie write timeout, skipping frame")
    except serial.SerialException as e:
        logger.warning(f"Crazyflie serial error: {e}")

def prepare_frame(frame):
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=(clahe_tile_size, clahe_tile_size)
    )
    l_channel = clahe.apply(l_channel)
    return cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

def create_stream_process():
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
        '-c:v', 'h264_v4l2m2m',  
        '-num_output_buffers', '32',
        '-num_capture_buffers', '16',
        '-b:v', BITRATE,
        '-pix_fmt', 'yuv420p',
        '-g', str(FRAME_RATE * 2),
        '-f', 'mpegts',
        stream_url
    ] 
    return subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

logger.info("Initializing...")

picam2 = None
cap = None
stream_proc = None
auto_exposure = False
exposure_time_us = 8000
analogue_gain = 2.0
ae_exposure_mode = 1
clahe_enabled = CLAHE_ENABLED
clahe_full_frame = CLAHE_FULL_FRAME
clahe_clip_limit = CLAHE_CLIP_LIMIT
clahe_tile_size = CLAHE_TILE_SIZE

if ENABLE_CRAZYFLIE:
    ser = connect_crazyflie()
else:
    ser = None

tracker = sv.ByteTrack(
    track_activation_threshold=CONFIDENCE_THRESHOLD,
    lost_track_buffer=MAX_AGE,
    minimum_matching_threshold=0.8,
    frame_rate=FRAME_RATE
)

detector = cv2.FaceDetectorYN.create(
    MODEL_PATH,
    "",
    (INFERENCE_W, INFERENCE_H),
    score_threshold=0.1,  
    nms_threshold=IOU_THRESHOLD,
    top_k=50
)

focal_length_px = (FRAME_WIDTH / 2.0) / math.tan(math.radians(HFOV_DEGREES / 2.0))
deadzone_pixels = int(math.tan(math.radians(DEADZONE_DEGREES)) * focal_length_px)
frame_center_x = FRAME_WIDTH // 2
frame_center_y = FRAME_HEIGHT // 2

if STREAM_PROTOCOL != "none":
    stream_proc = create_stream_process()
    logger.info(f">>> Streaming to {STREAM_PROTOCOL.upper()}:{PORT} <<<")

if RUN_TARGET == "rpi":
        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT)}
        )
        picam2.configure(config)
        picam2.start()
        apply_camera_controls()
else:
    cap = cv2.VideoCapture(PC_VIDEO_PATH)
    if not cap.isOpened():
        exit(1)

locked_target_id = None
last_target_pos = None
stream_active = True
angle_degx, angle_degy, isize = 0.0, 0.0, 0.0
last_seen_time = {}
TARGET_LOST_TIMEOUT = MAX_AGE / FRAME_RATE

scale_x = FRAME_WIDTH / INFERENCE_W
scale_y = FRAME_HEIGHT / INFERENCE_H

try:
    while True:
        t0 = time.perf_counter()
        process_camera_control_file(t0)

        if stop_requested:
            break
        
        if RUN_TARGET == "rpi":
            frame = picam2.capture_array()
        else:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        if frame is None: continue
        raw_image = prepare_frame(frame)
        
        # Apply CLAHE logic based on configuration
        if clahe_enabled and clahe_full_frame:
            orig_image = apply_clahe(raw_image)
            infer_image = cv2.resize(orig_image, (INFERENCE_W, INFERENCE_H))
        elif clahe_enabled and not clahe_full_frame:
            orig_image = raw_image.copy()
            infer_image = apply_clahe(cv2.resize(raw_image, (INFERENCE_W, INFERENCE_H)))
        else:
            orig_image = raw_image.copy()
            infer_image = cv2.resize(raw_image, (INFERENCE_W, INFERENCE_H))

        _, faces = detector.detect(infer_image)
        
        if faces is not None:
            x1 = faces[:, 0] * scale_x
            y1 = faces[:, 1] * scale_y
            x2 = x1 + (faces[:, 2] * scale_x)
            y2 = y1 + (faces[:, 3] * scale_y)
            conf = faces[:, 14]
            
            detections = sv.Detections(
                xyxy=np.column_stack((x1, y1, x2, y2)),
                confidence=conf
            )
        else:
            detections = sv.Detections.empty()
            
        tracked_dets = tracker.update_with_detections(detections)
        
        tracked_objects = []
        if len(tracked_dets) > 0:
            for i in range(len(tracked_dets)):
                box = tracked_dets.xyxy[i]
                trk_id = tracked_dets.tracker_id[i]
                tracked_objects.append([box[0], box[1], box[2], box[3], trk_id])
                
        cv2.line(orig_image, (frame_center_x - deadzone_pixels, 0), (frame_center_x - deadzone_pixels, FRAME_HEIGHT), (255, 0, 0), 1)
        cv2.line(orig_image, (frame_center_x + deadzone_pixels, 0), (frame_center_x + deadzone_pixels, FRAME_HEIGHT), (255, 0, 0), 1)
        
        if len(tracked_objects) > 0:
            for obj in tracked_objects:
                last_seen_time[int(obj[4])] = t0

            last_seen_time = {k: v for k, v in last_seen_time.items()
                      if t0 - v < TARGET_LOST_TIMEOUT * 2}
            
            current_ids = [int(obj[4]) for obj in tracked_objects]
            
            should_relock = (
                    locked_target_id is None
                    or (
                        locked_target_id not in current_ids
                        and (t0 - last_seen_time.get(locked_target_id, 0)) > TARGET_LOST_TIMEOUT
                    )
                )
            
            if should_relock:
                if last_target_pos is not None:
                    min_dist = float('inf')
                    best_id = None
                    
                    for obj in tracked_objects:
                        x1, y1, x2, y2, obj_id = [int(i) for i in obj]
                        current_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        
                        dist = math.sqrt(
                            (current_center[0] - last_target_pos[0])**2 + 
                            (current_center[1] - last_target_pos[1])**2
                        )
                        
                        if dist < min_dist:
                            min_dist = dist
                            best_id = int(obj_id)
                    
                    locked_target_id = best_id
                else:
                    center_dists = []
                    for obj in tracked_objects:
                        x1, y1, x2, y2, obj_id = [int(i) for i in obj]
                        center_x = (x1 + x2) // 2
                        dist_from_center = abs(center_x - frame_center_x)
                        center_dists.append((dist_from_center, int(obj_id)))
                    
                    locked_target_id = min(center_dists, key=lambda x: x[0])[1]
            
            for obj in tracked_objects:
                x1, y1, x2, y2, obj_id = [int(i) for i in obj]
                is_target = (obj_id == locked_target_id)
                color = (0, 0, 255) if is_target else (255, 0, 0)
                cv2.rectangle(orig_image, (x1, y1), (x2, y2), color, 2)
                
                if is_target:
                    last_target_pos = ((x1 + x2) // 2, (y1 + y2) // 2)

                    target_x = (x1 + x2) // 2
                    target_y = (y1 + y2) // 2

                    error_x = target_x - frame_center_x
                    error_y = target_y - frame_center_y

                    isize = math.sqrt((x2-x1)**2 + (y2-y1)**2)

                    angle_degx = math.degrees(math.atan(error_x / focal_length_px))
                    angle_degy = math.degrees(math.atan(error_y / focal_length_px))

                    pivot = (frame_center_x, FRAME_HEIGHT - 40)
                    arrow_col = (0, 255, 0) if abs(angle_degx) < DEADZONE_DEGREES else (0, 0, 255)
                    visual_angle = 270 + angle_degx
                    end_x = int(pivot[0] + 60 * math.cos(math.radians(visual_angle)))
                    end_y = int(pivot[1] + 60 * math.sin(math.radians(visual_angle)))
                    cv2.arrowedLine(orig_image, pivot, (end_x, end_y), arrow_col, 3)

        if STREAM_PROTOCOL != "none" and stream_proc is not None:
            try:
                stream_proc.stdin.write(orig_image.tobytes())
                if not stream_active:
                    stream_active = True
            except (BrokenPipeError, OSError):
                if stream_active:
                    stream_active = False
                
                if AUTO_RECONNECT:
                    try:
                        stream_proc.stdin.close()
                        stream_proc.terminate()
                        stream_proc.wait(timeout=1)
                    except:
                        pass
                    
                    time.sleep(RECONNECT_DELAY)
                    stream_proc = create_stream_process()
                    continue
                else:
                    break
        if RUN_TARGET == "pc" and STREAM_PROTOCOL == "none":
            cv2.imshow("RPI Camera Stream", orig_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if ENABLE_CRAZYFLIE:
            send_angle_to_crazyflie(ser, angle_degx, angle_degy, isize)
        
        fps = 1.0 / (time.perf_counter() - t0)
        status = "LIVE" if stream_active else "WAITING"
        print(f"[{status}] FPS: {fps:.1f} | Detections: {len(tracked_objects)} | Anglex: {angle_degx:>5.1f}° | Angley: {angle_degy:>5.1f}° | Size {isize:>5.1f}      ", end='\r')

        if RUN_TARGET == "pc":
            frame_time = time.perf_counter() - t0
            target_frame_time = 1.0 / FRAME_RATE
            if frame_time < target_frame_time:
                time.sleep(target_frame_time - frame_time)
                
        angle_degx, angle_degy, isize = 0.0, 0.0, 0.0

except KeyboardInterrupt:
    logger.info("\n\nStopping...")
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
    if ser is not None:
        ser.close()