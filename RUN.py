import math
import time
import cv2
import numpy as np
import vision.utils.box_utils_numpy as box_utils
import onnxruntime as ort
from sort import Sort
import socket

threshold = 0.7 # 100%
deadzone = 5 # dg
HFOV_degrees = 60.0  # dg
UDP_PORT = 5005
X = 640
Y = 480
QA = 50 # 100 %
BROADCAST_ADDR = '10.42.0.255'


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs, iou_threshold=iou_threshold, top_k=top_k)
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

# --- INIT ONNX 
tracker = Sort()
onnx_path = "models/onnx/version-RFB-320-perfect.onnx"
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 2
sess_options.inter_op_num_threads = 1
ort_session = ort.InferenceSession(onnx_path, sess_options)
input_name = ort_session.get_inputs()[0].name


locked_target_id = None
focal_length_px = (X / 2.0) / math.tan(math.radians(HFOV_degrees) / 2.0)

# --- INIT UDP 
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
sock.bind(('10.42.0.1', 0))
print(f">>> on port {UDP_PORT} <<<")

# --- INIT CAMERA 
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, X)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Y)

try:
    while True:
        print("1")
        ret, orig_image = cap.read()
        com_to_send = ""
        if not ret or orig_image is None:
            continue
        print("2")
        
        # Preprocessing
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 240))
        image = (image - np.array([127, 127, 127])) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0).astype(np.float32)

        print("3")
        
        # Inference
        confidences, boxes = ort_session.run(None, {input_name: image})
        boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)

        # Tracker
        if boxes.shape[0] > 0:
            scores = probs.reshape(-1, 1)
            dets = np.concatenate((boxes, scores), axis=1)
            tracked_objects = tracker.update(dets)
        else:
            tracked_objects = tracker.update(np.empty((0, 5)))
        print("4")
        
        # deadzone lines
        frame_center_x = X // 2
        deadzone_pixels = int(math.tan(math.radians(deadzone)) * focal_length_px)
        cv2.line(orig_image, (frame_center_x - deadzone_pixels, 0), (frame_center_x - deadzone_pixels, Y), (255, 0, 0), 1)
        cv2.line(orig_image, (frame_center_x + deadzone_pixels, 0), (frame_center_x + deadzone_pixels, Y), (255, 0, 0), 1)

        # Tracking Logic
        if len(tracked_objects) > 0:
            target_id_found = False
            
            # Check if current target is still visible
            if locked_target_id is not None:
                for obj in tracked_objects:
                    if int(obj[4]) == locked_target_id:
                        target_id_found = True
                        break

            # If no target or lost target, find the largest face
            if not target_id_found:
                largest_area = 0
                for obj in tracked_objects:
                    x1, y1, x2, y2, obj_id = [int(i) for i in obj]
                    area = (x2 - x1) * (y2 - y1)
                    if area > largest_area:
                        largest_area = area
                        locked_target_id = obj_id

            for obj in tracked_objects:
                x1, y1, x2, y2, obj_id = [int(i) for i in obj]
                if obj_id == locked_target_id:

                    # TARGET (Red)
                    cv2.rectangle(orig_image, (x1, y1), (x2, y2), (0, 0, 255), 4)
                    cv2.putText(orig_image, f"TARGET: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    face_center_x = int((x1 + x2) / 2)
                    cv2.circle(orig_image, (face_center_x, int((y1 + y2) / 2)), 5, (0, 255, 0), -1)
                    
                    pivot_x = frame_center_x
                    pivot_y = Y - 40 
                    arrow_length = 70

                    error_x = face_center_x - frame_center_x


                    angle_radians = math.atan(error_x / focal_length_px)
                    angle_degrees = math.degrees(angle_radians)
                    
                    
                    if abs(angle_degrees) > deadzone:

                        visual_angle = 270 + angle_degrees
                        end_x = int(pivot_x + arrow_length * math.cos(math.radians(visual_angle)))
                        end_y = int(pivot_y + arrow_length * math.sin(math.radians(visual_angle)))
                        cv2.arrowedLine(orig_image, (pivot_x, pivot_y), (end_x, end_y), (0, 0, 255), 4, tipLength=0.2)
                        cv2.circle(orig_image, (pivot_x, pivot_y), 6, (255, 255, 255), -1) 
                        
                        print(f"Command: angle {angle_degrees} |  ID: {locked_target_id}")
                    else:
                        end_y = pivot_y - arrow_length
                        cv2.arrowedLine(orig_image, (pivot_x, pivot_y), (pivot_x, end_y), (0, 255, 0), 4, tipLength=0.2)
                        cv2.circle(orig_image, (pivot_x, pivot_y), 6, (255, 255, 255), -1)
                        
                        print(f"Command: angle 0.0 |  ID: {locked_target_id}")
                else:

                    # OTHERS (Blue)
                    cv2.rectangle(orig_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        else:
            locked_target_id = None
            cv2.putText(orig_image, "NO TARGET", (X // 2 - 80, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print(f"NO TARGET")

        print(scores)

        # --- UDP BROADCAST 
        ret_encode, jpeg = cv2.imencode('.jpg', orig_image, [cv2.IMWRITE_JPEG_QUALITY, QA])
        if ret_encode:
            data = jpeg.tobytes()

            if len(data) > 65507:
                print(f"FRAME TOO BIG: {len(data)} bytes!")
            else:
                try:
                    sock.sendto(data, (BROADCAST_ADDR, UDP_PORT))
                except Exception as e:
                    print(f"UDP Send Error: {e}")

except KeyboardInterrupt:
    print("\nStopping Drone...")
finally:
    cap.release()
    sock.close()