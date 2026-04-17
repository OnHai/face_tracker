import time
import numpy as np
from picamera2 import Picamera2

# --- 1. INIT CAMERA ---
print("Initializing Picamera2...")
picam2 = Picamera2()

# Configure for your ov9281 native high-speed resolution
config = picam2.create_video_configuration(main={"size": (640, 400)})
picam2.configure(config)
picam2.start()

print(">>> Camera started. Capturing frames... Press Ctrl+C to stop. <<<")

try:
    while True:
        start_time = time.time()
        
        # --- 2. CAPTURE ---
        # Grabs the frame directly into memory as a NumPy array
        frame = picam2.capture_array()
        
        if frame is None:
            continue
            
        # --- 3. CALCULATE ---
        # Total brightness is the sum of all pixel values.
        # We cast to uint64 to prevent integer overflow since the sum is huge.
        total_brightness = np.sum(frame, dtype=np.uint64)
        
        # Average brightness (0-255) is usually easier to read for lighting changes
        avg_brightness = np.mean(frame)
        
        # --- 4. PRINT ---
        fps = 1.0 / (time.time() - start_time)
        print(f"[{fps:4.1f} FPS] Total: {total_brightness} | Avg: {avg_brightness:.1f} / 255")

except KeyboardInterrupt:
    print("\nStopping camera...")
finally:
    picam2.stop()