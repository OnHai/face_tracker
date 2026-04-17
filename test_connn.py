import socket
import time

UDP_PORT = 5005
# Change this from 255.255.255.255 to the AP's specific broadcast
BROADCAST_ADDR = '10.42.0.255' 

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

# FORCE the socket to use the Pi's Hotspot IP (10.42.0.1)
# This prevents the "Network is unreachable" error
sock.bind(('10.42.0.1', 0))

print(f"Sending 'ALIVE' to {BROADCAST_ADDR}:{UDP_PORT}...")

try:
    while True:
        sock.sendto(b"ALIVE", (BROADCAST_ADDR, UDP_PORT))
        print("Sent: ALIVE")
        time.sleep(1)
except Exception as e:
    print(f"Error: {e}")