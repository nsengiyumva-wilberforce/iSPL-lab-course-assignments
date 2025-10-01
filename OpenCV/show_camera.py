"""
show_camera.py
---------------
Opens your webcam (default index = 0) and displays the live camera feed.
Press 'q' to quit the window.
"""

import cv2

def show_camera(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Camera showing Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_camera(0)  # Change to 1, 2... if you have multiple cameras
