"""
show_video.py
---------------
Loads a video from your computer and displays it frame by frame using OpenCV.
Press 'q' to quit the video window early.
"""

import cv2

# Replace with your video file path
VIDEO_PATH = "F:\cat_images\\video.mp4"

def show_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not load video: {path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        cv2.imshow("A Cat video streaming using open CV", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_video(VIDEO_PATH)
