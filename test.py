from ultralytics import YOLO
import cv2

# Load YOLOv8 model (change 'm1.pt' to your model path)
model = YOLO("trained_model.pt")

# Input and output video paths
'''
input_video = "sample2.mp4"
output_video = "output_video.mp4"
'''
# Open video file
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open input video")
    exit()

# Video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# Create VideoWriter to save output
#out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Confidence threshold
conf_threshold = 0.4

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Run YOLO inference
    results = model(frame)

    # Filter detections by confidence and draw on frame
    for result in results:
        for box in result.boxes:
#            if box.conf[0] >= conf_threshold:
                frame = result.plot()


    # Write processed frame to output video
    cv2.imshow("YOLO Comparison (Press Q to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()