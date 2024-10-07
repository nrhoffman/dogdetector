import cv2
import numpy as np
from ultralytics import YOLO

model_dog = YOLO('src/dog_detector_runs/detect/train/weights/best.pt')
# model_posture = YOLO('src/dog_posture_runs/detect/train/weights/best.pt')

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cam.read()

    if not ret:
        break

    # Write the frame to the output file
    out.write(frame)

    #Resize
    resized_image = cv2.resize(frame, (640,640))

    results_dog = model_dog(resized_image)

    for result in results_dog:
        boxes = result.boxes
        for box in boxes:
            # Get the bounding box coordinates and confidence score
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int)  # Convert to int
            confidence = box.conf[0].item()  # Confidence score

            # Only draw the box if confidence is 80% or higher
            if confidence >= 0.5:
                label = int(box.cls[0])  # Class label
                # Draw the bounding box on the original frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Dog: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the captured frame with detections
        cv2.imshow('Camera', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()