from ultralytics import YOLO
import cv2
import math 
import serial
import time

# Define the interval in milliseconds
interval_ms = 150

# Initialize the webcam (change '0' if you have multiple webcams)
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set the webcam resolution (optional)
width_camera = 640
height_camera = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width_camera)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height_camera)

# Load the models
model_korban = YOLO("/home/rafi/Vision/2024/model/korban_best_RGB.pt")
model_tembok = YOLO("/home/rafi/Vision/2024/model/wall_best_DEPTH.pt")
model_sz = YOLO("/home/rafi/Vision/2024/model/savezone.pt")

# Time frame for FPS calculation
tm = cv2.TickMeter()
tm.start()

# Initialize variables
xerror_korban = 0
yerror_korban = 0
korban_size_pixels = 0
xerror_tembok = 0
yerror_tembok = 0
tembok_size_pixels = 0
modeDetect = 1

count = 0
max_count = 10
fps = 0
perintahTeensy = None
messagestring = '0,0,0'
countersend = 0

start_time = time.time()
last_sent_time = start_time

try:
    while True:

        # Capture frame from webcam
        ret, color_image = cap.read()
        if not ret:
            continue

        # Processing based on detection mode
        if modeDetect == 0:
            print("[INFO] Set deteksi ke Mode Tembok")
            highest_score = 0
            highest_score_box = None
            detectedTrack = model_tembok(color_image, stream=True)
            for r in detectedTrack:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    score = math.ceil((box.conf[0] * 100)) / 100
                    if score > highest_score:
                        highest_score = score
                        highest_score_box = (int(x1), int(y1), int(x2), int(y2))

            # Process the highest prediction
            if highest_score_box is not None:
                x1, y1, x2, y2 = highest_score_box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                center_x_frame = int(width_camera / 2)
                center_y_frame = int(height_camera / 2)
                cv2.line(color_image, (x1, y2), (x2, y2), (255, 0, 255), 10)
                cv2.circle(color_image, (center_x, y2), 30, (0, 255, 0), 5)
                xerror_tembok = center_x_frame - center_x
                yerror_tembok = center_y_frame - center_y
                toTeensyXframe = 512 + xerror_tembok
                toTeensyYframe = 512 + yerror_tembok
                messagestring = "a{:03d}{:03d}\n".format(toTeensyXframe, toTeensyYframe)
                print(messagestring)
            else:
                messagestring = 'a000000\n'
                print(messagestring)

        if modeDetect == 1:
            print("[INFO] Set deteksi ke Mode Korban")
            highest_score = 0
            highest_score_box = None
            detectedKorban = model_korban(color_image, stream=True)
            for r in detectedKorban:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    score = math.ceil((box.conf[0] * 100)) / 100
                    if score > highest_score:
                        highest_score = score
                        highest_score_box = (int(x1), int(y1), int(x2), int(y2))

            # Process the highest prediction
            if highest_score_box is not None:
                x1, y1, x2, y2 = highest_score_box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                center_x_frame = int(width_camera / 2)
                center_y_frame = int(height_camera / 2)
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 255, 0), 3)
                xerror_korban = center_x_frame - center_x
                yerror_korban = center_y_frame - center_y
                toTeensyXframe = 512 + xerror_korban
                toTeensyYframe = 512 + yerror_korban
                messagestring = "a{:03d}{:03d}\n".format(toTeensyXframe, toTeensyYframe)
                print(messagestring)
            else:
                messagestring = 'a000000\n'
                print(messagestring)

        if modeDetect == 3:
            print("[INFO] Set deteksi ke Mode SZ")
            highest_score_sz = 0
            highest_score_box = None
            detectedsz = model_sz(color_image, stream=True)
            for r in detectedsz:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    score = math.ceil((box.conf[0] * 100)) / 100
                    if score > highest_score_sz:
                        highest_score_sz = score
                        highest_score_box = (int(x1), int(y1), int(x2), int(y2))

            # Process the highest prediction
            if highest_score_box is not None:
                x1, y1, x2, y2 = highest_score_box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                center_x_frame = int(width_camera / 2)
                center_y_frame = int(height_camera / 2)
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 255, 0), 3)
                xerror = center_x_frame - center_x
                yerror = center_y_frame - center_y
                toTeensyXframe = 512 + xerror
                toTeensyYframe = 512 + yerror
                messagestring = "a{:03d}{:03d}\n".format(toTeensyXframe, toTeensyYframe)
                print(messagestring)
            else:
                messagestring = 'a000000\n'
                print(messagestring)

        # FPS calculation
        if count == max_count:
            tm.stop()
            fps = max_count / tm.getTimeSec()
            tm.reset()
            tm.start()
            count = 0

        cv2.putText(color_image, 'FPS: {:.2f}'.format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)
        cv2.imshow('Webcam', color_image)
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

