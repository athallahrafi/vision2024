from ultralytics import YOLO
import pyrealsense2 as rs
import cv2
import math 
import serial
import numpy as np
import time

# Define the interval in milliseconds
interval_ms = 150

pipeline = rs.pipeline()
config = rs.config()
# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
width_camera = 640 #848
height_camera = 480#480
config.enable_stream(rs.stream.depth, width_camera, height_camera, rs.format.z16, 30)  # 640,480
config.enable_stream(rs.stream.color, width_camera, height_camera, rs.format.bgr8, 30)  # 640,480

# Start streaming
pipeline.start(config)

# model
model_korban = YOLO("/home/aljazar2024-jrcoreos/ultralytics/aljazari_workspace/model/korban_best_RGB.pt")
model_tembok = YOLO("/home/aljazar2024-jrcoreos/ultralytics/aljazari_workspace/model/wall_best_DEPTH.pt")
model_sz = YOLO("/home/aljazar2024-jrcoreos/ultralytics/aljazari_workspace/model/savezone.pt")

#serial
teensy = serial.Serial('/dev/ttyUSB0', 115200, timeout=0)

#time framing
tm = cv2.TickMeter()
tm.start()

# Initialize variables
xerror_korban = 0
yerror_korban = 0
korban_size_pixels = 0
xerror_tembok = 0
yerror_tembok = 0
tembok_size_pixels = 0
modeDetect = 0

count = 0
max_count = 10
fps = 0
perintahTeensy = None
messagestring = '0,0,0'
countersend=0

start_time = time.time()
last_sent_time = start_time

try:
    while True:
        #check request data dari Teensy 
        try:
            perintahTeensy = teensy.readline().decode().strip()
        except:
            perintahTeensy = 'track'
        #perintahTeensy = 'track'
        if perintahTeensy == 'track':
            modeDetect = 0
        elif perintahTeensy == 'korban':
            modeDetect = 1
        elif perintahTeensy == 'savezone': 
            modeDetect = 3

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        # Convert depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.04), cv2.COLORMAP_JET)
        
        if modeDetect == 0:
            print ("[INFO] Set deteksi ke Mode  Tembok")
            highest_score = 0
            highest_score_box = None
            detectedTrack = model_tembok(depth_colormap, stream = True)
            for r in detectedTrack:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    score = math.ceil((box.conf[0]*100))/100
                    if score > highest_score:
                        highest_score = score
                        highest_score_box = (int(x1), int(y1), int(x2), int(y2))
             #proses nilai prediksi terbesar
            if highest_score_box is not None:
                x1, y1, x2, y2 = highest_score_box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                center_x_frame = int (width_camera/2)
                center_y_frame = int (height_camera/2)
                cv2.line(color_image, (x1, y2), (x2, y2), (255, 0, 255), 10)
                cv2.circle(color_image, (center_x, y2), 30, (0, 255, 0), 5)
                #cv2.line(color_image, (center_x_frame, center_y_frame), (center_x, center_y), (0, 0, 255), 3)
                xerror_tembok = center_x_frame - center_x
                yerror_tembok = center_y_frame - center_y
                toTeensyXframe = 512 + xerror_tembok
                toTeensyYframe = 512 + yerror_tembok
                tembok_size_pixels = 0 #tidak digunakan
                messagestring =  "a{:03d}{:03d}\n".format(toTeensyXframe,toTeensyYframe)
                teensy.write(messagestring.encode())
                print (messagestring)
            else:
                messagestring = 'a000000\n'
                teensy.write(messagestring.encode())
                #teensy.write(messagestring.encode())
                print (messagestring)

        if modeDetect == 1:
            print ("[INFO] Set deteksi ke Mode Korban")
            highest_score = 0
            highest_score_box = None
            detectedKorban = model_korban(color_image, stream = True)
            for r in detectedKorban:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    score = math.ceil((box.conf[0]*100))/100
                    if score > highest_score:
                        highest_score = score
                        highest_score_box = (int(x1), int(y1), int(x2), int(y2))
             #proses nilai prediksi terbesar
            if highest_score_box is not None:
                x1, y1, x2, y2 = highest_score_box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                center_x_frame = int (width_camera/2)
                center_y_frame = int (height_camera/2)
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 255, 0), 3)
                #cv2.circle(color_image, (center_x, center_y), 30, (0, 255, 0), 5)
                #cv2.line(color_image, (center_x_frame, center_y_frame), (center_x, center_y), (0, 0, 255), 3)
                xerror_korban = center_x_frame - center_x
                yerror_korban = center_y_frame - center_y
                toTeensyXframe = 512 + xerror_korban
                toTeensyYframe = 512 + yerror_korban
                messagestring =  "a{:03d}{:03d}\n".format(toTeensyXframe,toTeensyYframe)
                teensy.write(messagestring.encode())
                print (messagestring)
            else :
                messagestring = 'a000000\n'
                teensy.write(messagestring.encode())
                print (messagestring)
        
        if modeDetect == 3:
            print ("[INFO] Set deteksi ke Mode SZ")
            highest_score_sz = 0
            highest_score_box = None
            detectedsz = model_sz(color_image, stream = True)
            for r in detectedsz:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    score = math.ceil((box.conf[0]*100))/100
                    if score > highest_score_sz:
                        highest_score = score
                        highest_score_box = (int(x1), int(y1), int(x2), int(y2))
             #proses nilai prediksi terbesar
            if highest_score_box is not None:
                x1, y1, x2, y2 = highest_score_box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                center_x_frame = int (width_camera/2)
                center_y_frame = int (height_camera/2)
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 255, 0), 3)
                #cv2.circle(color_image, (center_x, center_y), 30, (0, 255, 0), 5)
                #cv2.line(color_image, (center_x_frame, center_y_frame), (center_x, center_y), (0, 0, 255), 3)
                xerror = center_x_frame - center_x
                yerror = center_y_frame - center_y
                toTeensyXframe = 512 + xerror
                toTeensyYframe = 512 + yerror
                messagestring =  "a{:03d}{:03d}\n".format(toTeensyXframe,toTeensyYframe)
                teensy.write(messagestring.encode())
                print (messagestring)
            else :
                messagestring = 'a000000\n'
                teensy.write(messagestring.encode())
                print (messagestring)


       
        #if time.time() - last_sent_time >= interval_ms / 1000:
       
        #last_sent_time = time.time()
        
        if count == max_count:
            tm.stop()
            fps = max_count / tm.getTimeSec()
            tm.reset()
            tm.start()
            count = 0
        
        #images = np.hstack((color_image, depth_colormap))
        cv2.putText(color_image, 'FPS: {:.2f}'.format(fps),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)
        cv2.imshow('RealSense', color_image)
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        
finally:
    pipeline.stop()
    teensy.close()
    cv2.destroyAllWindows()
    
