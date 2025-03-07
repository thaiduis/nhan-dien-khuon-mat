from flask import Flask, render_template, Response, request, jsonify
import cv2
from threading import Thread, Lock
import time
import numpy as np
import signal
import sys
import os

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# URL của luồng video
stream_url = 'http://172.16.47.13:4747/video'
cap = cv2.VideoCapture(stream_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Cấu hình kích thước bộ đệm để xử lý video mượt mà hơn

# Lấy kích thước khung hình từ luồng video
ret, frame = cap.read()
if ret:
    height, width = frame.shape[:2]
else:
    width, height = 640, 480  # Giá trị mặc định nếu không đọc được khung hình

video_frame = None  # Biến để lưu trữ khung hình hiện tại
lock = Lock()  # Khóa để đảm bảo đồng bộ khi truy cập khung hình

# Biến toàn cục để kiểm soát vòng lặp
running = True

# Biến để theo dõi trạng thái phát hiện đối tượng và quay video
object_detected = False
video_writer = None
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)

# Tải mô hình MobileNet SSD đã được huấn luyện trước để phát hiện đối tượng
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Khởi tạo background subtractor để phát hiện chuyển động
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

# Hàm để xử lý luồng video và phát hiện đối tượng
def camera_stream():
    global cap, video_frame, width, height, running, object_detected, video_writer

    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)  # Chờ một chút trước khi thử lại nếu không đọc được khung hình
            continue

        # Phát hiện chuyển động bằng background subtraction
        fgmask = fgbg.apply(frame)
        motion_detected = np.sum(fgmask) > 10000  # Ngưỡng phát hiện chuyển động

        # Nếu có chuyển động, thực hiện phát hiện đối tượng
        if motion_detected:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            # Kiểm tra xem có người nào được phát hiện không
            current_object_detected = False
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:  # Ngưỡng tin cậy cao hơn
                    idx = int(detections[0, 0, i, 1])
                    label = CLASSES[idx]  # Nhãn của đối tượng

                    # Chỉ xử lý nếu đối tượng là người
                    if label == "person":
                        current_object_detected = True
                        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                        (startX, startY, endX, endY) = box.astype("int")

                        # Vẽ bounding box và nhãn lên khung hình
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(frame, f"{label}: {confidence * 100:.2f}%", (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Nếu có người được phát hiện và chưa quay video, bắt đầu quay video
            if current_object_detected and not object_detected:
                object_detected = True
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                video_filename = os.path.join(output_folder, f"video_{timestamp}.avi")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (width, height))
                print(f"Bắt đầu quay video: {video_filename}")

                # Chụp ảnh khi phát hiện người
                image_filename = os.path.join(output_folder, f"image_{timestamp}.jpg")
                cv2.imwrite(image_filename, frame)
                print(f"Đã chụp ảnh: {image_filename}")

            # Nếu không còn người và đang quay video, dừng quay video
            if not current_object_detected and object_detected:
                object_detected = False
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                    print("Dừng quay video")

            # Nếu đang quay video, ghi khung hình vào video
            if object_detected and video_writer is not None:
                video_writer.write(frame)

        # Cập nhật khung hình hiện tại
        with lock:
            video_frame = frame.copy() if frame is not None else None

# Hàm để tạo khung hình gửi tới client
def gen_frames():
    global video_frame, running
    while running:
        with lock:
            if video_frame is None:
                time.sleep(0.1)
                continue
            ret, buffer = cv2.imencode('.jpg', video_frame)
            if not ret:
                continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Định tuyến Flask để truyền luồng video tới client
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Định tuyến Flask để phục vụ trang HTML chính
@app.route('/')
def index():
    return render_template('index_flask_server.html')

# Hàm xử lý tín hiệu Ctrl+C
def signal_handler(sig, frame):
    global running
    print("\nĐang dừng chương trình...")
    running = False
    if cap.isOpened():
        cap.release()
    if video_writer is not None:
        video_writer.release()
    sys.exit(0)

# Hàm chính để chạy ứng dụng Flask và xử lý luồng video
if __name__ == '__main__':
    # Đăng ký signal handler để bắt tín hiệu Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    camera_thread = Thread(target=camera_stream)
    camera_thread.start()
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        running = False
        if cap.isOpened():
            cap.release()
        if video_writer is not None:
            video_writer.release()