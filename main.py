import cv2
import numpy as np
import requests
import os


# Функция для скачивания файлов
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Скачиваю {filename}...")
        response = requests.get(url, stream=True)
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"{filename} загружен.")
    else:
        print(f"{filename} уже существует.")


# Скачиваем необходимые файлы
cfg_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
weights_url = "https://pjreddie.com/media/files/yolov4-tiny.weights"
coco_names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

download_file(cfg_url, "yolov4-tiny.cfg")
download_file(weights_url, "yolov4-tiny.weights")
download_file(coco_names_url, "coco.names")

# Загрузка имен классов
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Загрузка YOLO модели
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Индекс класса "person" в COCO
person_idx = classes.index("person")

# Настройка камеры
cap = cv2.VideoCapture("1338590-hd_1920_1080_30fps.mp4")  # 0 для веб-камеры или путь к видеофайлу

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    height, width = frame.shape[:2]

    # Подготовка входного кадра
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    # Сбор данных о детекциях
    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if class_id == person_idx and confidence > 0.5:
                # Получаем координаты бокса
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, w, h) = box.astype("int")
                x = int(centerX - w / 2)
                y = int(centerY - h / 2)

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Применяем non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Отображение результата
    count = 0
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {confidences[i]:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Отображение количества людей
    cv2.putText(frame, f"People count: {count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Real-time Crowd Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
