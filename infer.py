from ultralytics import YOLO
import cv2
import numpy as np
import yaml

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8s_playing_cards.pt")  # load a pretrained model (recommended for training)
# cap = cv2.VideoCapture("card.mp4")
cap = cv2.VideoCapture(0)

# Use the model
# results = model.train(data="coco128.yaml", epochs=3)  # train the model
# results = model.val()  # evaluate model performance on the validation set
# results = model("card4.png")  # predict on an image
# success = YOLO("yolov8n.pt").export(format="onnx")  # export a model to ONNX format
# print([x for x in results])

# Read class labels from the YAML file
with open("data_config/playing_cards.yaml", "r") as file:
    data = yaml.safe_load(file)
    class_labels = data["names"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device="mps")
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    
    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
        cv2.putText(frame, str(class_labels[cls]), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)
        print("xxxxxxx", str(class_labels[cls]))

    cv2.imshow("Img", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cam_quit = 1
        break

cap.release()
cv2.destroyAllWindows()
