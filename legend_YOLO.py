import cv2
import numpy as np

onnx_model_path = "best.onnx"
input_shape = (640, 640)
net = cv2.dnn.readNetFromONNX(onnx_model_path)
model_classify = ["legend"]


def recognize(img_path, threshold=0.5):
    img = cv2.imread(img_path)
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, input_shape, swapRB=True, crop=False)
    net.setInput(blob)

    output = net.forward()
    output = output.transpose((0, 2, 1))

    height, width, _ = img.shape
    x_factor, y_factor = width / input_shape[0], height / input_shape[1]

    classifys, scores, boxes = [], [], []
    for i in range(output[0].shape[0]):
        box = output[0][i]
        _, _, _, max_idx = cv2.minMaxLoc(box[4:])
        class_id = max_idx[1]
        score = box[4:][class_id]
        if (score > threshold):
            scores.append(score)
            classifys.append(model_classify[int(class_id)])
            x, y, w, h = box[0].item(), box[1].item(), box[2].item(), box[3].item()
            x = int((x - 0.5 * w) * x_factor)
            y = int((y - 0.5 * h) * y_factor)
            w = int(w * x_factor)
            h = int(h * y_factor)
            box = np.array([x, y, w, h])
            boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)
    for i in indexes:
        classify, score, box = classifys[i], scores[i], boxes[i]
        print(class_id, score, box)
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        label = f'{classify}: {score:.2f}'
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    predict = cv2.resize(img, (1600, 900))
    cv2.imshow("img", predict)
    cv2.waitKey(0)


if __name__ == '__main__':
    recognize('3.jpg',0.3)

