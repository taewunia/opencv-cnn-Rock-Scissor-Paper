import torch
import cv2
from PIL import Image

def camera(model, cap, pre_process, device, classes):
    cap = cv2.VideoCapture(cap)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame1 = Image.fromarray(frame1)
        frame1 = pre_process(frame1)
        frame1 = frame1.unsqueeze(0)
        frame1 = frame1.to(device)
        result = model(frame1)
        print(result)
        result = torch.softmax(result, dim=1)
        result, index = torch.max(result, dim=1)
        index = int(index)
        result = result.item()
        display_text = f"{classes[index]} {result:.3f}"
        cv2.putText(frame, display_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()