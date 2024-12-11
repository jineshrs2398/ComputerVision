import cv2
import mediapipe as mp
import argparse
import os

def process_img(img, face_detection):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)
    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box
            H, W, _ = img.shape

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            x1 = int(x1 *W)
            y1 = int(y1 *H)
            w = int(w *W)
            h = int(h *H)
            #cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0, 255, 0), 10)                
            face_region = img[y1:y1+h, x1:x1+w]

            #blur face
            img[y1:y1+h, x1:x1+w] = cv2.blur(img[y1:y1+h, x1:x1+w], (50, 50))

    return img

args = argparse.ArgumentParser()
args.add_argument("--mode", default='webcam') #mode selection of user what the user wants to do 
args.add_argument("--filePath", default=None)

args = args.parse_args()

#detect faces
mp_face_detection = mp.solutions.face_detection # created an object to detect all the faces
with mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detection:
    if args.mode in ["image"]:
        img = cv2.imread(args.filePath)

        img = process_img(img, face_detection)

        cv2.imshow('img',img)

    elif args.mode in ['webcam']:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        while ret:
            frame = process_img(frame, face_detection)
            cv2.imshow('video', frame)
            ret, frame = cap.read()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()




#save image
cv2.waitKey(0)
cv2.destroyAllWindows()