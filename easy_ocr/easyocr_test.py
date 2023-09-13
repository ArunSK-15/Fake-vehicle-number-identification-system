import cv2
import torch
import easyocr  # Import EasyOCR

def load_model(weights):
    model = torch.hub.load('E:/yolov5', 'custom', weights, source='local')  # Load YOLOv5 model
    return model

def ocr(img, reader):
    results = reader.readtext(img)
    text = ""
    for (bbox, text, prob) in results:
        text += text + " "
    return text

def identify(video_path, vehicle_detection_weights, plate_detection_weights):
    vehicle_detector, plate_detector = load_model(vehicle_detection_weights), load_model(plate_detection_weights)
    video = cv2.VideoCapture(video_path)
    frame_count = 0

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])

    while video.isOpened():
        ret, frame = video.read()
        vehicles = vehicle_detector(frame)
        if vehicles.pred[0] is not None and vehicles.pred[0].shape[0]>0:
            vehicle_df = (vehicles.pandas().xyxy[0])
            for i in range(len(vehicle_df.index)):
                vehicle = frame[int(vehicle_df['ymin'][i]):int(vehicle_df['ymax'][i]),
                          int(vehicle_df['xmin'][i]):int(vehicle_df['xmax'][i])]

                plate = plate_detector(vehicle)
                if plate.pred[0].shape[0]:
                    plate_df = (plate.pandas().xyxy[0])
                    for i in range(len(plate_df.index)):
                        plate_img = vehicle[int(plate_df['ymin'][i]):int(plate_df['ymax'][i]),
                                    int(plate_df['xmin'][i]):int(plate_df['xmax'][i])]
                        
                        if vehicle.shape[0] >= 5:
                            # Only process the image if it has a valid shape (CHW)
                            # Otherwise, skip processing or handle it differently
                            pass
                
                        vehicle_number = ocr(plate_img, reader)  # Use EasyOCR for OCR
                        frame = cv2.putText(frame, text=vehicle_number,
                                          org=[int(vehicle_df['xmin'][i]), int(vehicle_df['ymin'][i] - 3)],
                                          color=(255, 25, 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=4, thickness=2)
        final_image = vehicles.render()
        cv2.imshow('Video', final_image[0])
        frame_count = frame_count + 1
        print(frame_count)
        print(vehicle_number)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    vehicle_detection_weights="E:/SIH/TN_Hackathon/Weights/vehicles.pt"
    plate_detection_weights="E:/SIH/TN_Hackathon/Weights/number_plate.pt"
    video_path="E:/SIH/TN_Hackathon/seq.mp4"
    identify(video_path,vehicle_detection_weights,plate_detection_weights)