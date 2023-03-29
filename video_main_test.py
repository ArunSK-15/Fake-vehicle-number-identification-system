import cv2
import torch
import pytesseract

def load_model(weights):
    model=torch.hub.load('yolov5','custom',weights,source='local') #to load model
    return model

def ocr(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(img)
    return text

def identify(video_path,vehicle_detection_weights,plate_detection_weights):
    vehicle_detector,plate_detector=load_model(vehicle_detection_weights),load_model(plate_detection_weights)
    video = cv2.VideoCapture(video_path)
    frame_count=0
    while(video.isOpened()):
        ret, frame = video.read()
        vehicles=vehicle_detector(frame)
        if vehicles.pred[0].shape[0]:
            vehicle_df=(vehicles.pandas().xyxy[0])
            for i in range(len(vehicle_df.index)):
                vehicle=frame[int(vehicle_df['ymin'][i]):int(vehicle_df['ymax'][i]),int(vehicle_df['xmin'][i]):int(vehicle_df['xmax'][i])]
                plate=plate_detector(vehicle)    
                if plate.pred[0].shape[0]:
                    plate_df=(plate.pandas().xyxy[0])
                    for i in range(len(plate_df.index)):
                        plate_img = vehicle[int(plate_df['ymin'][i]):int(plate_df['ymax'][i]),int(plate_df['xmin'][i]):int(plate_df['xmax'][i])]
                        vehicle_number=ocr(plate_img)
                        frame=cv2.putText(frame,text=vehicle_number,org=[int(vehicle_df['xmin'][i]),int(vehicle_df['ymin'][i]-3)],color=(255,25,255),font=cv2.FONT_HERSHEY_SIMPLEX,fontScale=4,fontFace=4)
        final_image=vehicles.render()
        cv2.imshow('Video',final_image[0])
        frame_count=frame_count+1
        print(frame_count)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    vehicle_detection_weights="vehicles.pt"
    plate_detection_weights="number_plate.pt"
    video_path="test_image.jpg"
    identify(video_path,vehicle_detection_weights,plate_detection_weights)

