import cv2
import torch
import pytesseract

def load_model(model_weights):
    model=torch.hub.load('D:/Hackathon/tn_hack/python_scripts/yolov5','custom',model_weights,source='local') #to load model
    return model

def ocr(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(img)
    return text

def identify(image_path,vehicle_detection_weights,plate_detection_weights):
    vehicle_detector,plate_detector=load_model(vehicle_detection_weights),load_model(plate_detection_weights)
    img=cv2.imread(image_path)
    vehicles=vehicle_detector(img)
    if vehicles.pred[0].shape[0]:
        vehicles_df=(vehicles.pandas().xyxy[0])
        print(vehicles_df)
    #vehicles.save(save_dir="D:/Hackathon/tn_hack/test_folder")
    """for i in range(len(vehicles_df.index)):
            vehicle_img=img[int(vehicles_df['ymin'][i]):int(vehicles_df['ymax'][i]),int(vehicles_df['xmin'][i]):int(vehicles_df['xmax'][i])]    
            plate=plate_detector(vehicle_img)  
            if plate.pred[0].shape[0]:
                plate_df=(plate.pandas().xyxy[0])
                plate_img=vehicle_img[int(plate_df['ymin']):int(plate_df['ymax']),int(plate_df['xmin']):int(plate_df['xmax'])]
                vehicle_number=ocr(plate_img)
                print(vehicle_number)"""

if __name__ == "__main__":
    vehicle_detection_weights="D:/Hackathon/tn_hack/weights/vehicles.pt"
    plate_detection_weights="D:/Hackathon/tn_hack/weights/number_plate.pt"
    image_path="D:/Hackathon/tn_hack/test_test.jpeg"
    identify(image_path,vehicle_detection_weights,plate_detection_weights)
    
