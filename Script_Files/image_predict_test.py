import os
import sys
import cv2
import torch
import datetime
def load_model(model_weights):
    print(datetime.datetime.now())
    model=torch.hub.load('D:/git_repositories/yolov5','custom',model_weights,source='local',force_reload=True) #to load model
    print(datetime.datetime.now())
    return model

def predict_image(image_path,model_weights):
    model=load_model(model_weights)
    img=cv2.imread(image_path)
    print(datetime.datetime.now())
    result=model(img)
    print(datetime.datetime.now())
    result_image=result.render()  #to render predicted image
    cv2.imwrite("C:/Users/arun1/Downloads/test_image_result_pt.jpg",result_image[0])
    
if __name__ == "__main__":
    model_weights="D:/Hackathon/tn_hack/weights/yolov5s.pt"
    image_path="C:/Users/arun1/Downloads/test_image.jpg"
    predict_image(image_path,model_weights)
    