import torch 

def load_model(model_weights):
    model=torch.hub.load('D:/git_repositories/yolov5','custom',model_weights,source='local') #load model
    model.conf=0.35 #threshold value
    print("Model loaded")

if __name__ == "__main__":
    load_model(model_weights="D:/Hackathon/tn_hack/weights/yolov5s-fp16.tflite")