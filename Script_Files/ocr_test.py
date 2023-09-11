import cv2
import pytesseract

def ocr(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(image)
    print(text)

if __name__ == "__main__":
    ocr(img_path="C:/Users/arun1/Downloads/girl_on_chair_brighten.jpg")