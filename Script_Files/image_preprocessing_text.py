import pytesseract
import cv2

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
img="C:/Users/arun1/Downloads/number_test.jpeg"
image = cv2.imread(img)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#image=cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

image=cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
cv2.imwrite("C:/Users/arun1/Downloads/number_test_converted3.jpeg",image)
text = pytesseract.image_to_string(image,config='--psm 11 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',lang="ind",)
print(text)
