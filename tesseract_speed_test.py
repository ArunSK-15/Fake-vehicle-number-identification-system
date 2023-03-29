import pytesseract
from PIL import Image
import datetime

# Load the Tesseract engine
print(datetime.datetime.now())
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Define a function to process an image
def process_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

# Use the process_image function to process multiple images
print(datetime.datetime.now())
text1 = process_image('D:/Hackathon/tn_hack/number_test.jpeg')
print(text1)
print(datetime.datetime.now())
text2 = process_image('D:/Hackathon/tn_hack/number_test.jpeg')
print(text2)
print(datetime.datetime.now())
text3 = process_image('D:/Hackathon/tn_hack/number_test.jpeg')
print(text3)
print(datetime.datetime.now())
