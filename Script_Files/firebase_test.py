import cv2
import base64
import datetime
import firebase_admin
from io import BytesIO
from firebase_admin import credentials
from firebase_admin import firestore

img=cv2.imread("image.jpg")
img=cv2.resize(img,(640,640))
retval, buffer = cv2.imencode(".jpg", img)
image_base64 = base64.b64encode(buffer).decode('utf-8')
cred = credentials.Certificate('authentication_key.json')
app = firebase_admin.initialize_app(cred)
db = firestore.client()

current_time = str(datetime.datetime.now())
doc_ref = db.collection(current_time[:10]).document(current_time[11:19])
doc_ref.set({
    'Predicted vehicle number': 'ABCD1234',
    'Place': 'PLace1',
    'Image' : image_base64,
})
