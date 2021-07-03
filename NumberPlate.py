import cv2
import pytesseract
import numpy as np
import face_mask_detection
import pyrebase

config = {
	"apiKey": "AIzaSyDJ_0eynkJSKtdnoKOnwZrTM-k4wJGyMhY",
    "authDomain": "vehiclechallan-143ac.firebaseapp.com",
    "projectId": "vehiclechallan-143ac",
    "databaseURL":"https://vehiclechallan-143ac-default-rtdb.firebaseio.com/",
    "storageBucket": "vehiclechallan-143ac.appspot.com",
    "messagingSenderId": "950755837235",
    "appId": "1:950755837235:web:13cfad1abc208315226a46",
    "measurementId": "G-7ZFKXERR4B"
}

firebase = pyrebase.initialize_app(config)
database = firebase.database()


img = cv2.imread('./Data/img1.jpg')

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
states = {"TS":"Telangana","AP":"Andhra Pradesh","MH":"Maharashtra","KA":"Karnataka"}

bad_chars = ['ee', 'SE',' ']

def extract_num(img):
	original_img =cv2.resize(img,(400,400))
	y=600
	x=280
	h=400
	w=370
	img = img[y:y+h, x:x+w]
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	nplate = cascade.detectMultiScale(gray,1.1,4)
	for (x,y,w,h) in nplate:
		a,b = (int(0.02*img.shape[0]), int(0.025*img.shape[1]))
		plate = img[y+a:y+h-a, x+b:x+w-b, :]
		kernal = np.ones((1,1), np.uint8)
		plate = cv2.dilate(plate, kernal, iterations=1)
		plate = cv2.erode(plate, kernal, iterations=1)
		plate_gray = cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
		(thresh, plate) = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)

		read = pytesseract.image_to_string(plate)
		for i in bad_chars:
			read = read.replace(i, '')
		print(read)
		data={"Name":"Shashank","VehicleNumber":read,}
		database.push(data)
		stat = read[0:2]
		try:
			print('Bike Belongs to',states[stat])
		except:
			print('State not recognised!!')
		cv2.imshow('plate', plate)
	

	cv2.imshow("Result", original_img)
	#cv2.imwrite('result.jpg', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

b = face_mask_detection.face_mask_detector(img)
if(b==1):
	print("With Mask")
else:
	print("Vehicle fined for challan ")
	extract_num(img)