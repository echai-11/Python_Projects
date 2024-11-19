import cv2

#A Haar Cascade is basically a classifier which is used to detect particular objects from the source.

#The haarcascade_frontalface_default.xml is a haar cascade designed by OpenCV to detect the frontal face.
detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#imported image
#https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html#a473055e77dd7faa4d26d686226b292c1
imp_img= cv2.VideoCapture("images/elon.jpg")

#returns if image is read, and if so image pixel
#https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html#a473055e77dd7faa4d26d686226b292c1
res,img = imp_img.read()

#convert to greyscale image
#https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect faces of different sizes
#https://docs.opencv.org/3.4/d1/de5/classcv_1_1CascadeClassifier.html#aaf8181cb63968136476ec4204ffca498
#parameters: image, scaleFactor, minNeighbors
faces = detect.detectMultiScale(gray,1.3,5)

#draw a rectangle or square over the image
# (x, y+w), (x+w, y+w)
# (x,y), (x+w, y)

for (x,y,w,h) in faces:
    #paramater< image, pt1, p2 - righthand side top most, color of image, width of border
    cv2.rectangle(img, (x,y),(x+w, y+h), (255,255,0),2)

#show image
#parameters: title, image
cv2.imshow("Elon Image", img)
#milliseconds for how long to open image
cv2.waitKey(5000)
#release image
imp_img.release()
cv2.destroyAllWindows()




