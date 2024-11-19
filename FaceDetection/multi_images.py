import cv2
import glob

all_images = glob.glob("./images/*.jpg")

detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

for index,image in enumerate(all_images):
    img = cv2.imread(image)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect.detectMultiScale(gray_img, 1.3, 5)

    for (x, y, w, h) in faces:
        final_img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)

    cv2.imshow("Face Detection {}".format(index + 1), final_img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
