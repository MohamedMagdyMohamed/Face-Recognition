import cv2
import dlib
import numpy as np
import os


face_detector = dlib.get_frontal_face_detector()
#face_detector = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)        


def enhance_image(image):
    image_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    Y, Cr, Cb = cv2.split(image_YCrCb)
    Y = cv2.equalizeHist(Y)
    image_YCrCb = cv2.merge([Y, Cr, Cb])
    image = cv2.cvtColor(image_YCrCb, cv2.COLOR_YCR_CB2BGR)
    return image


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def scale_faces(face_rects, down_scale=1.5):
    faces = []
    for face in face_rects:
        scaled_face = dlib.rectangle(int(face.left() * down_scale),
                                    int(face.top() * down_scale),
                                    int(face.right() * down_scale),
                                    int(face.bottom() * down_scale))
        faces.append(scaled_face)
    return faces


def detect_faces(image, down_scale=1.5):
    image_scaled = cv2.resize(image, None, fx=1.0/down_scale, fy=1.0/down_scale, 
                              interpolation=cv2.INTER_LINEAR)
    faces = face_detector(image_scaled, 0)
    #faces = [face.rect for face in faces]
    faces = scale_faces(faces, down_scale)
    return faces


cam = cv2.VideoCapture(0)

assure_path_exists("dataSet/")

id = input('enter your id')
sampleNum = 1
assure_path_exists("dataSet/" + str(id) + "/")

while True:
   # Read the video frame
   ret, image =cam.read()
   image = cv2.flip(image,1)
   image1 = image.copy()

   enhance_image1 = enhance_image(image)

   adjust_gamma1 = adjust_gamma(enhance_image1, gamma=0.5)
   
   gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
   faces = detect_faces(gray, down_scale = 1.5)

   for face in faces:
       x,y,w,h = face.left(), face.top(), face.right(), face.bottom()
       cv2.rectangle(image, (x,y), (w,h), (255,200,150), 2)
       if sampleNum < 10:
           cv2.imwrite("dataSet/" + str(id) + "/" + str(id) + "_" + str(sampleNum) + ".jpg", image1)
           sampleNum=sampleNum+1
       
   cv2.imshow('Face Recognition',image)
   

   if cv2.waitKey(10) & 0xFF == ord('q'):
       break

cam.release()

# Close all windows
cv2.destroyAllWindows()