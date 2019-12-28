import numpy as np
import cv2
import _pickle as cPickle
import dlib

shape_predictor = dlib.shape_predictor("models/shape_predictor_5_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
face_detector = dlib.get_frontal_face_detector()
#face_detector = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")


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


def extract_face_embeddings(image, face_rect):
    shape = shape_predictor(image, face_rect)
    face_embedding = face_recognizer.compute_face_descriptor(image, shape)
    face_embedding = [x for x in face_embedding]
    face_embedding = np.array(face_embedding, dtype="float32")[np.newaxis, :]
    return face_embedding


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
  #  faces = [face.rect for face in faces]
    faces = scale_faces(faces, down_scale)
    return faces


def recognize_face(embedding, embeddings, labels, threshold=0.4):
    distances = np.linalg.norm(embeddings - embedding, axis=1)
    #print("distances")
    #print(distances)
    argmin = np.argmin(distances)
    #print("argmin")
    #print(argmin)
    minDistance = distances[argmin]
    #print("minDistance")
    #print(minDistance)
    if minDistance > threshold:
        label = "Unknown"
    else:
        label = labels[argmin]
    return (label, minDistance)


embeddings = np.load("face_embeddings.npy")
labels = cPickle.load(open("labels.pickle","rb"))

cam = cv2.VideoCapture(0)
 
while True:
   # Read the video frame
   ret, image =cam.read()
   image = cv2.flip(image,1)
   image_original = image.copy()

   enhance_image1 = enhance_image(image)

   adjust_gamma1 = adjust_gamma(enhance_image1, gamma=0.5)
   adjust_gamma2 = adjust_gamma(image, gamma=0.5)
#   cv2.imshow("enhance_image1", enhance_image1)
#   cv2.imshow("adjust_gamma1", adjust_gamma1)
#   cv2.imshow("adjust_gamma2", adjust_gamma2)
   #ret, image =cam.read()
   #image = cv2.flip(image,1)
   #image_original = image.copy()
   
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   faces = detect_faces(image)
   for face in faces:
       embedding = extract_face_embeddings(image, face)
       label, minDistance = recognize_face(embedding, embeddings, labels)
       (x1, y1, x2, y2) = face.left(), face.top(), face.right(), face.bottom()
       if label == "Unknown":
           cv2.rectangle(image_original, (x1, y1), (x2, y2), (0, 0, 255), 2)
           cv2.putText(image_original, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
       else:
           cv2.rectangle(image_original, (x1, y1), (x2, y2), (255, 120, 120), 2)
           cv2.putText(image_original, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
#       print("Confidece = {0:.2f}%".format(round(100 - (0.6 - minDistance), 2)))
       print("minDistance = ", minDistance)
       perCon = 100 * (1 - minDistance)
       if(perCon > 100):
           perCon = 100
       if label != "Unknown":
           print("Confidece = {0:.2f}%".format(round(perCon, 2)))
   cv2.imshow("Image", image_original)
   if cv2.waitKey(10) & 0xFF == ord('q'):
       break

cam.release()

# Close all windows
cv2.destroyAllWindows()