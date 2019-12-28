import numpy as np
import _pickle as cPickle
import cv2
import dlib
import glob
import time

start_time = time.time()

shape_predictor = dlib.shape_predictor("models/shape_predictor_5_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
face_detector = dlib.get_frontal_face_detector()
#face_detector = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")

def extract_face_embeddings(image, face_rect):
    shape = shape_predictor(image, face_rect)
    face_embedding = face_recognizer.compute_face_descriptor(image, shape)
    #print("face_embedding")
    #print(face_embedding)
    face_embedding = [x for x in face_embedding]
    #print("face_embedding  ->  x")
    #print(face_embedding)
    face_embedding = np.array(face_embedding, dtype="float32")[np.newaxis, :]
    #print("face_embedding  ->  array")
    #print(face_embedding)
    return face_embedding


def add_embeddings(embedding, label, 
                   embeddings_path="face_embeddings.npy", 
                   labels_path="labels.pickle"):
    first_time = False
    try:
        embeddings = np.load(embeddings_path)
        labels = cPickle.load(open(labels_path, "rb"))
        #print("labels")
        #print(labels)
    except IOError:
        first_time = True
    if first_time:
        embeddings = embedding
        labels = [label]
    else:
        embeddings = np.concatenate([embeddings, embedding])
        labels.append(label)
        #print("labels 2")
        #print(labels)
        #print(embeddings)
    np.save(embeddings_path, embeddings)
    with open(labels_path, "wb") as f:
        cPickle.dump(labels, f)
    return True


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


def enroll_face(image, label,
                embeddings_path="face_embeddings.npy",
                labels_path="labels.pickle", down_scale=1.5):
    faces = detect_faces(image, down_scale)
    if len(faces)<1:
        return False
    if len(faces)>1:
        raise ValueError("Multiple faces not allowed for enrolling")
    face = faces[0]
    face_embeddings = extract_face_embeddings(image, face)
    add_embeddings(face_embeddings, label)
    return True


filetypes = ["png", "jpg"]
dataset = "dataSet"
imPaths = []
for filetype in filetypes:
    imPaths += glob.glob("{}/*/*.{}".format(dataset, filetype))
for path in imPaths:
    label = path.split("/")[-2]
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    enroll_face(image, label)
print("--- %s seconds ---" % (time.time() - start_time))