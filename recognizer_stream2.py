
import pickle
import mxnet as mx 

from tensorflow.keras.models import load_model
# from mtcnn import MTCNN
from imutils import paths
from common import face_preprocess
import numpy as np
from deploy import face_model
import argparse
import time
import dlib
import cv2
import os
import facenet
import detect_face
import tensorflow.compat.v1 as tf

from imutils import face_utils
from Ultral_detection.box_utils import *

import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare

ap = argparse.ArgumentParser()

ap.add_argument("--mymodel", default="outputs/my_model.h5",
    help="Path to recognizer model")
ap.add_argument("--le", default="outputs/le.pickle",
    help="Path to label encoder")
ap.add_argument("--embeddings", default="outputs/embeddings.pickle",
    help='Path to embeddings')
ap.add_argument("--video-out", default="../datasets/videos_output/stream_test.mp4",
    help='Path to output video')


ap.add_argument('--image-size', default='112,112', help='')
ap.add_argument('--model', default='models/model-y1-test2/model,0', help='path to load model.')
ap.add_argument('--ga-model', default='', help='path to load model.')
ap.add_argument('--gpu', default=0, type=int, help='gpu id')
ap.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
ap.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
ap.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')

args = ap.parse_args()

onnx_path = 'Ultral_detection/UltraLight/models/ultra_light_640.onnx'
onnx_model = onnx.load(onnx_path)
predictor = prepare(onnx_model)
ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name

shape_predictor = dlib.shape_predictor('Ultral_detection/FacialLandmarks/shape_predictor_68_face_landmarks.dat')
fa = face_utils.facealigner.FaceAligner(shape_predictor, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))



npy='./npy'

# with tf.Graph().as_default():
#     config = tf.ConfigProto(log_device_placement=True,
#                             allow_soft_placement=True,  # Cho phép chuyển đổi tự động sang thiết bị được hỗ trợ khi không tìm thấy thiết bị 
#                             )
#     # config.gpu_options.allow_growth = True
#     config.gpu_options.per_process_gpu_memory_fraction = 0.1
#     sess = tf.Session(config=config)
#     with sess.as_default():
        # pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
        # Load embeddings and labels
data = pickle.loads(open(args.embeddings, "rb").read())
le = pickle.loads(open(args.le, "rb").read())

embeddings = np.array(data['embeddings'])
labels = le.fit_transform(data['names'])


# Initialize faces embedding model
embedding_model =face_model.FaceModel(args)

# Load the classifier model
model = load_model(args.mymodel)

# Define distance function
def findCosineDistance(vector1, vector2):
    """
    Calculate cosine distance between two vector
    """
    vec1 = vector1.flatten()
    vec2 = vector2.flatten()

    a = np.dot(vec1.T, vec2)
    b = np.dot(vec1.T, vec1)
    c = np.dot(vec2.T, vec2)
    return 1 - (a/(np.sqrt(b)*np.sqrt(c)))

def CosineSimilarity(test_vec, source_vecs):
    """
    Verify the similarity of one vector to group vectors of one class
    """
    cos_dist = 0
    for source_vec in source_vecs:
        cos_dist += findCosineDistance(test_vec, source_vec)
    return cos_dist/len(source_vecs)

# Initialize some useful arguments
cosine_threshold = 0.75
proba_threshold = 0.85
comparing_num = 5
trackers = []
texts = []
frames = 0

# Start streaming and recording
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
save_width = 600
save_height = int(600/frame_width*frame_height)
video_out = cv2.VideoWriter(args.video_out, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (save_width,save_height))

while True:
    ret, frame = cap.read()
    frames += 1
    # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (save_width, save_height))
    if frame is not None:
        h, w, _ = frame.shape

        # preprocess img acquired
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert bgr to rgb
        img = cv2.resize(img, (640, 480)) # resize
        img_mean = np.array([127, 127, 127])
        img = (img - img_mean) / 128
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        confidences, boxes = ort_session.run(None, {input_name: img})
        boxes, label, probs = predict(w, h, confidences, boxes, 0.7)
        for i in range(boxes.shape[0]):
            try:
                box = boxes[i, :]
                x1, y1, x2, y2 = box
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                shape = shape_predictor(gray, dlib.rectangle(left = x1, top=y1, right=x2, bottom=y2))
                
                shape = face_utils.shape_to_np(shape)
                five_point = np.array([(shape[37]+shape[38]+shape[40]+shape[41])/4,(shape[43]+shape[44]+shape[47]+shape[46])/4,  shape[30], shape[48],   shape[54]])

                # print(shape)
                landmarks = five_point.reshape(1,10)
                # print(five_point)
                bbox = np.array([box[0], box[1], box[2], box[3]])
                landmarks = np.array([landmarks[i][0], landmarks[i][2], landmarks[i][4], landmarks[i][6], landmarks[i][8],
                                landmarks[i][1], landmarks[i][3], landmarks[i][5], landmarks[i][7], landmarks[i][9]])

                landmarks = landmarks.reshape((2,5)).T
                nimg = face_preprocess.preprocess(frame, bbox, landmarks, image_size='112,112')
                nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                nimg = np.transpose(nimg, (2,0,1))
                embedding = embedding_model.get_feature(nimg).reshape(1,-1)

                text = "Unknown"

                # Predict class
                preds = model.predict(embedding)
                preds = preds.flatten()
                # Get the highest accuracy embedded vector
                j = np.argmax(preds)
                proba = preds[j]
                # Compare this vector to source class vectors to verify it is actual belong to this class
                match_class_idx = (labels == j)
                match_class_idx = np.where(match_class_idx)[0]
                # print(match_class_idx)
                selected_idx = np.random.choice(match_class_idx, comparing_num)
                compare_embeddings = embeddings[selected_idx]
                # # Calculate cosine similarity
                cos_similarity = CosineSimilarity(embedding, compare_embeddings)
                if cos_similarity < cosine_threshold and proba > proba_threshold:
                    name = le.classes_[j]
                    text = "{}".format(name)
                    print("Recognized: {} <{:.2f}>".format(name, proba*100))
                    # Start tracking
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                tracker.start_track(frame, rect)
                trackers.append(tracker)
                texts.append(text)

                y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
                cv2.putText(frame, text, (int(bbox[0]), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0), 2)
            except:
                pass
    else:
        for tracker, text in zip(trackers,texts):
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            cv2.rectangle(frame, (startX, startY), (endX, endY), (255,0,0), 2)
            cv2.putText(frame, text, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)


    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("Frame", frame)
    video_out.write(frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

video_out.release()
cap.release()
cv2.destroyAllWindows()
