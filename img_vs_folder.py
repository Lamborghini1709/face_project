

import cv2
import face
import time
import os
import numpy as np
import pickle
import os

import cv2
import numpy as np
import tensorflow as tf
from scipy import misc

# import align.detect_face
import detect_face
import facenet


# save_img_dir = './save_image'
# if not os.path.exists(save_img_dir):
#     os.mkdir(save_img_dir)
#
# def save_img(dir,img):
#     fold_name = dir.split('/')[-2]
#     filename = os.path.join(save_img_dir,fold_name,time.time()+'.png')
#     cv2.imwrite(filename,img)
#



# gpu_memory_fraction = 0.3
# # facenet_model_checkpoint = './models/20180408-102900'
# facenet_model_checkpoint = os.path.dirname(__file__) + "/models/20180408-102900"
# # classifier_model = './models/classifier.pkl'
# classifier_model = os.path.dirname(__file__) + "/models/classifier_align_en.pkl"
# debug = False
#
#
# class Face:
#     def __init__(self):
#         self.name = None
#         self.bounding_box = None
#         self.image = None
#         self.container_image = None
#         self.embedding = None
#
#
# class Recognition:
#     def __init__(self):
#         self.detect = Detection()
#         self.encoder = Encoder()
#         self.identifier = Identifier()
#
#     def add_identity(self, image, person_name):
#         faces = self.detect.find_faces(image)
#
#         if len(faces) == 1:
#             face = faces[0]
#             face.name = person_name
#             face.embedding = self.encoder.generate_embedding(face)
#             return faces
#
#     def identify(self, image):
#         faces = self.detect.find_faces(image)
#         prob = 0
#         for i, face in enumerate(faces):
#             if debug:
#                 cv2.imshow("Face: " + str(i), face.image)
#             face.embedding = self.encoder.generate_embedding(face)
#             # print(face.embedding.shape)
#             face.name ,prob= self.identifier.identify(face)
#
#         return faces,prob
#
#
# class Identifier:
#     def __init__(self):
#
#         with open(classifier_model, 'rb') as infile:
#             self.model, self.class_names = pickle.load(infile)
#
#     def identify(self, face):
#         if face.embedding is not None:
#             # face.embedding = face.embedding + 1
#             # print(face.embedding)
#             predictions = self.model.predict_proba([face.embedding])
#             # print(predictions)
#             # print(predictions[0][np.argmax(predictions, axis=1)])
#             prob = predictions[0][np.argmax(predictions, axis=1)]
#             best_class_indices = np.argmax(predictions, axis=1)
#
#             return self.class_names[best_class_indices[0]],prob
#
#
# class Encoder:
#     def __init__(self):
#
#         self.sess = tf.Session()
#         with self.sess.as_default():
#             facenet.load_model(facenet_model_checkpoint)
#
#     def generate_embedding(self, face):
#         # Get input and output tensors
#         images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
#         embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
#         phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
#
#         prewhiten_face = facenet.prewhiten(face.image)
#
#         # Run forward pass to calculate embeddings
#         feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
#         return self.sess.run(embeddings, feed_dict=feed_dict)[0]
#
#
# class Detection:
#     # face detection parameters
#     minsize = 20  # minimum size of face
#     threshold = [0.6, 0.7, 0.7]  # three steps's threshold
#     factor = 0.709  # scale factor
#
#     def __init__(self, face_crop_size=160, face_crop_margin=32):
#         self.pnet, self.rnet, self.onet = self._setup_mtcnn()
#         self.face_crop_size = face_crop_size
#         self.face_crop_margin = face_crop_margin
#
#     def _setup_mtcnn(self):
#         with tf.Graph().as_default():
#             gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
#             sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
#             with sess.as_default():
#                 return detect_face.create_mtcnn(sess, None)
#
#     def find_faces(self, image):
#         faces = []
#
#         bounding_boxes, _ = detect_face.detect_face(image, self.minsize,
#                                                           self.pnet, self.rnet, self.onet,
#                                                           self.threshold, self.factor)
#         for bb in bounding_boxes:
#             face = Face()
#             face.container_image = image
#             face.bounding_box = np.zeros(4, dtype=np.int32)
#
#             img_size = np.asarray(image.shape)[0:2]
#             face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
#             face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
#             face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
#             face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
#             cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
#             face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')
#
#             faces.append(face)
#
#         return faces
#



def save_or_not(frame,face_recognition):
    # face_recognition = face.Recognition()

    faces_0, prob = face_recognition.identify(frame)

    dir_list = []
    save_id = 1000


    root_dir = './save_image'
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    for root, dirs, files in os.walk(root_dir):
        for dir in dirs:
            # print(file)
            dir_list.append(os.path.join(root, dir))




    if len(dir_list) == 0:
        save_label = os.path.join(root_dir,str(save_id))
        os.mkdir(save_label)
        return True,str(save_id)

    if len(faces_0) == 1:
        # print(faces_0[0].embedding.shape)
        # print(dir_list)

        euc_list = []

        for i in dir_list:
            for root, dirs, files in os.walk(i):
                if len(files) <= 3:
                    for file in files:
                        # print(file)
                        vs_file = os.path.join(root, file)
                        vs_img = cv2.imread(vs_file)
                        save_label = vs_file.split('/')[-2]
                        faces_vs, probs = face_recognition.identify(vs_img)
                        if len(faces_vs) >= 1:
                            euc = float(
                                '%.2f' % np.sqrt(
                                    np.sum(np.square(np.subtract(faces_0[0].embedding, faces_vs[0].embedding)))))
                            euc_list.append(euc)

                            # if len(files) <3:
                            if np.mean(euc_list) <= 0.8:
                                if len(files) <= 200:
                                    return True, save_label
                                else:
                                    return False, None
                            else:
                                continue
                        else:

                            continue
                else:
                    for file in files[:3]:
                        # print(file)
                        vs_file = os.path.join(root, file)
                        vs_img = cv2.imread(vs_file)
                        save_label = vs_file.split('/')[-2]
                        faces_vs, probs = face_recognition.identify(vs_img)
                        if len(faces_vs) >= 1:
                            euc = float(
                                '%.2f' % np.sqrt(np.sum(np.square(np.subtract(faces_0[0].embedding, faces_vs[0].embedding)))))
                            euc_list.append(euc)

                            # if len(files) <3:
                            if len(euc_list) == 3 :
                                if  np.mean(euc_list) <= 1:
                                    if len(files) <= 100 :
                                        return True, str(save_label)
                                    else:
                                        return False,None
                                else:
                                    continue
                            else:
                                continue
                        else:

                            continue
            euc_list = []
                    # print(euc)

        save_id += len(dir_list)
        save_label = os.path.join(root_dir, str(save_id))
        os.mkdir(save_label)

        return True,str(save_id)


    else:
        return False,None
# print(file_list)



# euc = float('%.2f' % np.sqrt(np.sum(np.square(np.subtract(emb[0, :], emb[i, :])))))

# def save_img(frame,face_id):
#     save_img_rootdir = 'save_image'
#     save_img_dir = os.path.join(save_img_rootdir, face_id)
#     if not os.path.exists(save_img_dir):
#         os.mkdir(save_img_dir)
#     filename = os.path.join(save_img_dir, str(time.time())+'.png')
#     cv2.imwrite(filename,frame)
#     print('saved images')
#
# if __name__ == '__main__':
#     for i in range(250):
#         img = cv2.imread('./test/test1.jpeg')
#         result, save_label = save_or_not(img)
#         if result:
#             save_img(img, save_label)
#         else:
#             pass
