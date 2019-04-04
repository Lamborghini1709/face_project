import cv2
import face
import time
import os
from img_vs_folder import save_or_not
from classifier_job import schedule_jobs
import tensorflow as tf
import facenet
import schedule
import numpy as np
from datetime import datetime

def list2info(face_recognition):
    img_file = ['./test/test1.jpg','./test/test2.jpg','./test/test3.jpg','./test/test4.png','./test/test5.png','./test/test6.png',\
                './test/test7.png','./test/timg.jpeg','./test/test9.png']

    number = []
    names = []
    bb = []
    prob_list = []
    for i in range(9):
        img = cv2.imread(img_file[i])
        faces,prob,predictions = face2info(img,face_recognition)
        number.append(len(faces))
        face = faces[0]
        names.append(face.name)
        bb.append(face.bounding_box)
        # if prob > np.sum(predictions[0][-3:-1]):
        if prob > 0.6:
            prob = 1
        else:
            prob = 0
        prob_list.append(prob)
        # face = faces[0]
        # print('numbers of faces:')
        # print(len(faces))
    print(number)
    # face = faces[0]
    print('face name:')
    print(names)
    print(prob_list)
    print('face bounding_box:')
    print(bb)
    # print('face image:')
    # print(face.image.shape)
    # cv_show('face image',face.image)
    # print('face container_image:')
    # print(face.container_image.shape)
    # cv_show('face.container_image',face.container_image)
    # print('face emb shape:')
    # print(face.embedding.shape)

def img2info(face_recognition):
    img = cv2.imread('./test/test10.png')
    faces, prob, predictions = face2info(img, face_recognition)
    print(len(faces))
    for index,face in enumerate(faces):
        face_bb = face.bounding_box.astype(int)
        cv2.rectangle(img,
                      (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                      (0,0,255), 2)
        cv2.putText(img, str(index), (face_bb[0], face_bb[3]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),thickness=2, lineType=2)
    cv_show('face image', img)
    # face = faces[0]
    # print('face image:')
    # print(face.image.shape)
    # cv_show('face image',face.image)
    # print('face container_image:')
    # print(face.container_image.shape)
    # cv_show('face.container_image',face.container_image)
    # print('face emb shape:')
    # print(face.embedding.shape)
    # print('face bb:')
    # print(face.bounding_box.astype(int))


def img2onlyface(face_recognition):
    img = cv2.imread('./test/test31.jpg')
    faces, prob, predictions = face2info(img, face_recognition)
    print(len(faces))
    max_area = 0
    max_index = 0
    for index,face in enumerate(faces):
        face_bb = face.bounding_box.astype(int)
        area = (face_bb[2]-face_bb[0]) * (face_bb[3]-face_bb[1])
        if area > max_area:
            max_area = area
            max_index = index
    # print(faces[max_index])
    face = faces[max_index]
    print('face image:')
    print(face.image.shape)
    cv_show('face image', face.image)
    print('face container_image:')
    print(face.container_image.shape)
    cv_show('face.container_image', face.container_image)
    print('face emb shape:')
    print(face.embedding.shape)
    print('face bb:')
    print(face.bounding_box.astype(int))


def face2info(img,face_recognition):
    faces, prob, predictions = face_recognition.identify(img)
    # print(len(faces))
    return faces,prob,predictions


def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    face_recognition = face.Recognition()
    img2info(face_recognition)
    # img2onlyface(face_recognition)
