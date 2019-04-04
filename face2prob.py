
import cv2
import face
import os
import numpy as np

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#单张图片预测
def img2prob(face_recognition):

    file = './test/test26.jpg'
    img = cv2.imread(file)
    faces, prob, predictions = face_recognition.identify(img)
    print(prob)
    print(faces[0].name)
    print(len(faces))
    print(predictions)
    face_bb = faces[0].bounding_box.astype(int)
    cropped = img[face_bb[1]:face_bb[3], face_bb[0]:face_bb[2], :]
    # cv2.rectangle(img,
    #               (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
    #               (0, 255, 0), 2)
    cv_show('test26.jpg',cropped)


#一组图片预测
def group2prob(face_recognition):
    img_file = ['./test/test1.jpg','./test/test2.jpg','./test/test3.jpg','./test/test4.png','./test/test5.png','./test/test6.png',\
                './test/test7.png','./test/timg.jpeg','./test/test9.png']
    probs = []
    for file in img_file:

        img = cv2.imread(file)
        try:
            faces, prob, predictions = face_recognition.identify(img)
            probs.append(prob)
        except:
            print(file)
            continue
    print('max prob :')
    print(np.max(probs))
    print('mean prob:')
    print(np.mean(probs))


#一个文件夹图片预测
def dir2prob(face_recognition):
    input_data = './video_image/likaiming'
    img_files = []
    probs = []
    for root,dirs,files in os.walk(input_data):
        for file in files:
            # print(os.path.join(root,file))
            img_files.append(os.path.join(root,file))
    print(img_files)
    for file in img_files:
        img = cv2.imread(file)
        try:
            faces, prob, predictions = face_recognition.identify(img)
            # if prob > np.sum(predictions[0][-3:-1]):
            if faces[0].name == 'likaiming':
                prob = 1
            else:
                prob = 0
            probs.append(prob)
        except:
            print(file)
            continue
    print('max prob :')
    print(np.max(probs))
    print('mean prob:')
    print(np.mean(probs))
    print('min prob:')
    print(np.min(probs))
    print('acc:')
    print(np.sum(probs)/len(probs))


#整体做预测
def inputdir2prob(face_recognition):
    input_data = './video_image/'
    img_dir = []
    mean_accu = []
    for root,dirs,files in os.walk(input_data):
        for dir in dirs:
            img_dir.append(os.path.join(root,dir))
    # print(img_dir)
    for i in img_dir:
        img_file = []
        prob_list = []
        for root,dirs,files in os.walk(i):
            for file in files:
                img_file.append(os.path.join(root,file))
        # print(img_file)
        for file in img_file:
            # print(file)
            try:
                img = cv2.imread(file)
                faces, prob, predictions = face_recognition.identify(img)
                if prob > np.sum(predictions[0][-3:-1]):
                # if faces[0].name == i.split('/')[-1]:
                    prob = 1
                else:
                    prob = 0
                    print('预测错误')
                #     print(file)
                #     # os.remove(file)
                # prob_list.append(prob)
            except:
                print(file)
                print('发生异常')
                # os.remove(file)
                continue
        print(np.sum(prob_list)/len(prob_list))
        mean_accu.append(np.sum(prob_list)/len(prob_list))
    print('accu:',np.mean(mean_accu))


# 清洗数据
def cleardata(face_recognition):
    input_data = './data/'
    img_dir = []
    for root,dirs,files in os.walk(input_data):
        for dir in dirs:
            img_dir.append(os.path.join(root,dir))
    print(img_dir)
    img_file = []
    for i in img_dir:
        for root,dirs,files in os.walk(i):
            for file in files:
                img_file.append(os.path.join(root,file))
    print(len(img_file))
    prob_list = []
    for i in range(len(img_file)):
        img = cv2.imread(img_file[i])
        # print(img_file[i])

        # print(prob)
        # print(predictions)
        try:
            faces, prob, predictions = face_recognition.identify(img)
            np.sum(predictions[0][-3:-1])
        except:
            print(img_file[i])
            # os.remove(img_file[i])    #删除文件
            continue

        if prob > np.sum(predictions[0][-3:-1]):
            prob = 1
        else:
            prob = 0
        prob_list.append(prob)

    print(np.sum(prob_list)/len(prob_list))


if __name__ == '__main__':
    face_recognition = face.Recognition()
    dir2prob(face_recognition)















