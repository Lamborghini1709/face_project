import cv2
import face
import os


def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def img_add_flag(face_recognition):

    path = '/home/wayne/works/Projects/pythonProject/MP4edit/222/'
    filelist = []
    for root,dirs,files in os.walk(path):
        for file in files:
            filelist.append(os.path.join(root,file))
    for file in filelist:
        img =cv2.imread(file)

        faces, prob, predictions = face_recognition.identify(img)
        print(len(faces))
        for index,face in enumerate(faces):
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(img,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0,0,255), 2)
            cv2.putText(img, 'New customer', (face_bb[0], face_bb[3]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),thickness=2, lineType=2)
        cv_show('img',img)


if __name__ == '__main__':
    face_recognition = face.Recognition()
    img_add_flag(face_recognition)