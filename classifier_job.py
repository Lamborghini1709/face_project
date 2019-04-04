
"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os,shutil
import argparse
import tensorflow as tf
import numpy as np
import facenet
import detect_face
import random
from time import sleep
import math
import pickle
from sklearn.svm import SVC
import face
import cv2



def folder_vs_classifier(vs_dir,face_recognition):
    prob_list = []
    # classifier_filename = './models/classifier_align_en.pkl'
    # with open(classifier_filename, 'rb') as infile:
    #     model, class_names = pickle.load(infile)
    # threshold_prob = 2 / len(class_names)
    for root, dirs, files in os.walk(vs_dir):
        for file in files[3:]:
            file_img = os.path.join(root, file)
            img = cv2.imread(file_img)
            try:
                faces, prob,predictions = face_recognition.identify(img)
            except:
                faces = []
            if len(faces) >= 1:
                prob_list.append(prob)
                while len(prob_list) == 25:
                    if np.mean(prob_list) < np.sum(predictions[0][-4:-1]):
                        print('概率小于阈值，需要训练分类器')
                        return True
                    else:
                        print('概率大于0阈值，无需重新训练')
                        return False
            else:
                print('未检测到人脸')
                return False





def move_files(src_path,dst_path):
    files = os.listdir(src_path)
    for file in files:
        file_path = os.path.join(src_path,file)
        shutil.move(file_path,dst_path)



def job0():
    input_dir = './save_image'
    output_dir = './data'
    face_recognition = face.Recognition()
    if not os.path.exists(input_dir):
        return False
    saved_list = []
    for root, dirs, files in os.walk(input_dir):
        for dir in dirs:
            # print(file)
            saved_list.append(os.path.join(root, dir))
    print(saved_list)
    if len(saved_list) == 0:
        return False
    for dir in saved_list:
        for root, dirs, files in os.walk(dir):
            saved_imgs_lens = len(files)
            print('文件数',saved_imgs_lens)
            print(folder_vs_classifier(dir,face_recognition))
            if saved_imgs_lens >= 40 and folder_vs_classifier(dir,face_recognition):
                print('新用户',dir)
                new_dir = os.path.join(output_dir,dir.split('/')[-1])
                if not os.path.exists(new_dir):
                    os.mkdir(new_dir)
                print(new_dir)
                move_files(dir,new_dir)
                print('复制文件夹成功')
                shutil.rmtree(dir)
                print('复制后删除文件夹')
                return True
            else:
                print('直接删除文件夹')
                shutil.rmtree(dir)
                return False



#align data
def job1():
    print('start job1')
    # input_dir = './data'
    input_dir = './video_image'
    # output_dir = './data_align_en'
    output_dir = './video_image_align_410'
    image_size = 182
    margin = 44
    gpu_memory_fraction = 1.0
    detect_multiple_faces = False
    random_order = True

    sleep(random.random())
    output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Store some git revision info in a text file in the log directory
    src_path ,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    dataset = facenet.get_dataset(input_dir)

    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction, allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)

    with open(bounding_boxes_filename, "w" ,encoding='utf-8') as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        if random_order:
            random.shuffle(dataset)
        for cls in dataset:
            output_class_dir = os.path.join(output_dir, cls.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
                if random_order:
                    random.shuffle(cls.image_paths)
            for image_path in cls.image_paths:
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir, filename +'.png')
                print(image_path)
                if not os.path.exists(output_filename):
                    try:
                        img = misc.imread(image_path)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim <2:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                        img = img[: ,: ,0:3]

                        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                        nrof_faces = bounding_boxes.shape[0]
                        if nrof_faces >0:
                            det = bounding_boxes[: ,0:4]
                            det_arr = []
                            img_size = np.asarray(img.shape)[0:2]
                            if nrof_faces >1:
                                if detect_multiple_faces:
                                    for i in range(nrof_faces):
                                        det_arr.append(np.squeeze(det[i]))
                                else:
                                    bounding_box_size = (det[: ,2 ] -det[: ,0] ) *(det[: ,3 ] -det[: ,1])
                                    img_center = img_size / 2
                                    offsets = np.vstack([ (det[: ,0 ] +det[: ,2] ) / 2 -img_center[1], (det[: ,1 ] +det[: ,3] ) / 2 -img_center[0] ])
                                    offset_dist_squared = np.sum(np.power(offsets ,2.0) ,0)
                                    index = np.argmax \
                                        (bounding_box_size -offset_dist_squared *2.0) # some extra weight on the centering
                                    det_arr.append(det[index ,:])
                            else:
                                det_arr.append(np.squeeze(det))

                            for i, det in enumerate(det_arr):
                                det = np.squeeze(det)
                                bb = np.zeros(4, dtype=np.int32)
                                bb[0] = np.maximum(det[0] - margin / 2, 0)
                                bb[1] = np.maximum(det[1] - margin / 2, 0)
                                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                                cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                                scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                                nrof_successfully_aligned += 1
                                filename_base, file_extension = os.path.splitext(output_filename)
                                if detect_multiple_faces:
                                    output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                                else:
                                    output_filename_n = "{}{}".format(filename_base, file_extension)
                                misc.imsave(output_filename_n, scaled)
                                text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                        else:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)



#train classifier
def job2():
    print('start job2')
    mode = 'TRAIN'
    # data_dir = 'data_align_en'
    data_dir = './video_image_align_410'
    model = './models/20180408-102900'
    classifier_filename = './models/classifier_align_en_final_0404_1.pkl'
    use_split_dataset = False
    test_data_dir = './test'
    batch_size = 90
    image_size = 160
    seed = 666
    min_nrof_images_per_class = 20
    nrof_train_images_per_class = 10
    with tf.Graph().as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:

            np.random.seed(seed=seed)

            if use_split_dataset:
                dataset_tmp = facenet.get_dataset(data_dir)
                train_set, test_set = split_dataset(dataset_tmp, min_nrof_images_per_class,
                                                    nrof_train_images_per_class)
                if (mode == 'TRAIN'):
                    dataset = train_set
                elif (mode == 'CLASSIFY'):
                    dataset = test_set
            else:
                dataset = facenet.get_dataset(data_dir)

            # Check that there are at least one training image per class
            for cls in dataset:
                assert (len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')

            paths, labels = facenet.get_image_paths_and_labels(dataset)

            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

            classifier_filename_exp = os.path.expanduser(classifier_filename)

            if (mode == 'TRAIN'):
                # Train classifier
                print('Training classifier')
                model = SVC(kernel='linear', probability=True)
                model.fit(emb_array, labels)

                # Create a list of class names
                class_names = [cls.name.replace('_', ' ') for cls in dataset]

                # Saving classifier model
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)

            elif (mode == 'CLASSIFY'):
                # Classify images
                print('Testing classifier')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                for i in range(len(best_class_indices)):
                    print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

                accuracy = np.mean(np.equal(best_class_indices, labels))
                print('Accuracy: %.3f' % accuracy)


def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths) >= min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set



# if __name__ == '__main__':
#
#     # main(parse_arguments(sys.argv[1:]))
#     # job1()

def schedule_jobs():
    detect_dir = 'save_image'
    saved_list = []
    for root, dirs, files in os.walk(detect_dir):
        for dir in dirs:
            # print(file)
            saved_list.append(os.path.join(root, dir))
    print(saved_list)
    while len(saved_list) != 0:
        print('*'*40)
        if job0():
            job1()
            job2()
        else:
            pass

        saved_list = []
        for root, dirs, files in os.walk(detect_dir):
            for dir in dirs:
                # print(file)
                saved_list.append(os.path.join(root, dir))

# if __name__ == '__main__':
#     # schedule_jobs()
#     job1()
#     job2()