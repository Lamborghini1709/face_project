

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet
import detect_face


def main(image_files):
    margin = 44
    image_size = 160
    gpu_memory_fraction = 1.0
    model = './models/20180408-102900'
    images = load_and_align_data(image_files, image_size, margin, gpu_memory_fraction)
    with tf.Graph().as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # Load the model
            facenet.load_model(model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb = sess.run(embeddings, feed_dict=feed_dict)
    return emb


def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction, allow_growth=True)
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    # print(len(image_paths))
    tmp_image_paths = copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        print(img.shape)
        print('*************')
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
            # image_paths.remove(image)
            print("can't detect face, remove ")
            continue
        # print('type of bounding_boxes',type(bounding_boxes))
        # print('shape of bounding_boxes',bounding_boxes.shape)
        # print(bounding_boxes)
        det = np.squeeze(bounding_boxes[0, 0:4])
        # print('shape of det',det.shape)
        # print(det)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        # print(bb)
        # misc.imshow(img)
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        # misc.imshow(cropped)
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images


if __name__ == '__main__':
    img = ['./test/test2.jpg']
    img2 = ['./test/test6.png']
    emb = main(img)
    emb2 = main(img2)
    euc = float('%.2f' % np.sqrt(np.sum(np.square(np.subtract(emb[0,:], emb2[0,:])))))
    print(euc)