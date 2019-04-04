import os

input_dir = './video_image'


image_dirs = []
for root,dirs,files in os.walk(input_dir):
    for dir in dirs:
        image_dirs.append(os.path.join(root,dir))
print(image_dirs)


for image_dir in image_dirs:
    f = os.listdir(image_dir)
    n = 0
    for i in f:
        oldname = image_dir + '/' + f[n]
        newname = image_dir + '/' + 'a' + str(n+1) + '.jpg'
        os.rename(oldname,newname)
        n += 1