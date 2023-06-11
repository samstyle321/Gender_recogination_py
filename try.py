import cv2

im=cv2.imread('my_image.jpg')
#im = (1024,1024,3)
print(im.shape)
a=(im.shape[0]*2,im.shape[1]*2)
gpu = cv2.cuda_GpuMat()
gpu.upload(im)

b=cv2.cuda.resize(gpu,a)
print(b.size())
maxX=500
maxY=500
minX = 500
minY= 500
b.adjustROI(maxY,minY, minX, maxX)
print("Adjust ROI : ",b.size())
cropped = cv2.UMat(a, [minX, maxX], [minY, maxY])
print("Umat : ",cropped.size())