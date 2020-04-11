import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 

# 待预测图片
test_img_path = ['E:\\Paddle\\Paddle-3\\test\\1.jpg']
img = mpimg.imread(test_img_path[0]) 

# 展示待预测图片
plt.figure(figsize=(10,10))
plt.imshow(img) 
plt.axis('off') 
plt.show()

with open('test/test.txt', 'r') as f:
    test_img_path=[]
    for line in f:
        test_img_path.append(line.strip())
print(test_img_path)

import paddlehub as hub
module = hub.Module(name="pyramidbox_lite_mobile_mask")

input_dict = {"image": test_img_path}

# 口罩检测预测
results = module.face_detection(data=input_dict)
for result in results:
    print(result)

# 预测结果展示
img = mpimg.imread("E:\\Paddle\\Paddle-3\\detection_result\\1.jpg")
plt.figure(figsize=(10,10))
plt.imshow(img) 
plt.axis('off') 
plt.show()