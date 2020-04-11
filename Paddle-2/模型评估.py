import os
import time
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from multiprocessing import cpu_count
from paddle.fluid.dygraph import Pool2D,Conv2D
from paddle.fluid.dygraph import Linear

# 定义训练集和测试集的reader
def data_mapper(sample):
    img, label = sample
    img = Image.open(img)
    img = img.resize((100, 100), Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = img.transpose((2, 0, 1))
    img = img/255.0
    return img, label


def data_reader(data_list_path):
    def reader():
        with open(data_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img, label = line.split('\t')
                yield img, int(label)
    return paddle.reader.xmap_readers(data_mapper, reader, cpu_count(), 512)
 
# 用于测试的数据提供器
test_reader = paddle.batch(reader=data_reader('./test_data.list'), batch_size=32) 


#定义DNN网络
class MyDNN(fluid.dygraph.Layer):
    def __init__(self):
        super(MyDNN,self).__init__()
        self.hidden1 = Linear(100,100,act='relu')
        self.hidden2 = Linear(100,100,act='relu')
        self.hidden3 = Linear(100,100,act='relu')
        self.hidden4 = Linear(3*100*100,10,act='softmax')
    def forward(self,input):
        #print(input,shape)
        x = self.hidden1(input)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = fluid.layers.reshape(x, shape=[-1,3*100*100])
        y = self.hidden4(x)
       
        return y



#模型校验
with fluid.dygraph.guard():
    accs = []
    model_dict, _ = fluid.load_dygraph('MyDNN')
    model = MyDNN()
    model.load_dict(model_dict) #加载模型参数
    model.eval() #训练模式
    for batch_id,data in enumerate(test_reader()):#测试集
        images=np.array([x[0].reshape(3,100,100) for x in data],np.float32)
        labels = np.array([x[1] for x in data]).astype('int64')
        labels = labels[:, np.newaxis]

        image=fluid.dygraph.to_variable(images)
        label=fluid.dygraph.to_variable(labels)
        
        predict=model(image)       
        acc=fluid.layers.accuracy(predict,label)
        accs.append(acc.numpy()[0])
        avg_acc = np.mean(accs)
    print(avg_acc)
 