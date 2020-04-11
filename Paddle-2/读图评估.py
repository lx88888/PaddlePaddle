#读取预测图像，进行预测

def load_image(path):
    img = Image.open(path)
    img = img.resize((100, 100), Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = img.transpose((2, 0, 1))
    img = img/255.0
    print(img.shape)
    return img

#构建预测动态图过程
with fluid.dygraph.guard():
    infer_path = '手势.JPG'
    model=MyDNN()#模型实例化
    model_dict,_=fluid.load_dygraph('MyDNN')
    model.load_dict(model_dict)#加载模型参数
    model.eval()#评估模式
    infer_img = load_image(infer_path)
    infer_img=np.array(infer_img).astype('float32')
    infer_img=infer_img[np.newaxis,:, : ,:]
    infer_img = fluid.dygraph.to_variable(infer_img)
    result=model(infer_img)
    display(Image.open('手势.JPG'))
    print(np.argmax(result.numpy()))