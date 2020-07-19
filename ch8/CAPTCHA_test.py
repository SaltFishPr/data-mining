#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import transform as tf
from skimage.transform import resize
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops  # 用于图像分割
from sklearn.utils import check_random_state
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from pybrain.datasets.supervised import SupervisedDataSet  # 神经网络数据集
from pybrain.tools.shortcuts import buildNetwork  # 构建神经网络
from pybrain.supervised.trainers.backprop import BackpropTrainer  # 反向传播算法
from sklearn.metrics import f1_score
from nltk.corpus import words  # 导入语料库 用于生成单词
from sklearn.metrics import confusion_matrix  # 混淆矩阵
from nltk.metrics import edit_distance  # 编辑距离
from operator import itemgetter


# 用于生成验证码，接收一个单词和错切值，返回用numpy数组格式表示的图像
def create_captcha(text, shear=0.0, size=(100, 26)):
    im = Image.new("L", size, "black")
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("FiraCode-Medium.otf", 22)
    draw.text((0, 0), text, fill=1, font=font)
    image = np.array(im)
    affine_tf = tf.AffineTransform(shear=shear)
    image = tf.warp(image, affine_tf)
    # 对图像进行归一化处理，确保特征值落在0到1之间
    return image / image.max()


if __name__ == "__main__":
    image = create_captcha("GENE", shear=0.5)
    plt.imshow(image, cmap="Greys")
    plt.show()

    def segment_image(image):
        """
        接收图像，返回小图像列表
        :param image:
        :return:
        """
        # 找出像素值相同又连接在一起的像素块，类似上一节的连通分支
        labeled_image = label(image > 0)
        subimages = []
        for region in regionprops(labeled_image):
            start_x, start_y, end_x, end_y = region.bbox
            subimages.append(image[start_x:end_x, start_y:end_y])
        # 如果没有找到小图像，则将原图像作为子图返回
        if len(subimages) == 0:
            return [
                image,
            ]
        return subimages

    subimages = segment_image(image)
    f, axes = plt.subplots(1, len(subimages), figsize=(10, 3))
    for i in range(len(subimages)):
        axes[i].imshow(subimages[i], cmap="gray")
    plt.show()

    # 指定随机状态值
    random_state = check_random_state(14)
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    shear_values = np.arange(0, 0.5, 0.05)

    def generate_sample(random_state=None):
        random_state = check_random_state(random_state)
        letter = random_state.choice(letters)
        shear = random_state.choice(shear_values)
        return create_captcha(letter, shear=shear, size=(25, 25)), letters.index(letter)

    image, target = generate_sample(random_state)
    plt.imshow(image, cmap="Greys")
    print("The target for this image is {}".format(target))
    plt.show()

    dataset, targets = zip(*(generate_sample(random_state) for i in range(3000)))
    dataset = np.array(dataset, dtype=float)
    targets = np.array(targets)

    # 对26个字母类别进行编码
    onehot = OneHotEncoder()
    y = onehot.fit_transform(targets.reshape(targets.shape[0], 1))
    # 将稀疏矩阵转换为密集矩阵
    y = y.todense()

    # 调整图像大小
    dataset = np.array(
        [resize(segment_image(sample)[0], (20, 20)) for sample in dataset]
    )
    # 将最后三维的dataset的后二维扁平化
    X = dataset.reshape((dataset.shape[0], dataset.shape[1] * dataset.shape[2]))
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)

    training = SupervisedDataSet(X.shape[1], y.shape[1])
    for i in range(X_train.shape[0]):
        training.addSample(X_train[i], y_train[i])
    testing = SupervisedDataSet(X.shape[1], y.shape[1])
    for i in range(X_test.shape[0]):
        testing.addSample(X_test[i], y_test[i])
        # 指定维度，创建神经网络，第一个参数为输入层神经元数量，第二个参数隐含层神经元数量，第三个参数为输出层神经元数量
    # bias在每一层使用一个一直处于激活状态的偏置神经元
    net = buildNetwork(X.shape[1], 100, y.shape[1], bias=True)

    trainer = BackpropTrainer(net, training, learningrate=0.01, weightdecay=0.01)
    # 设定代码的运行步数
    trainer.trainEpochs(epochs=20)
    # 预测值
    predictions = trainer.testOnClassData(dataset=testing)
    # f1_score的average默认值为'binary'，如果不指定average则会发生ValueError
    print(
        "F-score:{0:.2f}".format(
            f1_score(y_test.argmax(axis=1), predictions, average="weighted")
        )
    )
    print(
        "F-score:{0:.2f}".format(
            f1_score(y_test.argmax(axis=1), predictions, average="micro")
        )
    )
    print(
        "F-score:{0:.2f}".format(
            f1_score(y_test.argmax(axis=1), predictions, average="macro")
        )
    )
    print("------------------------")

    def predict_captcha(captcha_image, neural_network):
        subimages = segment_image(captcha_image)
        predicted_word = ""
        for subimage in subimages:
            subimage = resize(subimage, (20, 20))
            outputs = net.activate(subimage.flatten())
            prediction = np.argmax(outputs)
            predicted_word += letters[prediction]
        return predicted_word

    word = "GENE"
    captcha = create_captcha(word, shear=0.2)
    print(predict_captcha(captcha, net))
    print("------------------------")

    def test_prediction(word, net, shear=0.2):
        captcha = create_captcha(word, shear=shear)
        prediction = predict_captcha(captcha, net)
        prediction = prediction[:4]
        # 返回预测结果是否正确，验证码中的单词和预测结果的前四个字符
        return word == prediction, word, prediction

    valid_words = [word.upper() for word in words.words() if len(word) == 4]
    num_correct = 0
    num_incorrect = 0
    for word in valid_words:
        correct, word, prediction = test_prediction(word, net, shear=0.2)
        if correct:
            num_correct += 1
        else:
            num_incorrect += 1
    print("Number correct is {}".format(num_correct))
    print("Number incorrect is {}".format(num_incorrect))
    print("------------------------")

    cm = confusion_matrix(np.argmax(y_test, axis=1), predictions)
    plt.figure(figsize=(20, 20))
    plt.imshow(cm)
    tick_marks = np.arange(len(letters))
    plt.xticks(tick_marks, letters)
    plt.yticks(tick_marks, letters)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

    steps = edit_distance("STEP", "STOP")
    print("The num of steps needed is: {}".format(steps))

    def compute_distance(prediction, word):
        return len(prediction) - sum(
            prediction[i] == word[i] for i in range(len(prediction))
        )

    def improved_prediction(word, net, dictionary, shear=0.2):
        captcha = create_captcha(word, shear=shear)
        prediction = predict_captcha(captcha, net)
        prediction = prediction[:4]
        if prediction not in dictionary:
            distance = sorted(
                [(w, compute_distance(prediction, w)) for w in dictionary],
                key=itemgetter(1),
            )
            best_word = distance[0]
            prediction = best_word[0]
        return word == prediction, word, prediction

    num_correct = 0
    num_incorrect = 0
    for word in valid_words:
        correct, word, prediction = improved_prediction(
            word, net, valid_words, shear=0.2
        )
        if correct:
            num_correct += 1
        else:
            num_incorrect += 1
    print("Number correct is {}".format(num_correct))
    print("Number incorrect is {}".format(num_incorrect))
