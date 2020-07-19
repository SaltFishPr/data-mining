# !/usr/bin/env python
# author: Salt Fish
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # --------加载、了解、预处理数据集--------
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    # 查看数据集
    print(
        train_images.shape,  # (60000，28，28)
        len(train_labels),  # 60000
        train_labels,
        test_images.shape,
        len(test_labels),
    )
    # 查看图像
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()
    # 预处理标准化
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # 查看数据集
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()
    print("------------------------")

    # --------建立模型--------
    # 建立神经网络所需要模型的各层
    # tf.keras.layers.Flatten将图像的格式从二维数组(28 * 28)转换为一维数组(28 * 28 = 784)
    # 可以将这个图层看作是图像中取消堆叠的像素行，并将它们排列起来
    # 这个层没有参数需要学习; 它只是重新格式化数据。
    #
    # 然后是两个稠密层（完全连接的层），中间一层有128个节点，
    # 最后一层返回长度为10的对数数组。每个神经元包含一个得分，指示当前图像对这一类的评分
    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(10),
        ]
    )
    # --------编译模型--------
    # 损失函数：这可以衡量训练期间模型的准确程度，希望最小化这个函数，以便将模型“引导”到正确的方向
    # 优化器：如何基于它看到的数据和它的损失函数更新模型
    # 指标：用于检测训练和测试步骤。下面的例子使用精确度，即正确分类的图像的分数
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    # --------训练模型--------
    model.fit(train_images, train_labels, epochs=100, verbose=1)
    # --------评估表现--------
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print("\nTest accuracy:", test_acc)

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    # prediction是由10个数字组成的数组。它们表示模型对图像对应于10种不同衣服各自的置信度
    predictions = probability_model.predict(test_images)
    print(predictions[0])
    print("------------------------")

    # --------验证模型--------
    # 制作图表来观察十个类别预测的完整集合
    def plot_image(i, predictions_array, true_label, img):
        predictions_array, true_label, img = predictions_array, true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap=plt.cm.binary)
        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = "blue"
        else:
            color = "red"
        # 置信度百分比
        plt.xlabel(
            "{} {:2.0f}% ({})".format(
                class_names[predicted_label],
                100 * np.max(predictions_array),
                class_names[true_label],
            ),
            color=color,
        )

    # 绘制置信度柱状图
    def plot_value_array(i, predictions_array, true_label):
        predictions_array, true_label = predictions_array, true_label[i]
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)
        # 错误的预测标签为红色
        thisplot[predicted_label].set_color("red")
        # 正确的标签为蓝色
        thisplot[true_label].set_color("blue")

    i = 0
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions[i], test_labels)
    plt.show()

    i = 12
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions[i], test_labels)
    plt.show()

    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show()
    print("------------------------")
    # --------使用模型--------
    img = test_images[1]
    print(img.shape)

    img = np.expand_dims(img, 0)
    print(img.shape)
    # 为该图像预测
    predictions_single = probability_model.predict(img)
    print(predictions_single)
    plot_value_array(1, predictions_single[0], test_labels)
    _ = plt.xticks(range(10), class_names, rotation=45)
    # 返回预测的种类
    np.argmax(predictions_single[0])
