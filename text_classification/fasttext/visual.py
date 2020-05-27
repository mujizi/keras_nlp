# -*- coding: utf-8 -*-
# @Time    : 2020/5/20 下午8:07
# @Author  : Benqi
import matplotlib.pyplot as plt
import pickle

with open('trainHistoryDict.txt', 'rb') as file_pi:
    history = pickle.load(file_pi)

print(history)
plt.subplot(211)
plt.title("Accuracy")
plt.plot(history["accuracy"], color="g", label="Train")
plt.plot(history["val_accuracy"], color="b", label="Test")
plt.legend(loc="best")

plt.subplot(212)
plt.title("Loss")
plt.plot(history["loss"], color="g", label="Train")
plt.plot(history["val_loss"], color="b", label="Test")
plt.legend(loc="best")

plt.tight_layout()
plt.show()