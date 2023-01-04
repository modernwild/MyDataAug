import matplotlib.pyplot as plt
import pandas as pd


def draw_loss(history_file, save_path, show=False):
    data = pd.read_csv(history_file)
    # sns.relplot(x="epoch", y="train_loss", ci=None, kind="line", data=data)
    # plt.show()

    fig = plt.figure(figsize=(10, 10))
    # 第一幅图的下标从1开始，设置6张子图
    plt.subplot(2, 1, 1)
    data["train_loss"].plot.line()
    plt.xlabel("train_loss")  # 给x轴命名标签
    plt.ylabel("epoch")  # 给y轴命名标签

    plt.subplot(2, 1, 2)
    data["valid_acc"].plot.line()
    plt.xlabel("valid_acc")  # 给x轴命名标签
    plt.ylabel("epoch")  # 给y轴命名标签

    # 显示画布
    if show:
        plt.show()
    plt.savefig(save_path)
