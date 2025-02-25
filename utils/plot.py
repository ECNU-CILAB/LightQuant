# author:Liu Yu
# time:2025/2/11 19:41
import matplotlib.pyplot as plt
from my_parser import args
def plot_figure(avg_train_loss_list=[], avg_val_loss_list=[], avg_acc_list=[], avg_mcc_list=[], epochs=100):
    # 创建一个图形对象
    plt.figure(figsize=(12, 4))

    # 绘制train损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(range(1, epochs + 1), avg_train_loss_list, label='Average Train Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Average Train Loss')
    plt.legend()

    # 绘制验证损失曲线
    plt.subplot(2, 2, 2)
    plt.plot(range(1, epochs + 1, 5), avg_val_loss_list, label='Average Validation Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Average Validation Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(2, 2, 3)
    plt.plot(range(1, epochs + 1, 5), avg_acc_list, label='Average Validation Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Average Accuracy')
    plt.legend()

    # 绘制MCC曲线
    plt.subplot(2, 2, 4)
    plt.plot(range(1, epochs + 1, 5), avg_mcc_list, label='Average Validation MCC', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('MCC')
    plt.title('Average MCC')
    plt.legend()

    # 调整子图间距
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    # 自动调整子图参数
    plt.tight_layout()

    plt.savefig(args.figure_save_folder + args.model + '.png')


