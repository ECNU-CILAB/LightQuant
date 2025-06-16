import matplotlib.pyplot as plt
def plot_figure(avg_train_loss_list=[], avg_val_loss_list=[], avg_acc_list=[], avg_mcc_list=[], epochs=100, figure_save_folder=None, model=None):

    plt.figure(figsize=(12, 4))


    plt.subplot(2, 2, 1)
    plt.plot(range(1, epochs + 1), avg_train_loss_list, label='Average Train Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Average Train Loss')
    plt.legend()


    plt.subplot(2, 2, 2)
    plt.plot(range(5, epochs + 1, 5), avg_val_loss_list, label='Average Validation Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Average Validation Loss')
    plt.legend()


    plt.subplot(2, 2, 3)
    plt.plot(range(5, epochs + 1, 5), avg_acc_list, label='Average Validation Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Average Accuracy')
    plt.legend()


    plt.subplot(2, 2, 4)
    plt.plot(range(5, epochs + 1, 5), avg_mcc_list, label='Average Validation MCC', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('MCC')
    plt.title('Average MCC')
    plt.legend()


    plt.subplots_adjust(wspace=0.4, hspace=0.4)


    plt.tight_layout()

    plt.savefig(figure_save_folder + f'{model}' + '.png')


