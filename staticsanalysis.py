import matplotlib

# This needs to be done *before* importing pyplot or pylab
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


class HistoryAnalysis:

    @staticmethod
    def plot_history(history, namefile):
        """
        Collects the history, returned from training the model and creates two charts:
        A plot of accuracy on the training and validation datasets over training epochs.
        A plot of loss on the training and validation datasets over training epochs.
        :param history: (dict) from keras fit
        :param namefile: (str) set name save file
        :return: plt(object) plot
        """
        # make new directory
        new_dir = 'ResultPlot'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        # plot's name
        loss_plot = os.path.join(new_dir, "{:s}_loss.png".format(namefile))
        acc_plot = os.path.join(new_dir, "{:s}_accuracy.png".format(namefile))

        loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
        val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
        acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
        val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

        if len(loss_list) == 0:
            print('Loss is missing in history')
            return
        else:
            pass

        # As loss always exists
        epochs = range(1, len(history.history[loss_list[0]]) + 1)

        # Loss
        plt.figure(1)
        for l in loss_list:
            plt.plot(epochs, history.history[l], 'b',
                     label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

        for l in val_loss_list:
            plt.plot(epochs, history.history[l], 'g',
                     label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # save figure loss
        plt.savefig(loss_plot)

        # Accuracy
        plt.figure(2)
        for l in acc_list:
            plt.plot(epochs, history.history[l], 'b',
                     label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
        for l in val_acc_list:
            plt.plot(epochs, history.history[l], 'g',
                     label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')

        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # save figure loss
        plt.savefig(acc_plot)
        return plt
