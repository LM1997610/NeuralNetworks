
import matplotlib.pyplot as plt

def do_plot(history, n_epochs, xticks):

    fig, ax = plt.subplots(1, 2, figsize=(11, 3))
    fig.tight_layout(pad=4, w_pad=6.5)
    
    x = list(range(1, n_epochs + 1))
    
    plot_info = [{'label_prefix': '', 'color': 'red'},
                 {'label_prefix': 'val_', 'color': 'green'}]

    for i, ax_index in enumerate([0, 1]):

        for info in plot_info:
            label = info['label_prefix'] + 'loss' if i == 0 else info['label_prefix'] + 'sparse_categorical_accuracy'
            value = history.history[label]
            ax[ax_index].plot(x, value, label=label, color=info['color'])
        
        ax[ax_index].set_title(f'\n {"Loss" if i == 0 else "Classification Accuracy"} History\n', fontsize=14)
        ax[ax_index].set_xlabel('\n Epochs')
        ax[ax_index].set_ylabel('Loss' if i == 0 else 'Classification accuracy')
        ax[ax_index].set_xticks(list(range(1, n_epochs + 1, xticks)))
        ax[ax_index].grid(linewidth=0.5)
        
        ax[ax_index].legend(loc="best", prop={'size': 8})

    plt.show()
    print(); print()

def show_two_images(immagine, vanilla_map):

    fig, ax = plt.subplots(1, 2, figsize=(4, 4))

    for i in range(2):

        ax[i].imshow(immagine if i == 0 else vanilla_map, cmap="gray", aspect="equal")
        ax[i].axis('off')

    cax = fig.add_axes([0.95, 0.3, 0.02, 0.39])
    fig.colorbar(ax[1].imshow(vanilla_map, cmap="gray", aspect="equal"), cax, orientation='vertical')

    plt.subplots_adjust(wspace=0.04)
    plt.show()


def show_images(images_index_list, evaluate_on_sigma, plot_in_rows = 5):
  
  fig, ax = plt.subplots(len(images_index_list), plot_in_rows, figsize=(7, 5))

  for a in range(len(images_index_list)):

      this_list, titles = evaluate_on_sigma(images_index_list[a])

      for i in range(len(this_list)):

          im = ax[a][i].imshow(this_list[i], cmap="gray", aspect="auto")
          ax[a][i].axis('off')

          if i<5 and a<1:
              ax[a][i].set_title(titles[i])

  plt.subplots_adjust(wspace=0.04, hspace=0.05)
  plt.show()

