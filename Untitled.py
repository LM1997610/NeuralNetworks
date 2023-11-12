
import matplotlib.pyplot as plt

def  do_plot(history, n_epochs):

    fig, ax = plt.subplots(1, 2, figsize=(11, 3))
    fig.tight_layout(pad=4, w_pad = 6.5)
    
    x = list(range(1, n_epochs + 1))
    
    ax[0].plot(x, history.history['loss'], label='train_loss', color = "red")
    ax[0].plot(x, history.history['val_loss'], label='val_loss', color = "green")

    ax[0].set_title('\n Loss History\n', fontsize=14)
    ax[0].set_xlabel('\n Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_xticks(x)
    ax[0].grid(linewidth=0.5)
    ax[0].legend(loc="best", prop={'size': 8})


    ax[1].plot(x, history.history['sparse_categorical_accuracy'], label='train_accuracy', color = "red")
    ax[1].plot(x, history.history['val_sparse_categorical_accuracy'], label = 'val_accuracy', color = "green")
    
    ax[1].set_title('\n Classification Accuracy History\n', fontsize=14)
    ax[1].set_xlabel('\n Epochs')
    ax[1].set_ylabel('Classification accuracy')
    ax[1].grid(linewidth=0.5)

    ax[1].set_xticks(x)
    ax[1].legend(loc ="lower right", prop={'size': 8})
    
    #plt.savefig("history_plot.png")
    plt.show()

def show_images(immagine, vanilla_map, smooth_map):

  #mappa, g = silency_map(test_images[indx], model)

  if smooth_map is not None:

    fig, ax = plt.subplots(1, 3, figsize=(7, 7))
    a = ax[2].imshow(smooth_map, cmap="gary")
    ax[2].axis('off')

  else:
    fig, ax = plt.subplots(1, 2, figsize=(5, 5))

  ax[0].imshow(immagine)
  ax[0].axis('off')

  a = ax[1].imshow(vanilla_map, cmap="gray")
  ax[1].axis('off')


  plt.colorbar(a, ax = ax, fraction=0.022, pad=0.08)
  plt.show()


