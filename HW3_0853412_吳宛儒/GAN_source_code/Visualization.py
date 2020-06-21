import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

##Write down your visualization code here
def plot_result(save_dir,G_loss,D_loss):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_loss,label="G")
    plt.plot(D_loss,label="D")
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.legend(loc='upper right')
    plt.savefig(save_dir+'loss.png')
    plt.show()
## Animation for your generation
##input : image_list (size = (the number of sample times, how many samples created each time, image )   )
img_list = []

fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

plt.show()
# https://matplotlib.org/api/_as_gen/matplotlib.animation.Animation.html#matplotlib.animation.Animation.save