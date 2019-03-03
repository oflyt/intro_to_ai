import matplotlib.pyplot as plt
from IPython import display

def show_state(observation, env_id, step=0, info=""):
    plt.figure(3)
    plt.clf()
    
    if len(observation)==4:
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10,4))
        for i in range(4):
            axes[i].imshow(observation[i], cmap='gray')
            axes[i].set_axis_off()
            axes[i].set_aspect('auto')
            fig.tight_layout()
    else:
        plt.imshow(observation, cmap='gray')
        plt.axis('off')

    plt.title("%s | Step: %d %s" % (env_id, step, info))
    display.clear_output(wait=True)
    display.display(plt.gcf())
plt.close()
