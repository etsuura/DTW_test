import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def get_max(data1, data2):
    array = np.zeros(2)
    array[0] = np.amax(data1)
    array[1] = np.amax(data2)

    return np.amax(array)

def get_min(data1, data2):
    array = np.zeros(2)
    array[0] = np.amin(data1)
    array[1] = np.amin(data2)

def plot_1figure(data, title="fig", figure_name="fig", save=True):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(data)

    max_val = np.amax(data) #縦軸の最大値決定のため
    min_val = np.amin(data)
    if min_val < 0:
        ax.set_ylim(min_val + min_val / 10, max_val + max_val / 10)     #軸設定
    else:
        ax.set_ylim(0 - min_val / 10, max_val + max_val / 10)     #軸設定
    ax.set_title(title)           #グラフ名決め

    fig.tight_layout()                  # タイトルとラベルが被るのを解消

    if save==True:
        figure_name = figure_name + ".png"
        plt.savefig(figure_name)      # save as png

    plt.show()  # グラフを画面表示

# def save_animation(array, title="animation", animation_name="animation"):
#     # アニメーション保存
#     max_val = np.amax(array)
#     fig = plt.figure()  # figure objectを取得
#     ax1 = fig.add_subplot(1, 1, 1)
#     ims = []
#     for i in range(array.shape[0]):
#         artist1 = ax1.plot(array[i, :], "b")
#         ax1.set_title(title)  # グラフ名決め
#         ax1.set_ylim(0, max_val + max_val / 10)  # 軸設定
#         ims.append(artist1)
#
#     ani = animation.ArtistAnimation(fig, ims, interval=10)
#     animation_name += ".gif"
#     ani.save(animation_name, writer="pillow")

# def plot_2figure(data1, data2, plot_name1="fig1", plot_name2="fig2", figure_name="fig"):
#     # plot 2fig
#     fig = plt.figure()
#     ax1 = fig.add_subplot(2, 1, 1)
#     ax2 = fig.add_subplot(2, 1, 2)
#
#     ax1.plot(data1)
#     ax2.plot(data2)
#
#     max_val = get_max(data1, data2)         #縦軸の最大値決定のため
#     ax1.set_ylim(0, max_val + max_val / 10)     #軸設定
#     ax2.set_ylim(0, max_val + max_val / 10)
#
#     ax1.set_title(plot_name1)           #グラフ名決め
#     ax2.set_title(plot_name2)
#
#     fig.tight_layout()                  # タイトルとラベルが被るのを解消
#
#     figure_name = figure_name + ".png"
#     plt.savefig(figure_name)      # save as png
#
#
# def save_2plot(array1, array2, title1, title2, file_name):
#     # アニメーション保存
#     max_val = get_max(array1, array2)
#     fig = plt.figure()  # figure objectを取得
#     ax1 = fig.add_subplot(2, 1, 1)
#     ax2 = fig.add_subplot(2, 1, 2)
#     ims = []
#     for i in range(array1.shape[0]):
#         im1,  = ax1.plot(array1[i, :], "b")
#         im2,  = ax2.plot(array2[i, :], "b")
#         ax1.set_title(title1)  # グラフ名決め
#         ax2.set_title(title2)
#         ax1.set_ylim(0, max_val + max_val / 10)  # 軸設定
#         ax2.set_ylim(0, max_val + max_val / 10)
#         fig.tight_layout()
#         ims.append([im1, im2])
#     ani = animation.ArtistAnimation(fig, ims, interval=10)
#     file_name += ".gif"
#     ani.save(file_name, writer="pillow")