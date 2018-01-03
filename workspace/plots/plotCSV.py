#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np 

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def tf_csv_to_plt(tf_csv_file_name, line_width, color, alpha, ax):
    csv_np = np.genfromtxt(tf_csv_file_name, delimiter=',')
    x = csv_np[1:,1]
    y = csv_np[1:,2]
    line = ax.plot(x,y)[0]
    plt.setp(line, linewidth=line_width, color=color, alpha=alpha)
    return [x, y]

def tf_csvs_to_plt(env_name, ax):
    line_width = 4
    alpha = 0.4
    a2c_color = 'lightgreen'
    a2s_color = 'lightblue'
    a2sx1, a2sy1 = tf_csv_to_plt(env_name+'/run_A2S-run1-tag-average_reward_1.csv',line_width,a2s_color,alpha,ax)
    a2sx2, a2sy2 = tf_csv_to_plt(env_name+'/run_A2S-run2-tag-average_reward_1.csv',line_width,a2s_color,alpha,ax)
    a2sx3, a2sy3 = tf_csv_to_plt(env_name+'/run_A2S-run3-tag-average_reward_1.csv',line_width,a2s_color,alpha,ax)
    a2cx1, a2cy1 = tf_csv_to_plt(env_name+'/run_A2C-run1-tag-average_reward_1.csv',line_width,a2c_color,alpha,ax)
    a2cx2, a2cy2 = tf_csv_to_plt(env_name+'/run_A2C-run2-tag-average_reward_1.csv',line_width,a2c_color,alpha,ax)
    a2cx3, a2cy3 = tf_csv_to_plt(env_name+'/run_A2C-run3-tag-average_reward_1.csv',line_width,a2c_color,alpha,ax)

    a2sy = moving_average(np.mean(np.array([a2sy1,a2sy2,a2sy3]),axis=0))
    a2sx = np.mean(np.array([a2sx1,a2sx2,a2sx3]),axis=0)[:a2sy.shape[0]]

    a2cy = moving_average(np.mean(np.array([a2cy1,a2cy2,a2cy3]),axis=0))
    a2cx = np.mean(np.array([a2cx1,a2cx2,a2cx3]),axis=0)[:a2cy.shape[0]]
    
    a2sL = ax.plot(a2sx, a2sy)[0]
    a2cL = ax.plot(a2cx, a2cy)[0]

    line_width = 2
    a2c_color = 'green'
    a2s_color = 'blue'
    alpha = 1.0

    plt.setp(a2sL, linewidth=line_width, color=a2s_color, alpha=alpha)
    plt.setp(a2cL, linewidth=line_width, color=a2c_color, alpha=alpha)

    return [a2sL, a2cL]

if __name__ == "__main__":
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure()

    # f, axs = plt.subplots(2,2)

    # ax1 = axs[0,0]
    # ax2 = axs[1,0]
    # ax3 = axs[0,1]
    # ax4 = axs[1,1]
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[0, 1])

    ax1.grid(color='lightgrey',linestyle='-')
    ax1.set_axisbelow(True)
    ax2.grid(color='lightgrey',linestyle='-')
    ax2.set_axisbelow(True)
    ax3.grid(color='lightgrey',linestyle='-')
    ax3.set_axisbelow(True)
    # ax4.grid(color='lightgrey',linestyle='-')
    # ax4.set_axisbelow(True)

    ax1.set_title('Walker2d')
    ax2.set_title('Hopper')
    ax3.set_title('HalfCheetah')
    tf_csvs_to_plt('Walker2d', ax1)
    tf_csvs_to_plt('Hopper', ax2)
    a2sL, a2cL = tf_csvs_to_plt('HalfCheetah', ax3)
    # a2sL, a2cL = tf_csvs_to_plt('Walker2d', ax4)

    ax1.set_xlim(xmax=1000000)
    ax2.set_xlim(xmax=1000000)
    ax3.set_xlim(xmax=1000000)
    # ax4.set_xlim(xmax=1000000)
    ax1.set_ylim(ymin=-100)
    ax2.set_ylim(ymin=-100)
    ax3.set_ylim(ymin=-100)
    # ax4.set_ylim(ymin=-100)

    ax1.xaxis.set_ticks(np.array([0,1000000]))
    ax2.xaxis.set_ticks(np.array([0,1000000]))
    ax3.xaxis.set_ticks(np.array([0,1000000]))
    # ax4.xaxis.set_ticks(np.array([0,1000000]))

    plt.figlegend((a2sL, a2cL),('A2S', 'A2C'), loc='upper right', bbox_to_anchor=(0.7, 0.45), frameon=False)

    plt.show()