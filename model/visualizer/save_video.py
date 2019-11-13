import cv2, tqdm, sys, os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from vlib import save_pattern, save_curve, plot_frame, plot_frames

def save_frames(filename='./new/c_20x20_100s.csv', target='20x20_100s_bending', nx=20, ny=20):

    c = pd.read_csv(filename).values
    nx = nx
    ny = ny

    # domains
    x = np.linspace(-nx/2,nx/2,nx)    
    theta = np.linspace(0, 2*np.pi, ny)
    y = ny/2/np.pi * np.cos(theta)
    z = ny/2/np.pi * np.sin(theta)
    X, Y = np.meshgrid(x, y) 
    X, Z = np.meshgrid(x, z) 

    for j in tqdm.tqdm(range(len(c))):

        C = np.reshape(c[j], (nx, ny))
        color = C

        norm = matplotlib.colors.Normalize(vmin=0, vmax=0.8)
        m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
        m.set_array([])

        fcolors = m.to_rgba(color.T)

        fig = plt.figure(figsize=(20, 10))
        ax = fig.gca(projection='3d')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.8, edgecolor='k', linewidth=0.5, facecolors=fcolors, vmin=0, vmax=0.8, shade=False)

        # ax.view_init(100, 180)
        ax.view_init(180, 100)
        plt.savefig('./save/animations/'+target+'/frames/img' + str(j) + '.jpg')
        plt.close(fig)

def save_video(target):
    
    file_to_save = './save/animations/'+target+'/movie/movie_4x.avi'
    frames_loc = './save/animations/'+target + '/frames/'
    
    frames = os.listdir(frames_loc)
    total_num = len(frames)

    fps = 400

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter = cv2.VideoWriter(file_to_save, fourcc, fps, (1000, 1000))

    for i in tqdm.tqdm(range(total_num)):
        frame = cv2.imread(frames_loc+'img'+str(i+1)+'.jpg')
        videoWriter.write(frame)
    videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    filename = '~/Downloads/c/c_20x20_100s_elongation.csv'
    target = '20x20_100s_elongation2'
    frames_saved = sys.argv[1]
    if frames_saved == 'False': 
        save_frames(filename, target, nx=20, ny=20)
    save_video(target)
