import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from visualizer.vlib import save_pattern, save_curve, plot_frame, plot_frames
import pandas as pd
import cv2, tqdm, sys, os

def save_frames(filename='./new/c_20x20_100s.csv', target='20x20_100s_bending', nx=20, ny=20):

    c = pd.read_csv(filename)
    nx = nx
    ny = ny

    # domains
    x = np.linspace(-nx/2,nx/2,nx) 
    y = np.linspace(-ny/np.pi,ny/np.pi,ny)           
    X, Y = np.meshgrid(x, y) 

    for j in tqdm.tqdm(range(len(c))):

        C = np.reshape(c.values[j], (nx, ny))
        color = C # change to desired fourth dimension
        norm = matplotlib.colors.Normalize(vmin=0, vmax=0.8)
        m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
        m.set_array([])
        fcolors = m.to_rgba(color)

        # plot
        fig = plt.figure(figsize = (10, 10))
        ax = fig.gca(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        im = ax.plot_surface(X,Y,np.sqrt((ny/2)**2-Y**2), rstride=1, cstride=1, facecolors=fcolors, vmin = 0, vmax = 0.8, shade=False)
        ax.view_init(100, 180)
        plt.savefig('./save/animations/'+target+'/frames/img' + str(j) + '.jpg')
        plt.close(fig)

def save_video(target, ):
    
    file_to_save = './save/animations/'+target+'/movie/movie_4x.avi'
    frames_loc = './save/animations/'+target + '/frames/'
    
    frames = os.listdir(frames_loc)
    total_num = len(frames)

    fps = 400

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter = cv2.VideoWriter(file_to_save,fourcc,fps,(1000,1000))

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
