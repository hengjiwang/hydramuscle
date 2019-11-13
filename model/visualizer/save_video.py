import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from vlib import save_pattern, save_curve, plot_frame, plot_frames
import pandas as pd
import cv2, tqdm, sys, os

def save_frames(source, target, nx, ny):

    c = pd.read_csv(source)
    nx = nx
    ny = ny

    # domains
    x = np.linspace(-nx/2,nx/2,nx) 
    y = np.linspace(-ny/np.pi/2,ny/np.pi/2,ny)           
    X, Y = np.meshgrid(x, y) 

    for j in tqdm.tqdm(range(len(c))):

        C = np.reshape(c.values[j], (nx, ny))
        color1 = C[:, 0:int(ny/2)] # change to desired fourth dimension
        color2 = C[:, int(ny/2):]
        norm = matplotlib.colors.Normalize(vmin=0, vmax=0.8)
        m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
        m.set_array([])
        fcolors1 = m.to_rgba(color1)
        fcolors2 = m.to_rgba(color2)

        # plot
        fig = plt.figure(figsize = (10, 10))
        ax = fig.gca(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.plot_surface(X,Y,np.sqrt((ny/4)**2-Y**2), rstride=1, cstride=1, facecolors=fcolors1, vmin = 0, vmax = 0.8, shade=False)
        ax.plot_surface(X,Y,-np.sqrt((ny/4)**2-Y**2), rstride=1, cstride=1, facecolors=fcolors2, vmin = 0, vmax = 0.8, shade=False)
        ax.view_init(100, 180)
        try:
            plt.savefig('../../save/animations/'+target+'/frames/img' + str(j) + '.jpg')
        except FileNotFoundError:
            os.makedirs('../../save/animations/'+target+'/frames/')
            os.makedirs('../../save/animations/'+target+'/movie/')
            plt.savefig('../../save/animations/'+target+'/frames/img' + str(j) + '.jpg')
        plt.close(fig)

def save_video(target, target_fps):
    
    file_to_save = '../../save/animations/'+target+'/movie/movie.avi'
    frames_loc = '../../save/animations/'+target + '/frames/'
    
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
    df = pd.read_json('config.json')
    source = df.SourceFile.values[0]
    target = df.TargetFile.values[0]
    nx = df.NumX.values[0]
    ny = df.NumY.values[0]
    target_fps = df.TargetFPS.values[0]
    frames_saved = df.FramesSaved.values[0]
    if frames_saved == 'False': 
        save_frames(source, target, nx, ny)
    save_video(target, target_fps)
