import cv2, tqdm, sys, os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from vlib import save_pattern, save_curve, plot_frame, plot_frames

def save_frames(source, target, nx, ny):

    c = pd.read_csv(source).values
    nx = nx
    ny = ny

    # domains         
    x = np.linspace(-nx/2, nx/2, nx)    
    theta = np.linspace(0, 2*np.pi, ny)
    y = ny/2/np.pi * np.cos(theta)
    z = ny/2/np.pi * np.sin(theta)
    X, Y = np.meshgrid(x, y) 
    X, Z = np.meshgrid(x, z) 
        

    for j in range(len(c)):
        # print(j)
        fig = plt.figure(figsize=(10, 10))

        C = np.reshape(c[j], (nx, ny))
        color = C

        norm = matplotlib.colors.Normalize(vmin=0, vmax=0.8)
        m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
        m.set_array([])

        fcolors = m.to_rgba(color)

        plt.clf()
        
        ax = fig.gca(projection='3d')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.8, edgecolor='k',
            linewidth=0.5, facecolors=fcolors, vmin=0, vmax=0.8, shade=False)
        ax.view_init(180, 100)
        try:
            plt.savefig('./save/animations/'+target+'/frames/img' + str(j) + '.jpg')
        except FileNotFoundError:
            os.makedirs('./save/animations/'+target+'/frames/')
            os.makedirs('./save/animations/'+target+'/movie/')
            plt.savefig('./save/animations/'+target+'/frames/img' + str(j) + '.jpg', orientation='landscape')
        
        plt.close(fig)
        
def save_video(target, fps):
    
    file_to_save = './save/animations/'+target+'/movie/movie.avi'
    frames_loc = './save/animations/'+target + '/frames/'
    
    frames = os.listdir(frames_loc)
    total_num = len(frames)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter = cv2.VideoWriter(file_to_save, fourcc, fps, (1000, 1000))

    for i in tqdm.tqdm(range(total_num)):
        frame = cv2.imread(frames_loc+'img'+str(i+1)+'.jpg')
        videoWriter.write(frame)
    videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    df = pd.read_json('config.json')
    source = df.SourceFile.values[0]
    target = df.TargetFile.values[0]
    frames_saved = df.FramesSaved.values[0]
    nx = df.NumX.values[0]
    ny = df.NumY.values[0]
    fps = df.TargetFPS.values[0]
    if frames_saved == 'False':
        save_frames(source, target, nx, ny)
    save_video(target, fps)
