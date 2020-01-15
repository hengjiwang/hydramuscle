import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2, tqdm, sys, os, time, multiprocessing
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from hydramuscle.postprocessing.visualizer.vlib import save_pattern, save_curve, plot_frame, plot_frames
from multiprocessing import Pool
import multiprocessing

def save_frame(j, c, X, Y, Z, filename):

    # print(os.getpid())

    fig = plt.figure(figsize=(5, 5))
    fig.subplots_adjust(bottom=0, 
                        top=1, 
                        left=0, 
                        right=1)

    color = c[j]

    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.8)
    m = plt.cm.ScalarMappable(norm=norm, cmap='hot')
    m.set_array([])

    fcolors = m.to_rgba(color)

    # plt.clf()
    
    ax = fig.gca(projection='3d')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.8, edgecolor='k',
        linewidth=0.5, facecolors=fcolors, vmin=0, vmax=0.8, shade=False)
    ax.view_init(180, 100)

    print('saving img' + str(j) + '...')
    plt.savefig('/media/hengji/DATA/Data/Documents/hydramuscle/results/animations/'+filename+'/frames/img' + str(j) + '.jpg', 
                orientation='landscape')
                #bbox_inches='tight')

    print('saved ' + str(j))

    plt.close(fig)

def save_frames(c, filename, nx, ny):

    # c = pd.read_csv(source).values
    nx = nx
    ny = ny
    c = np.reshape(c, (-1, nx, ny))
    # c = multiprocessing.Array("i", c)

    # domains         
    x = np.linspace(-nx/2, nx/2, nx)    
    theta = np.linspace(0, 2*np.pi, ny)
    y = ny/2/np.pi * np.cos(theta)
    z = ny/2/np.pi * np.sin(theta)
    X, Y = np.meshgrid(x, y) 
    X, Z = np.meshgrid(x, z) 

    print("The number of CPU is:" + str(multiprocessing.cpu_count()))
    print('saving frames...')
    for j in range(0, len(c)-10, 10):
        # pool.apply_async(save_frame, args=(j,c,X,Y,Z))

        process_list = []

        for k in range(10):
            process_list.append(multiprocessing.Process(target = save_frame, args=(j+k,c,X,Y,Z,filename)))

        for k in range(10):
            process_list[k].start()

        for k in range(10):
            process_list[k].join()
        

def save_video(filename, fps):
    
    file_to_save = '/media/hengji/DATA/Data/Documents/hydramuscle/results/animations/'+filename+'/movie/movie.avi'
    frames_loc = '/media/hengji/DATA/Data/Documents/hydramuscle/results/animations/'+filename+'/frames/'
    
    frames = os.listdir(frames_loc)
    total_num = len(frames)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter = cv2.VideoWriter(file_to_save, fourcc, fps, (360, 360))

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

    print("Reading data...")
    c = pd.read_csv(source).values
    print("Data read.")

    if not os.path.exists('./save/animations/'+target+'/frames/'):
        os.makedirs('./save/animations/'+target+'/frames/')
        os.makedirs('./save/animations/'+target+'/movie/')

    if frames_saved == 'False':
        save_frames(c, target, nx, ny)
    save_video(target, fps)
