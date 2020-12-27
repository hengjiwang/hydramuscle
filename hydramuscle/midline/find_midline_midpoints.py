import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import xml.etree.ElementTree as ET
from cv2 import cv2
from tqdm import tqdm

def load_contour(filename):
    "Load and reformat contour"

    file_format = filename.split('.')[-1]
    if file_format == 'pkl':
        with open(filename, 'rb') as pickle_file:
            contours = pickle.load(pickle_file)
        # Reformat data
        for iframe in range(len(contours)):
            pts = [(pt[0][0], pt[0][1]) for pt in contours[iframe][0]]
            contours[iframe] = pts

    elif file_format == 'xml':
        try:
            root = ET.parse(filename).getroot()
            rois = root.find('rois').findall('roi')
            contours = []
            for roi in rois[1:]:
                points = roi.find('points').findall('point')
                contour = []
                for point in points:
                    pos_x = float(point.find('pos_x').text)
                    pos_y = float(point.find('pos_y').text)
                    contour.append((pos_x, pos_y))
                contours.append(contour)
        except:
            root = ET.parse(filename).getroot()
            rois = root.findall('roi')
            contours = []
            for i in range(len(rois)):
                contours.append(0)

            for roi in rois:
                id = int(roi.find('t').text)
                points = roi.find('points').findall('point')
                contour = []
                for point in points:
                    pos_x = float(point.find('pos_x').text)
                    pos_y = float(point.find('pos_y').text)
                    contour.append((pos_x, pos_y))
                try:
                    contours[id] = contour
                except:
                    print(id)

    return contours

def intp_seq(seq, nintp):
    "Interpolate sequence"
    seq_new = []
    for j in range(1, len(seq)):
        x_prev = seq[j-1][0]
        y_prev = seq[j-1][1]
        x_next = seq[j][0]
        y_next = seq[j][1]
        xintp = np.linspace(x_prev, x_next, nintp, endpoint=False)
        yintp = np.linspace(y_prev, y_next, nintp, endpoint=False)
        for k in range(len(xintp)):
            seq_new.append((xintp[k], yintp[k]))

    seq_new.append(seq[-1])
    return seq_new


def load_marker(filename):
    "Load tracked points"
    df = pd.read_csv(filename)
    df.columns = ['scorer', 'hypostome_x', 'hypostome_y', 'hypostome_likelihood',
                  'armpit1_x', 'armpit1_y', 'armpit1_likelihood',
                  'armpit2_x', 'armpit2_y', 'armpit2_likelihood',
                  'peduncle_x', 'peduncle_y', 'peduncle_likelihood']
    df = df.drop(index=[0, 1]).drop(columns='scorer').reset_index(drop=True)
    df = df.astype(float)
    return df

def locate_point(marker, contour):
    "Locate the corresponding index of the marker on contour"

    index = 0
    mindist = np.inf
    for j in range(len(contour)):
        dist = (marker[0] - contour[j][0])**2 + (marker[1] - contour[j][1])**2
        if dist < mindist:
            mindist = dist
            index = j

    return index

def length_segment(seg):
    "Returns length of segment seg"
    length = 0
    for j in range(len(seg)-1):
        p1, p2 = seg[j], seg[j+1]
        length += np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    return length

def extract_midline(contour, marker_mat, nseg=20, play=False):
    # Reformat marker
    marker = defaultdict(tuple)
    marker['hypostome'] = (marker_mat[0], marker_mat[1])
    marker['armpit1'] = (marker_mat[3], marker_mat[4])
    marker['armpit2'] = (marker_mat[6], marker_mat[7])
    marker['peduncle'] = (marker_mat[9], marker_mat[10])

    # Locate the peduncle on contour
    ind_ped = locate_point(marker['peduncle'], contour)

    # Reindex the contour -- start from the peduncle
    contour = contour[ind_ped:] + contour[:ind_ped]

    # Locate other markers
    ind_ped = 0
    ind_arp1 = locate_point(marker['armpit1'], contour)
    ind_arp2 = locate_point(marker['armpit2'], contour)
    ind_hyp = locate_point(marker['hypostome'], contour)

    # Separate contour to two parts
    ind_arp1, ind_arp2 = min(ind_arp1, ind_arp2), max(ind_arp1, ind_arp2)
    contour_half_1 = np.array(contour[:ind_arp1])
    contour_half_2 = np.array([contour[0]] + contour[ind_arp2:][::-1])

    if len(contour_half_1) == 0 or len(contour_half_2) == 0:
        # continue
        raise Exception("Half contour is 0")

    # if play:
    #     plt.plot(contour_half_1[:, 0], contour_half_1[:, 1], 'g.')
    #     plt.plot(contour_half_2[:, 0], contour_half_2[:, 1], 'g.')

    # if play:
    #     contour = np.array(contour)
    #     plt.plot(contour[:, 0], contour[:, 1], 'g.')

    # plt.plot(contour[ind_arp1][0], contour[ind_arp1][1], 'b.', markersize=20)
    # plt.plot(contour[ind_arp2][0], contour[ind_arp2][1], 'b.', markersize=20)
    # plt.plot(contour[ind_hyp][0], contour[ind_hyp][1], 'r.', markersize=20)
    # plt.plot(contour[ind_ped][0], contour[ind_ped][1], 'r.', markersize=20)

    # plt.plot(marker['hypostome'][0], marker['hypostome'][1], 'g.', markersize=20)
    # plt.plot(marker['armpit1'][0], marker['armpit1'][1], 'b.', markersize=20)
    # plt.plot(marker['armpit2'][0], marker['armpit2'][1], 'b.', markersize=20)
    # plt.plot(marker['peduncle'][0], marker['peduncle'][1], 'r.', markersize=20)


    # Find the midpoints
    midpoints = []
    len_contour_1 = length_segment(contour_half_1)
    len_contour_2 = length_segment(contour_half_2)
    ind_seg_pt1 = 0
    ind_seg_pt2 = 0
    cum_len_1 = 0
    cum_len_2 = 0

    for j in range(1, nseg):

        # Locate the segment points
        while cum_len_1 < j/nseg * len_contour_1:
            cum_len_1 += length_segment(contour_half_1[ind_seg_pt1:ind_seg_pt1+2])
            ind_seg_pt1 += 1

        while cum_len_2 < j/nseg * len_contour_2:
            cum_len_2 += length_segment(contour_half_2[ind_seg_pt2:ind_seg_pt2+2])
            ind_seg_pt2 += 1

        seg_pt_1 = contour_half_1[ind_seg_pt1]
        seg_pt_2 = contour_half_2[ind_seg_pt2]
        if play:
            plt.plot([seg_pt_1[0], seg_pt_2[0]], [seg_pt_1[1], seg_pt_2[1]], 'brown')
        midpoint = ((seg_pt_1[0] + seg_pt_2[0]) // 2, (seg_pt_1[1] + seg_pt_2[1]) // 2)
        midpoints.append(midpoint[0])
        midpoints.append(midpoint[1])
        if play:
            plt.plot(midpoint[0], midpoint[1], 'r.', markersize=10)

    # plt.plot(contour[ind_hyp][0], contour[ind_hyp][1], 'r.', markersize=10)
    midpoints.append(marker['hypostome'][0])
    midpoints.append(marker['hypostome'][1])

    return midpoints, contour_half_1, contour_half_2

def find_midline(file_contour, file_marker, file_video="", nseg=40, play=False):
    "Find midline"

    # Load files
    contours = load_contour(file_contour)

    print('Contour loaded, the size is: ' + str(len(contours)))

    markers = load_marker(file_marker).values

    print('Markers loaded, the size is: ' + str(len(markers)))

    missed_contour = [] # list(range(12096, 12145))
    print('Number of missed contours is: ' + str(len(missed_contour)))

    markers = [x for i,x in enumerate(markers) if i not in missed_contour]
    # contours = contours[7:]

    midpoints_all = []

    # Align data
    nframes = min(len(contours), len(markers))
    # contours = contours[:nframes]
    # markers = markers[:nframes].values

    # Loop over frames
    # for iframe in tqdm(range(nframes)):

    # cap = cv2.VideoCapture(file_video)
    # ret, frame = cap.read()
    # ny, nx, _ = frame.shape

    # if play:
    #     plt.figure(figsize=(nx/50, ny/50))

    # iframe = 0
    # while(ret):

    for iframe in tqdm(range(len(contours))):

        # if play:
        #     plt.clf()

        # plt.imshow(frame)

        # Extract contour and marker
        contour = contours[iframe]
        marker_mat = markers[iframe]

        midpoints, _, _ = extract_midline(contour, marker_mat, nseg, play)

        # plt.plot(midpoints[::2], midpoints[1::2], 'r-')

        # if play:
        #     plt.xlim(0, nx)
        #     plt.ylim(0, ny)
        #     plt.pause(0.001)

        # ret, frame = cap.read()
        # iframe += 1

        # print(iframe)

        midpoints_all.append(midpoints)

        # plt.xticks([])
        # plt.yticks([])

        # plt.savefig('../img'+str(iframe)+'.jpg', bbox_inches='tight')

        # input()


    return midpoints_all

if __name__ == "__main__":

    FILENAME = "Pre_Bisect_40x_4fps_ex4_1100-3040_enhanced"

    midpoints = find_midline("../data/contour/Pre_Bisect_40x_4fps_ex4.xml",
                             "../data/marker/Pre_Bisect_40x_4fps_ex4DeepCut_resnet50_Hydra2Nov17shuffle1_1030000.csv",
                             "../data/videos/NGCaMP/Pre_Bisect_40x_4fps_ex4_1100-3040_enhanced.avi",
                             nseg=20,
                             play=False)

    df = pd.DataFrame(midpoints)
    df.to_csv("../data/midpoints/midpoints_"+FILENAME+".csv", index=False)

