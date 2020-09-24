%% Set up

close all; clear all; clc;

%% 

v1 = VideoReader('./data/ca_mo_658838.mp4');
v2 = VideoReader('./data/spikes_658838.avi');
w = VideoWriter('./data/n_ca_mo_658838.avi');

w.FrameRate = 20;
open(w);

%% 

for j = 1:250
    
    disp(j);
    im_ca_mo = read(v1, j);
%     im_ca_mo = imresize(im_ca_mo, [ny ny+1]);
    im_spk = read(v2, j);
    
    montage({im_spk, im_ca_mo}, 'Size', [2 1]);
    F = getframe(gcf);
    [im_all, Map] = frame2im(F);
    
    writeVideo(w, im_all);
    
end

close(w)