%% Set up

close all; clear all; clc;

%%
v1 = VideoReader('./data/ecto_calcium_834573.avi');
% v2 = VideoReader('./data/ecto_force_834573.avi');
% v3 = VideoReader('./data/endo_force_834573.avi');
w = VideoWriter('./data/motion_ca_834573.avi');
w.FrameRate = 50;
open(w);

[X,Y,Z] = cylinder(ones(1,5),100);

motionFolder = './data/motion_834573/';
motionFiles = dir(fullfile(motionFolder, '*.png'));

%% 

for j = 1:500
    
    disp(j);
    
    % Create 3d image of calcium frame
    calcium = read(v1, 1 + (j-1)*10);
    calcium = flipdim(calcium, 1);
    figure('visible','off');
    warp(X,Y,Z, calcium);
    view([-66 20])
    set(gcf, 'position', [0, 0, 1623, 1623])
    F = getframe(gcf);
    [im_ca, Map] = frame2im(F);
    im_ca = imresize(im_ca, [1374 1623]);
    clf;
    close;
    
%     % Create 3d image of ecto stress
%     ectoForce = read(v2, 1 + (j-1)*10);
%     ectoForce = flipdim(ectoForce, 1);
%     figure('visible','off');
%     warp(X,Y,Z, ectoForce);
%     view([-66 20])
%     set(gcf, 'position', [0, 0, 1623, 1623])
%     F = getframe(gcf);
%     [im_ec, Map] = frame2im(F);
%     im_ec = imresize(im_ec, [1374 1623]);
%     clf;
%     close;
%     
%     % Create 3d image of endo stress
%     endoForce = read(v3, 1 + (j-1)*10);
%     endoForce = flipdim(endoForce, 1);
%     figure('visible','off');
%     warp(X,Y,Z, endoForce);
%     view([-66 20])
%     set(gcf, 'position', [0, 0, 1623, 1623])
%     F = getframe(gcf);
%     [im_en, Map] = frame2im(F);
%     im_en = imresize(im_en, [1374 1623]);
%     clf;
%     close;
    
    % Read motion frame
    im_mo = imread([motionFolder, motionFiles(j).name]);
    
    % Combine figures
    figure('visible','off');
    montage({im_mo, im_ca});
    F = getframe(gcf);
    [im_all, Map] = frame2im(F);
    
    % Write frame
    writeVideo(w, im_all);
    
end

close(w);