%% Set up

close all; clear all; clc;

%%
v1 = VideoReader('./data/ecto_calcium_389459.avi');
% v2 = VideoReader('./data/ecto_force_389459.avi');
% v3 = VideoReader('./data/endo_force_389459.avi');
w = VideoWriter('./data/motion_ca_389459.avi');
w.FrameRate = 20;
open(w);

[X,Y,Z] = cylinder(ones(1,5),100);

motionFolder = './data/motion_389459/';
motionFiles = dir(fullfile(motionFolder, '*.png'));

%% 

nx = 1263;
ny = 1142;

for j = 1:100
    
    disp(j);
    
    % Create 3d image of calcium frame
    calcium = read(v1, 1000 + (j-1)*20);
    calcium = flipdim(calcium, 1);
    figure('visible','off');
    warp(X,Y,Z, calcium);
    view([180 17])
    set(gcf, 'position', [0, 0, nx, nx])
    F = getframe(gcf);
    [im_ca, Map] = frame2im(F);
    im_ca = imresize(im_ca, [ny nx]);
    clf;
    close;
    
%     % Create 3d image of ecto stress
%     ectoForce = read(v2, 1000 + (j-1)*20);
%     ectoForce = flipdim(ectoForce, 1);
%     figure('visible','off');
%     warp(X,Y,Z, ectoForce);
%     view([180 17])
%     set(gcf, 'position', [0, 0, nx, nx])
%     F = getframe(gcf);
%     [im_ec, Map] = frame2im(F);
%     im_ec = imresize(im_ec, [ny nx]);
%     clf;
%     close;
%     
%     % Create 3d image of endo stress
%     endoForce = read(v3, 1000 + (j-1)*20);
%     endoForce = flipdim(endoForce, 1);
%     figure('visible','off');
%     warp(X,Y,Z, endoForce);
%     view([180 17])
%     set(gcf, 'position', [0, 0, nx, nx])
%     F = getframe(gcf);
%     [im_en, Map] = frame2im(F);
%     im_en = imresize(im_en, [ny nx]);
%     clf;
%     close;
    
    % Read motion frame
    im_mo = imread([motionFolder, motionFiles(j).name]);
    
    % Combine figures
    figure('visible','off');
%     montage({im_mo, im_ca, im_ec, im_en});
    montage({im_mo, im_ca});
    F = getframe(gcf);
    [im_all, Map] = frame2im(F);
    
    % Write frame
    writeVideo(w, im_all);
    
end

close(w);