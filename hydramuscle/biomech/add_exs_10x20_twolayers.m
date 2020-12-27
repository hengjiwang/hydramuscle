%% Setup
close all; clear all; clc;
addpath('/usr/local/comsol53a/multiphysics/mli');
% addpath('C:\Program Files\COMSOL\COMSOL53a\Multiphysics\mli');
mphstart;
import com.comsol.model.util.*;
mphopen model  

NAME = '017767';

mphsave(model, ['map_', NAME, '.m'])

%% Load data

FILENAME = NAME;

disp("Loading data...")

ecto_filepath = ['../force/',FILENAME,'_ecto.csv'];
endo_filepath = ['../force/',FILENAME,'_endo.csv'];
% endo_filepath = ['..\force\',FILENAME,'_ecto.csv'];

force_ecto = csvread(ecto_filepath, 1);
force_endo = csvread(endo_filepath, 1);

% Reformat and save data

disp("Reformating data...")  

% Construct time axis
time = zeros(6500, 1);

for j = 1:length(time)
    time(j) = 0.1 * (j-1);
end

% Reformat force data
len = length(force_ecto(1, :)); 

disp("Saving data...")

for j = 1:len
    f = [time, force_ecto(:, j)];
    csvwrite(['../force/cycle_twolayers_300s/force_ecto_', num2str(j), '.txt'], f);
    f = [time, force_endo(:, j)];  
    csvwrite(['../force/cycle_twolayers_300s/force_endo_', num2str(j), '.txt'], f);
end

clear force_ecto;
clear force_endo;

% Import data into COMSOL model

len = 200;

disp("Importing data...")

for j = 1:len
    
    intname = ['int', num2str(j)];
    model.func.create(intname, 'Interpolation');
    model.func(intname).model('comp1');
    model.func(intname).set('source', 'file');
    model.func(intname).set('filename', ...
        ['../force/cycle_twolayers_300s/force_ecto_', num2str(j), '.txt']);
    model.func(intname).importData;
end

for j = 1:len
    intname = ['int', num2str(j+len)];
    model.func.create(intname, 'Interpolation');
    model.func(intname).model('comp1');
    model.func(intname).set('source', 'file');
    model.func(intname).set('filename', ...
        ['../force/cycle_twolayers_300s/force_endo_', num2str(j), '.txt']);
    model.func(intname).importData;
end

%% Add external stresses

for j = 1:200
    disp(j);
    
    % For the cylinder part
    if 3 <= mod(j, 20) && mod(j, 20) <= 18
        % Create external stress
        model.physics('solid').feature('hmm1').create(['exs', ...
            num2str(j)], 'ExternalStress', 3);
        % Choose sys4
        model.physics('solid').feature('hmm1').feature(['exs', ...
            num2str(j)]).set('coordinateSystem', 'sys4');
        % Choose elememts
        model.physics('solid').feature('hmm1').feature(['exs', ...
            num2str(j)]).selection.named(['geom1_sel', num2str(j)]);
        % Set stress formula
        model.physics('solid').feature('hmm1').feature(['exs', ...
            num2str(j)]).set('Sext', ...
                {'0' '0' '0' ...
                 '0' exs_endo(j) '0' ...
                 '0' '0' exs_ecto(j)});
    % For the bottom sphere part
    elseif 1 == mod(j, 20) || mod(j, 20) == 2
        model.physics('solid').feature('hmm1').create(['exs', ...
            num2str(j)], 'ExternalStress', 3);
        model.physics('solid').feature('hmm1').feature(['exs', ...
            num2str(j)]).set('coordinateSystem', 'sys5');
        model.physics('solid').feature('hmm1').feature(['exs', ...
            num2str(j)]).selection.named(['geom1_sel', num2str(j)]);
        model.physics('solid').feature('hmm1').feature(['exs', ...
            num2str(j)]).set('Sext', ...
                {'0' '0' '0' ...
                 '0' exs_ecto(j) '0' ... 
                 '0' '0' exs_endo(j)});
    % For the top sphere part
    elseif 19 == mod(j, 20) || mod(j, 20) == 0
        model.physics('solid').feature('hmm1').create(['exs', ...
            num2str(j)], 'ExternalStress', 3);
        model.physics('solid').feature('hmm1').feature(['exs', ...
            num2str(j)]).set('coordinateSystem', 'sys6');
        model.physics('solid').feature('hmm1').feature(['exs', ...
            num2str(j)]).selection.named(['geom1_sel', num2str(j)]);
        model.physics('solid').feature('hmm1').feature(['exs', ...
            num2str(j)]).set('Sext', ...
                {'0' '0' '0' ...
                 '0' exs_ecto(j) '0' ... 
                 '0' '0' exs_endo(j)});
    end
    
end

%% Save model
disp('Saving model...');
% mphsave(model, ['map_',NAME,'.mph']);
mphsave(model, [NAME, '.mph']);
