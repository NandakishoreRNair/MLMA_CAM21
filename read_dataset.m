clear;
clc;
close all;

%% ========================================================
%% USER SETTINGS
%% ========================================================

base_folder = ...
    'D:\CAM21\data\Classification\levelground';

subject = 'ab14';


%% ========================================================
%% LOAD DATA
%% ========================================================

fprintf('=====================================\n');
fprintf('LOADING DATA\n');
fprintf('=====================================\n\n');

input_file = fullfile( ...
    base_folder, ...
    ['full_' subject '_input_4.mat']);

output_file = fullfile( ...
    base_folder, ...
    ['full_' subject '_output_4.mat']);

input_data = load(input_file);

output_data = load(output_file);

X = table2array(input_data.alldata);

output_table = output_data.alldata;

fprintf('Original data:\n');
fprintf('Samples : %d\n', size(X,1));
fprintf('Features: %d\n\n', size(X,2));
unique(output_data.alldata.trial_feat_last)
groupsummary(output_data.alldata, 'trial_feat_last')
output_table