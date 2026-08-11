%% ========================================================
%% CONVERT ALL SUBJECT MAT FILES TO CSV
%% ========================================================

clear;
clc;

base_folder = ...
    'D:\CAM21\data\Classification\levelground\';
output_folder='D:\CAM21\data\csv\'

%% Find all MAT files

%mat_files = dir(fullfile(base_folder,'*.mat'));
mat_files = dir('D:\CAM21\data\Classification\levelground\full_ab08_input_4.mat');
fprintf('Found %d MAT files\n\n', length(mat_files));

%% Convert each file

for k = 1:length(mat_files)

    mat_name = mat_files(k).name;

    mat_path = fullfile(base_folder, mat_name);

    fprintf('Processing %s\n', mat_name);

    data = load(mat_path);

    %% Check if variable alldata exists

    if isfield(data,'alldata')

        csv_name = strrep(mat_name,'.mat','.csv');

        csv_path = fullfile(output_folder,csv_name);

        writetable(data.alldata,csv_path);

        fprintf(' -> Saved %s\n', csv_name);

    else

        fprintf(' -> No variable "alldata" found\n');

    end

end

fprintf('\nDone.\n');