%% ========================================================
%% CONVERT MAT FILES TO CSV FOR PYTHON LSTM
%% Converts full_subject_input_4.mat and output files
%% to CSV format readable by Python/pandas
%% ========================================================

clear all; close all; clc;

ground_to_test = 'ramp';

base_folder = sprintf('/home/eeiww/ut55iqoh/MLMA_CAM21/Classification/%s/', ...
    ground_to_test);

output_folder = sprintf('/home/eeiww/ut55iqoh/MLMA_CAM21/csv_data/%s/', ...
    ground_to_test);

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

subjects = {'ab07', 'ab08', 'ab09', 'ab12', 'ab13', 'ab14', 'ab17', 'ab18', ...
            'ab19', 'ab20', 'ab21', 'ab23', 'ab24', 'ab27', 'ab28'};

%% Load DBN features to save only selected features
dbn_mat = fullfile( ...
    sprintf('/home/eeiww/ut55iqoh/MLMA_CAM21/result_feature_selection/%s/', ...
    ground_to_test), ...
    'DBN_selected_features_no_improve_stop.mat');

dbn_data          = load(dbn_mat);
selected_features = dbn_data.features;

fprintf('=======================================================\n');
fprintf('CONVERTING MAT FILES TO CSV\n');
fprintf('Ground   : %s\n', ground_to_test);
fprintf('Features : %d (DBN-selected)\n', length(selected_features));
fprintf('Output   : %s\n', output_folder);
fprintf('=======================================================\n\n');

for subj_idx = 1:length(subjects)
    subject = subjects{subj_idx};

    input_file  = fullfile(base_folder, ['full_' subject '_input_4.mat']);
    output_file = fullfile(base_folder, ['full_' subject '_output_4.mat']);

    if ~isfile(input_file) || ~isfile(output_file)
        fprintf('WARNING: Files not found for %s. Skipping.\n', subject);
        continue;
    end

    try
        input_data  = load(input_file);
        output_data = load(output_file);

        % Get feature matrix — select DBN features only
        X         = table2array(input_data.alldata);
        X_sel     = X(:, selected_features);

        % Get output columns
        out_table  = output_data.alldata;
        labels_col = out_table.labels_feat_last;
        trial_col  = out_table.trial_feat_last;
        gait_col   = out_table.gait_feat_last;

        % Build combined table for CSV
        % Columns: feat_1, feat_2, ..., feat_N, label, trial, gait
        n_rows = size(X_sel, 1);
        n_feat = size(X_sel, 2);

        % Create feature column names
        feat_names = cell(1, n_feat);
        for f = 1:n_feat
            feat_names{f} = sprintf('feat_%d', selected_features(f));
        end

        % Write INPUT CSV (features only)
        input_csv = fullfile(output_folder, [subject '_input.csv']);

        fid = fopen(input_csv, 'w');

        % Header
        fprintf(fid, '%s', strjoin(feat_names, ','));
        fprintf(fid, '\n');

        % Data rows
        for row = 1:n_rows
            fprintf(fid, '%f', X_sel(row, 1));
            for col = 2:n_feat
                fprintf(fid, ',%f', X_sel(row, col));
            end
            fprintf(fid, '\n');
        end
        fclose(fid);

        % Write OUTPUT CSV (labels, trial, gait)
        output_csv = fullfile(output_folder, [subject '_output.csv']);

        fid = fopen(output_csv, 'w');

        % Header
        fprintf(fid, 'label,trial,gait\n');

        % Data rows
        for row = 1:n_rows
            fprintf(fid, '%s,%d,%d\n', ...
                labels_col{row}, trial_col(row), gait_col(row));
        end
        fclose(fid);

        fprintf('Converted: %s  (%d rows, %d features)\n', ...
            subject, n_rows, n_feat);

    catch ME
        fprintf('ERROR: %s - %s\n', subject, ME.message);
    end
end

fprintf('\n=======================================================\n');
fprintf('CONVERSION COMPLETE!\n');
fprintf('CSV files saved to: %s\n', output_folder);
fprintf('=======================================================\n');