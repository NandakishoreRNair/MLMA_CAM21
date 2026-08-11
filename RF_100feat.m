%% ========================================================
%% TEST ACTUAL ACCURACY OF SELECTED RF FEATURES
%%
%% PURPOSE:
%% Evaluate REAL classification accuracy using:
%% - selected features from SFS
%% - proper Leave-One-Trial-Out CV
%%
%% NO DATA LEAKAGE
%% ========================================================

clear;
clc;
close all;

%% ========================================================
%% USER SETTINGS
%% ========================================================

base_folder = ...
'D:\CAM21\data\Classification\levelground\85\400\';

subject = 'ab14';

%% ========================================================
%% SELECTED FEATURES
%% Paste from feature selection report
%% ========================================================

selected_features = [ ...
151, 228, 3, 170, 157, 188, 15, 158, 13, 150, ...
140, 190, 187, 141, 9, 138, 5, 10, 225, 181, ...
162, 178, 224, 160, 180, 159, 200, 20, 154, 204, ...
165, 149, 37, 164, 189, 45, 205, 38, 184, 240, ...
168, 194, 163, 218, 175, 198, 185, 43, 148, 39, ...
199, 195, 215, 46, 145, 40, 191, 18, 214, 146, ...
210, 19, 206, 230, 174, 144, 250, 220, 183, 254, ...
197, 169, 222, 248, 107, 179, 11, 255, 166, 203, ...
25, 213, 41, 209, 208, 245, 161, 127, 242, 238, ...
249, 256, 44, 177, 173, 134, 155, 14, 137, 21 ...
];

%% ========================================================
%% LOAD DATA
%% ========================================================

fprintf('=====================================\n');
fprintf('LOADING DATA\n');
fprintf('=====================================\n\n');

input_file = fullfile( ...
    base_folder, ...
    [subject '_input.mat']);

output_file = fullfile( ...
    base_folder, ...
    [subject '_output.mat']);

input_data = load(input_file);

output_data = load(output_file);

X = table2array(input_data.alldata);

output_table = output_data.alldata;

fprintf('Original data:\n');
fprintf('Samples : %d\n', size(X,1));
fprintf('Features: %d\n\n', size(X,2));

%% ========================================================
%% USE ONLY SELECTED FEATURES
%% ========================================================

X = X(:, selected_features);

fprintf('Using selected features only\n');
fprintf('Selected features: %d\n\n', size(X,2));

%% ========================================================
%% EXTRACT LABELS / TRIALS / GAIT
%% ========================================================

labels = output_table.labels_feat_last;

trial = output_table.trial_feat_last;

gait = output_table.gait_feat_last;

%% ========================================================
%% MERGE TURN LABELS
%% ========================================================

for i = 1:length(labels)

    if strcmp(labels{i}, 'turn1') || ...
       strcmp(labels{i}, 'turn2')

        labels{i} = 'turn';
    end
end

%% ========================================================
%% KEEP ONLY TRIALS 1-5
%% ========================================================

mask = trial >= 1 & trial <= 5;

X = X(mask,:);

labels = labels(mask);

trial = trial(mask);

gait = gait(mask);

fprintf('Remaining samples after filtering: %d\n\n', ...
    size(X,1));

%% ========================================================
%% SORT DATA
%% Trial --> Gait
%% ========================================================

fprintf('Sorting data...\n');

sort_table = table( ...
    trial, ...
    gait, ...
    (1:length(gait))', ...
    'VariableNames', ...
    {'trial','gait','rowInd'});

sort_table = sortrows(sort_table, ...
    {'trial','gait'});

sortedInds = sort_table.rowInd;

X = X(sortedInds,:);

labels = labels(sortedInds);

trial = trial(sortedInds);

gait = gait(sortedInds);

fprintf('Sorting complete\n\n');

%% ========================================================
%% CONVERT LABELS TO NUMERIC
%% ========================================================

unique_labels = unique(labels);

num_classes = length(unique_labels);

label_map = containers.Map();

for i = 1:num_classes

    label_map(unique_labels{i}) = i;
end

y = zeros(length(labels),1);

for i = 1:length(labels)

    y(i) = label_map(labels{i});
end

fprintf('Classes:\n');

for i = 1:num_classes

    fprintf('%d --> %s\n', ...
        i, unique_labels{i});
end

fprintf('\n');

%% ========================================================
%% LEAVE-ONE-TRIAL-OUT CROSS VALIDATION
%% ========================================================

unique_trials = unique(trial);

num_folds = length(unique_trials);

fold_accuracies = zeros(num_folds,1);

all_pred = [];

all_truth = [];

fprintf('=====================================\n');
fprintf('RUNNING TRUE EVALUATION\n');
fprintf('Leave-One-Trial-Out CV\n');
fprintf('=====================================\n\n');

%% ========================================================
%% CROSS VALIDATION LOOP
%% ========================================================

for fold = 1:num_folds

    fprintf('=====================================\n');
    fprintf('Fold %d / %d\n', fold, num_folds);

    test_trial = unique_trials(fold);

    fprintf('Testing Trial %d\n', test_trial);

    %% ----------------------------------------------------
    %% TRAIN / TEST SPLIT
    %% ----------------------------------------------------

    train_mask = trial ~= test_trial;

    test_mask = trial == test_trial;

    X_train = X(train_mask,:);

    y_train = y(train_mask);

    X_test = X(test_mask,:);

    y_test = y(test_mask);

    %% ----------------------------------------------------
    %% NORMALIZATION
    %% IMPORTANT:
    %% ONLY USING TRAINING STATISTICS
    %% ----------------------------------------------------

    fprintf('Normalizing...\n');

    mu = mean(X_train);

    sigma = std(X_train);

    sigma(sigma == 0) = 1;

    X_train = (X_train - mu) ./ sigma;

    X_test = (X_test - mu) ./ sigma;

    %% ----------------------------------------------------
    %% TRAIN RANDOM FOREST
    %% ----------------------------------------------------

    fprintf('Training Random Forest...\n');

    rf_model = TreeBagger( ...
        100, ...
        X_train, ...
        y_train, ...
        'Method','classification', ...
        'OOBPrediction','off');

    %% ----------------------------------------------------
    %% TEST MODEL
    %% ----------------------------------------------------

    fprintf('Testing model...\n');

    y_pred = predict(rf_model, X_test);

    %% Convert cell output to numeric

    if iscell(y_pred)

        y_pred = str2double(y_pred);
    end

    %% ----------------------------------------------------
    %% ACCURACY
    %% ----------------------------------------------------

    accuracy = ...
        sum(y_pred == y_test) / length(y_test);

    fold_accuracies(fold) = accuracy;

    fprintf('Fold Accuracy: %.2f %%\n\n', ...
        accuracy * 100);

    %% Store results

    all_pred = [all_pred; y_pred];

    all_truth = [all_truth; y_test];
end

%% ========================================================
%% FINAL RESULTS
%% ========================================================

fprintf('\n');
fprintf('=====================================\n');
fprintf('FINAL RESULTS\n');
fprintf('=====================================\n\n');

mean_acc = mean(fold_accuracies);

std_acc = std(fold_accuracies);

fprintf('TRUE Accuracy: %.2f %%\n', ...
    mean_acc * 100);

fprintf('Std Accuracy : %.2f %%\n\n', ...
    std_acc * 100);

fprintf('Fold Results:\n');

for i = 1:num_folds

    fprintf('Trial %d --> %.2f %%\n', ...
        unique_trials(i), ...
        fold_accuracies(i) * 100);
end

%% ========================================================
%% CONFUSION MATRIX
%% ========================================================

pred_labels = unique_labels(all_pred);

truth_labels = unique_labels(all_truth);

figure;

confusionchart( ...
    truth_labels, ...
    pred_labels);

title('Random Forest - TRUE Evaluation');

%% ========================================================
%% DONE
%% ========================================================

fprintf('\nDone.\n');