
%% ========================================================
%% LDA FOR LOCOMOTION CLASSIFICATION
%%
%% Leave-One-Trial-Out Cross Validation
%%
%% ========================================================

clear;
clc;
close all;

%% ========================================================
%% USER SETTINGS
%% ========================================================

base_folder = ...
'D:\CAM21\data\Classification\stair\';

subject = 'ab12';
%% ========================================================
%% SELECTED FEATURES
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
    ['full_' subject '_input_4.mat']);

output_file = fullfile( ...
    base_folder, ...
    ['full_' subject '_output_4.mat']);

input_data = load(input_file);
output_data = load(output_file);

X = table2array(input_data.alldata);

%% Keep only selected features

X = X(:, selected_features);


output_table = output_data.alldata;

fprintf('Samples : %d\n', size(X,1));
fprintf('Features Used : %d\n', size(X,2));
fprintf('Selected Features : %d\n\n', ...
length(selected_features));


%% ========================================================
%% EXTRACT LABELS / TRIAL / GAIT
%% ========================================================

labels = output_table.labels_feat_last;
trial  = output_table.trial_feat_last;
gait   = output_table.gait_feat_last;

%% ========================================================
%% MERGE TURN LABELS
%% ========================================================

% fprintf('Merging turn labels...\n');
% 
% for i = 1:length(labels)
% 
%     if strcmp(labels{i},'turn1') || ...
%        strcmp(labels{i},'turn2')
% 
%         labels{i} = 'turn';
%     end
% end

%% ========================================================
%% KEEP TRIALS 1-5
%% ========================================================

% mask = trial >= 1 & trial <= 5;
% 
% X = X(mask,:);
% labels = labels(mask);
% trial = trial(mask);
% gait = gait(mask);
% 
% fprintf('Remaining samples: %d\n\n', size(X,1));
disp(unique(trial)');
%% ========================================================
%% SORT BY TRIAL THEN GAIT %
%% ========================================================

% fprintf('Sorting by Trial and Gait %% ...\n');
% 
% sort_table = table( ...
%     trial, ...
%     gait, ...
%     (1:length(trial))', ...
%     'VariableNames', ...
%     {'trial','gait','rowInd'});
% 
% sort_table = sortrows( ...
%     sort_table, ...
%     {'trial','gait'});
% 
% sortedInds = sort_table.rowInd;
% 
% X = X(sortedInds,:);
% labels = labels(sortedInds);
% trial = trial(sortedInds);
% gait = gait(sortedInds);
% 
% fprintf('Sorting complete.\n\n');

%% ========================================================
%% LABEL ENCODING
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
%% LEAVE-ONE-TRIAL-OUT CV
%% ========================================================

unique_trials = unique(trial);

num_folds = length(unique_trials);

fold_accuracies = zeros(num_folds,1);

all_pred = [];
all_truth = [];

fprintf('=====================================\n');
fprintf('RUNNING CROSS VALIDATION\n');
fprintf('=====================================\n\n');

%% ========================================================
%% CROSS VALIDATION LOOP
%% ========================================================

for fold = 1:num_folds

    fprintf('-------------------------------------\n');
    fprintf('Fold %d / %d\n', ...
        fold, num_folds);

    test_trial = unique_trials(fold);

    fprintf('Testing Trial %d\n', ...
        test_trial);

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
    %% ----------------------------------------------------

    fprintf('Normalizing features...\n');

    mu = mean(X_train);

    sigma = std(X_train);

    sigma(sigma == 0) = 1;

    X_train = (X_train - mu) ./ sigma;
    X_test  = (X_test  - mu) ./ sigma;

    %% ----------------------------------------------------
    %% TRAIN LDA
    %% ----------------------------------------------------

    fprintf('Training LDA...\n');

    lda_model = fitcdiscr( ...
        X_train, ...
        y_train, ...
        'DiscrimType','linear');

    %% ----------------------------------------------------
    %% PREDICT
    %% ----------------------------------------------------

    fprintf('Predicting...\n');

    [y_pred, ~] = predict( ...
        lda_model, ...
        X_test);

    %% ----------------------------------------------------
    %% ACCURACY
    %% ----------------------------------------------------

    accuracy = ...
        mean(y_pred == y_test);

    fold_accuracies(fold) = accuracy;

    fprintf('Accuracy = %.2f %%\n\n', ...
        accuracy * 100);

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

fprintf('Mean Accuracy : %.2f %%\n', ...
    mean_acc * 100);

fprintf('Std Accuracy  : %.2f %%\n\n', ...
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

title('LDA');

%% ========================================================
%% DONE
%% ========================================================

fprintf('\nDone.\n');

