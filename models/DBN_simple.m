%% ========================================================
%% LDA-HMM (DBN) FOR LOCOMOTION CLASSIFICATION
%%
%% Online Prediction
%% LDA Emissions + HMM Temporal Filtering
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
%    if strcmp(labels{i},'turn1') || ...
%       strcmp(labels{i},'turn2')
% 
%        labels{i} = 'turn';
%    end
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
%% SORT BY TRIAL -> GAIT %
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



%% =====================================
%% PRINT TRANSITIONS OF ONE TRIAL--test
%% =====================================

trial_to_print = 1;

idx = find(trial == trial_to_print);

fprintf('\nTransitions in Trial %d\n', trial_to_print);
fprintf('-----------------------------------\n');

for k = 1:length(idx)-1

    fprintf('%s --> %s\n', ...
        labels{idx(k)}, ...
        labels{idx(k+1)});

end
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
        i, ...
        unique_labels{i});
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
all_pred_lda = [];

fprintf('=====================================\n');
fprintf('RUNNING CROSS VALIDATION\n');
fprintf('=====================================\n\n');

%% ========================================================
%% CROSS VALIDATION
%% ========================================================

for fold = 1:num_folds

    fprintf('-------------------------------------\n');
    fprintf('Fold %d / %d\n', ...
        fold, ...
        num_folds);

    test_trial = unique_trials(fold);

    fprintf('Testing Trial %d\n', ...
        test_trial);

    %% ----------------------------------------------------
    %% TRAIN / TEST SPLIT
    %% ----------------------------------------------------

    train_mask = trial ~= test_trial;
    test_mask  = trial == test_trial;

    X_train = X(train_mask,:);
    y_train = y(train_mask);

    X_test = X(test_mask,:);
    y_test = y(test_mask);

    %% ----------------------------------------------------
    %% NORMALIZATION
    %% ----------------------------------------------------

    mu = mean(X_train);

    sigma = std(X_train);

    sigma(sigma == 0) = 1;

    X_train = (X_train - mu) ./ sigma;
    X_test  = (X_test - mu) ./ sigma;

    %% ----------------------------------------------------
    %% LDA EMISSION MODEL
    %% ----------------------------------------------------

    fprintf('Training LDA...\n');

    lda_model = fitcdiscr( ...
        X_train, ...
        y_train, ...
        'DiscrimType','linear');

    %% ----------------------------------------------------
    %% EMISSION PROBABILITIES
    %% ----------------------------------------------------

    fprintf('Computing emission probabilities...\n');

    [~,scores] = predict(lda_model, X_test);

    emission_probs = scores;

    emission_probs(emission_probs < 1e-10) = 1e-10;

    emission_probs = ...
        emission_probs ./ ...
        sum(emission_probs,2);

    [~, y_pred_lda] = max(emission_probs, [], 2);

    %% ----------------------------------------------------
    %% TRANSITION MATRIX
    %% ----------------------------------------------------

    fprintf('Building transition matrix...\n');

    transition_matrix = ones(num_classes);

    train_trials = unique(trial(train_mask));

    for tt = 1:length(train_trials)

        current_trial = train_trials(tt);

        current_mask = ...
            train_mask & ...
            (trial == current_trial);

        seq = y(current_mask);

        for k = 1:length(seq)-1

            from_state = seq(k);
            to_state = seq(k+1);

            transition_matrix( ...
                from_state, ...
                to_state) = ...
                transition_matrix( ...
                from_state, ...
                to_state) + 1;
        end
    end
   disp(transition_matrix)

    transition_matrix = ...
        transition_matrix ./ ...
        sum(transition_matrix,2);

    disp(transition_matrix)

    %% ----------------------------------------------------
    %% HMM / DBN ONLINE FILTERING
    %% ----------------------------------------------------

    fprintf('Applying temporal filtering...\n');

    filtered_probs = ...
        zeros(size(emission_probs));

    filtered_probs(1,:) = ...
        emission_probs(1,:) ./ ...
        sum(emission_probs(1,:));

    for t = 2:size(emission_probs,1)

        temporal_prior = ...
            filtered_probs(t-1,:) * ...
            transition_matrix;

        filtered_probs(t,:) = ...
            emission_probs(t,:) .* ...
            temporal_prior;

        filtered_probs(t,:) = ...
            filtered_probs(t,:) ./ ...
            sum(filtered_probs(t,:));
    end

    %% ----------------------------------------------------
    %% PREDICTIONS
    %% ----------------------------------------------------

    [~,y_pred] = max( ...
        filtered_probs, ...
        [], ...
        2);

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
    all_pred_lda = [all_pred_lda; y_pred_lda]; % LDA

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

title('LDA-HMM (DBN)');

%% ========================================================
%% DONE
%% ========================================================

%% Check: Is DBN smoother than LDA?

% Your variable name in DBN code
fprintf('\n=== TEMPORAL SMOOTHING CHECK ===\n');
% Count consecutive same predictions across ALL folds

dbn_smooth = sum(diff(all_pred) == 0);

lda_smooth = sum(diff(all_pred_lda) == 0);

fprintf('LDA smooth: %d / %d\n', ...
    lda_smooth, ...
    length(all_pred_lda)-1);

fprintf('DBN smooth: %d / %d\n', ...
    dbn_smooth, ...
    length(all_pred)-1);

fprintf('Smoothness gain: %d (%.1f%%)\n', ...
    dbn_smooth - lda_smooth, ...
    100*(dbn_smooth - lda_smooth)/(length(all_pred)-1));

if dbn_smooth > lda_smooth
    fprintf('✓ DBN is SMOOTHER (temporal working!)\n');
else
    fprintf('⚠️ DBN is NOT smoother\n');
end

% Check accuracy difference

lda_acc = mean(all_pred_lda == all_truth);

dbn_acc = mean(all_pred == all_truth);

fprintf('\nAccuracy difference:\n');

fprintf('LDA: %.4f\n', lda_acc);

fprintf('DBN: %.4f\n', dbn_acc);

fprintf('Difference: %.4f (%.2f%%)\n', ...
    dbn_acc - lda_acc, ...
    100*(dbn_acc - lda_acc));

disp(scores(1:5,:))
unique(output_table.trial_feat_last)