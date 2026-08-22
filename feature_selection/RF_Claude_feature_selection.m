%% ========================================================
%% FEATURE SELECTION: RANDOM FOREST MODEL
%% Sequential Forward Selection
%% Leave-One-Trial-Out Cross Validation
%% Run overnight
%% ========================================================

clear all; close all; clc;

%% Parallel pool for speed
if isempty(gcp('nocreate'))
    parpool('local');
end

%% ========================================================
%% USER SETTINGS
%% ========================================================

ground_to_test = 'stair';  % change to 'levelground' or 'ramp' as needed

base_folder   = sprintf('/home/eeiww/ut55iqoh/MLMA_CAM21/Classification/%s/', ground_to_test);
output_folder = sprintf('/home/eeiww/ut55iqoh/MLMA_CAM21/result_feature_selection/%s/', ground_to_test);

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

subjects = {'ab07', 'ab08', 'ab09', 'ab12', 'ab13', 'ab14', 'ab17', 'ab18', ...
            'ab19', 'ab20', 'ab21', 'ab23', 'ab24', 'ab27', 'ab28'};

%% ========================================================
%% PARAMETERS
%% ========================================================

MAX_FEATURES_TARGET      = 150;
MIN_FEATURES_FOR_INITIAL = 100;
NUM_TREES                = 20;   % reduced for speed — still reliable for selection

% Early stopping
PATIENCE        = 5;
MIN_IMPROVEMENT = 0.001;

fprintf('=======================================================\n');
fprintf('RANDOM FOREST FEATURE SELECTION\n');
fprintf('Ground           : %s\n', ground_to_test);
fprintf('Target features  : %d\n', MAX_FEATURES_TARGET);
fprintf('Trees per fold   : %d\n', NUM_TREES);
fprintf('Patience         : %d\n', PATIENCE);
fprintf('Min improvement  : %.4f\n', MIN_IMPROVEMENT);
fprintf('=======================================================\n\n');

%% ========================================================
%% STEP 1: LOAD DATA
%% ========================================================

fprintf('STEP 1: LOADING DATA\n\n');

X_combined      = [];
y_combined      = {};
trial_combined  = [];

for subj_idx = 1:length(subjects)
    subject     = subjects{subj_idx};
    input_file  = fullfile(base_folder, ['full_' subject '_input_4.mat']);
    output_file = fullfile(base_folder, ['full_' subject '_output_4.mat']);

    if ~isfile(input_file) || ~isfile(output_file)
        fprintf('WARNING: %s not found. Skipping.\n', subject);
        continue;
    end

    try
        input_data  = load(input_file);
        output_data = load(output_file);

        X          = table2array(input_data.alldata);
        out_table  = output_data.alldata;
        labels_col = out_table.labels_feat_last;
        trial_col  = out_table.trial_feat_last;

        X_combined     = [X_combined;     X];
        y_combined     = [y_combined;     labels_col];
        trial_combined = [trial_combined; trial_col + subj_idx*1000];

        fprintf('Loaded: %s  (%d samples)\n', subject, size(X,1));
    catch ME
        fprintf('ERROR: %s - %s\n', subject, ME.message);
    end
end

fprintf('\nTotal: %d samples, %d features\n\n', ...
    size(X_combined,1), size(X_combined,2));

% Label encoding
unique_labels = unique(y_combined);
num_classes   = length(unique_labels);
label_map     = containers.Map(unique_labels, 1:num_classes);

y_combined_numeric = zeros(length(y_combined), 1);
for i = 1:length(y_combined)
    y_combined_numeric(i) = label_map(y_combined{i});
end

%% ========================================================
%% STEP 2: FEATURE RANKING
%% ========================================================

fprintf('STEP 2: RANKING FEATURES\n\n');

mu_g   = mean(X_combined);
sig_g  = std(X_combined);
sig_g(sig_g == 0) = 1;
X_rank = (X_combined - mu_g) ./ sig_g;

feature_scores = zeros(size(X_combined,2), 1);
tic;
for feat = 1:size(X_combined,2)
    if mod(feat,50)==0 || feat==1
        fprintf('[%s] Ranking feature %d/%d  (%.1f sec)\n', ...
            datetime('now','Format','HH:mm:ss'), feat, size(X_combined,2), toc);
    end
    var_score            = var(X_rank(:,feat));
    corr_score           = abs(corr(X_rank(:,feat), y_combined_numeric));
    feature_scores(feat) = var_score * corr_score;
end

[~, ranked_features] = sort(feature_scores, 'descend');
fprintf('[%s] Ranking complete!\n\n', datetime('now','Format','HH:mm:ss'));

%% ========================================================
%% STEP 3: RF SEQUENTIAL FORWARD SELECTION
%%
%% NOTE: RF is scale-invariant — no normalisation needed
%% Tree splits are rank-based not magnitude-based
%% ========================================================

fprintf('STEP 3: RF FEATURE SELECTION\n\n');

unique_trials = unique(trial_combined);
num_folds     = length(unique_trials);

selected_features  = zeros(0,1);
remaining_features = ranked_features(1:min(MIN_FEATURES_FOR_INITIAL, length(ranked_features)));

history_feature  = zeros(MAX_FEATURES_TARGET, 1);
history_accuracy = zeros(MAX_FEATURES_TARGET, 1);

best_accuracy_so_far = -inf;
no_improve_count     = 0;
stop_reason          = 'max features reached';

% Parallel pool local copies
num_folds_par      = num_folds;
unique_trials_par  = unique_trials;
trial_par          = trial_combined;
y_numeric_par      = y_combined_numeric;
num_trees_par      = NUM_TREES;

iteration = 0;
tic;

while length(selected_features) < MAX_FEATURES_TARGET && ~isempty(remaining_features)
    iteration = iteration + 1;

    if iteration == 1 || mod(iteration,5) == 0
        fprintf('[%s] Iter %3d | selected: %3d | patience: %d/%d  (%.1f sec)\n', ...
            datetime('now','Format','HH:mm:ss'), iteration, ...
            length(selected_features), no_improve_count, PATIENCE, toc);
    end

    best_candidate_feature  = -1;
    best_candidate_accuracy = -inf;

    for k = 1:length(remaining_features)
        feat = remaining_features(k);

        current_features = [selected_features; feat];
        X_subset = X_combined(:, current_features);

        fold_accuracies = zeros(num_folds_par, 1);
        valid_fold      = false(num_folds_par, 1);

        parfor fold = 1:num_folds_par
            test_trial = unique_trials_par(fold);
            train_mask = trial_par ~= test_trial;
            test_mask  = trial_par == test_trial;

            X_train = X_subset(train_mask, :);
            y_train = y_numeric_par(train_mask);
            X_test  = X_subset(test_mask,  :);
            y_test  = y_numeric_par(test_mask);

            % RF is scale-invariant — no normalisation needed
            try
                rf_model     = TreeBagger(num_trees_par, X_train, y_train, ...
                                          'Method',        'classification', ...
                                          'OOBPrediction', 'off');
                y_pred_cell  = predict(rf_model, X_test);
                y_pred       = str2double(y_pred_cell);
                fold_accuracies(fold) = sum(y_pred == y_test) / length(y_test);
                valid_fold(fold)      = true;
            catch
            end
        end

        if any(valid_fold)
            avg_acc = mean(fold_accuracies(valid_fold));
            if avg_acc > best_candidate_accuracy
                best_candidate_accuracy = avg_acc;
                best_candidate_feature  = feat;
            end
        end
    end

    if best_candidate_feature == -1
        stop_reason = 'no valid candidate';
        fprintf('[%s] No valid candidate. Stopping.\n', ...
            datetime('now','Format','HH:mm:ss'));
        break;
    end

    improvement = best_candidate_accuracy - best_accuracy_so_far;

    if improvement >= MIN_IMPROVEMENT
        best_accuracy_so_far = best_candidate_accuracy;
        no_improve_count     = 0;
    else
        no_improve_count = no_improve_count + 1;
        fprintf('[%s] Iter %3d: No improvement (gain=%.5f, patience %d/%d)\n', ...
            datetime('now','Format','HH:mm:ss'), iteration, ...
            improvement, no_improve_count, PATIENCE);

        if no_improve_count >= PATIENCE
            stop_reason = sprintf('no improvement for %d consecutive iterations', PATIENCE);
            fprintf('[%s] Early stopping triggered.\n', ...
                datetime('now','Format','HH:mm:ss'));
            break;
        end
    end

    selected_features  = [selected_features; best_candidate_feature];
    remaining_features = remaining_features(remaining_features ~= best_candidate_feature);

    history_feature(iteration)  = best_candidate_feature;
    history_accuracy(iteration) = best_candidate_accuracy;
end

% Trim history
history_feature  = history_feature(1:length(selected_features));
history_accuracy = history_accuracy(1:length(selected_features));

fprintf('\n[%s] RF feature selection complete!\n', datetime('now','Format','HH:mm:ss'));
fprintf('Stop reason    : %s\n', stop_reason);
fprintf('Total selected : %d features\n\n', length(selected_features));

%% ========================================================
%% STEP 4: SAVE RESULTS
%% ========================================================

fprintf('STEP 4: SAVING RESULTS\n\n');

txt_filename = fullfile(output_folder, 'RF_selected_features.txt');
fid = fopen(txt_filename, 'w');

fprintf(fid, '=======================================================\n');
fprintf(fid, 'RF SELECTED FEATURES (Ground: %s)\n', ground_to_test);
fprintf(fid, '=======================================================\n\n');
fprintf(fid, 'Model  : Random Forest (%d trees per fold)\n', NUM_TREES);
fprintf(fid, 'Method : Sequential Forward Selection + Leave-One-Trial-Out CV\n');
fprintf(fid, 'Stop   : %s\n', stop_reason);
fprintf(fid, 'Total features available : %d\n', size(X_combined,2));
fprintf(fid, 'Features selected        : %d\n\n', length(selected_features));

fprintf(fid, 'Selected feature indices:\n[\n');
for j = 1:length(selected_features)
    if mod(j,10) == 0 || j == length(selected_features)
        fprintf(fid, ' %d\n', selected_features(j));
    else
        fprintf(fid, ' %d,', selected_features(j));
    end
end
fprintf(fid, ']\n\n');

fprintf(fid, 'Selection history:\n');
fprintf(fid, 'Iteration  Feature  Accuracy\n');
fprintf(fid, '---------  -------  --------\n');
for j = 1:length(history_feature)
    fprintf(fid, '%9d  %7d  %.6f\n', j, history_feature(j), history_accuracy(j));
end

fclose(fid);
fprintf('Saved: %s\n', txt_filename);

mat_filename = fullfile(output_folder, 'RF_selected_features.mat');
features = selected_features;
save(mat_filename, 'features', 'history_feature', 'history_accuracy');
fprintf('Saved: %s\n\n', mat_filename);

fprintf('=======================================================\n');
fprintf('RF FEATURE SELECTION COMPLETE!\n');
fprintf('Ground: %s | Features: %d | Stop: %s\n', ...
    ground_to_test, length(selected_features), stop_reason);
fprintf('=======================================================\n');