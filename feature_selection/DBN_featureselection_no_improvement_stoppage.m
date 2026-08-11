%% ========================================================
%% FEATURE SELECTION: DBN MODEL
%% Sequential Forward Selection using Dynamic Bayesian Network
%% Compare Temporal (DBN) vs Non-Temporal (LDA, SVM, RF)
%% ========================================================

clear all; close all; clc;

%% ========================================================
%% USER SETTINGS - CHOOSE PHASE
%% ========================================================

ground_to_test = 'stair';  % Options: 35 or 85

fprintf('=======================================================\n');
fprintf('DBN FEATURE SELECTION\n');
fprintf('Ground: %s\n', ground_to_test);
fprintf('=======================================================\n\n');

%% ========================================================
%% SETUP PATHS
%% ========================================================

base_folder = sprintf('D:\\CAM21\\data\\Classification\\%s\\', ground_to_test);

subjects = {'ab07', 'ab08', 'ab09', 'ab12', 'ab13', 'ab14', 'ab17', 'ab18', ...
            'ab19', 'ab20', 'ab21', 'ab23', 'ab24', 'ab27', 'ab28'};

output_folder = sprintf('D:\\CAM21\\code\\my_code\\result_feature_selection\\%s\\', ground_to_test);
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% ========================================================
%% PARAMETERS
%% ========================================================

MAX_FEATURES_TARGET  = 150;
MIN_FEATURES_FOR_INITIAL = 100;

% --- Early stopping ---
% Stop if accuracy does not improve by at least MIN_IMPROVEMENT
% for PATIENCE consecutive iterations.
PATIENCE     = 5;     % how many iterations without improvement before stopping
MIN_IMPROVEMENT = 1e-4; % minimum meaningful accuracy gain

fprintf('Configuration:\n');
fprintf('  Target features  : %d\n', MAX_FEATURES_TARGET);
fprintf('  Gound            : %s\n', ground_to_test);
fprintf('  Early-stop patience : %d iterations\n', PATIENCE);
fprintf('  Min improvement  : %.4f\n', MIN_IMPROVEMENT);
fprintf('  Base folder      : %s\n', base_folder);
fprintf('  Output folder    : %s\n\n', output_folder);

%% ========================================================
%% STEP 1: LOAD AND COMBINE DATA
%% ========================================================

fprintf('=======================================================\n');
fprintf('STEP 1: LOADING DATA FROM ALL SUBJECTS\n');
fprintf('=======================================================\n\n');

X_combined      = [];
y_combined      = {};
trial_combined  = [];
subject_indices = [];

for subj_idx = 1:length(subjects)
    subject = subjects{subj_idx};

    input_file  = fullfile(base_folder, ['full_' subject '_input_4.mat']);
    output_file = fullfile(base_folder, ['full_' subject '_output_4.mat']);

    if ~isfile(input_file) || ~isfile(output_file)
        fprintf('WARNING: Files not found for %s. Skipping...\n', subject);
        continue;
    end

    try
        input_data  = load(input_file);
        output_data = load(output_file);

        X            = table2array(input_data.alldata);
        output_table = output_data.alldata;

        labels_col = output_table.labels_feat_last;
        trial_col  = output_table.trial_feat_last;

        % Merge turn1 and turn2 into a single 'turn' class
        for i = 1:length(labels_col)
            if strcmp(labels_col{i}, 'turn1') || strcmp(labels_col{i}, 'turn2')
                labels_col{i} = 'turn';
            end
        end

        X_combined      = [X_combined;      X];
        y_combined      = [y_combined;      labels_col];
        trial_combined  = [trial_combined;  trial_col];
        subject_indices = [subject_indices; repmat(subj_idx, size(X,1), 1)];

        fprintf('Loaded: %s  (%d samples, %d features)\n', ...
            subject, size(X,1), size(X,2));

    catch ME
        fprintf('ERROR loading %s: %s\n', subject, ME.message);
    end
end

fprintf('\nTotal combined data: %d samples, %d features\n', ...
    size(X_combined,1), size(X_combined,2));

% Convert labels to numeric
unique_labels = unique(y_combined);
label_map     = containers.Map(unique_labels, 1:length(unique_labels));

y_combined_numeric = zeros(length(y_combined), 1);
for i = 1:length(y_combined)
    y_combined_numeric(i) = label_map(y_combined{i});
end

num_classes = length(unique_labels);
fprintf('Classes: %d\n\n', num_classes);

%% ========================================================
%% STEP 2: FEATURE RANKING
%%   NOTE: only z-score here for ranking purposes.
%%   The CV loop will re-normalise per fold (no leakage).
%% ========================================================

fprintf('=======================================================\n');
fprintf('STEP 2: RANKING FEATURES\n');
fprintf('=======================================================\n\n');

fprintf('Computing feature scores (variance x label-correlation)...\n\n');

% Global z-score used ONLY for ranking, NOT passed into CV
mu_global    = mean(X_combined);
sigma_global = std(X_combined);
sigma_global(sigma_global == 0) = 1;
X_normalized_rank = (X_combined - mu_global) ./ sigma_global;

feature_scores = zeros(size(X_combined, 2), 1);
tic;

for feat = 1:size(X_combined, 2)
    if mod(feat, 50) == 0 || feat == 1
        fprintf('[%s] Feature %3d / %d  (Elapsed: %.1f sec)\n', ...
            datetime('now','Format','HH:mm:ss'), feat, size(X_combined,2), toc);
    end
    var_score        = var(X_normalized_rank(:, feat));
    corr_with_labels = abs(corr(X_normalized_rank(:, feat), y_combined_numeric));
    feature_scores(feat) = var_score * corr_with_labels;
end

fprintf('[%s] Feature ranking complete!\n\n', datetime('now','Format','HH:mm:ss'));

[sorted_scores, ranked_features] = sort(feature_scores, 'descend');

fprintf('Top 20 ranked features:\n');
for i = 1:min(20, length(ranked_features))
    fprintf('  Rank %2d: Feature %3d  (Score = %.6f)\n', ...
        i, ranked_features(i), sorted_scores(i));
end
fprintf('\n');

%% ========================================================
%% STEP 3: DBN FEATURE SELECTION  (Leave-One-Trial-Out CV)
%% ========================================================

fprintf('=======================================================\n');
fprintf('STEP 3: DBN FEATURE SELECTION\n');
fprintf('Sequential Forward Selection with Leave-One-Trial-Out CV\n');
fprintf('Early stopping: patience=%d, min_improvement=%.4f\n', ...
    PATIENCE, MIN_IMPROVEMENT);
fprintf('=======================================================\n\n');

unique_trials = unique(trial_combined);
num_folds     = length(unique_trials);

% Use column vectors throughout to avoid shape-mismatch errors
selected_features  = zeros(0, 1);                          % column vector
remaining_features = ranked_features(1:min(MIN_FEATURES_FOR_INITIAL, ...
                        length(ranked_features)));         % already column from sort

% Pre-allocate history arrays (faster than struct concatenation)
history_feature  = zeros(MAX_FEATURES_TARGET, 1);
history_accuracy = zeros(MAX_FEATURES_TARGET, 1);

% Early-stopping state
best_accuracy_so_far = -inf;
no_improve_count     = 0;
stop_reason          = 'max features reached';

iteration = 0;
tic;

while length(selected_features) < MAX_FEATURES_TARGET && ~isempty(remaining_features)
    iteration = iteration + 1;

    if iteration == 1 || mod(iteration, 5) == 0
        fprintf('[%s] Iter %3d | selected: %3d | patience: %d/%d  (%.1f sec)\n', ...
            datetime('now','Format','HH:mm:ss'), iteration, ...
            length(selected_features), no_improve_count, PATIENCE, toc);
    end

    best_candidate_feature  = -1;
    best_candidate_accuracy = -inf;

    % ---- test each candidate feature ----
    for k = 1:length(remaining_features)
        feat = remaining_features(k);

        current_features = [selected_features; feat];   % both column vectors → safe
        X_subset = X_combined(:, current_features);     % use RAW data; normalise per fold

        fold_accuracies = zeros(num_folds, 1);
        valid_fold      = false(num_folds, 1);

        for fold = 1:num_folds
            test_trial = unique_trials(fold);

            train_mask = trial_combined ~= test_trial;
            test_mask  = trial_combined == test_trial;

            X_train = X_subset(train_mask, :);
            y_train = y_combined_numeric(train_mask);
            X_test  = X_subset(test_mask,  :);
            y_test  = y_combined_numeric(test_mask);

            % ---- Normalise using TRAINING statistics only (no leakage) ----
            mu_tr    = mean(X_train);
            sigma_tr = std(X_train);
            sigma_tr(sigma_tr == 0) = 1;

            X_train = (X_train - mu_tr) ./ sigma_tr;
            X_test  = (X_test  - mu_tr) ./ sigma_tr;

            try
                % Train LDA for emission probabilities
                lda_model = fitcdiscr(X_train, y_train, 'DiscrimType', 'linear');
                [~, scores] = predict(lda_model, X_test);

                % Clamp and row-normalise emission probabilities
                emission_probs = max(scores, 1e-10);
                emission_probs = emission_probs ./ sum(emission_probs, 2);

                % Build transition matrix from TRAINING sequence
                transition_matrix = ones(num_classes, num_classes);
                for t = 1:length(y_train)-1
                    transition_matrix(y_train(t), y_train(t+1)) = ...
                        transition_matrix(y_train(t), y_train(t+1)) + 1;
                end
                transition_matrix = transition_matrix ./ sum(transition_matrix, 2);

                % HMM forward filtering
                T = size(emission_probs, 1);
                filtered_probs = zeros(T, num_classes);
                filtered_probs(1, :) = emission_probs(1, :) / sum(emission_probs(1, :));

                for t = 2:T
                    temporal_prior      = filtered_probs(t-1, :) * transition_matrix;
                    filtered_probs(t,:) = emission_probs(t,:) .* temporal_prior;
                    s = sum(filtered_probs(t,:));
                    if s < 1e-300
                        % underflow guard: fall back to emission only
                        filtered_probs(t,:) = emission_probs(t,:);
                        s = sum(filtered_probs(t,:));
                    end
                    filtered_probs(t,:) = filtered_probs(t,:) ./ s;
                end

                [~, y_pred] = max(filtered_probs, [], 2);
                fold_accuracies(fold) = sum(y_pred == y_test) / length(y_test);
                valid_fold(fold)      = true;

            catch
                % skip fold on error (e.g. rank-deficient LDA)
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

    % ---- No valid candidate found at all ----
    if best_candidate_feature == -1
        stop_reason = 'no valid candidate (all folds errored)';
        fprintf('[%s] Iter %3d: No valid candidate found. Stopping.\n', ...
            datetime('now','Format','HH:mm:ss'), iteration);
        break;
    end

    % ---- Check whether accuracy actually improved ----
    improvement = best_candidate_accuracy - best_accuracy_so_far;

    if improvement >= MIN_IMPROVEMENT
        % Genuine improvement
        best_accuracy_so_far = best_candidate_accuracy;
        no_improve_count     = 0;
    else
        % No meaningful improvement this iteration
        no_improve_count = no_improve_count + 1;
        fprintf('[%s] Iter %3d: No improvement (gain=%.5f, patience %d/%d)\n', ...
            datetime('now','Format','HH:mm:ss'), iteration, ...
            improvement, no_improve_count, PATIENCE);

        if no_improve_count >= PATIENCE
            stop_reason = sprintf('no improvement for %d consecutive iterations', PATIENCE);
            fprintf('[%s] Early stopping triggered after %d iterations without improvement.\n', ...
                datetime('now','Format','HH:mm:ss'), PATIENCE);
            break;
        end
    end

    % ---- Accept the best feature found this iteration ----
    selected_features  = [selected_features; best_candidate_feature];
    remaining_features = remaining_features(remaining_features ~= best_candidate_feature);

    history_feature(iteration)  = best_candidate_feature;
    history_accuracy(iteration) = best_candidate_accuracy;
end

% Trim pre-allocated arrays to actual length
history_feature  = history_feature(1:iteration);
history_accuracy = history_accuracy(1:iteration);

fprintf('\n[%s] DBN feature selection complete!\n', datetime('now','Format','HH:mm:ss'));
fprintf('Stop reason    : %s\n', stop_reason);
fprintf('Total selected : %d features\n\n', length(selected_features));

%% ========================================================
%% STEP 4: SAVE RESULTS
%% ========================================================

fprintf('=======================================================\n');
fprintf('STEP 4: SAVING RESULTS\n');
fprintf('=======================================================\n\n');

txt_filename = fullfile(output_folder, 'DBN_selected_features_no_improve_stop.txt');
fid = fopen(txt_filename, 'w');

fprintf(fid, '=======================================================\n');
fprintf(fid, 'DBN SELECTED FEATURES (Ground: %s)\n', ground_to_test);
fprintf(fid, '=======================================================\n\n');
fprintf(fid, 'Method : Sequential Forward Selection + Leave-One-Trial-Out CV\n');
fprintf(fid, 'Stop   : %s\n', stop_reason);
fprintf(fid, 'Total features available : 256\n');
fprintf(fid, 'Features selected        : %d\n\n', length(selected_features));

fprintf(fid, 'Selected feature indices:\n[\n');
for j = 1:length(selected_features)
    if mod(j, 10) == 0 || j == length(selected_features)
        fprintf(fid, ' %d\n', selected_features(j));
    else
        fprintf(fid, ' %d,', selected_features(j));
    end
end
fprintf(fid, ']\n\n');

fprintf(fid, 'Feature importance (by selection order):\n');
for j = 1:min(30, length(selected_features))
    feat = selected_features(j);
    fprintf(fid, '  %2d. Feature %3d  (Initial rank: %d)\n', ...
        j, feat, find(ranked_features == feat, 1));
end
if length(selected_features) > 30
    fprintf(fid, '  ... and %d more features\n', length(selected_features) - 30);
end

fprintf(fid, '\n\nSelection history:\n');
fprintf(fid, 'Iteration  Feature  Accuracy\n');
fprintf(fid, '---------  -------  --------\n');
for j = 1:length(history_feature)
    fprintf(fid, '%9d  %7d  %.6f\n', j, history_feature(j), history_accuracy(j));
end

fclose(fid);
fprintf('Saved: %s\n', txt_filename);

mat_filename = fullfile(output_folder, 'DBN_selected_features_no_improve_stop.mat');
features = selected_features;
save(mat_filename, 'features', 'history_feature', 'history_accuracy');
fprintf('Saved: %s\n\n', mat_filename);

%% ========================================================
%% STEP 5: SUMMARY
%% ========================================================

fprintf('=======================================================\n');
fprintf('SUMMARY\n');
fprintf('=======================================================\n\n');
fprintf('ground          : %s\n', ground_to_test);
fprintf('Features selected : %d / 256\n', length(selected_features));
fprintf('Iterations run : %d\n', iteration);
fprintf('Stop reason    : %s\n', stop_reason);
if ~isempty(history_accuracy)
    fprintf('Final accuracy : %.4f\n', history_accuracy(end));
    fprintf('Best accuracy  : %.4f\n', max(history_accuracy));
end
fprintf('\nOutput saved to: %s\n', output_folder);
fprintf('\n=======================================================\n');
fprintf('FEATURE SELECTION COMPLETE!\n');
fprintf('=======================================================\n');