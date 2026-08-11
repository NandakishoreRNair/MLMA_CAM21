%% ========================================================
%% FEATURE SELECTION: DBN MODEL
%% Sequential Forward Selection using Dynamic Bayesian Network
%% Compare Temporal (DBN) vs Non-Temporal (LDA, SVM, RF)
%% ========================================================

clear all; close all; clc;

%% ========================================================
%% USER SETTINGS - CHOOSE PHASE
%% ========================================================

% Choose which phase to test
phase_to_test = 85;  % Options: 35 or 85
% phase_to_test = 35;  % Uncomment for 35% phase

fprintf('=======================================================\n');
fprintf('DBN FEATURE SELECTION\n');
fprintf('Phase: %d%%\n', phase_to_test);
fprintf('=======================================================\n\n');

%% ========================================================
%% SETUP PATHS
%% ========================================================

base_folder = sprintf('D:\\CAM21\\data\\Classification\\levelground\\%d\\400\\', phase_to_test);

subjects = {'ab07', 'ab08', 'ab09', 'ab12', 'ab13', 'ab14', 'ab17', 'ab18', ...
            'ab19', 'ab20', 'ab21', 'ab23', 'ab24', 'ab27', 'ab28'};

output_folder = sprintf('D:\\CAM21\\feature_selection\\phase_%d\\', phase_to_test);
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% ========================================================
%% PARAMETERS
%% ========================================================

MAX_FEATURES_TARGET = 150;
MIN_FEATURES_FOR_INITIAL = 100;

fprintf('Configuration:\n');
fprintf('  Target features: %d\n', MAX_FEATURES_TARGET);
fprintf('  Phase: %d%%\n', phase_to_test);
fprintf('  Base folder: %s\n', base_folder);
fprintf('  Output folder: %s\n\n', output_folder);

%% ========================================================
%% STEP 1: LOAD AND COMBINE DATA
%% ========================================================

fprintf('=======================================================\n');
fprintf('STEP 1: LOADING DATA FROM ALL SUBJECTS\n');
fprintf('=======================================================\n\n');

X_combined = [];
y_combined = [];
trial_combined = [];
subject_indices = [];

for subj_idx = 1:length(subjects)
    subject = subjects{subj_idx};
    
    input_file = fullfile(base_folder, [subject '_input.mat']);
    output_file = fullfile(base_folder, [subject '_output.mat']);
    
    if ~isfile(input_file) || ~isfile(output_file)
        fprintf('WARNING: Files not found for %s. Skipping...\n', subject);
        continue;
    end
    
    try
        input_data = load(input_file);
        output_data = load(output_file);
        
        X = table2array(input_data.alldata);
        output_table = output_data.alldata;
        
        labels_col = output_table.labels_feat_last;
        trial_col = output_table.trial_feat_last;
        
        % Merge turn1 and turn2
        for i = 1:length(labels_col)
            if strcmp(labels_col{i}, 'turn1') || strcmp(labels_col{i}, 'turn2')
                labels_col{i} = 'turn';
            end
        end
        
        X_combined = [X_combined; X];
        y_combined = [y_combined; labels_col];
        trial_combined = [trial_combined; trial_col];
        subject_indices = [subject_indices; repmat(subj_idx, size(X, 1), 1)];
        
        fprintf('Loaded: %s (%d samples, %d features)\n', subject, size(X, 1), size(X, 2));
        
    catch ME
        fprintf('ERROR loading %s: %s\n', subject, ME.message);
        continue;
    end
end

fprintf('\n');
fprintf('Total combined data: %d samples, %d features\n', size(X_combined, 1), size(X_combined, 2));

% Convert labels to numeric
unique_labels = unique(y_combined);
label_map = containers.Map(unique_labels, 1:length(unique_labels));

y_combined_numeric = zeros(length(y_combined), 1);
for i = 1:length(y_combined)
    y_combined_numeric(i) = label_map(y_combined{i});
end

num_classes = length(unique_labels);
fprintf('Classes: %d\n\n', num_classes);

%% ========================================================
%% STEP 2: FEATURE RANKING
%% ========================================================

fprintf('=======================================================\n');
fprintf('STEP 2: RANKING FEATURES\n');
fprintf('=======================================================\n\n');

fprintf('Computing feature scores using variance and correlation...\n\n');

X_normalized = (X_combined - mean(X_combined)) ./ std(X_combined);

feature_scores = [];
tic;

for feat = 1:size(X_combined, 2)
    if mod(feat, 50) == 0 || feat == 1
        elapsed = toc;
        fprintf('[%s] Feature %3d / %d  (Elapsed: %.1f sec)\n', ...
            datetime('now', 'Format', 'HH:mm:ss'), feat, size(X_combined, 2), elapsed);
    end
    
    var_score = var(X_normalized(:, feat));
    corr_with_labels = abs(corr(X_normalized(:, feat), y_combined_numeric));
    score = var_score * corr_with_labels;
    
    feature_scores = [feature_scores; score];
end

fprintf('[%s] Feature ranking complete!\n\n', datetime('now', 'Format', 'HH:mm:ss'));

[sorted_scores, ranked_features] = sort(feature_scores, 'descend');

fprintf('Top 20 ranked features:\n');
for i = 1:min(20, length(ranked_features))
    feat_idx = ranked_features(i);
    fprintf('  Rank %2d: Feature %3d (Score = %.6f)\n', i, feat_idx, sorted_scores(i));
end

fprintf('\n');

%% ========================================================
%% STEP 3: DBN FEATURE SELECTION (LEAVE-ONE-TRIAL-OUT CV)
%% ========================================================

fprintf('=======================================================\n');
fprintf('STEP 3: DBN FEATURE SELECTION\n');
fprintf('Sequential Forward Selection with Leave-One-Trial-Out CV\n');
fprintf('=======================================================\n\n');

unique_trials = unique(trial_combined);
num_folds = length(unique_trials);

selected_features = [];
remaining_features = ranked_features(1:min(MIN_FEATURES_FOR_INITIAL, length(ranked_features)))';

selection_history = [];
iteration = 0;

fprintf('Starting with top %d features from ranking\n\n', length(remaining_features));

tic;

while length(selected_features) < MAX_FEATURES_TARGET && ~isempty(remaining_features)
    iteration = iteration + 1;
    
    if iteration == 1 || mod(iteration, 5) == 0
        elapsed = toc;
        fprintf('[%s] Iteration %3d: %3d features selected (Elapsed: %.1f sec)\n', ...
            datetime('now', 'Format', 'HH:mm:ss'), iteration, length(selected_features), elapsed);
    end
    
    best_feature = [];
    best_accuracy = -inf;
    
    % Test each remaining feature
    for feat_idx_pos = 1:length(remaining_features)
        feat = remaining_features(feat_idx_pos);
        
        % Create feature subset
        current_features = [selected_features; feat];
        X_subset = X_normalized(:, current_features);
        
        % Leave-one-trial-out cross-validation
        fold_accuracies = [];
        
        for fold = 1:num_folds
            test_trial = unique_trials(fold);
            
            train_mask = trial_combined ~= test_trial;
            test_mask = trial_combined == test_trial;
            
            X_train = X_subset(train_mask, :);
            y_train = y_combined_numeric(train_mask);
            X_test = X_subset(test_mask, :);
            y_test = y_combined_numeric(test_mask);
            
            % Normalize using training statistics
            mu_train = mean(X_train);
            sigma_train = std(X_train);
            sigma_train(sigma_train == 0) = 1;
            
            X_train = (X_train - mu_train) ./ sigma_train;
            X_test = (X_test - mu_train) ./ sigma_train;
            
            try
                % Train LDA for emission probabilities
                lda_model = fitcdiscr(X_train, y_train, 'DiscrimType', 'linear');
                
                % Get emission probabilities
                [~, scores] = predict(lda_model, X_test);
                emission_probs = scores;
                emission_probs(emission_probs < 1e-10) = 1e-10;
                emission_probs = emission_probs ./ sum(emission_probs, 2);
                
                % Build transition matrix from training data
                transition_matrix = ones(num_classes);
                for t = 1:length(y_train)-1
                    from_state = y_train(t);
                    to_state = y_train(t+1);
                    transition_matrix(from_state, to_state) = transition_matrix(from_state, to_state) + 1;
                end
                transition_matrix = transition_matrix ./ sum(transition_matrix, 2);
                
                % Apply HMM filtering
                filtered_probs = zeros(size(emission_probs));
                filtered_probs(1, :) = emission_probs(1, :) ./ sum(emission_probs(1, :));
                
                for t = 2:size(emission_probs, 1)
                    temporal_prior = filtered_probs(t-1, :) * transition_matrix;
                    filtered_probs(t, :) = emission_probs(t, :) .* temporal_prior;
                    filtered_probs(t, :) = filtered_probs(t, :) ./ sum(filtered_probs(t, :));
                end
                
                % Get predictions
                [~, y_pred] = max(filtered_probs, [], 2);
                
                % Compute accuracy
                accuracy = sum(y_pred == y_test) / length(y_test);
                fold_accuracies = [fold_accuracies; accuracy];
                
            catch
                % If error, skip this feature
                continue;
            end
        end
        
        % Average accuracy across folds
        if ~isempty(fold_accuracies)
            avg_accuracy = mean(fold_accuracies);
            
            if avg_accuracy > best_accuracy
                best_accuracy = avg_accuracy;
                best_feature = feat;
            end
        end
    end
    
    if isempty(best_feature)
        fprintf('[%s] Iteration %3d: No improvement found. Stopping.\n', ...
            datetime('now', 'Format', 'HH:mm:ss'), iteration);
        break;
    end
    
    % Add best feature
    selected_features = [selected_features; best_feature];
    remaining_features = remaining_features(remaining_features ~= best_feature);
    
    selection_history = [selection_history; struct(...
        'iteration', iteration, ...
        'feature', best_feature, ...
        'accuracy', best_accuracy)];
    
    if length(selected_features) >= MAX_FEATURES_TARGET
        break;
    end
end

fprintf('[%s] DBN feature selection complete!\n', datetime('now', 'Format', 'HH:mm:ss'));
fprintf('Total features selected: %d\n\n', length(selected_features));

%% ========================================================
%% STEP 4: SAVE RESULTS
%% ========================================================

fprintf('=======================================================\n');
fprintf('STEP 4: SAVING RESULTS\n');
fprintf('=======================================================\n\n');

% Save to text file
filename = fullfile(output_folder, 'DBN_selected_features.txt');
fid = fopen(filename, 'w');

fprintf(fid, '=======================================================\n');
fprintf(fid, 'DBN SELECTED FEATURES (Phase: %d%%)\n', phase_to_test);
fprintf(fid, '=======================================================\n\n');

fprintf(fid, 'Method: Sequential Forward Selection with Leave-One-Trial-Out CV\n');
fprintf(fid, 'Total features available: 256\n');
fprintf(fid, 'Features selected: %d\n\n', length(selected_features));

fprintf(fid, 'Selected feature indices:\n');
fprintf(fid, '[\n');
for j = 1:length(selected_features)
    if mod(j, 10) == 0 || j == length(selected_features)
        fprintf(fid, ' %d\n', selected_features(j));
    else
        fprintf(fid, ' %d,', selected_features(j));
    end
end
fprintf(fid, ']\n\n');

fprintf(fid, 'Feature importance ranking (by selection order):\n');
for j = 1:min(30, length(selected_features))
    feat = selected_features(j);
    fprintf(fid, '  %2d. Feature %3d (Initial rank: %d)\n', j, feat, find(ranked_features == feat));
end

if length(selected_features) > 30
    fprintf(fid, '  ... and %d more features\n', length(selected_features) - 30);
end

fprintf(fid, '\n\nSelection history:\n');
fprintf(fid, 'Iteration  Feature  Accuracy\n');
fprintf(fid, '----------  -------  --------\n');

for j = 1:min(50, length(selection_history))
    h = selection_history(j);
    fprintf(fid, '%10d  %7d  %.6f\n', h.iteration, h.feature, h.accuracy);
end

if length(selection_history) > 50
    fprintf(fid, '... and %d more iterations\n', length(selection_history) - 50);
end

fclose(fid);

fprintf('Saved: %s\n', filename);

% Save as .mat file
mat_filename = fullfile(output_folder, 'DBN_selected_features.mat');
features = selected_features;
save(mat_filename, 'features');
fprintf('Saved: %s\n\n', mat_filename);

%% ========================================================
%% STEP 5: SUMMARY
%% ========================================================

fprintf('=======================================================\n');
fprintf('SUMMARY\n');
fprintf('=======================================================\n\n');

fprintf('Phase: %d%%\n', phase_to_test);
fprintf('Features selected: %d / 256\n', length(selected_features));
fprintf('Iterations: %d\n', iteration);
fprintf('Final accuracy: %.4f\n\n', selection_history(end).accuracy);

fprintf('=======================================================\n');
fprintf('FEATURE SELECTION COMPLETE!\n');
fprintf('=======================================================\n\n');

fprintf('Output files saved in: %s\n', output_folder);
fprintf('\nNext step: Run same process for SVM and Random Forest\n');
fprintf('Then compare feature overlap between models\n');
