%% ========================================================
%% FEATURE SELECTION: SEQUENTIAL FORWARD SELECTION (FIXED)
%% For multiple models: LDA, SVM, Random Forest, DBN
%% ========================================================

clear all; close all;

% Define base folder path
base_folder = 'D:\CAM21\data\Classification\levelground\35\400\';

% List of all subjects
subjects = {'ab07', 'ab08', 'ab09', 'ab12', 'ab13', 'ab14', 'ab17', 'ab18', ...
            'ab19', 'ab20', 'ab21', 'ab23', 'ab24', 'ab27', 'ab28'};

% Create output folder for results
output_folder = 'D:\CAM21\feature_selection\';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Global parameters
MAX_FEATURES_TARGET = 150;
MIN_FEATURES_FOR_INITIAL = 100;  % Start with top 40 from ranking

fprintf('=======================================================\n');
fprintf('FEATURE SELECTION: SEQUENTIAL FORWARD SELECTION (FIXED)\n');
fprintf('=======================================================\n\n');

fprintf('Configuration:\n');
fprintf('  Target features: %d\n', MAX_FEATURES_TARGET);
fprintf('  Models to test: LDA, SVM, Random Forest\n');
fprintf('  Base folder: %s\n', base_folder);
fprintf('  Output folder: %s\n\n', output_folder);

%% STEP 1: LOAD AND COMBINE DATA FROM ALL SUBJECTS
fprintf('=======================================================\n');
fprintf('STEP 1: LOADING DATA FROM ALL SUBJECTS\n');
fprintf('=======================================================\n\n');

X_combined = [];
y_combined = [];
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
        
        input_table = input_data.alldata;
        output_table = output_data.alldata;
        
        X = table2array(input_table);
        labels_col = output_table.labels_feat_last;
        
        % Merge turn1 and turn2
        labels_col_merged = labels_col;
        for i = 1:length(labels_col_merged)
            if strcmp(labels_col_merged{i}, 'turn1') || strcmp(labels_col_merged{i}, 'turn2')
                labels_col_merged{i} = 'turn';
            end
        end
        
        % Combine
        X_combined = [X_combined; X];
        y_combined = [y_combined; labels_col_merged];
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

fprintf('Classes: %d\n\n', length(unique_labels));

%% STEP 2: BETTER INITIAL FEATURE RANKING
fprintf('=======================================================\n');
fprintf('STEP 2: RANKING FEATURES USING VARIANCE & CORRELATION\n');
fprintf('=======================================================\n\n');

fprintf('Computing feature scores using variance and correlation...\n\n');

% Normalize features
X_normalized = (X_combined - mean(X_combined)) ./ std(X_combined);

feature_scores = [];
tic;

for feat = 1:size(X_combined, 2)
    if mod(feat, 50) == 0 || feat == 1
        elapsed = toc;
        fprintf('[%s] Feature %3d / %d  (Elapsed: %.1f sec)\n', ...
            datetime('now', 'Format', 'HH:mm:ss'), feat, size(X_combined, 2), elapsed);
    end
    
    % Calculate variance of this feature
    var_score = var(X_normalized(:, feat));
    
    % Calculate correlation with labels (using absolute correlation)
    corr_with_labels = abs(corr(X_normalized(:, feat), y_combined_numeric));
    
    % Combined score
    score = var_score * corr_with_labels;
    
    feature_scores = [feature_scores; score];
end

fprintf('[%s] Feature ranking complete!\n\n', datetime('now', 'Format', 'HH:mm:ss'));

% Sort features by score
[sorted_scores, ranked_features] = sort(feature_scores, 'descend');

fprintf('Top 20 ranked features:\n');
for i = 1:min(20, length(ranked_features))
    feat_idx = ranked_features(i);
    fprintf('  Rank %2d: Feature %3d (Score = %.6f)\n', i, feat_idx, sorted_scores(i));
end

fprintf('\nBottom 5 ranked features:\n');
for i = 1:5
    feat_idx = ranked_features(end - i + 1);
    fprintf('  Rank %d: Feature %3d (Score = %.6f)\n', length(ranked_features) - i + 1, feat_idx, sorted_scores(end - i + 1));
end

%% STEP 3: SEQUENTIAL FORWARD SELECTION FOR EACH MODEL
fprintf('\n=======================================================\n');
fprintf('STEP 3: SEQUENTIAL FORWARD SELECTION\n');
fprintf('=======================================================\n\n');

% Models to test
model_names = {'LDA', 'SVM', 'RandomForest'};

% Store results
selected_features_all_models = {};

for model_idx = 1:length(model_names)
    model_name = model_names{model_idx};
    
    fprintf('\n');
    fprintf('=======================================================\n');
    fprintf('MODEL: %s\n', model_name);
    fprintf('=======================================================\n\n');
    
    % Start with top features from ranking
    top_features_idx = ranked_features(1:min(MIN_FEATURES_FOR_INITIAL, length(ranked_features)))';
    
    selected_features = [];
    remaining_features = top_features_idx;
    
    % Get baseline error with all top features
    X_all_top = X_normalized(:, top_features_idx);
    
    try
        if strcmp(model_name, 'LDA')
            model = fitcdiscr(X_all_top, y_combined_numeric);
        elseif strcmp(model_name, 'SVM')
            model = fitcecoc(X_all_top, y_combined_numeric, 'Learner', 'svm');
        elseif strcmp(model_name, 'RandomForest')
            model = TreeBagger(50, X_all_top, y_combined_numeric, 'Method', 'classification');
        end
        
        y_pred_baseline = predict(model, X_all_top);
        if istable(y_pred_baseline)
            y_pred_baseline = str2double(y_pred_baseline{:,1});
        elseif iscell(y_pred_baseline)
            y_pred_baseline = str2double(y_pred_baseline);
        end
        
        baseline_error = sum(y_pred_baseline ~= y_combined_numeric) / length(y_combined_numeric);
        baseline_accuracy = 1 - baseline_error;
        
        fprintf('Baseline (all top %d features): %.4f accuracy\n\n', length(top_features_idx), baseline_accuracy);
        
    catch ME
        fprintf('Error with baseline: %s\n\n', ME.message);
        baseline_accuracy = 0;
    end
    
    % Sequential forward selection
    iteration = 0;
    selection_history = [];
    tic;
    
    while length(selected_features) < MAX_FEATURES_TARGET && ~isempty(remaining_features)
        iteration = iteration + 1;
        
        if iteration == 1 || mod(iteration, 10) == 0
            elapsed = toc;
            fprintf('[%s] Iteration %3d: %3d features selected (Elapsed: %.1f sec)\n', ...
                datetime('now', 'Format', 'HH:mm:ss'), iteration, length(selected_features), elapsed);
        end
        
        best_feature = [];
        best_accuracy = -inf;
        
        % Test each remaining feature
        for feat_idx_pos = 1:length(remaining_features)
            feat = remaining_features(feat_idx_pos);
            
            % Create feature subset: selected + candidate
            current_features = [selected_features; feat];
            X_subset = X_normalized(:, current_features);
            
            try
                % Train model
                if strcmp(model_name, 'LDA')
                    model = fitcdiscr(X_subset, y_combined_numeric);
                    y_pred = predict(model, X_subset);
                elseif strcmp(model_name, 'SVM')
                    model = fitcecoc(X_subset, y_combined_numeric, 'Learner', 'svm');
                    y_pred = predict(model, X_subset);
                elseif strcmp(model_name, 'RandomForest')
                    model = TreeBagger(50, X_subset, y_combined_numeric, 'Method', 'classification');
                    y_pred = predict(model, X_subset);
                    y_pred = str2double(y_pred);
                end
                
                % Handle cell array output
                if iscell(y_pred)
                    y_pred = str2double(y_pred);
                end
                
                accuracy = sum(y_pred == y_combined_numeric) / length(y_combined_numeric);
                
                if accuracy > best_accuracy
                    best_accuracy = accuracy;
                    best_feature = feat;
                end
            catch
                % Skip if error
                continue;
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
        
        % Stop if target reached
        if length(selected_features) >= MAX_FEATURES_TARGET
            break;
        end
    end
    
    fprintf('[%s] %s feature selection complete!\n', ...
        datetime('now', 'Format', 'HH:mm:ss'), model_name);
    fprintf('    Total features selected: %d\n\n', length(selected_features));
    
    % Store results
    selected_features_all_models{model_idx} = struct(...
        'model', model_name, ...
        'features', selected_features, ...
        'num_features', length(selected_features), ...
        'history', selection_history);
end

%% STEP 4: SAVE RESULTS
fprintf('\n=======================================================\n');
fprintf('STEP 4: SAVING RESULTS\n');
fprintf('=======================================================\n\n');

for i = 1:length(selected_features_all_models)
    result = selected_features_all_models{i};
    model_name = result.model;
    features = result.features;
    num_features = result.num_features;
    history = result.history;
    
    % Save to text file
    filename = fullfile(output_folder, [model_name '_selected_features.txt']);
    fid = fopen(filename, 'w');
    
    fprintf(fid, '=======================================================\n');
    fprintf(fid, 'SELECTED FEATURES FOR MODEL: %s\n', model_name);
    fprintf(fid, '=======================================================\n\n');
    
    fprintf(fid, 'Method: Sequential Forward Selection\n');
    fprintf(fid, 'Total features available: 256\n');
    fprintf(fid, 'Features selected: %d\n\n', num_features);
    
    fprintf(fid, 'Selected feature indices:\n');
    fprintf(fid, '[\n');
    for j = 1:length(features)
        if mod(j, 10) == 0 || j == length(features)
            fprintf(fid, ' %d\n', features(j));
        else
            fprintf(fid, ' %d,', features(j));
        end
    end
    fprintf(fid, ']\n\n');
    
    fprintf(fid, 'Feature importance ranking (by selection order):\n');
    for j = 1:min(30, length(features))
        feat = features(j);
        fprintf(fid, '  %2d. Feature %3d (Rank in initial: %d)\n', j, feat, find(ranked_features == feat));
    end
    
    if length(features) > 30
        fprintf(fid, '  ... and %d more features\n', length(features) - 30);
    end
    
    fprintf(fid, '\n\nSelection history:\n');
    fprintf(fid, 'Iteration  Feature  Accuracy\n');
    fprintf(fid, '----------  -------  --------\n');
    
    for j = 1:min(50, length(history))
        h = history(j);
        fprintf(fid, '%10d  %7d  %.6f\n', h.iteration, h.feature, h.accuracy);
    end
    
    if length(history) > 50
        fprintf(fid, '... and %d more iterations\n', length(history) - 50);
    end
    
    fclose(fid);
    
    fprintf('Saved: %s\n', filename);
    
    % Save as .mat file
    mat_filename = fullfile(output_folder, [model_name '_selected_features.mat']);
    save(mat_filename, 'features');
    fprintf('Saved: %s\n', mat_filename);
end

% Create summary
summary_filename = fullfile(output_folder, 'FEATURE_SELECTION_SUMMARY.txt');
fid = fopen(summary_filename, 'w');

fprintf(fid, '=======================================================\n');
fprintf(fid, 'FEATURE SELECTION SUMMARY\n');
fprintf(fid, '=======================================================\n\n');

fprintf(fid, 'Method: Sequential Forward Selection with Variance-Correlation Ranking\n');
fprintf(fid, 'Target features: %d\n', MAX_FEATURES_TARGET);
fprintf(fid, 'Total available features: 256\n\n');

fprintf(fid, 'RESULTS:\n');
for i = 1:length(selected_features_all_models)
    result = selected_features_all_models{i};
    fprintf(fid, '%s: %d features selected\n', result.model, result.num_features);
end

fclose(fid);

fprintf('Saved: %s\n\n', summary_filename);

fprintf('=======================================================\n');
fprintf('FEATURE SELECTION COMPLETE!\n');
fprintf('=======================================================\n\n');

fprintf('Output files created in: %s\n', output_folder);