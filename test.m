%% ========================================================
%% BATCH PROCESSING: ALL PARTICIPANTS
%% Train LDA on each subject separately, save results
% List of all subjects
%subjects = {'ab07', 'ab08', 'ab09', 'ab12', 'ab13', 'ab14', 'ab17', 'ab18', 'ab19', 'ab20', ...
 %   'ab21', 'ab23', 'ab24', 'ab27', 'ab28'};
%% ========================================================

%% ========================================================
%% BATCH PROCESSING: ALL PARTICIPANTS (FIXED VERSION)
%% Train LDA on each subject separately, save results
%% ========================================================

clear all; close all;

% Define base folder path
base_folder = 'D:\CAM21\data\Classification\levelground\35\400\';

% List of all subjects
subjects = {'ab07', 'ab08', 'ab09', 'ab12', 'ab13', 'ab14', 'ab17', 'ab18', 'ab19', 'ab20', ...
    'ab21', 'ab23', 'ab24', 'ab27', 'ab28'};

% Create output folder for results
output_folder = 'D:\CAM21\results\';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

fprintf('=======================================================\n');
fprintf('BATCH PROCESSING ALL SUBJECTS\n');
fprintf('=======================================================\n\n');

% Store results for all subjects
all_results = {};
overall_accuracies = [];

%% LOOP THROUGH EACH SUBJECT
for subj_idx = 1:length(subjects)
    subject = subjects{subj_idx};
    
    fprintf('Processing subject: %s\n', subject);
    fprintf('-------------------------------------------------------\n');
    
    %% Load data for this subject
    input_file = fullfile(base_folder, [subject '_input.mat']);
    output_file = fullfile(base_folder, [subject '_output.mat']);
    
    % Check if files exist
    if ~isfile(input_file) || ~isfile(output_file)
        fprintf('WARNING: Files not found for %s. Skipping...\n\n', subject);
        continue;
    end
    
    try
        input_data = load(input_file);
        output_data = load(output_file);
        
        input_table = input_data.alldata;
        output_table = output_data.alldata;
        
        X = table2array(input_table);
        trial_col = output_table.trial_feat_last;
        labels_col = output_table.labels_feat_last;
        
        %% MERGE TURN1 AND TURN2 INTO TURN
        labels_col_merged = labels_col;
        for i = 1:length(labels_col_merged)
            label = labels_col_merged{i};
            if strcmp(label, 'turn1') || strcmp(label, 'turn2')
                labels_col_merged{i} = 'turn';
            end
        end
        labels_col = labels_col_merged;
        
        %% GET ALL ROWS FOR EACH TRIAL
        fprintf('Finding trials for %s...\n', subject);
        num_trials = max(trial_col);
        trial_info = {};
        
        for trial_num = 1:num_trials
            trial_idx = find(trial_col == trial_num);
            trial_info{trial_num} = trial_idx;
        end
        
        fprintf('Found %d trials\n', num_trials);
        
        %% LEAVE-ONE-TRIAL-OUT CROSS-VALIDATION
        fprintf('Performing leave-one-trial-out CV for %s...\n\n', subject);
        
        fold_accuracies = [];
        fold_results = {};
        unique_labels = unique(labels_col);
        
        for test_fold = 1:num_trials
            % Create train and test indices
            train_idx = [];
            for train_trial = 1:num_trials
                if train_trial ~= test_fold
                    train_idx = [train_idx; trial_info{train_trial}];
                end
            end
            test_idx = trial_info{test_fold};
            
            train_idx = sort(train_idx);
            test_idx = sort(test_idx);
            
            % Extract features and labels
            X_train = X(train_idx, :);
            y_train = labels_col(train_idx);
            
            X_test = X(test_idx, :);
            y_test = labels_col(test_idx);
            
            % Convert labels to numeric
            label_map = containers.Map(unique_labels, 1:length(unique_labels));
            
            y_train_numeric = zeros(length(y_train), 1);
            for i = 1:length(y_train)
                y_train_numeric(i) = label_map(y_train{i});
            end
            
            y_test_numeric = zeros(length(y_test), 1);
            for i = 1:length(y_test)
                y_test_numeric(i) = label_map(y_test{i});
            end
            
            % Train LDA
            model = fitcdiscr(X_train, y_train_numeric);
            
            % Test
            y_pred = predict(model, X_test);
            
            % Calculate accuracy
            accuracy = sum(y_pred == y_test_numeric) / length(y_test_numeric);
            fold_accuracies = [fold_accuracies; accuracy];
            
            % Store per-class results
            per_class = {};
            for class = 1:length(unique_labels)
                class_mask = y_test_numeric == class;
                if sum(class_mask) > 0
                    class_acc = sum(y_pred(class_mask) == class) / sum(class_mask);
                    num_correct = sum(y_pred(class_mask) == class);
                    num_samples = sum(class_mask);
                    
                    % Convert label to string if needed
                    label_name = unique_labels{class};
                    if iscell(label_name)
                        label_name = label_name{1};
                    end
                    
                    per_class{class} = struct('name', label_name, ...
                                              'accuracy', class_acc, ...
                                              'correct', num_correct, ...
                                              'total', num_samples);
                end
            end
            
            fold_results{test_fold} = struct('fold', test_fold, ...
                                             'accuracy', accuracy, ...
                                             'per_class', per_class);
        end
        
        %% CALCULATE OVERALL STATISTICS
        mean_accuracy = mean(fold_accuracies);
        std_accuracy = std(fold_accuracies);
        
        fprintf('Done! Mean CV Accuracy: %.2f%% (±%.2f%%)\n', ...
            mean_accuracy*100, std_accuracy*100);
        
        %% SAVE RESULTS TO TEXT FILE
        output_file_txt = fullfile(output_folder, [subject '_results.txt']);
        
        fid = fopen(output_file_txt, 'w');
        
        fprintf(fid, '=======================================================\n');
        fprintf(fid, 'LDA CLASSIFICATION RESULTS FOR SUBJECT: %s\n', subject);
        fprintf(fid, '=======================================================\n\n');
        
        fprintf(fid, 'EXPERIMENTAL SETUP:\n');
        fprintf(fid, '  Data folder: %s\n', base_folder);
        fprintf(fid, '  Gait phase: 35%%\n');
        fprintf(fid, '  Window size: 400ms\n');
        fprintf(fid, '  Model: LDA (Linear Discriminant Analysis)\n');
        fprintf(fid, '  Cross-validation: Leave-One-Trial-Out\n');
        fprintf(fid, '  Total trials: %d\n', num_trials);
        fprintf(fid, '  Total samples: %d\n', size(X, 1));
        fprintf(fid, '  Total features: %d\n', size(X, 2));
        fprintf(fid, '  Classes: %d (merged turn1 and turn2)\n\n', length(unique_labels));
        
        fprintf(fid, '=======================================================\n');
        fprintf(fid, 'OVERALL PERFORMANCE (ALL FOLDS)\n');
        fprintf(fid, '=======================================================\n\n');
        
        fprintf(fid, 'Mean Accuracy:     %.2f%%\n', mean_accuracy*100);
        fprintf(fid, 'Std Dev:           %.2f%%\n', std_accuracy*100);
        fprintf(fid, 'Min Accuracy:      %.2f%% (Fold %d)\n', min(fold_accuracies)*100, find(fold_accuracies == min(fold_accuracies), 1));
        fprintf(fid, 'Max Accuracy:      %.2f%% (Fold %d)\n\n', max(fold_accuracies)*100, find(fold_accuracies == max(fold_accuracies), 1));
        
        fprintf(fid, '=======================================================\n');
        fprintf(fid, 'DETAILED RESULTS BY FOLD\n');
        fprintf(fid, '=======================================================\n\n');
        
        for fold = 1:length(fold_results)
            result = fold_results{fold};
            
            fprintf(fid, '--- FOLD %d (Test on Trial %d) ---\n', fold, fold);
            fprintf(fid, 'Overall Accuracy: %.2f%%\n\n', result.accuracy*100);
            
            fprintf(fid, 'Per-Class Accuracy:\n');
            for class_idx = 1:length(result.per_class)
                if ~isempty(result.per_class{class_idx})
                    pc = result.per_class{class_idx};
                    % Make sure name is a string
                    name_str = pc.name;
                    if iscell(name_str)
                        name_str = name_str{1};
                    end
                    fprintf(fid, '  %s: %.2f%% (%d/%d)\n', ...
                        name_str, pc.accuracy*100, pc.correct, pc.total);
                end
            end
            fprintf(fid, '\n');
        end
        
        fprintf(fid, '=======================================================\n');
        fprintf(fid, 'AVERAGE CLASS ACCURACY (ACROSS ALL FOLDS)\n');
        fprintf(fid, '=======================================================\n\n');
        
        % Calculate average per-class accuracy across all folds
        all_class_accuracies = containers.Map();
        all_class_counts = containers.Map();
        
        for fold = 1:length(fold_results)
            result = fold_results{fold};
            for class_idx = 1:length(result.per_class)
                if ~isempty(result.per_class{class_idx})
                    pc = result.per_class{class_idx};
                    class_name = pc.name;
                    if iscell(class_name)
                        class_name = class_name{1};
                    end
                    
                    if ~isKey(all_class_accuracies, class_name)
                        all_class_accuracies(class_name) = 0;
                        all_class_counts(class_name) = 0;
                    end
                    
                    all_class_accuracies(class_name) = all_class_accuracies(class_name) + pc.accuracy;
                    all_class_counts(class_name) = all_class_counts(class_name) + 1;
                end
            end
        end
        
        class_names = keys(all_class_accuracies);
        for class_idx = 1:length(class_names)
            class_name = class_names{class_idx};
            avg_acc = all_class_accuracies(class_name) / all_class_counts(class_name);
            fprintf(fid, '%s: %.2f%%\n', class_name, avg_acc*100);
        end
        
        fprintf(fid, '\n=======================================================\n');
        fprintf(fid, 'END OF RESULTS FOR %s\n', subject);
        fprintf(fid, '=======================================================\n');
        
        fclose(fid);
        
        % Store for summary
        all_results{subj_idx} = struct('subject', subject, ...
                                        'mean_acc', mean_accuracy, ...
                                        'std_acc', std_accuracy, ...
                                        'num_trials', num_trials);
        overall_accuracies = [overall_accuracies; mean_accuracy];
        
        fprintf('Results saved to: %s\n\n', output_file_txt);
        
    catch ME
        fprintf('ERROR processing %s: %s\n\n', subject, ME.message);
        continue;
    end
end

%% CREATE SUMMARY FILE FOR ALL SUBJECTS
fprintf('=======================================================\n');
fprintf('CREATING SUMMARY FILE\n');
fprintf('=======================================================\n\n');

summary_file = fullfile(output_folder, 'ALL_SUBJECTS_SUMMARY.txt');
fid = fopen(summary_file, 'w');

fprintf(fid, '=======================================================\n');
fprintf(fid, 'SUMMARY: LDA CLASSIFICATION FOR ALL SUBJECTS\n');
fprintf(fid, '=======================================================\n\n');

fprintf(fid, 'Experimental Setup:\n');
fprintf(fid, '  Data folder: %s\n', base_folder);
fprintf(fid, '  Gait phase: 35%%\n');
fprintf(fid, '  Window size: 400ms\n');
fprintf(fid, '  Model: LDA\n');
fprintf(fid, '  Cross-validation: Leave-One-Trial-Out\n\n');

fprintf(fid, '=======================================================\n');
fprintf(fid, 'RESULTS BY SUBJECT\n');
fprintf(fid, '=======================================================\n\n');

fprintf(fid, 'Subject    Mean Accuracy    Std Dev    Trials\n');
fprintf(fid, '---        -----------      -------    ------\n');

for subj_idx = 1:length(all_results)
    result = all_results{subj_idx};
    fprintf(fid, '%-10s %6.2f%%         %6.2f%%      %d\n', ...
        result.subject, result.mean_acc*100, result.std_acc*100, result.num_trials);
end

fprintf(fid, '\n=======================================================\n');
fprintf(fid, 'OVERALL STATISTICS\n');
fprintf(fid, '=======================================================\n\n');

if ~isempty(overall_accuracies)
    fprintf(fid, 'Mean Accuracy (All Subjects): %.2f%%\n', mean(overall_accuracies)*100);
    fprintf(fid, 'Std Dev (All Subjects):      %.2f%%\n', std(overall_accuracies)*100);
    fprintf(fid, 'Min Accuracy:                %.2f%%\n', min(overall_accuracies)*100);
    fprintf(fid, 'Max Accuracy:                %.2f%%\n', max(overall_accuracies)*100);
    fprintf(fid, 'Number of Subjects:          %d\n', length(all_results));
    fprintf(fid, '\nDetailed Results by Subject:\n');
    fprintf(fid, '\n');
    
    for subj_idx = 1:length(all_results)
        result = all_results{subj_idx};
        fprintf(fid, '%s: Mean Accuracy = %.2f%% (±%.2f%%), Trials = %d\n', ...
            result.subject, result.mean_acc*100, result.std_acc*100, result.num_trials);
    end
end

fprintf(fid, '\n=======================================================\n');
fprintf(fid, 'Individual result files:\n');
fprintf(fid, '=======================================================\n\n');

for subj_idx = 1:length(all_results)
    result = all_results{subj_idx};
    fprintf(fid, '%s_results.txt\n', result.subject);
end

fclose(fid);

fprintf('Summary file saved to: %s\n\n', summary_file);

%% FINAL DISPLAY
fprintf('=======================================================\n');
fprintf('BATCH PROCESSING COMPLETE!\n');
fprintf('=======================================================\n\n');

fprintf('Results Summary:\n');
for subj_idx = 1:length(all_results)
    result = all_results{subj_idx};
    fprintf('%s: Mean Accuracy = %.2f%% (±%.2f%%)\n', ...
        result.subject, result.mean_acc*100, result.std_acc*100);
end

fprintf('\nAll results saved in: %s\n\n', output_folder);
fprintf('Files created:\n');
fprintf('  - Individual files: [subject]_results.txt\n');
fprintf('  - Summary file: ALL_SUBJECTS_SUMMARY.txt\n');