%{
data = load('D:\CAM21\data\Classification\levelground\35\400\ab07_input.mat')
data = load('D:\CAM21\data\Classification\levelground\full_ab08_input_4.mat')
fieldnames(data)
disp(data)
data.alldata.Properties.VariableNames
data.alldata
%h = data.alldata.Header;
%plot(h)

h = data.alldata.Header;

reset_idx = find(diff(h) < 0);

trial_starts = [1; reset_idx+1];
trial_ends   = [reset_idx; length(h)];

trial_starts
trial_ends

for i = 1:length(trial_starts)
    fprintf(' %d: Start = %d, End = %d\n', i, trial_starts(i), trial_ends(i));
end


disp("##################")
clear all
full_output_file = 'D:\CAM21\data\Classification\levelground\full_ab07_output_4.mat';
full_output = load(full_output_file);


% Get the output table
output_table = full_output.alldata;

% Show all columns
fprintf('Output table columns:\n');
disp(output_table.Properties.VariableNames);

fprintf('\n=== DATA TYPES ===\n');
disp(output_table.Properties.VariableTypes);

fprintf('\n=== FIRST 20 ROWS ===\n');
disp(output_table(1:20, :));

fprintf('\n=== CHECKING EACH COLUMN ===\n');
col_names = output_table.Properties.VariableNames;
for col = 1:length(col_names)
    col_name = col_names{col};
    col_data = output_table.(col_name);

    fprintf('\nColumn %d: %s\n', col, col_name);
    fprintf('  Type: %s\n', class(col_data));

    if isnumeric(col_data)
        fprintf('  Unique values: %s\n', mat2str(unique(col_data)'));
    elseif iscell(col_data)
        fprintf('  Cell array with %d elements\n', length(col_data));
        % Show first 5 cells
        for i = 1:min(5, length(col_data))
            fprintf('  Cell %d: %s\n', i, mat2str(col_data{i}));
        end
    end
end

fprintf('\n=== LOOKING FOR TRIAL IDENTIFIER ===\n');
% 'trial_feat_last' likely contains trial info
trial_col = output_table.trial_feat_last;
fprintf('Trial column first 30 values:\n');
disp(trial_col(1:30));

% Check if numeric or cell
if isnumeric(trial_col)
    unique_trials = unique(trial_col);
    fprintf('\nUnique trial values: %s\n', mat2str(unique_trials'));

    % Find trial boundaries
    trial_diff = diff(trial_col);
    change_points = find(trial_diff ~= 0);
    fprintf('\nTrial changes at indices: %s\n', mat2str(change_points'));

    fprintf('\nNumber of trials: %d\n', length(change_points) + 1);

    % Show trial sizes
    fprintf('\nTrial boundaries:\n');
    trial_starts = [1; change_points + 1];
    trial_ends = [change_points; length(trial_col)];
    for i = 1:length(trial_starts)
        fprintf('Trial %d: samples %d to %d (size = %d)\n', ...
            i, trial_starts(i), trial_ends(i), trial_ends(i) - trial_starts(i) + 1);
    end
elseif iscell(trial_col)
    fprintf('Trial column is cell array\n');
    fprintf('First 10 values:\n');
    for i = 1:min(10, length(trial_col))
        fprintf('  %d: %s\n', i, mat2str(trial_col{i}));
    end
end
%}
clear all

% Load data
input_file = 'D:\CAM21\data\Classification\levelground\35\400\ab07_input.mat';
output_file = 'D:\CAM21\data\Classification\levelground\35\400\ab07_output.mat';

input_data = load(input_file);
output_data = load(output_file);

% Get tables
input_table = input_data.alldata;      % [1636, 256]
output_table = output_data.alldata;    % [1636, 5]

% Convert input to array (all numeric)
X = table2array(input_table);          % Features [1636, 256]

% For output, extract columns separately (one is cell array)
trial_col = output_table.trial_feat_last;        % Column 3 - trial numbers
labels_col = output_table.labels_feat_last;      % Column 4 - activity labels (CELL!)

fprintf('Unique trials: %s\n', mat2str(unique(trial_col)'));

%% STEP 1: Identify which rows belong to which trial

% Find indices for each trial
trial_1_idx = find(trial_col == 1);
trial_2_idx = find(trial_col == 2);
trial_3_idx = find(trial_col == 3);
trial_4_idx = find(trial_col == 4);
trial_5_idx = find(trial_col == 5);

fprintf('\n=== TRIAL BREAKDOWN ===\n');
fprintf('Trial 1: rows %d to %d (%d samples)\n', trial_1_idx(1), trial_1_idx(end), length(trial_1_idx));
fprintf('Trial 2: rows %d to %d (%d samples)\n', trial_2_idx(1), trial_2_idx(end), length(trial_2_idx));
fprintf('Trial 3: rows %d to %d (%d samples)\n', trial_3_idx(1), trial_3_idx(end), length(trial_3_idx));
fprintf('Trial 4: rows %d to %d (%d samples)\n', trial_4_idx(1), trial_4_idx(end), length(trial_4_idx));
fprintf('Trial 5: rows %d to %d (%d samples)\n', trial_5_idx(1), trial_5_idx(end), length(trial_5_idx));

%% STEP 2: Create training and test sets

% Training set: combine trials 1, 2, 3, 4
train_idx = [trial_1_idx; trial_2_idx; trial_3_idx; trial_4_idx];
X_train = X(train_idx, :);             % Features for training
y_train = labels_col(train_idx);       % Labels for training (CELL array)

% Test set: trial 5 only
test_idx = trial_5_idx;
X_test = X(test_idx, :);               % Features for testing
y_test = labels_col(test_idx);         % Labels for testing (CELL array)

fprintf('\n=== TRAINING/TEST SPLIT ===\n');
fprintf('Training set: %d samples\n', size(X_train, 1));
fprintf('Test set: %d samples\n', size(X_test, 1));
fprintf('Features: %d\n', size(X_train, 2));

%% STEP 3: Convert cell labels to numeric (required for LDA)

fprintf('\n=== CONVERTING LABELS ===\n');

% Get unique labels
unique_labels = unique(labels_col);
fprintf('Unique activity labels: ');
for i = 1:length(unique_labels)
    fprintf('%s ', unique_labels{i});
end
fprintf('\n');

% Create mapping: cell label → numeric
label_map = containers.Map(unique_labels, 1:length(unique_labels));

% Convert training labels
y_train_numeric = zeros(length(y_train), 1);
for i = 1:length(y_train)
    y_train_numeric(i) = label_map(y_train{i});  % Note: y_train{i} because it's cell
end

% Convert test labels
y_test_numeric = zeros(length(y_test), 1);
for i = 1:length(y_test)
    y_test_numeric(i) = label_map(y_test{i});    % Note: y_test{i} because it's cell
end

fprintf('Labels converted to numeric format\n');

%% STEP 4: Train LDA model

fprintf('\n=== TRAINING LDA MODEL ===\n');

model = fitcdiscr(X_train, y_train_numeric);

fprintf('Model trained successfully!\n');

%% STEP 5: Test the model

fprintf('\n=== TESTING MODEL ===\n');

% Make predictions
y_pred = predict(model, X_test);

% Calculate accuracy
accuracy = sum(y_pred == y_test_numeric) / length(y_test_numeric);

fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Correct predictions: %d / %d\n', sum(y_pred == y_test_numeric), length(y_test_numeric));

% Show per-class accuracy
fprintf('\n=== PER-CLASS ACCURACY ===\n');
for class = 1:length(unique_labels)
    class_mask = y_test_numeric == class;
    if sum(class_mask) > 0
        class_acc = sum(y_pred(class_mask) == class) / sum(class_mask);
        fprintf('%s: %.2f%% (%d/%d)\n', unique_labels{class}, class_acc*100, ...
            sum(y_pred(class_mask) == class), sum(class_mask));
    end
end


%Data sorte by gatphase grouped by trail

clear all

% Load data
input_file = 'D:\CAM21\data\Classification\levelground\35\400\ab07_input.mat';
output_file = 'D:\CAM21\data\Classification\levelground\35\400\ab07_output.mat';

input_data = load(input_file);
output_data = load(output_file);

output_table = output_data.alldata;

fprintf('=== ORIGINAL DATA ===\n');
fprintf('Total rows: %d\n', height(output_table));
fprintf('Total columns: %d\n\n', width(output_table));
disp(output_table);

%% Sort by trial first, then by gait phase

sorted_output = sortrows(output_table, {'trial_feat_last', 'gait_feat_last'});

fprintf('\n\n=== SORTED DATA (By Trial, then Gait Phase) ===\n');
fprintf('Total rows: %d\n', height(sorted_output));
fprintf('Total columns: %d\n\n', width(sorted_output));
disp(sorted_output);

%% Show structure by trial

fprintf('\n\n=== DATA ORGANIZED BY TRIAL (DETAILED VIEW) ===\n');

unique_trials = unique(sorted_output.trial_feat_last);

for trial_num = unique_trials'
    trial_mask = sorted_output.trial_feat_last == trial_num;
    trial_data = sorted_output(trial_mask, :);

    fprintf('\n========================================\n');
    fprintf('TRIAL %d (%d samples)\n', trial_num, height(trial_data));
    fprintf('========================================\n\n');
    disp(trial_data);
end
