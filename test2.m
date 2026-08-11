%{
input_file = 'D:\CAM21\data\Classification\levelground\35\400\ab07_input.mat';
output_file = 'D:\CAM21\data\Classification\levelground\35\400\ab07_output.mat';
input_data = load(input_file);
output_data = load(output_file);

output_table = output_data.alldata;
trial_col = output_table.trial_feat_last;

% Look at the FIRST 50 values
fprintf('First 50 trial values:\n');
disp(trial_col(1:50)');

% Find where trial actually CHANGES
fprintf('\n=== WHERE TRIAL COLUMN CHANGES ===\n');
trial_diff = diff(trial_col);
change_points = find(trial_diff ~= 0);

fprintf('Trial changes at rows: %s\n', mat2str(change_points'));
fprintf('Number of segments: %d\n', length(change_points) + 1);

% Show trial boundaries
fprintf('\n=== ACTUAL TRIAL SEGMENTS ===\n');
segment_starts = [1; change_points + 1];
segment_ends = [change_points; length(trial_col)];

for i = 1:length(segment_starts)
    start = segment_starts(i);
    finish = segment_ends(i);
    trial_val = trial_col(start);
    fprintf('Segment %d: rows %d to %d (size=%d), trial_value=%d\n', ...
        i, start, finish, finish-start+1, trial_val);
end
%}
clear all

input_file = 'D:\CAM21\data\Classification\levelground\35\400\ab07_input.mat';
output_file = 'D:\CAM21\data\Classification\levelground\35\400\ab07_output.mat';

input_data = load(input_file);
output_data = load(output_file);

input_table = input_data.alldata;
output_table = output_data.alldata;

X = table2array(input_table);
trial_col = output_table.trial_feat_last;
labels_col = output_table.labels_feat_last;

%% MERGE TURN1 AND TURN2
fprintf('Before: %d unique labels\n', length(unique(labels_col)));

labels_col_merged = labels_col;
for i = 1:length(labels_col_merged)
    if strcmp(labels_col_merged{i}, 'turn1') || strcmp(labels_col_merged{i}, 'turn2')
        labels_col_merged{i} = 'turn';
    end
end

fprintf('After: %d unique labels\n', length(unique(labels_col_merged)));
labels_col = labels_col_merged;

%% STEP 1: Get ALL rows that belong to each trial (not just first occurrence)

fprintf('=== COLLECTING ROWS FOR EACH TRIAL ===\n');

% For each trial number, find ALL rows that have that trial value
for trial_num = 1:6
    trial_idx = find(trial_col == trial_num);
    fprintf('Trial %d: %d rows total\n', trial_num, length(trial_idx));
end

%% STEP 2: Train on trials 1-4, test on trial 5

train_idx = [];
for trial_num = 1:4
    trial_idx = find(trial_col == trial_num);
    train_idx = [train_idx; trial_idx];
end

test_idx = find(trial_col == 5);

% Sort indices to maintain order
train_idx = sort(train_idx);
test_idx = sort(test_idx);

X_train = X(train_idx, :);
y_train = labels_col(train_idx);

X_test = X(test_idx, :);
y_test = labels_col(test_idx);

fprintf('\n=== TRAINING/TEST SPLIT ===\n');
fprintf('Training set (Trials 1-4): %d samples\n', size(X_train, 1));
fprintf('Test set (Trial 5): %d samples\n', size(X_test, 1));
fprintf('Features: %d\n', size(X_train, 2));

%% STEP 3: Convert labels to numeric

unique_labels = unique(labels_col);
label_map = containers.Map(unique_labels, 1:length(unique_labels));

y_train_numeric = zeros(length(y_train), 1);
for i = 1:length(y_train)
    y_train_numeric(i) = label_map(y_train{i});
end

y_test_numeric = zeros(length(y_test), 1);
for i = 1:length(y_test)
    y_test_numeric(i) = label_map(y_test{i});
end

%% STEP 4: Train and test LDA

fprintf('\n=== TRAINING LDA MODEL ===\n');
model = fitcdiscr(X_train, y_train_numeric);
fprintf('Model trained!\n');

fprintf('\n=== TESTING MODEL ===\n');
y_pred = predict(model, X_test);

accuracy = sum(y_pred == y_test_numeric) / length(y_test_numeric);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

fprintf('\n=== PER-CLASS ACCURACY ===\n');
for class = 1:length(unique_labels)
    class_mask = y_test_numeric == class;
    if sum(class_mask) > 0
        class_acc = sum(y_pred(class_mask) == class) / sum(class_mask);
        fprintf('%s: %.2f%% (%d/%d)\n', unique_labels{class}, class_acc*100, ...
            sum(y_pred(class_mask) == class), sum(class_mask));
    end
end