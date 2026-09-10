% Feature importance comparison across LDA, SVM, RF and DBN
% Compares the top 80 features selected by each model to see
% how much overlap there is between model types
% Run this separately for each terrain by changing ground_to_test

clear all; close all; clc;

% change this to run on different terrains
ground_to_test = 'ramp';
top_n = 80;   % how many top features to compare

base_folder   = sprintf('/home/eeiww/ut55iqoh/MLMA_CAM21/Classification/%s/', ground_to_test);
output_folder = sprintf('/home/eeiww/ut55iqoh/MLMA_CAM21/result_feature_selection/%s/', ground_to_test);

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

subjects = {'ab07', 'ab08', 'ab09', 'ab12', 'ab13', 'ab14', 'ab17', 'ab18', ...
            'ab19', 'ab20', 'ab21', 'ab23', 'ab24', 'ab27', 'ab28'};

fprintf('running feature comparison for %s (top %d features)\n\n', ground_to_test, top_n);

% load all subject data
fprintf('loading data...\n');

X_combined     = [];
y_combined     = {};
trial_combined = [];

for subj_idx = 1:length(subjects)
    subject     = subjects{subj_idx};
    input_file  = fullfile(base_folder, ['full_' subject '_input_4.mat']);
    output_file = fullfile(base_folder, ['full_' subject '_output_4.mat']);

    if ~isfile(input_file) || ~isfile(output_file)
        fprintf('skipping %s - file not found\n', subject);
        continue;
    end

    try
        input_data  = load(input_file);
        output_data = load(output_file);

        X          = table2array(input_data.alldata);
        out_table  = output_data.alldata;
        labels_col = out_table.labels_feat_last;
        trial_col  = out_table.trial_feat_last;

        % merge turn1 and turn2 into a single turn class
        for i = 1:length(labels_col)
            if strcmp(labels_col{i},'turn1') || strcmp(labels_col{i},'turn2')
                labels_col{i} = 'turn';
            end
        end

        X_combined     = [X_combined;     X];
        y_combined     = [y_combined;     labels_col];
        trial_combined = [trial_combined; trial_col];

        fprintf('  loaded %s (%d samples)\n', subject, size(X,1));
    catch ME
        fprintf('  error loading %s: %s\n', subject, ME.message);
    end
end

% convert labels to numbers
unique_labels = unique(y_combined);
num_classes   = length(unique_labels);
label_map     = containers.Map(unique_labels, 1:num_classes);

y_numeric = zeros(length(y_combined), 1);
for i = 1:length(y_combined)
    y_numeric(i) = label_map(y_combined{i});
end

num_features = size(X_combined, 2);

% normalise for LDA and SVM importance calculation
mu_g   = mean(X_combined);
sig_g  = std(X_combined);
sig_g(sig_g == 0) = 1;
X_norm = (X_combined - mu_g) ./ sig_g;

fprintf('\ntotal: %d samples, %d features, %d classes\n\n', ...
    size(X_combined,1), num_features, num_classes);


% LDA importance - Fisher criterion (between/within class variance ratio)
fprintf('computing LDA feature importance...\n');

lda_scores = zeros(num_features, 1);

for f = 1:num_features
    x_f = X_norm(:, f);
    grand_mean  = mean(x_f);
    between_var = 0;
    within_var  = 0;

    for c = 1:num_classes
        class_mask = y_numeric == c;
        x_class    = x_f(class_mask);
        n_c        = sum(class_mask);
        class_mean = mean(x_class);
        between_var = between_var + n_c * (class_mean - grand_mean)^2;
        within_var  = within_var  + sum((x_class - class_mean).^2);
    end

    if within_var > 0
        lda_scores(f) = between_var / within_var;
    end
end

[~, lda_ranked] = sort(lda_scores, 'descend');
lda_top = lda_ranked(1:top_n);
fprintf('  done\n\n');


% SVM importance - sum of absolute weights across all binary classifiers
fprintf('training SVM for feature importance (takes a moment)...\n');

svm_template = templateSVM('KernelFunction', 'linear', 'Standardize', false);
svm_model    = fitcecoc(X_norm, y_numeric, ...
                        'Learners', svm_template, ...
                        'Coding',   'onevsone');

svm_weights = zeros(num_features, 1);
for b = 1:length(svm_model.BinaryLearners)
    w = svm_model.BinaryLearners{b}.Beta;
    svm_weights = svm_weights + abs(w);
end

[~, svm_ranked] = sort(svm_weights, 'descend');
svm_top = svm_ranked(1:top_n);
fprintf('  done\n\n');


% RF importance - OOB permutation importance (50 trees)
fprintf('training random forest for feature importance...\n');

rf_model = TreeBagger(50, X_combined, y_numeric, ...
                      'Method',                'classification', ...
                      'OOBPrediction',         'on', ...
                      'OOBPredictorImportance', 'on');

rf_importance = rf_model.OOBPermutedPredictorDeltaError;

[~, rf_ranked] = sort(rf_importance, 'descend');
rf_top = rf_ranked(1:top_n);
fprintf('  done\n\n');


% DBN features - loaded from the sequential forward selection results
fprintf('loading DBN features from saved file...\n');

dbn_mat = fullfile(sprintf('/home/eeiww/ut55iqoh/MLMA_CAM21/result_feature_selection/%s/', ...
    ground_to_test), 'DBN_selected_features_no_improve_stop.mat');

if isfile(dbn_mat)
    dbn_data     = load(dbn_mat);
    dbn_features = dbn_data.features;
    dbn_top      = dbn_features(1:min(top_n, length(dbn_features)));
    fprintf('  loaded %d DBN features, using top %d\n\n', ...
        length(dbn_features), length(dbn_top));
else
    fprintf('  DBN file not found, using hardcoded fallback\n\n');
    dbn_features = [19,5,13,157,244,200,14,20,197,234, ...
                    205,153,158,173,187,169,16,246,2,189, ...
                    4,109,249,178,3,8,148,7,225,222];
    dbn_top = dbn_features(1:min(top_n, length(dbn_features)))';
end


% compute pairwise overlap between all 4 models
fprintf('computing feature overlap...\n\n');

tops  = {lda_top(:), svm_top(:), rf_top(:), dbn_top(:)};
names = {'LDA', 'SVM', 'RF', 'DBN'};

overlap_matrix = zeros(4, 4);
for i = 1:4
    for j = 1:4
        overlap_matrix(i,j) = length(intersect(tops{i}, tops{j}));
    end
end

fprintf('overlap matrix (shared features in top %d):\n\n', top_n);
fprintf('         LDA    SVM     RF    DBN\n');
fprintf('       -----  -----  -----  -----\n');
for i = 1:4
    fprintf('%-6s ', names{i});
    for j = 1:4
        fprintf('  %3d  ', overlap_matrix(i,j));
    end
    fprintf('\n');
end

fprintf('\npairwise overlap percentages:\n\n');
for i = 1:4
    for j = i+1:4
        ov  = overlap_matrix(i,j);
        pct = 100 * ov / top_n;
        fprintf('  %s vs %s : %d/%d shared (%.1f%%)\n', ...
            names{i}, names{j}, ov, top_n, pct);
    end
end

% find universally agreed features and model-specific ones
common_all4 = intersect(intersect(lda_top(:), svm_top(:)), ...
                        intersect(rf_top(:),  dbn_top(:)));

dbn_unique = setdiff(dbn_top(:), union(union(lda_top(:), svm_top(:)), rf_top(:)));

linear_only = intersect(lda_top(:), svm_top(:));
linear_only = setdiff(linear_only, union(rf_top(:), dbn_top(:)));

fprintf('\nfeatures in all 4 models: %d\n', length(common_all4));
fprintf('features unique to DBN:   %d\n', length(dbn_unique));
fprintf('linear-only features:     %d\n\n', length(linear_only));


% plots
% heatmap of feature overlap
figure('Name', 'Feature Overlap', 'Position', [100 100 500 420]);
imagesc(overlap_matrix);
colorbar;
colormap(hot);
xticks(1:4); xticklabels(names);
yticks(1:4); yticklabels(names);
title(sprintf('Feature Overlap — Top %d Features per Model', top_n));
xlabel('Model'); ylabel('Model');

for i = 1:4
    for j = 1:4
        text(j, i, sprintf('%d', overlap_matrix(i,j)), ...
            'HorizontalAlignment', 'center', ...
            'Color', 'white', 'FontWeight', 'bold', 'FontSize', 12);
    end
end

saveas(gcf, fullfile(output_folder, 'overlap_heatmap.png'));

% stacked bar showing unique vs shared per model
figure('Name', 'Feature Uniqueness', 'Position', [100 600 600 400]);

unique_count = zeros(1, 4);
shared_count = zeros(1, 4);

for i = 1:4
    other_idx = setdiff(1:4, i);
    others = union(union(tops{other_idx(1)}, tops{other_idx(2)}), tops{other_idx(3)});
    unique_count(i) = length(setdiff(tops{i}, others));
    shared_count(i) = top_n - unique_count(i);
end

bar_data = [unique_count; shared_count]';
b = bar(bar_data, 'stacked');
b(1).FaceColor = [0.85 0.33 0.10];
b(2).FaceColor = [0.00 0.45 0.74];
xticks(1:4); xticklabels(names);
ylabel('Number of Features');
title(sprintf('Unique vs Shared Features (Top %d)', top_n));
legend({'Unique to model', 'Shared with others'}, 'Location', 'northeast');
ylim([0 top_n + 5]);

for i = 1:4
    text(i, top_n + 1, sprintf('%d unique', unique_count(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10);
end

saveas(gcf, fullfile(output_folder, 'feature_uniqueness.png'));

% save summary text file
txt_out = fullfile(output_folder, 'feature_comparison_summary.txt');
fid = fopen(txt_out, 'w');

fprintf(fid, 'feature importance comparison - %s (top %d)\n\n', ground_to_test, top_n);

fprintf(fid, 'LDA top %d : ', top_n); fprintf(fid, '%d ', lda_top); fprintf(fid, '\n');
fprintf(fid, 'SVM top %d : ', top_n); fprintf(fid, '%d ', svm_top); fprintf(fid, '\n');
fprintf(fid, 'RF  top %d : ', top_n); fprintf(fid, '%d ', rf_top);  fprintf(fid, '\n');
fprintf(fid, 'DBN top %d : ', top_n); fprintf(fid, '%d ', dbn_top); fprintf(fid, '\n\n');

fprintf(fid, 'overlap matrix:\n');
fprintf(fid, '         LDA    SVM     RF    DBN\n');
for i = 1:4
    fprintf(fid, '%-6s ', names{i});
    for j = 1:4
        fprintf(fid, '  %3d  ', overlap_matrix(i,j));
    end
    fprintf(fid, '\n');
end

fprintf(fid, '\npairwise overlap:\n');
for i = 1:4
    for j = i+1:4
        ov  = overlap_matrix(i,j);
        pct = 100 * ov / top_n;
        fprintf(fid, '  %s vs %s : %d/%d (%.1f%%)\n', ...
            names{i}, names{j}, ov, top_n, pct);
    end
end

fprintf(fid, '\nfeatures in all 4 models: %d\n', length(common_all4));
fprintf(fid, 'DBN unique features: %d\n', length(dbn_unique));
fprintf(fid, 'linear-only features: %d\n', length(linear_only));

fclose(fid);

fprintf('saved results to %s\n', txt_out);
fprintf('done!\n');