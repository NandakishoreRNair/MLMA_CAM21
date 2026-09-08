%% ========================================================
%% FAST FEATURE IMPORTANCE COMPARISON
%% LDA vs SVM vs RF vs DBN
%% Goal: Test if linear/nonlinear/temporal models select
%%       different features
%% ========================================================

clear all; close all; clc;

%% ========================================================
%% SETTINGS
%% ========================================================

ground_to_test = 'ramp';
TOP_N          = 80;   % compare top N features across models

base_folder   = sprintf('/home/eeiww/ut55iqoh/MLMA_CAM21/Classification/%s/', ground_to_test);
output_folder = sprintf('/home/eeiww/ut55iqoh/MLMA_CAM21/result_feature_selection/%s/', ground_to_test);

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

subjects = {'ab07', 'ab08', 'ab09', 'ab12', 'ab13', 'ab14', 'ab17', 'ab18', ...
            'ab19', 'ab20', 'ab21', 'ab23', 'ab24', 'ab27', 'ab28'};

fprintf('=======================================================\n');
fprintf('FAST FEATURE IMPORTANCE COMPARISON\n');
fprintf('Ground: %s  |  Top N: %d\n', ground_to_test, TOP_N);
fprintf('=======================================================\n\n');

%% ========================================================
%% STEP 1: LOAD DATA
%% ========================================================

fprintf('STEP 1: LOADING DATA\n\n');

X_combined     = [];
y_combined     = {};
trial_combined = [];

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

        for i = 1:length(labels_col)
            if strcmp(labels_col{i},'turn1') || strcmp(labels_col{i},'turn2')
                labels_col{i} = 'turn';
            end
        end

        X_combined     = [X_combined;     X];
        y_combined     = [y_combined;     labels_col];
        trial_combined = [trial_combined; trial_col];

        fprintf('Loaded: %s  (%d samples)\n', subject, size(X,1));
    catch ME
        fprintf('ERROR: %s — %s\n', subject, ME.message);
    end
end

% Label encoding
unique_labels = unique(y_combined);
num_classes   = length(unique_labels);
label_map     = containers.Map(unique_labels, 1:num_classes);

y_numeric = zeros(length(y_combined), 1);
for i = 1:length(y_combined)
    y_numeric(i) = label_map(y_combined{i});
end

num_features = size(X_combined, 2);

% Global normalisation (used for LDA and SVM importance — not for CV)
mu_g    = mean(X_combined);
sig_g   = std(X_combined);
sig_g(sig_g == 0) = 1;
X_norm  = (X_combined - mu_g) ./ sig_g;

fprintf('\nTotal: %d samples, %d features, %d classes\n\n', ...
    size(X_combined,1), num_features, num_classes);

%% ========================================================
%% STEP 2: LDA FEATURE IMPORTANCE
%% Train one LDA on all data, extract Fisher criterion score
%% per feature (between-class / within-class variance ratio)
%% ========================================================

fprintf('=======================================================\n');
fprintf('STEP 2: LDA FEATURE IMPORTANCE\n');
fprintf('=======================================================\n\n');

lda_scores = zeros(num_features, 1);

for f = 1:num_features
    x_f = X_norm(:, f);

    % Between-class variance
    grand_mean    = mean(x_f);
    between_var   = 0;
    within_var    = 0;

    for c = 1:num_classes
        class_mask  = y_numeric == c;
        x_class     = x_f(class_mask);
        n_c         = sum(class_mask);
        class_mean  = mean(x_class);
        between_var = between_var + n_c * (class_mean - grand_mean)^2;
        within_var  = within_var  + sum((x_class - class_mean).^2);
    end

    if within_var > 0
        lda_scores(f) = between_var / within_var;
    else
        lda_scores(f) = 0;
    end
end

[~, lda_ranked] = sort(lda_scores, 'descend');
lda_top = lda_ranked(1:TOP_N);

fprintf('LDA top %d features computed.\n\n', TOP_N);

%% ========================================================
%% STEP 3: SVM FEATURE IMPORTANCE
%% Train one multiclass SVM, extract |w| weight per feature
%% Linear SVM: feature importance = sum of |weights| across
%% all binary classifiers in ECOC
%% ========================================================

fprintf('=======================================================\n');
fprintf('STEP 3: SVM FEATURE IMPORTANCE\n');
fprintf('=======================================================\n\n');

fprintf('Training SVM on full data (one-time, no CV)...\n');

svm_template = templateSVM('KernelFunction', 'linear', 'Standardize', false);
svm_model    = fitcecoc(X_norm, y_numeric, ...
                        'Learners', svm_template, ...
                        'Coding',   'onevsone');

% Extract weight vectors from each binary SVM
svm_weights = zeros(num_features, 1);
for b = 1:length(svm_model.BinaryLearners)
    w = svm_model.BinaryLearners{b}.Beta;
    svm_weights = svm_weights + abs(w);
end

[~, svm_ranked] = sort(svm_weights, 'descend');
svm_top = svm_ranked(1:TOP_N);

fprintf('SVM top %d features computed.\n\n', TOP_N);

%% ========================================================
%% STEP 4: RF FEATURE IMPORTANCE
%% Train one RF with OOB permutation importance
%% Much faster than sequential selection
%% ========================================================

fprintf('=======================================================\n');
fprintf('STEP 4: RF FEATURE IMPORTANCE\n');
fprintf('=======================================================\n\n');

fprintf('Training Random Forest (50 trees, OOB importance)...\n');

rf_model = TreeBagger(50, X_combined, y_numeric, ...
                      'Method',           'classification', ...
                      'OOBPrediction',    'on', ...
                      'OOBPredictorImportance', 'on');

rf_importance = rf_model.OOBPermutedPredictorDeltaError;

[~, rf_ranked] = sort(rf_importance, 'descend');
rf_top = rf_ranked(1:TOP_N);

fprintf('RF top %d features computed.\n\n', TOP_N);

%% ========================================================
%% STEP 5: DBN FEATURES
%% Load from your already-completed feature selection
%% ========================================================

fprintf('=======================================================\n');
fprintf('STEP 5: LOADING DBN FEATURES\n');
fprintf('=======================================================\n\n');

dbn_mat = fullfile(sprintf('/home/eeiww/ut55iqoh/MLMA_CAM21/result_feature_selection/%s/', ...
    ground_to_test), 'DBN_selected_features_no_improve_stop.mat');

if isfile(dbn_mat)
    dbn_data = load(dbn_mat);
    dbn_features_all = dbn_data.features;
    dbn_top  = dbn_features_all(1:min(TOP_N, length(dbn_features_all)));
    fprintf('DBN features loaded: %d total, using top %d\n\n', ...
        length(dbn_features_all), length(dbn_top));
else
    fprintf('WARNING: DBN .mat file not found at:\n%s\n', dbn_mat);
    fprintf('Enter DBN features manually below.\n\n');
    % Paste your DBN features here as fallback
    dbn_features_all = [19,5,13,157,244,200,14,20,197,234, ...
                        205,153,158,173,187,169,16,246,2,189, ...
                        4,109,249,178,3,8,148,7,225,222];
    dbn_top = dbn_features_all(1:min(TOP_N, length(dbn_features_all)))';
end

%% ========================================================
%% STEP 6: OVERLAP ANALYSIS
%% ========================================================

fprintf('=======================================================\n');
fprintf('STEP 6: OVERLAP ANALYSIS (Top %d features)\n', TOP_N);
fprintf('=======================================================\n\n');

% Pairwise overlaps
pairs = {'LDA','SVM'; 'LDA','RF'; 'LDA','DBN'; ...
         'SVM','RF';  'SVM','DBN'; 'RF','DBN'};

tops = {lda_top(:), svm_top(:), rf_top(:), dbn_top(:)};
names = {'LDA', 'SVM', 'RF', 'DBN'};

overlap_matrix = zeros(4, 4);

for i = 1:4
    for j = 1:4
        overlap_matrix(i,j) = length(intersect(tops{i}, tops{j}));
    end
end

fprintf('Overlap matrix (number of shared features in top %d):\n\n', TOP_N);
fprintf('         LDA    SVM     RF    DBN\n');
fprintf('       -----  -----  -----  -----\n');
for i = 1:4
    fprintf('%-6s ', names{i});
    for j = 1:4
        fprintf('  %3d  ', overlap_matrix(i,j));
    end
    fprintf('\n');
end

fprintf('\nPairwise overlap percentages:\n\n');
for i = 1:4
    for j = i+1:4
        ov  = overlap_matrix(i,j);
        pct = 100 * ov / TOP_N;
        fprintf('  %s vs %s : %d / %d features shared  (%.1f%%)\n', ...
            names{i}, names{j}, ov, TOP_N, pct);
    end
end

%% ========================================================
%% STEP 7: VISUALISE — Venn-style bar chart
%% ========================================================

fprintf('\n=======================================================\n');
fprintf('STEP 7: VISUALISATION\n');
fprintf('=======================================================\n\n');

% ---- Plot 1: Overlap heatmap ----
figure('Name', 'Feature Overlap Heatmap', 'Position', [100 100 500 420]);
imagesc(overlap_matrix);
colorbar;
colormap(hot);
xticks(1:4); xticklabels(names);
yticks(1:4); yticklabels(names);
title(sprintf('Feature Overlap — Top %d Features per Model', TOP_N));
xlabel('Model'); ylabel('Model');

for i = 1:4
    for j = 1:4
        text(j, i, sprintf('%d', overlap_matrix(i,j)), ...
            'HorizontalAlignment', 'center', ...
            'Color', 'white', 'FontWeight', 'bold', 'FontSize', 12);
    end
end

saveas(gcf, fullfile(output_folder, 'overlap_heatmap.png'));

% ---- Plot 2: Feature rank comparison for shared features ----
% Find features in ALL 4 models
common_all4 = intersect(intersect(lda_top(:), svm_top(:)), ...
                        intersect(rf_top(:),  dbn_top(:)));

fprintf('Features in TOP %d of ALL 4 models: %d\n', TOP_N, length(common_all4));
if ~isempty(common_all4)
    fprintf('  Indices: ');
    fprintf('%d  ', common_all4);
    fprintf('\n');
end

% Features unique to DBN (temporal-only)
dbn_unique = setdiff(dbn_top(:), union(union(lda_top(:), svm_top(:)), rf_top(:)));
fprintf('\nFeatures UNIQUE to DBN (not in LDA/SVM/RF top %d): %d\n', TOP_N, length(dbn_unique));
if ~isempty(dbn_unique)
    fprintf('  Indices: ');
    fprintf('%d  ', dbn_unique);
    fprintf('\n');
end

% Features in linear models only (LDA + SVM, not RF/DBN)
linear_only = intersect(lda_top(:), svm_top(:));
linear_only = setdiff(linear_only, union(rf_top(:), dbn_top(:)));
fprintf('\nFeatures shared by LDA+SVM but NOT in RF/DBN: %d\n', length(linear_only));
if ~isempty(linear_only)
    fprintf('  Indices: ');
    fprintf('%d  ', linear_only);
    fprintf('\n');
end

% ---- Plot 3: Stacked bar showing unique vs shared per model ----
figure('Name', 'Feature Uniqueness per Model', 'Position', [100 600 600 400]);

for i = 1:4
    other_idx = setdiff(1:4, i);          % e.g. [2,3,4] when i=1
    others = union(union(tops{other_idx(1)}, tops{other_idx(2)}), ...
                         tops{other_idx(3)});
    unique_count(i) = length(setdiff(tops{i}, others));
    shared_count(i) = TOP_N - unique_count(i);
end

bar_data = [unique_count; shared_count]';
b = bar(bar_data, 'stacked');
b(1).FaceColor = [0.85 0.33 0.10];  % unique = orange
b(2).FaceColor = [0.00 0.45 0.74];  % shared = blue
xticks(1:4); xticklabels(names);
ylabel('Number of Features');
title(sprintf('Unique vs Shared Features (Top %d)', TOP_N));
legend({'Unique to model', 'Shared with others'}, 'Location', 'northeast');
ylim([0 TOP_N + 5]);

for i = 1:4
    text(i, TOP_N + 1, sprintf('%d unique', unique_count(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10);
end

saveas(gcf, fullfile(output_folder, 'feature_uniqueness.png'));

%% ========================================================
%% STEP 8: SAVE SUMMARY
%% ========================================================

txt_out = fullfile(output_folder, 'feature_comparison_summary.txt');
fid = fopen(txt_out, 'w');

fprintf(fid, '=======================================================\n');
fprintf(fid, 'FEATURE IMPORTANCE COMPARISON SUMMARY\n');
fprintf(fid, 'Ground: %s  |  Top N: %d\n', ground_to_test, TOP_N);
fprintf(fid, '=======================================================\n\n');

fprintf(fid, 'LDA top %d : ', TOP_N); fprintf(fid, '%d ', lda_top); fprintf(fid, '\n');
fprintf(fid, 'SVM top %d : ', TOP_N); fprintf(fid, '%d ', svm_top); fprintf(fid, '\n');
fprintf(fid, 'RF  top %d : ', TOP_N); fprintf(fid, '%d ', rf_top);  fprintf(fid, '\n');
fprintf(fid, 'DBN top %d : ', TOP_N); fprintf(fid, '%d ', dbn_top); fprintf(fid, '\n\n');

fprintf(fid, 'Overlap matrix (shared features in top %d):\n\n', TOP_N);
fprintf(fid, '         LDA    SVM     RF    DBN\n');
for i = 1:4
    fprintf(fid, '%-6s ', names{i});
    for j = 1:4
        fprintf(fid, '  %3d  ', overlap_matrix(i,j));
    end
    fprintf(fid, '\n');
end

fprintf(fid, '\nPairwise overlap:\n');
for i = 1:4
    for j = i+1:4
        ov  = overlap_matrix(i,j);
        pct = 100 * ov / TOP_N;
        fprintf(fid, '  %s vs %s : %d/%d  (%.1f%%)\n', ...
            names{i}, names{j}, ov, TOP_N, pct);
    end
end

fprintf(fid, '\nFeatures in ALL 4 models: %d\n', length(common_all4));
fprintf(fid, 'DBN-unique features      : %d\n', length(dbn_unique));
fprintf(fid, 'Linear-only features     : %d\n', length(linear_only));

fclose(fid);

fprintf('\nSaved: %s\n', txt_out);
fprintf('\n=======================================================\n');
fprintf('COMPARISON COMPLETE!\n');
fprintf('=======================================================\n');