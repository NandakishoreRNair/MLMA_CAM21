%% ========================================================
%% HYPOTHESIS TEST: Temporal Modeling at Gait Phase Transitions
%%
%% "At steady-state (35% gait), simpler models suffice.
%%  At transitions (85% gait), DBN temporal advantage is critical."
%%
%% Method:
%%   - Per-subject Leave-One-Trial-Out CV (matches paper)
%%   - Separate models trained at 35% and 85% gait phase
%%   - DBN uses 35% prediction as prior for 85% classification
%%   - Transition step = label differs between 35% and 85%
%% ========================================================

clear all; close all; clc;

%% ========================================================
%% SETTINGS
%% ========================================================

ground_to_test = 'levelground';

base_35       = sprintf('/home/eeiww/ut55iqoh/MLMA_CAM21/Classification/%s/35/400/', ground_to_test);
base_85       = sprintf('/home/eeiww/ut55iqoh/MLMA_CAM21/Classification/%s/85/400/', ground_to_test);
output_folder = sprintf('/home/eeiww/ut55iqoh/MLMA_CAM21/result_hypothesis_35vs85/%s/', ground_to_test);

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

subjects = {'ab07', 'ab08', 'ab09', 'ab12', 'ab13', 'ab14', 'ab17', 'ab18', ...
            'ab19', 'ab20', 'ab21', 'ab23', 'ab24', 'ab27', 'ab28'};

NUM_TREES = 30;

fprintf('=======================================================\n');
fprintf('HYPOTHESIS TEST: Gait Phase Temporal Advantage\n');
fprintf('Ground  : %s\n', ground_to_test);
fprintf('Method  : Per-subject Leave-One-Trial-Out CV\n');
fprintf('35%%     = steady state\n');
fprintf('85%%     = transition\n');
fprintf('Models  : LDA, SVM, RF, DBN\n');
fprintf('=======================================================\n\n');

%% ========================================================
%% STEP 1: LOAD DBN FEATURES
%% ========================================================

fprintf('STEP 1: LOADING DBN FEATURES\n\n');

dbn_mat = fullfile( ...
    sprintf('/home/eeiww/ut55iqoh/MLMA_CAM21/result_feature_selection/%s/', ground_to_test), ...
    'DBN_selected_features_no_improve_stop.mat');

if ~isfile(dbn_mat)
    error('DBN features not found at:\n%s', dbn_mat);
end

dbn_data          = load(dbn_mat);
selected_features = dbn_data.features;
fprintf('DBN features loaded: %d features\n\n', length(selected_features));

%% ========================================================
%% STEP 2: LOAD DATA PER SUBJECT
%% ========================================================

fprintf('STEP 2: LOADING DATA\n\n');

subj_X35    = cell(length(subjects), 1);
subj_X85    = cell(length(subjects), 1);
subj_y35    = cell(length(subjects), 1);
subj_y85    = cell(length(subjects), 1);
subj_trials = cell(length(subjects), 1);
subj_trans  = cell(length(subjects), 1);
valid_subj  = false(length(subjects), 1);

% Collect all labels for global consistent encoding
all_labels_pool = {};

for subj_idx = 1:length(subjects)
    subject  = subjects{subj_idx};
    input35  = fullfile(base_35, [subject '_input.mat']);
    output35 = fullfile(base_35, [subject '_output.mat']);
    input85  = fullfile(base_85, [subject '_input.mat']);
    output85 = fullfile(base_85, [subject '_output.mat']);

    if ~isfile(input35) || ~isfile(output35) || ...
       ~isfile(input85) || ~isfile(output85)
        fprintf('WARNING: Files not found for %s. Skipping.\n', subject);
        continue;
    end

    try
        in35  = load(input35);
        out35 = load(output35);
        in85  = load(input85);
        out85 = load(output85);

        X35_all      = table2array(in35.alldata);
        X85_all      = table2array(in85.alldata);
        labels35_all = out35.alldata.labels_feat_last;
        labels85_all = out85.alldata.labels_feat_last;
        trial35_all  = out35.alldata.trial_feat_last;
        trial85_all  = out85.alldata.trial_feat_last;

        % Select DBN features
        X35_all = X35_all(:, selected_features);
        X85_all = X85_all(:, selected_features);

        % Align by trial — handle row count mismatches
        common_trials = intersect(unique(trial35_all), unique(trial85_all));

        X35_s = []; X85_s = [];
        y35_s = {}; y85_s = {};
        trial_s = []; trans_s = [];

        for tt = 1:length(common_trials)
            tr    = common_trials(tt);
            idx35 = find(trial35_all == tr);
            idx85 = find(trial85_all == tr);
            n     = min(length(idx35), length(idx85));
            if n == 0, continue; end

            idx35 = idx35(1:n);
            idx85 = idx85(1:n);
            lbl35 = labels35_all(idx35);
            lbl85 = labels85_all(idx85);

            X35_s   = [X35_s;   X35_all(idx35, :)];
            X85_s   = [X85_s;   X85_all(idx85, :)];
            y35_s   = [y35_s;   lbl35];
            y85_s   = [y85_s;   lbl85];
            trial_s = [trial_s; repmat(tr, n, 1)];
            trans_s = [trans_s; ~strcmp(lbl35, lbl85)];
        end

        if isempty(X35_s)
            fprintf('WARNING: No aligned data for %s.\n', subject);
            continue;
        end

        subj_X35{subj_idx}    = X35_s;
        subj_X85{subj_idx}    = X85_s;
        subj_y35{subj_idx}    = y35_s;
        subj_y85{subj_idx}    = y85_s;
        subj_trials{subj_idx} = trial_s;
        subj_trans{subj_idx}  = trans_s;
        valid_subj(subj_idx)  = true;

        all_labels_pool = [all_labels_pool; y35_s; y85_s];

        fprintf('Loaded: %s  (%d steps | steady: %d | transition: %d)\n', ...
            subject, size(X35_s,1), sum(~trans_s), sum(trans_s));

    catch ME
        fprintf('ERROR: %s - %s\n', subject, ME.message);
    end
end

% Global label encoding — consistent numeric IDs across all subjects
all_labels  = unique(all_labels_pool);
num_classes = length(all_labels);
label_map   = containers.Map(all_labels, 1:num_classes);

fprintf('\nClasses (%d total):\n', num_classes);
for i = 1:num_classes
    fprintf('  %d -> %s\n', i, all_labels{i});
end

% Convert string labels to numeric for each subject
for subj_idx = 1:length(subjects)
    if ~valid_subj(subj_idx), continue; end

    y35_str = subj_y35{subj_idx};
    y85_str = subj_y85{subj_idx};
    n       = length(y35_str);

    y35_num = zeros(n, 1);
    y85_num = zeros(n, 1);
    for i = 1:n
        y35_num(i) = label_map(y35_str{i});
        y85_num(i) = label_map(y85_str{i});
    end

    subj_y35{subj_idx} = y35_num;
    subj_y85{subj_idx} = y85_num;
end

fprintf('\n');

%% ========================================================
%% STEP 3: PER-SUBJECT LEAVE-ONE-TRIAL-OUT CV
%%
%% For each subject:
%%   For each trial (left out as test):
%%     Train M35 on remaining trials -> get prior at 35%
%%     Train M85 on remaining trials -> get emission at 85%
%%     Build T matrix P(class@85 | class@35) from training
%%     DBN = emission@85 x (prior@35 x T)
%%     LDA/SVM/RF = 85% features only (no prior)
%% ========================================================

fprintf('=======================================================\n');
fprintf('STEP 3: PER-SUBJECT LEAVE-ONE-TRIAL-OUT CV\n');
fprintf('=======================================================\n\n');

all_truth_85   = [];
all_trans_mask = [];
all_pred_LDA   = [];
all_pred_SVM   = [];
all_pred_RF    = [];
all_pred_DBN   = [];

% Per-subject storage for reporting
subj_overall_lda = nan(length(subjects), 1);
subj_overall_dbn = nan(length(subjects), 1);
subj_trans_lda   = nan(length(subjects), 1);
subj_trans_dbn   = nan(length(subjects), 1);
subj_steady_lda  = nan(length(subjects), 1);
subj_steady_dbn  = nan(length(subjects), 1);

svm_template = templateSVM('KernelFunction', 'linear', 'Standardize', false);

tic;

for subj_idx = 1:length(subjects)
    if ~valid_subj(subj_idx), continue; end

    subject  = subjects{subj_idx};
    X35_s    = subj_X35{subj_idx};
    X85_s    = subj_X85{subj_idx};
    y35_s    = subj_y35{subj_idx};
    y85_s    = subj_y85{subj_idx};
    trial_s  = subj_trials{subj_idx};
    trans_s  = subj_trans{subj_idx};

    unique_trials_s = unique(trial_s);
    num_folds_s     = length(unique_trials_s);

    fprintf('[%s] Subject %-5s | %d trials | %d steps  (%.1f sec)\n', ...
        datetime('now','Format','HH:mm:ss'), subject, ...
        num_folds_s, length(y85_s), toc);

    % Per-subject accumulation
    subj_truth   = [];
    subj_trans_m = [];
    subj_lda     = [];
    subj_svm_p   = [];
    subj_rf_p    = [];
    subj_dbn_p   = [];

    for fold = 1:num_folds_s
        test_trial = unique_trials_s(fold);

        train_m = trial_s ~= test_trial;
        test_m  = trial_s == test_trial;

        if sum(train_m) < 5 || sum(test_m) < 1
            continue;
        end

        X35_train = X35_s(train_m, :);
        X85_train = X85_s(train_m, :);
        y35_train = y35_s(train_m);
        y85_train = y85_s(train_m);

        X35_test  = X35_s(test_m, :);
        X85_test  = X85_s(test_m, :);
        y85_test  = y85_s(test_m);
        trans_test = trans_s(test_m);

        if length(unique(y85_train)) < 2
            continue;
        end

        % Normalise using 85% training statistics only (no leakage)
        mu_tr    = mean(X85_train);
        sigma_tr = std(X85_train);
        sigma_tr(sigma_tr == 0) = 1;

        X35_train_n = (X35_train - mu_tr) ./ sigma_tr;
        X85_train_n = (X85_train - mu_tr) ./ sigma_tr;
        X35_test_n  = (X35_test  - mu_tr) ./ sigma_tr;
        X85_test_n  = (X85_test  - mu_tr) ./ sigma_tr;

        % Build transition matrix P(class@85 | class@35)
        % from this subject's training steps only
        transition_matrix = ones(num_classes, num_classes);
        for k = 1:length(y35_train)
            from_cls = y35_train(k);
            to_cls   = y85_train(k);
            transition_matrix(from_cls, to_cls) = ...
                transition_matrix(from_cls, to_cls) + 1;
        end
        transition_matrix = transition_matrix ./ sum(transition_matrix, 2);

        %% ---- LDA at 85% ----
        lda85_ok = false;
        try
            lda85_model = fitcdiscr(X85_train_n, y85_train, 'DiscrimType', 'linear');
            pred_LDA    = predict(lda85_model, X85_test_n);
            lda85_ok    = true;
        catch
            pred_LDA = mode(y85_train) * ones(size(y85_test));
        end

        %% ---- SVM at 85% ----
        try
            svm85_model = fitcecoc(X85_train_n, y85_train, ...
                                   'Learners', svm_template, ...
                                   'Coding',   'onevsone');
            pred_SVM    = predict(svm85_model, X85_test_n);
        catch
            pred_SVM = mode(y85_train) * ones(size(y85_test));
        end

        %% ---- RF at 85% ----
        try
            rf85_model   = TreeBagger(NUM_TREES, X85_train, y85_train, ...
                                      'Method',        'classification', ...
                                      'OOBPrediction', 'off');
            pred_RF_cell = predict(rf85_model, X85_test);
            pred_RF      = str2double(pred_RF_cell);
        catch
            pred_RF = mode(y85_train) * ones(size(y85_test));
        end

        %% ---- DBN: M35 prior + M85 emission ----
        % Key fix: use ClassNames directly as column indices
        % since labels are already encoded as 1..num_classes
        try
            % Train model at 35% gait phase
            m35_model = fitcdiscr(X35_train_n, y35_train, 'DiscrimType', 'linear');

            % Get scores from M35 (prior) and M85 LDA (emission)
            [~, scores35] = predict(m35_model,    X35_test_n);
            [~, scores85] = predict(lda85_model,  X85_test_n);

            % Class lists — already numeric 1..num_classes
            classes_35 = m35_model.ClassNames;
            classes_85 = lda85_model.ClassNames;

            n_test = size(scores35, 1);

            % Initialise full-size matrices with uniform small probability
            prior_full    = ones(n_test, num_classes) * 1e-10;
            emission_full = ones(n_test, num_classes) * 1e-10;

            % Fill known class columns from M35 scores
            for ci = 1:length(classes_35)
                col = classes_35(ci);
                if col >= 1 && col <= num_classes
                    prior_full(:, col) = scores35(:, ci);
                end
            end

            % Fill known class columns from M85 scores
            for ci = 1:length(classes_85)
                col = classes_85(ci);
                if col >= 1 && col <= num_classes
                    emission_full(:, col) = scores85(:, ci);
                end
            end

            % Clamp negatives and normalise rows
            prior_full    = max(prior_full,    1e-10);
            emission_full = max(emission_full, 1e-10);
            prior_full    = prior_full    ./ sum(prior_full,    2);
            emission_full = emission_full ./ sum(emission_full, 2);

            % Propagate prior through gait-phase transition matrix
            % temporal_prior(i,c) = P(class c at 85% | what happened at 35%)
            temporal_prior = prior_full * transition_matrix;
            temporal_prior = temporal_prior ./ sum(temporal_prior, 2);

            % DBN posterior = emission x temporal prior (element-wise)
            posterior_85  = emission_full .* temporal_prior;
            posterior_85  = posterior_85  ./ sum(posterior_85, 2);

            [~, pred_DBN] = max(posterior_85, [], 2);

        catch ME
            pred_DBN = pred_LDA;
            fprintf('    DBN failed fold %d: %s\n', fold, ME.message);
        end

        % Accumulate this fold into subject storage
        subj_truth   = [subj_truth;   y85_test];
        subj_trans_m = [subj_trans_m; trans_test];
        subj_lda     = [subj_lda;     pred_LDA];
        subj_svm_p   = [subj_svm_p;   pred_SVM];
        subj_rf_p    = [subj_rf_p;    pred_RF];
        subj_dbn_p   = [subj_dbn_p;   pred_DBN];
    end

    % Per-subject summary
    if ~isempty(subj_truth)
        s_mask = ~logical(subj_trans_m);
        t_mask =  logical(subj_trans_m);

        subj_overall_lda(subj_idx) = mean(subj_lda   == subj_truth) * 100;
        subj_overall_dbn(subj_idx) = mean(subj_dbn_p == subj_truth) * 100;

        fprintf('  Overall    LDA:%.1f%%  SVM:%.1f%%  RF:%.1f%%  DBN:%.1f%%\n', ...
            mean(subj_lda  ==subj_truth)*100, ...
            mean(subj_svm_p==subj_truth)*100, ...
            mean(subj_rf_p ==subj_truth)*100, ...
            mean(subj_dbn_p==subj_truth)*100);

        if sum(s_mask) > 0
            subj_steady_lda(subj_idx) = mean(subj_lda(s_mask)   == subj_truth(s_mask)) * 100;
            subj_steady_dbn(subj_idx) = mean(subj_dbn_p(s_mask) == subj_truth(s_mask)) * 100;
            fprintf('  Steady     LDA:%.1f%%  SVM:%.1f%%  RF:%.1f%%  DBN:%.1f%%\n', ...
                mean(subj_lda(s_mask)  ==subj_truth(s_mask))*100, ...
                mean(subj_svm_p(s_mask)==subj_truth(s_mask))*100, ...
                mean(subj_rf_p(s_mask) ==subj_truth(s_mask))*100, ...
                mean(subj_dbn_p(s_mask)==subj_truth(s_mask))*100);
        end

        if sum(t_mask) > 0
            subj_trans_lda(subj_idx) = mean(subj_lda(t_mask)   == subj_truth(t_mask)) * 100;
            subj_trans_dbn(subj_idx) = mean(subj_dbn_p(t_mask) == subj_truth(t_mask)) * 100;
            fprintf('  Transition LDA:%.1f%%  SVM:%.1f%%  RF:%.1f%%  DBN:%.1f%%\n', ...
                mean(subj_lda(t_mask)  ==subj_truth(t_mask))*100, ...
                mean(subj_svm_p(t_mask)==subj_truth(t_mask))*100, ...
                mean(subj_rf_p(t_mask) ==subj_truth(t_mask))*100, ...
                mean(subj_dbn_p(t_mask)==subj_truth(t_mask))*100);
        end
        fprintf('\n');
    end

    % Accumulate across all subjects
    all_truth_85   = [all_truth_85;   subj_truth];
    all_trans_mask = [all_trans_mask; subj_trans_m];
    all_pred_LDA   = [all_pred_LDA;   subj_lda];
    all_pred_SVM   = [all_pred_SVM;   subj_svm_p];
    all_pred_RF    = [all_pred_RF;    subj_rf_p];
    all_pred_DBN   = [all_pred_DBN;   subj_dbn_p];
end

fprintf('[%s] All subjects complete! (%.1f sec)\n\n', ...
    datetime('now','Format','HH:mm:ss'), toc);

%% ========================================================
%% STEP 4: AGGREGATE RESULTS
%% ========================================================

fprintf('=======================================================\n');
fprintf('STEP 4: AGGREGATE RESULTS\n');
fprintf('=======================================================\n\n');

trans_mask  = logical(all_trans_mask);
steady_mask = ~trans_mask;

fprintf('Total test steps — steady: %d | transition: %d\n\n', ...
    sum(steady_mask), sum(trans_mask));

models    = {'LDA', 'SVM', 'RF', 'DBN'};
all_preds = {all_pred_LDA, all_pred_SVM, all_pred_RF, all_pred_DBN};

overall_acc    = zeros(4,1);
steady_acc     = zeros(4,1);
transition_acc = zeros(4,1);

fprintf('%-6s  %12s  %15s  %15s\n', ...
    'Model','Overall (%)','Steady St.(%)','Transition(%)');
fprintf('%s\n', repmat('-',1,55));

for i = 1:4
    pred = all_preds{i};
    overall_acc(i)    = mean(pred == all_truth_85)                            * 100;
    steady_acc(i)     = mean(pred(steady_mask) == all_truth_85(steady_mask))  * 100;
    transition_acc(i) = mean(pred(trans_mask)  == all_truth_85(trans_mask))   * 100;
    fprintf('%-6s  %12.2f  %15.2f  %15.2f\n', ...
        models{i}, overall_acc(i), steady_acc(i), transition_acc(i));
end
fprintf('%s\n\n', repmat('-',1,55));

fprintf('DBN advantage AT TRANSITIONS:\n');
for i = 1:3
    fprintf('  DBN vs %-4s : %+.2f%%\n', models{i}, ...
        transition_acc(4) - transition_acc(i));
end
fprintf('\nDBN advantage AT STEADY STATE:\n');
for i = 1:3
    fprintf('  DBN vs %-4s : %+.2f%%\n', models{i}, ...
        steady_acc(4) - steady_acc(i));
end

%% ========================================================
%% STEP 5: VISUALISATION
%% ========================================================

fprintf('\nSTEP 5: PLOTTING\n\n');

% Plot 1: Grouped bar — steady vs transition per model
figure('Name','Gait Phase Hypothesis','Position',[100 100 800 550]);
bar_data = [steady_acc, transition_acc];
b = bar(bar_data, 'grouped');
b(1).FaceColor = [0.00 0.45 0.74];
b(2).FaceColor = [0.85 0.33 0.10];
xticks(1:4); xticklabels(models);
ylabel('Accuracy (%)');
xlabel('Model');
title(sprintf(['Steady State vs Transition Accuracy\n' ...
    '(%s | Per-subject LOTO-CV | DBN uses 35%% as prior for 85%%)'], ...
    ground_to_test));
legend({'Steady steps (same label @35%%&85%%)', ...
        'Transition steps (label changes)'}, 'Location','southeast');
ylim([0 108]); grid on;
for i = 1:4
    text(i-0.15, steady_acc(i)+1, sprintf('%.1f%%',steady_acc(i)), ...
        'HorizontalAlignment','center','FontSize',9);
    text(i+0.15, transition_acc(i)+1, sprintf('%.1f%%',transition_acc(i)), ...
        'HorizontalAlignment','center','FontSize',9,'Color',[0.6 0 0]);
end
saveas(gcf, fullfile(output_folder, 'gait_phase_hypothesis.png'));

% Plot 2: DBN advantage at transitions
figure('Name','DBN Transition Advantage','Position',[100 680 600 420]);
dbn_adv = transition_acc(4) - transition_acc(1:3);
b2 = bar(dbn_adv, 'FaceColor','flat');
for i = 1:3
    if dbn_adv(i) >= 0
        b2.CData(i,:) = [0.47 0.67 0.19];
    else
        b2.CData(i,:) = [0.85 0.33 0.10];
    end
end
xticks(1:3); xticklabels({'vs LDA','vs SVM','vs RF'});
ylabel('DBN Accuracy Gain at Transitions (%)');
title({'DBN Advantage at Transition Steps (85% Gait Phase)', ...
    'Positive = DBN better | Negative = DBN worse'});
yline(0,'k--','LineWidth',1.5); grid on;
for i = 1:3
    text(i, dbn_adv(i) + sign(dbn_adv(i))*0.4, ...
        sprintf('%+.2f%%', dbn_adv(i)), ...
        'HorizontalAlignment','center','FontSize',12,'FontWeight','bold');
end
saveas(gcf, fullfile(output_folder, 'dbn_transition_advantage.png'));

% Plot 3: Per-subject DBN vs LDA transition accuracy
valid_idx   = find(valid_subj);
plot_names  = {};
plot_lda_t  = [];
plot_dbn_t  = [];
plot_lda_s  = [];
plot_dbn_s  = [];

for k = 1:length(valid_idx)
    si = valid_idx(k);
    if ~isnan(subj_trans_lda(si))
        plot_names{end+1} = subjects{si};
        plot_lda_t(end+1) = subj_trans_lda(si);
        plot_dbn_t(end+1) = subj_trans_dbn(si);
        plot_lda_s(end+1) = subj_steady_lda(si);
        plot_dbn_s(end+1) = subj_steady_dbn(si);
    end
end

figure('Name','Per-Subject Results','Position',[750 100 1000 800]);

subplot(2,1,1);
hold on;
plot(1:length(plot_names), plot_lda_t, 'bo-', 'LineWidth',1.5, 'MarkerSize',8);
plot(1:length(plot_names), plot_dbn_t, 'r^-', 'LineWidth',1.5, 'MarkerSize',8);
xticks(1:length(plot_names)); xticklabels(plot_names);
ylabel('Transition Accuracy (%)');
title(sprintf('Per-Subject TRANSITION Accuracy at 85%% Gait Phase (%s)', ground_to_test));
legend({'LDA (no temporal)','DBN (35%% prior)'}, 'Location','southeast');
ylim([0 105]); grid on;

subplot(2,1,2);
hold on;
plot(1:length(plot_names), plot_lda_s, 'bo-', 'LineWidth',1.5, 'MarkerSize',8);
plot(1:length(plot_names), plot_dbn_s, 'r^-', 'LineWidth',1.5, 'MarkerSize',8);
xticks(1:length(plot_names)); xticklabels(plot_names);
ylabel('Steady State Accuracy (%)');
title(sprintf('Per-Subject STEADY STATE Accuracy at 35%% Gait Phase (%s)', ground_to_test));
legend({'LDA (no temporal)','DBN (35%% prior)'}, 'Location','southeast');
ylim([0 105]); grid on;

saveas(gcf, fullfile(output_folder, 'per_subject_accuracy.png'));

%% ========================================================
%% STEP 6: SAVE RESULTS
%% ========================================================

fprintf('STEP 6: SAVING RESULTS\n\n');

txt_out = fullfile(output_folder, 'hypothesis_test_results.txt');
fid = fopen(txt_out, 'w');

fprintf(fid, '=======================================================\n');
fprintf(fid, 'HYPOTHESIS TEST: GAIT PHASE TEMPORAL ADVANTAGE\n');
fprintf(fid, 'Ground   : %s\n', ground_to_test);
fprintf(fid, 'Features : DBN-selected (%d features)\n', length(selected_features));
fprintf(fid, 'CV       : Per-subject Leave-One-Trial-Out\n');
fprintf(fid, 'DBN      : 35%% prediction as prior for 85%% classification\n');
fprintf(fid, 'Transition: steps where label differs @35%% vs @85%%\n');
fprintf(fid, '=======================================================\n\n');

fprintf(fid, 'Test steps — steady: %d | transition: %d\n\n', ...
    sum(steady_mask), sum(trans_mask));

fprintf(fid, '%-6s  %12s  %15s  %15s\n', ...
    'Model','Overall (%)','Steady St.(%)','Transition(%)');
fprintf(fid, '%s\n', repmat('-',1,55));
for i = 1:4
    fprintf(fid, '%-6s  %12.2f  %15.2f  %15.2f\n', ...
        models{i}, overall_acc(i), steady_acc(i), transition_acc(i));
end
fprintf(fid, '%s\n\n', repmat('-',1,55));

fprintf(fid, 'DBN advantage AT TRANSITIONS:\n');
for i = 1:3
    fprintf(fid, '  DBN vs %-4s : %+.2f%%\n', models{i}, ...
        transition_acc(4)-transition_acc(i));
end
fprintf(fid, '\nDBN advantage AT STEADY STATE:\n');
for i = 1:3
    fprintf(fid, '  DBN vs %-4s : %+.2f%%\n', models{i}, ...
        steady_acc(4)-steady_acc(i));
end

fprintf(fid, '\nPer-subject results:\n');
fprintf(fid, '%-8s  %11s  %11s  %11s  %11s  %11s  %11s\n', ...
    'Subject','Overall LDA','Overall DBN',...
    'Steady LDA','Steady DBN','Trans LDA','Trans DBN');
fprintf(fid, '%s\n', repmat('-',1,80));
for k = 1:length(valid_idx)
    si = valid_idx(k);
    fprintf(fid, '%-8s  %11.2f  %11.2f  %11.2f  %11.2f  %11.2f  %11.2f\n', ...
        subjects{si}, ...
        subj_overall_lda(si), subj_overall_dbn(si), ...
        subj_steady_lda(si),  subj_steady_dbn(si), ...
        subj_trans_lda(si),   subj_trans_dbn(si));
end

fprintf(fid, '\nHYPOTHESIS VERDICT:\n');
trans_gap  = min(transition_acc(4) - transition_acc(1:3));
steady_gap = max(abs(steady_acc(4) - steady_acc(1:3)));
if trans_gap > 2 && steady_gap < 2
    fprintf(fid, 'SUPPORTED:\n');
    fprintf(fid, '  DBN better at transitions (+%.2f%% min advantage)\n', trans_gap);
    fprintf(fid, '  Similar at steady state (%.2f%% max difference)\n', steady_gap);
elseif trans_gap > 0
    fprintf(fid, 'PARTIALLY SUPPORTED:\n');
    fprintf(fid, '  DBN better at transitions (+%.2f%% min)\n', trans_gap);
    fprintf(fid, '  Steady state difference: %.2f%%\n', steady_gap);
else
    fprintf(fid, 'NOT SUPPORTED:\n');
    fprintf(fid, '  DBN shows no clear transition advantage\n');
    fprintf(fid, '  Max DBN gain at transitions: %.2f%%\n', ...
        max(transition_acc(4) - transition_acc(1:3)));
end

fclose(fid);

fprintf('Saved: %s\n', txt_out);
fprintf('\n=======================================================\n');
fprintf('HYPOTHESIS TEST COMPLETE!\n');
fprintf('=======================================================\n');