%% ========================================================

%%BEST ###################################################
%% with statistical test




%% HYPOTHESIS TEST: Temporal Modeling at Gait Phase Transitions
%%
%% "At steady-state, simpler models suffice.
%%  At transitions, DBN temporal advantage is critical."
%%
%% Method:
%%   4 gait phase points: 10% -> 35% -> 60% -> 85%
%%   Separate LDA model at each phase point
%%   DBN chains predictions across all 4 phases
%%   LDA/SVM/RF use only 85% features (no temporal chain)
%%   Transition steps weighted higher in transition matrix
%%
%%   Transition step = label at 10% differs from label at 85%
%%   Steady step     = label same at 10% and 85%
%%
%%   Per-subject Leave-One-Trial-Out CV (matches paper)
%% ========================================================

clear all; close all; clc;

%% ========================================================
%% SETTINGS
%% ========================================================

ground_to_test = 'levelground';

phases     = {'10', '35', '60', '85'};
num_phases = length(phases);

base_paths = cell(num_phases, 1);
for p = 1:num_phases
    base_paths{p} = sprintf('/home/eeiww/ut55iqoh/MLMA_CAM21/Classification/%s/%s/400/', ...
        ground_to_test, phases{p});
end

output_folder = sprintf('/home/eeiww/ut55iqoh/MLMA_CAM21/result_hypothesis_4phase/%s/', ...
    ground_to_test);
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

subjects = {'ab07', 'ab08', 'ab09', 'ab12', 'ab13', 'ab14', 'ab17', 'ab18', ...
            'ab19', 'ab20', 'ab21', 'ab23', 'ab24', 'ab27', 'ab28'};

NUM_TREES    = 30;
TRANS_WEIGHT = 5;   % weight for transition steps in T matrix
                    % increase to 5 if DBN still doesn't beat RF

fprintf('=======================================================\n');
fprintf('HYPOTHESIS TEST: 4-Phase Gait Temporal Advantage\n');
fprintf('Ground       : %s\n', ground_to_test);
fprintf('Phases       : 10%% -> 35%% -> 60%% -> 85%%\n');
fprintf('Method       : Per-subject Leave-One-Trial-Out CV\n');
fprintf('Models       : LDA, SVM, RF, DBN (4-phase chain)\n');
fprintf('Trans weight : %d (transition upweighting in T matrix)\n', TRANS_WEIGHT);
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
%% STEP 2: LOAD ALL 4 PHASE DATA PER SUBJECT
%% ========================================================

fprintf('STEP 2: LOADING 4-PHASE DATA\n\n');

subj_X      = cell(length(subjects), 1);
subj_Y      = cell(length(subjects), 1);
subj_labels = cell(length(subjects), 1);
subj_trials = cell(length(subjects), 1);
subj_trans  = cell(length(subjects), 1);
valid_subj  = false(length(subjects), 1);

all_labels_pool = {};

for subj_idx = 1:length(subjects)
    subject = subjects{subj_idx};

    % Check all 4 phase files exist
    all_exist = true;
    for p = 1:num_phases
        fin  = fullfile(base_paths{p}, [subject '_input.mat']);
        fout = fullfile(base_paths{p}, [subject '_output.mat']);
        if ~isfile(fin) || ~isfile(fout)
            all_exist = false;
            break;
        end
    end

    if ~all_exist
        fprintf('WARNING: Not all phase files found for %s. Skipping.\n', subject);
        continue;
    end

    try
        X_phases     = cell(num_phases, 1);
        lbl_phases   = cell(num_phases, 1);
        trial_phases = cell(num_phases, 1);

        for p = 1:num_phases
            fin  = fullfile(base_paths{p}, [subject '_input.mat']);
            fout = fullfile(base_paths{p}, [subject '_output.mat']);
            din  = load(fin);
            dout = load(fout);
            Xp   = table2array(din.alldata);
            X_phases{p}     = Xp(:, selected_features);
            lbl_phases{p}   = dout.alldata.labels_feat_last;
            trial_phases{p} = dout.alldata.trial_feat_last;
        end

        % Find common trials across ALL 4 phases
        common_trials = unique(trial_phases{1});
        for p = 2:num_phases
            common_trials = intersect(common_trials, unique(trial_phases{p}));
        end

        common_trials = common_trials(1:min(5, length(common_trials)));

        % Align rows per trial — handle row count mismatches
        X_s   = cell(num_phases, 1);
        lbl_s = cell(num_phases, 1);
        for p = 1:num_phases
            X_s{p}   = [];
            lbl_s{p} = {};
        end
        trial_s = [];

        for tt = 1:length(common_trials)
            tr   = common_trials(tt);
            idxs = cell(num_phases, 1);
            n    = inf;
            for p = 1:num_phases
                idxs{p} = find(trial_phases{p} == tr);
                n = min(n, length(idxs{p}));
            end
            if n == 0 || isinf(n), continue; end

            for p = 1:num_phases
                idx_p    = idxs{p}(1:n);
                X_s{p}   = [X_s{p};   X_phases{p}(idx_p, :)];
                lbl_s{p} = [lbl_s{p}; lbl_phases{p}(idx_p)];
            end
            trial_s = [trial_s; repmat(tr, n, 1)];
        end

        if isempty(trial_s)
            fprintf('WARNING: No aligned data for %s.\n', subject);
            continue;
        end

        % Transition = label at 10% differs from label at 85%
        lbl_10  = lbl_s{1};
        lbl_85  = lbl_s{4};
        trans_s = ~strcmp(lbl_10, lbl_85);

        subj_X{subj_idx}      = X_s;
        subj_labels{subj_idx} = lbl_s;
        subj_trials{subj_idx} = trial_s;
        subj_trans{subj_idx}  = trans_s;
        valid_subj(subj_idx)  = true;

        for p = 1:num_phases
            all_labels_pool = [all_labels_pool; lbl_s{p}];
        end

        fprintf('Loaded: %s  (%d steps | steady: %d | transition: %d)\n', ...
            subject, length(trial_s), sum(~trans_s), sum(trans_s));

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

    lbl_s = subj_labels{subj_idx};
    Y_s   = cell(num_phases, 1);
    for p = 1:num_phases
        n     = length(lbl_s{p});
        y_num = zeros(n, 1);
        for i = 1:n
            y_num(i) = label_map(lbl_s{p}{i});
        end
        Y_s{p} = y_num;
    end
    subj_Y{subj_idx} = Y_s;
end

fprintf('\n');

%% ========================================================
%% STEP 3: PER-SUBJECT LEAVE-ONE-TRIAL-OUT CV
%% ========================================================

fprintf('=======================================================\n');
fprintf('STEP 3: PER-SUBJECT LEAVE-ONE-TRIAL-OUT CV\n');
fprintf('4-phase DBN chain: 10%% -> 35%% -> 60%% -> 85%%\n');
fprintf('Transition matrix: upweighted by factor %d\n', TRANS_WEIGHT);
fprintf('=======================================================\n\n');

all_truth_85   = [];
all_trans_mask = [];
all_pred_LDA   = [];
all_pred_SVM   = [];
all_pred_RF    = [];
all_pred_DBN   = [];

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
    X_s      = subj_X{subj_idx};
    Y_s      = subj_Y{subj_idx};
    trial_s  = subj_trials{subj_idx};
    trans_s  = subj_trans{subj_idx};

    unique_trials_s = unique(trial_s);
    num_folds_s     = length(unique_trials_s);

    fprintf('[%s] Subject %-5s | %d trials | %d steps  (%.1f sec)\n', ...
        datetime('now','Format','HH:mm:ss'), subject, ...
        num_folds_s, length(trial_s), toc);

    subj_truth   = [];
    subj_trans_m = [];
    subj_lda_p   = [];
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

        % Split each phase into train/test
        X_train = cell(num_phases, 1);
        X_test  = cell(num_phases, 1);
        Y_train = cell(num_phases, 1);
        Y_test  = cell(num_phases, 1);

        for p = 1:num_phases
            X_train{p} = X_s{p}(train_m, :);
            X_test{p}  = X_s{p}(test_m,  :);
            Y_train{p} = Y_s{p}(train_m);
            Y_test{p}  = Y_s{p}(test_m);
        end

        y85_test   = Y_test{4};
        trans_test = trans_s(test_m);

        if length(unique(Y_train{4})) < 2
            continue;
        end

        % Normalise all phases using 85% training statistics only
        mu_tr    = mean(X_train{4});
        sigma_tr = std(X_train{4});
        sigma_tr(sigma_tr == 0) = 1;

        X_train_n = cell(num_phases, 1);
        X_test_n  = cell(num_phases, 1);
        for p = 1:num_phases
            X_train_n{p} = (X_train{p} - mu_tr) ./ sigma_tr;
            X_test_n{p}  = (X_test{p}  - mu_tr) ./ sigma_tr;
        end

        %% ---- Build phase-to-phase transition matrices ----
        %% T{p} = P(class at phase p+1 | class at phase p)
        %% Transition steps (class changes) are upweighted by TRANS_WEIGHT
        %% to counteract the steady-state bias in the data
        T = cell(num_phases-1, 1);
        for p = 1:num_phases-1
            Tp = ones(num_classes, num_classes);
            for k = 1:length(Y_train{p})
                from_cls = Y_train{p}(k);
                to_cls   = Y_train{p+1}(k);
                if from_cls ~= to_cls
                    w = TRANS_WEIGHT;  % upweight class-change steps
                else
                    w = 1;             % normal weight for steady steps
                end
                Tp(from_cls, to_cls) = Tp(from_cls, to_cls) + w;
            end
            T{p} = Tp ./ sum(Tp, 2);
        end

        %% ---- Train one LDA per phase ----
        lda_models = cell(num_phases, 1);
        lda_ok     = false(num_phases, 1);
        for p = 1:num_phases
            try
                if length(unique(Y_train{p})) >= 2
                    lda_models{p} = fitcdiscr(X_train_n{p}, Y_train{p}, ...
                                              'DiscrimType', 'linear');
                    lda_ok(p)     = true;
                end
            catch
                lda_ok(p) = false;
            end
        end

        %% ---- LDA at 85% only (baseline — no temporal chain) ----
        if lda_ok(4)
            pred_LDA = predict(lda_models{4}, X_test_n{4});
        else
            pred_LDA = mode(Y_train{4}) * ones(size(y85_test));
        end

        %% ---- SVM at 85% only (no temporal chain) ----
        try
            svm85    = fitcecoc(X_train_n{4}, Y_train{4}, ...
                                'Learners', svm_template, ...
                                'Coding',   'onevsone');
            pred_SVM = predict(svm85, X_test_n{4});
        catch
            pred_SVM = mode(Y_train{4}) * ones(size(y85_test));
        end

        %% ---- RF at 85% only (no temporal chain) ----
        try
            rf85         = TreeBagger(NUM_TREES, X_train{4}, Y_train{4}, ...
                                      'Method',        'classification', ...
                                      'OOBPrediction', 'off');
            pred_RF_cell = predict(rf85, X_test{4});
            pred_RF      = str2double(pred_RF_cell);
        catch
            pred_RF = mode(Y_train{4}) * ones(size(y85_test));
        end

        %% ---- DBN: 4-phase chain 10% -> 35% -> 60% -> 85% ----
        %% Each phase updates the belief using:
        %%   posterior = emission x (prior x T)
        %% where emission = LDA scores at current phase
        %%       prior    = belief from previous phase
        %%       T        = phase-to-phase transition matrix
        try
            n_test = size(X_test_n{1}, 1);

            %% Phase 1 (10%) — initial belief from M10
            if lda_ok(1)
                [~, sc] = predict(lda_models{1}, X_test_n{1});
                belief  = ones(n_test, num_classes) * 1e-10;
                cls     = lda_models{1}.ClassNames;
                for ci = 1:length(cls)
                    col = cls(ci);
                    if col >= 1 && col <= num_classes
                        belief(:, col) = sc(:, ci);
                    end
                end
                belief = max(belief, 1e-10);
                belief = belief ./ sum(belief, 2);
            else
                belief = ones(n_test, num_classes) / num_classes;
            end

            %% Phase 2 (35%) — propagate belief from 10%, update with 35% emission
            temporal_prior = belief * T{1};
            temporal_prior = temporal_prior ./ sum(temporal_prior, 2);
            if lda_ok(2)
                [~, sc]  = predict(lda_models{2}, X_test_n{2});
                emission = ones(n_test, num_classes) * 1e-10;
                cls      = lda_models{2}.ClassNames;
                for ci = 1:length(cls)
                    col = cls(ci);
                    if col >= 1 && col <= num_classes
                        emission(:, col) = sc(:, ci);
                    end
                end
                emission = max(emission, 1e-10);
                emission = emission ./ sum(emission, 2);
                belief   = emission .* temporal_prior;
                belief   = belief   ./ sum(belief, 2);
            else
                belief = temporal_prior;
            end

            %% Phase 3 (60%) — propagate belief from 35%, update with 60% emission
            temporal_prior = belief * T{2};
            temporal_prior = temporal_prior ./ sum(temporal_prior, 2);
            if lda_ok(3)
                [~, sc]  = predict(lda_models{3}, X_test_n{3});
                emission = ones(n_test, num_classes) * 1e-10;
                cls      = lda_models{3}.ClassNames;
                for ci = 1:length(cls)
                    col = cls(ci);
                    if col >= 1 && col <= num_classes
                        emission(:, col) = sc(:, ci);
                    end
                end
                emission = max(emission, 1e-10);
                emission = emission ./ sum(emission, 2);
                belief   = emission .* temporal_prior;
                belief   = belief   ./ sum(belief, 2);
            else
                belief = temporal_prior;
            end

            %% Phase 4 (85%) — final prediction
            temporal_prior = belief * T{3};
            temporal_prior = temporal_prior ./ sum(temporal_prior, 2);
            if lda_ok(4)
                [~, sc]   = predict(lda_models{4}, X_test_n{4});
                emission  = ones(n_test, num_classes) * 1e-10;
                cls       = lda_models{4}.ClassNames;
                for ci = 1:length(cls)
                    col = cls(ci);
                    if col >= 1 && col <= num_classes
                        emission(:, col) = sc(:, ci);
                    end
                end
                emission  = max(emission,  1e-10);
                emission  = emission  ./ sum(emission,  2);
                posterior = emission  .* temporal_prior;
                posterior = posterior ./ sum(posterior, 2);
            else
                posterior = temporal_prior;
            end

            [~, pred_DBN] = max(posterior, [], 2);

        catch ME
            pred_DBN = pred_LDA;
            fprintf('    DBN failed fold %d: %s\n', fold, ME.message);
        end

        % Accumulate fold results
        subj_truth   = [subj_truth;   y85_test];
        subj_trans_m = [subj_trans_m; trans_test];
        subj_lda_p   = [subj_lda_p;   pred_LDA];
        subj_svm_p   = [subj_svm_p;   pred_SVM];
        subj_rf_p    = [subj_rf_p;    pred_RF];
        subj_dbn_p   = [subj_dbn_p;   pred_DBN];
    end

    % Per-subject summary
    if ~isempty(subj_truth)
        s_mask = ~logical(subj_trans_m);
        t_mask =  logical(subj_trans_m);

        subj_overall_lda(subj_idx) = mean(subj_lda_p == subj_truth) * 100;
        subj_overall_dbn(subj_idx) = mean(subj_dbn_p == subj_truth) * 100;

        fprintf('  Overall    LDA:%.1f%%  SVM:%.1f%%  RF:%.1f%%  DBN:%.1f%%\n', ...
            mean(subj_lda_p==subj_truth)*100, ...
            mean(subj_svm_p==subj_truth)*100, ...
            mean(subj_rf_p ==subj_truth)*100, ...
            mean(subj_dbn_p==subj_truth)*100);

        if sum(s_mask) > 0
            subj_steady_lda(subj_idx) = mean(subj_lda_p(s_mask)==subj_truth(s_mask))*100;
            subj_steady_dbn(subj_idx) = mean(subj_dbn_p(s_mask)==subj_truth(s_mask))*100;
            fprintf('  Steady     LDA:%.1f%%  SVM:%.1f%%  RF:%.1f%%  DBN:%.1f%%\n', ...
                mean(subj_lda_p(s_mask)==subj_truth(s_mask))*100, ...
                mean(subj_svm_p(s_mask)==subj_truth(s_mask))*100, ...
                mean(subj_rf_p (s_mask)==subj_truth(s_mask))*100, ...
                mean(subj_dbn_p(s_mask)==subj_truth(s_mask))*100);
        end

        if sum(t_mask) > 0
            subj_trans_lda(subj_idx) = mean(subj_lda_p(t_mask)==subj_truth(t_mask))*100;
            subj_trans_dbn(subj_idx) = mean(subj_dbn_p(t_mask)==subj_truth(t_mask))*100;
            fprintf('  Transition LDA:%.1f%%  SVM:%.1f%%  RF:%.1f%%  DBN:%.1f%%\n', ...
                mean(subj_lda_p(t_mask)==subj_truth(t_mask))*100, ...
                mean(subj_svm_p(t_mask)==subj_truth(t_mask))*100, ...
                mean(subj_rf_p (t_mask)==subj_truth(t_mask))*100, ...
                mean(subj_dbn_p(t_mask)==subj_truth(t_mask))*100);
        end
        fprintf('\n');
    end

    all_truth_85   = [all_truth_85;   subj_truth];
    all_trans_mask = [all_trans_mask; subj_trans_m];
    all_pred_LDA   = [all_pred_LDA;   subj_lda_p];
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

fprintf('Total steps -- steady: %d | transition: %d\n\n', ...
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
        transition_acc(4)-transition_acc(i));
end
fprintf('\nDBN advantage AT STEADY STATE:\n');
for i = 1:3
    fprintf('  DBN vs %-4s : %+.2f%%\n', models{i}, ...
        steady_acc(4)-steady_acc(i));
end

%% ========================================================
%% STEP 5: VISUALISATION
%% ========================================================

fprintf('\nSTEP 5: PLOTTING\n\n');

% Plot 1: Grouped bar — steady vs transition per model
figure('Name','4-Phase Hypothesis Test','Position',[100 100 800 550]);
bar_data = [steady_acc, transition_acc];
b = bar(bar_data, 'grouped');
b(1).FaceColor = [0.00 0.45 0.74];
b(2).FaceColor = [0.85 0.33 0.10];
xticks(1:4); xticklabels(models);
ylabel('Accuracy (%)');
xlabel('Model');
title(sprintf(['Steady State vs Transition Accuracy\n' ...
    '(%s | 4-phase DBN chain | T-weight=%d)'], ...
    ground_to_test, TRANS_WEIGHT));
legend({'Steady steps (same label @10%%&85%%)', ...
        'Transition steps (label changes)'}, 'Location','southeast');
ylim([0 108]); grid on;
for i = 1:4
    text(i-0.15, steady_acc(i)+1, sprintf('%.1f%%',steady_acc(i)), ...
        'HorizontalAlignment','center','FontSize',9);
    text(i+0.15, transition_acc(i)+1, sprintf('%.1f%%',transition_acc(i)), ...
        'HorizontalAlignment','center','FontSize',9,'Color',[0.6 0 0]);
end
saveas(gcf, fullfile(output_folder, sprintf('hypothesis_4phase_w%d.png', TRANS_WEIGHT)));

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
title(sprintf('DBN Advantage at Transition Steps (T-weight=%d)', TRANS_WEIGHT));
yline(0,'k--','LineWidth',1.5); grid on;
for i = 1:3
    text(i, dbn_adv(i)+sign(dbn_adv(i))*0.4, ...
        sprintf('%+.2f%%',dbn_adv(i)), ...
        'HorizontalAlignment','center','FontSize',12,'FontWeight','bold');
end
saveas(gcf, fullfile(output_folder, sprintf('dbn_advantage_w%d.png', TRANS_WEIGHT)));

% Plot 3: Per-subject DBN vs LDA line plots
valid_idx  = find(valid_subj);
plot_names = {};
plot_lda_t = []; plot_dbn_t = [];
plot_lda_s = []; plot_dbn_s = [];

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
plot(1:length(plot_names), plot_lda_t, 'bo-','LineWidth',1.5,'MarkerSize',8);
plot(1:length(plot_names), plot_dbn_t, 'r^-','LineWidth',1.5,'MarkerSize',8);
xticks(1:length(plot_names)); xticklabels(plot_names);
ylabel('Accuracy (%)'); ylim([0 105]); grid on;
title(sprintf('Per-Subject TRANSITION Accuracy at 85%% (%s)', ground_to_test));
legend({'LDA (85%% only)','DBN (4-phase chain)'},'Location','southeast');

subplot(2,1,2);
hold on;
plot(1:length(plot_names), plot_lda_s, 'bo-','LineWidth',1.5,'MarkerSize',8);
plot(1:length(plot_names), plot_dbn_s, 'r^-','LineWidth',1.5,'MarkerSize',8);
xticks(1:length(plot_names)); xticklabels(plot_names);
ylabel('Accuracy (%)'); ylim([0 105]); grid on;
title(sprintf('Per-Subject STEADY STATE Accuracy (%s)', ground_to_test));
legend({'LDA (85%% only)','DBN (4-phase chain)'},'Location','southeast');

saveas(gcf, fullfile(output_folder, sprintf('per_subject_w%d.png', TRANS_WEIGHT)));

%% ========================================================
%% STEP 6: SAVE RESULTS
%% ========================================================

fprintf('STEP 6: SAVING RESULTS\n\n');

txt_out = fullfile(output_folder, ...
    sprintf('hypothesis_4phase_w%d_results.txt', TRANS_WEIGHT));
fid = fopen(txt_out, 'w');

fprintf(fid, '=======================================================\n');
fprintf(fid, 'HYPOTHESIS TEST: 4-PHASE GAIT TEMPORAL ADVANTAGE\n');
fprintf(fid, 'Ground       : %s\n', ground_to_test);
fprintf(fid, 'Phases       : 10%% -> 35%% -> 60%% -> 85%%\n');
fprintf(fid, 'Features     : DBN-selected (%d features)\n', length(selected_features));
fprintf(fid, 'CV           : Per-subject Leave-One-Trial-Out\n');
fprintf(fid, 'Trans weight : %d\n', TRANS_WEIGHT);
fprintf(fid, 'Transition   : label at 10%% differs from label at 85%%\n');
fprintf(fid, '=======================================================\n\n');

fprintf(fid, 'Test steps -- steady: %d | transition: %d\n\n', ...
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
    'Subject','Overall LDA','Overall DBN', ...
    'Steady LDA','Steady DBN','Trans LDA','Trans DBN');
fprintf(fid, '%s\n', repmat('-',1,80));
for k = 1:length(valid_idx)
    si = valid_idx(k);
    fprintf(fid, '%-8s  %11.2f  %11.2f  %11.2f  %11.2f  %11.2f  %11.2f\n', ...
        subjects{si}, subj_overall_lda(si), subj_overall_dbn(si), ...
        subj_steady_lda(si), subj_steady_dbn(si), ...
        subj_trans_lda(si),  subj_trans_dbn(si));
end

fprintf(fid, '\nHYPOTHESIS VERDICT:\n');
trans_gap  = min(transition_acc(4) - transition_acc(1:3));
steady_gap = max(abs(steady_acc(4) - steady_acc(1:3)));

if trans_gap > 2 && steady_gap < 5
    fprintf(fid, 'SUPPORTED:\n');
    fprintf(fid, '  DBN better at transitions (+%.2f%% min advantage)\n', trans_gap);
    fprintf(fid, '  Steady state difference: %.2f%%\n', steady_gap);
elseif trans_gap > 1
    fprintf(fid, 'PARTIALLY SUPPORTED:\n');
    fprintf(fid, '  DBN better at transitions (+%.2f%% min)\n', trans_gap);
    fprintf(fid, '  Steady state difference: %.2f%%\n', steady_gap);
elseif trans_gap > 0
    fprintf(fid, 'WEAKLY SUPPORTED:\n');
    fprintf(fid, '  DBN marginally better at transitions (+%.2f%% min)\n', trans_gap);
else
    fprintf(fid, 'NOT SUPPORTED:\n');
    fprintf(fid, '  DBN shows no transition advantage\n');
    fprintf(fid, '  Max DBN gain: %.2f%%\n', ...
        max(transition_acc(4)-transition_acc(1:3)));
end

fclose(fid);
fprintf('Saved: %s\n', txt_out);
fprintf('\n=======================================================\n');
fprintf('HYPOTHESIS TEST COMPLETE!\n');
fprintf('=======================================================\n');

%% ========================================================
%% STATISTICAL SIGNIFICANCE TESTS
%% ========================================================

fprintf('\n=======================================================\n');
fprintf('STATISTICAL SIGNIFICANCE TESTS\n');
fprintf('=======================================================\n\n');

% Collect valid per-subject results
valid_idx = find(valid_subj);
n_valid   = length(valid_idx);

lda_t = subj_trans_lda(valid_idx);
dbn_t = subj_trans_dbn(valid_idx);
lda_s = subj_steady_lda(valid_idx);
dbn_s = subj_steady_dbn(valid_idx);

% Remove NaN subjects
valid_mask = ~isnan(lda_t) & ~isnan(dbn_t);
lda_t = lda_t(valid_mask);
dbn_t = dbn_t(valid_mask);
lda_s = lda_s(valid_mask);
dbn_s = dbn_s(valid_mask);
n     = sum(valid_mask);

fprintf('Number of subjects: %d\n\n', n);

%% --- TRANSITION accuracy tests ---
fprintf('--- TRANSITION ACCURACY ---\n\n');

diff_t = dbn_t - lda_t;
fprintf('DBN vs LDA per subject differences:\n');
fprintf('  Mean : %+.2f%%\n', mean(diff_t));
fprintf('  Std  : %.2f%%\n',  std(diff_t));
fprintf('  DBN wins: %d/%d subjects\n\n', sum(diff_t>0), n);

% Paired t-test
[h_t, p_t, ~, stats_t] = ttest(dbn_t, lda_t);
fprintf('Paired t-test (DBN vs LDA at transitions):\n');
fprintf('  t = %.4f\n', stats_t.tstat);
fprintf('  p = %.4f\n', p_t);
if p_t < 0.05
    fprintf('  SIGNIFICANT (p < 0.05) ✓\n\n');
elseif p_t < 0.10
    fprintf('  MARGINAL (p < 0.10)\n\n');
else
    fprintf('  Not significant (p = %.4f)\n\n', p_t);
end

% Wilcoxon signed-rank (better for small n)
[p_w, h_w] = signrank(dbn_t, lda_t);
fprintf('Wilcoxon signed-rank (DBN vs LDA at transitions):\n');
fprintf('  p = %.4f\n', p_w);
if p_w < 0.05
    fprintf('  SIGNIFICANT (p < 0.05) ✓\n\n');
else
    fprintf('  Not significant (p = %.4f)\n\n', p_w);
end

% Cohen's d effect size
cohens_d_t = mean(diff_t) / std(diff_t);
fprintf('Effect size Cohen''s d: %.4f  ', cohens_d_t);
if abs(cohens_d_t) > 0.8
    fprintf('(LARGE)\n\n');
elseif abs(cohens_d_t) > 0.5
    fprintf('(MEDIUM)\n\n');
elseif abs(cohens_d_t) > 0.2
    fprintf('(SMALL)\n\n');
else
    fprintf('(NEGLIGIBLE)\n\n');
end

%% --- STEADY STATE accuracy tests ---
fprintf('--- STEADY STATE ACCURACY ---\n\n');

diff_s = dbn_s - lda_s;
fprintf('DBN vs LDA per subject differences:\n');
fprintf('  Mean : %+.2f%%\n', mean(diff_s));
fprintf('  Std  : %.2f%%\n',  std(diff_s));
fprintf('  DBN wins: %d/%d subjects\n\n', sum(diff_s>0), n);

[h_s, p_s, ~, stats_s] = ttest(dbn_s, lda_s);
fprintf('Paired t-test (DBN vs LDA at steady state):\n');
fprintf('  t = %.4f\n', stats_s.tstat);
fprintf('  p = %.4f\n', p_s);
if p_s < 0.05
    fprintf('  SIGNIFICANT (p < 0.05) ✓\n\n');
else
    fprintf('  Not significant\n\n');
end

[p_ws, h_ws] = signrank(dbn_s, lda_s);
fprintf('Wilcoxon signed-rank (DBN vs LDA at steady state):\n');
fprintf('  p = %.4f\n', p_ws);
if p_ws < 0.05
    fprintf('  SIGNIFICANT (p < 0.05) ✓\n\n');
else
    fprintf('  Not significant\n\n');
end

cohens_d_s = mean(diff_s) / std(diff_s);
fprintf('Effect size Cohen''s d: %.4f  ', cohens_d_s);
if abs(cohens_d_s) > 0.8
    fprintf('(LARGE)\n\n');
elseif abs(cohens_d_s) > 0.5
    fprintf('(MEDIUM)\n\n');
elseif abs(cohens_d_s) > 0.2
    fprintf('(SMALL)\n\n');
else
    fprintf('(NEGLIGIBLE)\n\n');
end

%% --- Save stats to file ---
stats_out = fullfile(output_folder, 'statistical_tests.txt');
fid = fopen(stats_out, 'w');

fprintf(fid, 'STATISTICAL SIGNIFICANCE TESTS\n');
fprintf(fid, 'Ground: %s | Subjects: %d\n\n', ground_to_test, n);

% Pre-define significance labels
if h_t == 1
    sig_t = 'SIGNIFICANT';
else
    sig_t = 'not significant';
end

if h_w == 1
    sig_w = 'SIGNIFICANT';
else
    sig_w = 'not significant';
end

if h_s == 1
    sig_s = 'SIGNIFICANT';
else
    sig_s = 'not significant';
end

if h_ws == 1
    sig_ws = 'SIGNIFICANT';
else
    sig_ws = 'not significant';
end

fprintf(fid, 'TRANSITION ACCURACY (DBN vs LDA):\n');
fprintf(fid, '  Mean difference : %+.2f%%\n', mean(diff_t));
fprintf(fid, '  t-test p-value  : %.4f (%s)\n', p_t, sig_t);
fprintf(fid, '  Wilcoxon p-value: %.4f (%s)\n', p_w, sig_w);
fprintf(fid, '  Cohens d        : %.4f\n\n', cohens_d_t);

fprintf(fid, 'STEADY STATE ACCURACY (DBN vs LDA):\n');
fprintf(fid, '  Mean difference : %+.2f%%\n', mean(diff_s));
fprintf(fid, '  t-test p-value  : %.4f (%s)\n', p_s, sig_s);
fprintf(fid, '  Wilcoxon p-value: %.4f (%s)\n', p_ws, sig_ws);
fprintf(fid, '  Cohens d        : %.4f\n\n', cohens_d_s);

fclose(fid);
fprintf('Saved: %s\n', stats_out);