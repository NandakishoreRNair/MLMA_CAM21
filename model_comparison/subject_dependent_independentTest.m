%% ========================================================
%% SUBJECT-DEPENDENT vs SUBJECT-INDEPENDENT COMPARISON
%%
%% Tests how well each model generalizes to new subjects
%% Paper only tested subject-dependent models
%% This is a novel contribution extending the paper
%%
%% Subject-Dependent:
%%   Train on subject's OWN trials → test on same subject
%%   Leave-One-Trial-Out CV per subject
%%
%% Subject-Independent:
%%   Train on OTHER subjects → test on new subject
%%   Leave-One-Subject-Out CV
%%
%% Models: LDA, SVM, RF, DBN
%% Data: 4 gait phases (10%, 35%, 60%, 85%)
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

output_folder = sprintf('/home/eeiww/ut55iqoh/MLMA_CAM21/result_subj_dep_indep/%s/', ...
    ground_to_test);
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

subjects = {'ab07', 'ab08', 'ab09', 'ab12', 'ab13', 'ab14', 'ab17', 'ab18', ...
            'ab19', 'ab20', 'ab21', 'ab23', 'ab24', 'ab27', 'ab28'};

NUM_TREES    = 30;
TRANS_WEIGHT = 3;

fprintf('=======================================================\n');
fprintf('SUBJECT-DEPENDENT vs SUBJECT-INDEPENDENT COMPARISON\n');
fprintf('Ground  : %s\n', ground_to_test);
fprintf('Phases  : 10%% -> 35%% -> 60%% -> 85%%\n');
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

        % Find common trials across all 4 phases
        common_trials = unique(trial_phases{1});
        for p = 2:num_phases
            common_trials = intersect(common_trials, unique(trial_phases{p}));
        end

        % Keep only first 5 trials
        common_trials = common_trials(1:min(5, length(common_trials)));

        % Align rows per trial
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

% Global label encoding
all_labels  = unique(all_labels_pool);
num_classes = length(all_labels);
label_map   = containers.Map(all_labels, 1:num_classes);

fprintf('\nClasses (%d): ', num_classes);
for i = 1:num_classes
    fprintf('%s ', all_labels{i});
end
fprintf('\n\n');

% Convert to numeric
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

%% ========================================================
%% STEP 3: SUBJECT-DEPENDENT CV
%%
%% For each subject:
%%   Train on OWN trials (leave one trial out)
%%   Test on left-out trial
%%   → Best case scenario — model knows this subject
%% ========================================================

fprintf('=======================================================\n');
fprintf('STEP 3: SUBJECT-DEPENDENT (Leave-One-Trial-Out)\n');
fprintf('=======================================================\n\n');

svm_template = templateSVM('KernelFunction','linear','Standardize',false);

% Storage — per subject overall, steady, transition accuracy
dep_lda  = nan(length(subjects), 3);  % cols: overall, steady, transition
dep_svm  = nan(length(subjects), 3);
dep_rf   = nan(length(subjects), 3);
dep_dbn  = nan(length(subjects), 3);

valid_idx = find(valid_subj);

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

    fprintf('[%s] DEP — Subject %-5s  (%.1f sec)\n', ...
        datetime('now','Format','HH:mm:ss'), subject, toc);

    all_truth   = [];
    all_trans_m = [];
    all_lda     = [];
    all_svm_p   = [];
    all_rf_p    = [];
    all_dbn_p   = [];

    for fold = 1:num_folds_s
        test_trial = unique_trials_s(fold);
        train_m    = trial_s ~= test_trial;
        test_m     = trial_s == test_trial;

        if sum(train_m) < 5 || sum(test_m) < 1, continue; end

        X_train = cell(num_phases,1); X_test = cell(num_phases,1);
        Y_train = cell(num_phases,1); Y_test = cell(num_phases,1);
        for p = 1:num_phases
            X_train{p} = X_s{p}(train_m,:);
            X_test{p}  = X_s{p}(test_m,:);
            Y_train{p} = Y_s{p}(train_m);
            Y_test{p}  = Y_s{p}(test_m);
        end

        y85_test   = Y_test{4};
        trans_test = trans_s(test_m);

        if length(unique(Y_train{4})) < 2, continue; end

        % Normalise using 85% training stats
        mu_tr    = mean(X_train{4});
        sigma_tr = std(X_train{4});
        sigma_tr(sigma_tr==0) = 1;
        X_train_n = cell(num_phases,1);
        X_test_n  = cell(num_phases,1);
        for p = 1:num_phases
            X_train_n{p} = (X_train{p} - mu_tr) ./ sigma_tr;
            X_test_n{p}  = (X_test{p}  - mu_tr) ./ sigma_tr;
        end

        % Transition matrix with upweighting
        T = cell(num_phases-1,1);
        for p = 1:num_phases-1
            Tp = ones(num_classes,num_classes);
            for k = 1:length(Y_train{p})
                fc = Y_train{p}(k); tc = Y_train{p+1}(k);
                if fc ~= tc, w = TRANS_WEIGHT; else, w = 1; end
                Tp(fc,tc) = Tp(fc,tc) + w;
            end
            T{p} = Tp ./ sum(Tp,2);
        end

        % Train LDA per phase
        lda_models = cell(num_phases,1);
        lda_ok     = false(num_phases,1);
        for p = 1:num_phases
            try
                if length(unique(Y_train{p})) >= 2
                    lda_models{p} = fitcdiscr(X_train_n{p}, Y_train{p}, 'DiscrimType','linear');
                    lda_ok(p)     = true;
                end
            catch
            end
        end

        % LDA at 85%
        if lda_ok(4)
            pred_LDA = predict(lda_models{4}, X_test_n{4});
        else
            pred_LDA = mode(Y_train{4})*ones(size(y85_test));
        end

        % SVM at 85%
        try
            svm85    = fitcecoc(X_train_n{4}, Y_train{4}, 'Learners',svm_template,'Coding','onevsone');
            pred_SVM = predict(svm85, X_test_n{4});
        catch
            pred_SVM = mode(Y_train{4})*ones(size(y85_test));
        end

        % RF at 85%
        try
            rf85         = TreeBagger(NUM_TREES, X_train{4}, Y_train{4}, 'Method','classification','OOBPrediction','off');
            pred_RF      = str2double(predict(rf85, X_test{4}));
        catch
            pred_RF = mode(Y_train{4})*ones(size(y85_test));
        end

        % DBN 4-phase chain
        try
            n_test = size(X_test_n{1},1);

            % Phase 1
            if lda_ok(1)
                [~,sc] = predict(lda_models{1}, X_test_n{1});
                belief = ones(n_test,num_classes)*1e-10;
                cls    = lda_models{1}.ClassNames;
                for ci = 1:length(cls)
                    if cls(ci)>=1 && cls(ci)<=num_classes
                        belief(:,cls(ci)) = sc(:,ci);
                    end
                end
                belief = max(belief,1e-10); belief = belief./sum(belief,2);
            else
                belief = ones(n_test,num_classes)/num_classes;
            end

            % Phase 2
            tp = belief*T{1}; tp = tp./sum(tp,2);
            if lda_ok(2)
                [~,sc] = predict(lda_models{2}, X_test_n{2});
                em = ones(n_test,num_classes)*1e-10;
                cls = lda_models{2}.ClassNames;
                for ci=1:length(cls), if cls(ci)>=1&&cls(ci)<=num_classes, em(:,cls(ci))=sc(:,ci); end, end
                em = max(em,1e-10); em = em./sum(em,2);
                belief = em.*tp; belief = belief./sum(belief,2);
            else
                belief = tp;
            end

            % Phase 3
            tp = belief*T{2}; tp = tp./sum(tp,2);
            if lda_ok(3)
                [~,sc] = predict(lda_models{3}, X_test_n{3});
                em = ones(n_test,num_classes)*1e-10;
                cls = lda_models{3}.ClassNames;
                for ci=1:length(cls), if cls(ci)>=1&&cls(ci)<=num_classes, em(:,cls(ci))=sc(:,ci); end, end
                em = max(em,1e-10); em = em./sum(em,2);
                belief = em.*tp; belief = belief./sum(belief,2);
            else
                belief = tp;
            end

            % Phase 4
            tp = belief*T{3}; tp = tp./sum(tp,2);
            if lda_ok(4)
                [~,sc] = predict(lda_models{4}, X_test_n{4});
                em = ones(n_test,num_classes)*1e-10;
                cls = lda_models{4}.ClassNames;
                for ci=1:length(cls), if cls(ci)>=1&&cls(ci)<=num_classes, em(:,cls(ci))=sc(:,ci); end, end
                em = max(em,1e-10); em = em./sum(em,2);
                post = em.*tp; post = post./sum(post,2);
            else
                post = tp;
            end
            [~,pred_DBN] = max(post,[],2);
        catch
            pred_DBN = pred_LDA;
        end

        all_truth   = [all_truth;   y85_test];
        all_trans_m = [all_trans_m; trans_test];
        all_lda     = [all_lda;     pred_LDA];
        all_svm_p   = [all_svm_p;   pred_SVM];
        all_rf_p    = [all_rf_p;    pred_RF];
        all_dbn_p   = [all_dbn_p;   pred_DBN];
    end

    if ~isempty(all_truth)
        s_m = ~logical(all_trans_m);
        t_m =  logical(all_trans_m);

        for mi = 1:4
            preds = {all_lda, all_svm_p, all_rf_p, all_dbn_p};
            pred  = preds{mi};
            switch mi
                case 1, dep_lda(subj_idx,:) = [mean(pred==all_truth) mean(pred(s_m)==all_truth(s_m)) mean(pred(t_m)==all_truth(t_m))]*100;
                case 2, dep_svm(subj_idx,:) = [mean(pred==all_truth) mean(pred(s_m)==all_truth(s_m)) mean(pred(t_m)==all_truth(t_m))]*100;
                case 3, dep_rf(subj_idx,:)  = [mean(pred==all_truth) mean(pred(s_m)==all_truth(s_m)) mean(pred(t_m)==all_truth(t_m))]*100;
                case 4, dep_dbn(subj_idx,:) = [mean(pred==all_truth) mean(pred(s_m)==all_truth(s_m)) mean(pred(t_m)==all_truth(t_m))]*100;
            end
        end

        fprintf('  Overall: LDA=%.1f%% SVM=%.1f%% RF=%.1f%% DBN=%.1f%%\n', ...
            dep_lda(subj_idx,1), dep_svm(subj_idx,1), ...
            dep_rf(subj_idx,1),  dep_dbn(subj_idx,1));
    end
end

fprintf('\n[%s] Subject-dependent complete! (%.1f sec)\n\n', ...
    datetime('now','Format','HH:mm:ss'), toc);

%% ========================================================
%% STEP 4: SUBJECT-INDEPENDENT CV
%%
%% For each subject:
%%   Train on ALL OTHER subjects
%%   Test on this subject (all trials)
%%   → Real-world scenario — model never saw this person
%% ========================================================

fprintf('=======================================================\n');
fprintf('STEP 4: SUBJECT-INDEPENDENT (Leave-One-Subject-Out)\n');
fprintf('=======================================================\n\n');

% Combine all subject data into one pool
X_all      = cell(num_phases,1);
Y_all      = cell(num_phases,1);
trans_all  = [];
subj_all   = [];

for p = 1:num_phases
    X_all{p} = [];
    Y_all{p} = [];
end

for subj_idx = 1:length(subjects)
    if ~valid_subj(subj_idx), continue; end
    X_s   = subj_X{subj_idx};
    Y_s   = subj_Y{subj_idx};
    trans_s = subj_trans{subj_idx};
    n_s   = length(trans_s);

    for p = 1:num_phases
        X_all{p} = [X_all{p}; X_s{p}];
        Y_all{p} = [Y_all{p}; Y_s{p}];
    end
    trans_all = [trans_all; trans_s];
    subj_all  = [subj_all;  repmat(subj_idx, n_s, 1)];
end

% Storage
indep_lda = nan(length(subjects), 3);
indep_svm = nan(length(subjects), 3);
indep_rf  = nan(length(subjects), 3);
indep_dbn = nan(length(subjects), 3);

tic;

for subj_idx = 1:length(subjects)
    if ~valid_subj(subj_idx), continue; end

    subject = subjects{subj_idx};
    fprintf('[%s] INDEP — Subject %-5s  (%.1f sec)\n', ...
        datetime('now','Format','HH:mm:ss'), subject, toc);

    train_m = subj_all ~= subj_idx;
    test_m  = subj_all == subj_idx;

    if sum(train_m) < 10 || sum(test_m) < 1, continue; end

    X_train = cell(num_phases,1); X_test = cell(num_phases,1);
    Y_train = cell(num_phases,1); Y_test = cell(num_phases,1);
    for p = 1:num_phases
        X_train{p} = X_all{p}(train_m,:);
        X_test{p}  = X_all{p}(test_m,:);
        Y_train{p} = Y_all{p}(train_m);
        Y_test{p}  = Y_all{p}(test_m);
    end

    y85_test   = Y_test{4};
    trans_test = trans_all(test_m);

    if length(unique(Y_train{4})) < 2, continue; end

    % Normalise
    mu_tr    = mean(X_train{4});
    sigma_tr = std(X_train{4});
    sigma_tr(sigma_tr==0) = 1;
    X_train_n = cell(num_phases,1);
    X_test_n  = cell(num_phases,1);
    for p = 1:num_phases
        X_train_n{p} = (X_train{p} - mu_tr) ./ sigma_tr;
        X_test_n{p}  = (X_test{p}  - mu_tr) ./ sigma_tr;
    end

    % Transition matrix
    T = cell(num_phases-1,1);
    for p = 1:num_phases-1
        Tp = ones(num_classes,num_classes);
        for k = 1:length(Y_train{p})
            fc = Y_train{p}(k); tc = Y_train{p+1}(k);
            if fc ~= tc, w = TRANS_WEIGHT; else, w = 1; end
            Tp(fc,tc) = Tp(fc,tc) + w;
        end
        T{p} = Tp ./ sum(Tp,2);
    end

    % Train LDA per phase
    lda_models = cell(num_phases,1);
    lda_ok     = false(num_phases,1);
    for p = 1:num_phases
        try
            if length(unique(Y_train{p})) >= 2
                lda_models{p} = fitcdiscr(X_train_n{p}, Y_train{p}, 'DiscrimType','linear');
                lda_ok(p)     = true;
            end
        catch
        end
    end

    % LDA
    if lda_ok(4)
        pred_LDA = predict(lda_models{4}, X_test_n{4});
    else
        pred_LDA = mode(Y_train{4})*ones(size(y85_test));
    end

    % SVM
    try
        svm85    = fitcecoc(X_train_n{4}, Y_train{4}, 'Learners',svm_template,'Coding','onevsone');
        pred_SVM = predict(svm85, X_test_n{4});
    catch
        pred_SVM = mode(Y_train{4})*ones(size(y85_test));
    end

    % RF
    try
        rf85    = TreeBagger(NUM_TREES, X_train{4}, Y_train{4}, 'Method','classification','OOBPrediction','off');
        pred_RF = str2double(predict(rf85, X_test{4}));
    catch
        pred_RF = mode(Y_train{4})*ones(size(y85_test));
    end

    % DBN 4-phase chain
    try
        n_test = size(X_test_n{1},1);

        if lda_ok(1)
            [~,sc] = predict(lda_models{1}, X_test_n{1});
            belief = ones(n_test,num_classes)*1e-10;
            cls = lda_models{1}.ClassNames;
            for ci=1:length(cls), if cls(ci)>=1&&cls(ci)<=num_classes, belief(:,cls(ci))=sc(:,ci); end, end
            belief = max(belief,1e-10); belief = belief./sum(belief,2);
        else
            belief = ones(n_test,num_classes)/num_classes;
        end

        tp = belief*T{1}; tp = tp./sum(tp,2);
        if lda_ok(2)
            [~,sc] = predict(lda_models{2}, X_test_n{2});
            em = ones(n_test,num_classes)*1e-10; cls = lda_models{2}.ClassNames;
            for ci=1:length(cls), if cls(ci)>=1&&cls(ci)<=num_classes, em(:,cls(ci))=sc(:,ci); end, end
            em = max(em,1e-10); em = em./sum(em,2);
            belief = em.*tp; belief = belief./sum(belief,2);
        else
            belief = tp;
        end

        tp = belief*T{2}; tp = tp./sum(tp,2);
        if lda_ok(3)
            [~,sc] = predict(lda_models{3}, X_test_n{3});
            em = ones(n_test,num_classes)*1e-10; cls = lda_models{3}.ClassNames;
            for ci=1:length(cls), if cls(ci)>=1&&cls(ci)<=num_classes, em(:,cls(ci))=sc(:,ci); end, end
            em = max(em,1e-10); em = em./sum(em,2);
            belief = em.*tp; belief = belief./sum(belief,2);
        else
            belief = tp;
        end

        tp = belief*T{3}; tp = tp./sum(tp,2);
        if lda_ok(4)
            [~,sc] = predict(lda_models{4}, X_test_n{4});
            em = ones(n_test,num_classes)*1e-10; cls = lda_models{4}.ClassNames;
            for ci=1:length(cls), if cls(ci)>=1&&cls(ci)<=num_classes, em(:,cls(ci))=sc(:,ci); end, end
            em = max(em,1e-10); em = em./sum(em,2);
            post = em.*tp; post = post./sum(post,2);
        else
            post = tp;
        end
        [~,pred_DBN] = max(post,[],2);
    catch
        pred_DBN = pred_LDA;
    end

    s_m = ~logical(trans_test);
    t_m =  logical(trans_test);

    preds_all = {pred_LDA, pred_SVM, pred_RF, pred_DBN};
    stores    = {indep_lda, indep_svm, indep_rf, indep_dbn};

    for mi = 1:4
        pred = preds_all{mi};
        acc  = [mean(pred==y85_test) ...
                mean(pred(s_m)==y85_test(s_m)) ...
                mean(pred(t_m)==y85_test(t_m))] * 100;
        switch mi
            case 1, indep_lda(subj_idx,:) = acc;
            case 2, indep_svm(subj_idx,:) = acc;
            case 3, indep_rf(subj_idx,:)  = acc;
            case 4, indep_dbn(subj_idx,:) = acc;
        end
    end

    fprintf('  Overall: LDA=%.1f%% SVM=%.1f%% RF=%.1f%% DBN=%.1f%%\n', ...
        indep_lda(subj_idx,1), indep_svm(subj_idx,1), ...
        indep_rf(subj_idx,1),  indep_dbn(subj_idx,1));
end

fprintf('\n[%s] Subject-independent complete! (%.1f sec)\n\n', ...
    datetime('now','Format','HH:mm:ss'), toc);

%% ========================================================
%% STEP 5: COMPARE RESULTS
%% ========================================================

fprintf('=======================================================\n');
fprintf('STEP 5: SUBJECT-DEPENDENT vs INDEPENDENT COMPARISON\n');
fprintf('=======================================================\n\n');

models   = {'LDA', 'SVM', 'RF', 'DBN'};
dep_all  = {dep_lda,   dep_svm,   dep_rf,   dep_dbn};
indep_all = {indep_lda, indep_svm, indep_rf, indep_dbn};
col_names = {'Overall', 'Steady', 'Transition'};

for col = 1:3
    fprintf('--- %s ACCURACY ---\n\n', col_names{col});
    fprintf('%-6s  %12s  %15s  %10s\n', ...
        'Model','Dependent (%)','Independent (%)','Drop (%)');
    fprintf('%s\n', repmat('-',1,50));

    for mi = 1:4
        dep_acc   = nanmean(dep_all{mi}(:,col));
        indep_acc = nanmean(indep_all{mi}(:,col));
        drop      = dep_acc - indep_acc;
        fprintf('%-6s  %12.2f  %15.2f  %10.2f\n', ...
            models{mi}, dep_acc, indep_acc, drop);
    end
    fprintf('\n');
end

%% ========================================================
%% STEP 6: VISUALISATION
%% ========================================================

fprintf('STEP 6: PLOTTING\n\n');

figure('Name','Subject Dep vs Indep','Position',[100 100 1000 600]);

metric_names = {'Overall', 'Steady State', 'Transition'};
colors_dep   = [0.00 0.45 0.74];
colors_indep = [0.85 0.33 0.10];

for col = 1:3
    subplot(1,3,col);

    dep_vals   = zeros(4,1);
    indep_vals = zeros(4,1);
    for mi = 1:4
        dep_vals(mi)   = nanmean(dep_all{mi}(:,col));
        indep_vals(mi) = nanmean(indep_all{mi}(:,col));
    end

    bar_data = [dep_vals, indep_vals];
    b = bar(bar_data, 'grouped');
    b(1).FaceColor = colors_dep;
    b(2).FaceColor = colors_indep;

    xticks(1:4); xticklabels(models);
    ylabel('Accuracy (%)');
    title(metric_names{col});
    legend({'Subject-Dependent','Subject-Independent'}, ...
        'Location','southeast','FontSize',7);
    ylim([0 105]); grid on;

    for mi = 1:4
        text(mi-0.15, dep_vals(mi)+1, sprintf('%.1f',dep_vals(mi)), ...
            'HorizontalAlignment','center','FontSize',7);
        text(mi+0.15, indep_vals(mi)+1, sprintf('%.1f',indep_vals(mi)), ...
            'HorizontalAlignment','center','FontSize',7);
    end
end

sgtitle(sprintf('Subject-Dependent vs Independent (%s)', ground_to_test));
saveas(gcf, fullfile(output_folder, 'dep_vs_indep.png'));

% Plot 2: Accuracy drop per model
figure('Name','Accuracy Drop','Position',[100 700 800 400]);
drop_overall = zeros(4,1);
drop_trans   = zeros(4,1);
for mi = 1:4
    drop_overall(mi) = nanmean(dep_all{mi}(:,1)) - nanmean(indep_all{mi}(:,1));
    drop_trans(mi)   = nanmean(dep_all{mi}(:,3)) - nanmean(indep_all{mi}(:,3));
end

bar_data = [drop_overall, drop_trans];
b = bar(bar_data, 'grouped');
b(1).FaceColor = [0.47 0.67 0.19];
b(2).FaceColor = [0.85 0.33 0.10];
xticks(1:4); xticklabels(models);
ylabel('Accuracy Drop (%)');
title('Accuracy Drop: Subject-Dependent to Independent');
legend({'Overall drop','Transition drop'},'Location','northeast');
grid on;
saveas(gcf, fullfile(output_folder, 'accuracy_drop.png'));

%% ========================================================
%% STEP 7: SAVE RESULTS
%% ========================================================

fprintf('STEP 7: SAVING RESULTS\n\n');

txt_out = fullfile(output_folder, 'dep_vs_indep_results.txt');
fid = fopen(txt_out, 'w');

fprintf(fid, '=======================================================\n');
fprintf(fid, 'SUBJECT-DEPENDENT vs SUBJECT-INDEPENDENT\n');
fprintf(fid, 'Ground   : %s\n', ground_to_test);
fprintf(fid, 'Features : DBN-selected (%d)\n', length(selected_features));
fprintf(fid, 'Dep CV   : Leave-One-Trial-Out per subject\n');
fprintf(fid, 'Indep CV : Leave-One-Subject-Out\n');
fprintf(fid, '=======================================================\n\n');

for col = 1:3
    fprintf(fid, '--- %s ACCURACY ---\n', col_names{col});
    fprintf(fid, '%-6s  %12s  %15s  %10s\n', ...
        'Model','Dependent','Independent','Drop');
    fprintf(fid, '%s\n', repmat('-',1,50));
    for mi = 1:4
        dep_acc   = nanmean(dep_all{mi}(:,col));
        indep_acc = nanmean(indep_all{mi}(:,col));
        fprintf(fid, '%-6s  %12.2f  %15.2f  %10.2f\n', ...
            models{mi}, dep_acc, indep_acc, dep_acc-indep_acc);
    end
    fprintf(fid, '\n');
end

fprintf(fid, 'Per-subject results:\n\n');
for mi = 1:4
    fprintf(fid, 'Model: %s\n', models{mi});
    fprintf(fid, '%-8s  %10s  %10s  %10s  %10s  %10s  %10s\n', ...
        'Subject','Dep-Ov','Indep-Ov','Dep-St','Indep-St','Dep-Tr','Indep-Tr');
    fprintf(fid, '%s\n', repmat('-',1,75));
    for k = 1:length(valid_idx)
        si = valid_idx(k);
        fprintf(fid, '%-8s  %10.2f  %10.2f  %10.2f  %10.2f  %10.2f  %10.2f\n', ...
            subjects{si}, ...
            dep_all{mi}(si,1),   indep_all{mi}(si,1), ...
            dep_all{mi}(si,2),   indep_all{mi}(si,2), ...
            dep_all{mi}(si,3),   indep_all{mi}(si,3));
    end
    fprintf(fid, '\n');
end

fclose(fid);
fprintf('Saved: %s\n', txt_out);
fprintf('\n=======================================================\n');
fprintf('COMPARISON COMPLETE!\n');
fprintf('=======================================================\n');