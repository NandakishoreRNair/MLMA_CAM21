#!/home/eeiww/ut55iqoh/lstm_env/bin/python

# LSTM comparison with DBN and LDA for locomotion classification
# Uses the same 4-phase gait data (10%, 35%, 60%, 85%) as the DBN
# CSV files were converted from MATLAB .mat files using convert_mat_to_csv.m
# Transition steps are defined by class label at 85% gait phase

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# change this to run on different terrains
ground = 'levelground'

subjects = ['ab07', 'ab08', 'ab09', 'ab12', 'ab13', 'ab14',
            'ab17', 'ab18', 'ab19', 'ab20', 'ab21', 'ab23',
            'ab24', 'ab27', 'ab28']

csv_dir    = f'/home/eeiww/ut55iqoh/MLMA_CAM21/csv_data/{ground}/'
output_dir = f'/home/eeiww/ut55iqoh/MLMA_CAM21/result_lstm_corrected/{ground}/'
os.makedirs(output_dir, exist_ok=True)

# LSTM settings - tried a few values, these worked well
hidden_size  = 64
num_layers   = 2
epochs       = 50
learning_rate = 0.001
batch_size   = 32

# DBN transition matrix upweighting
trans_weight = 5
max_trials   = 5   # only use first 5 trials per subject

# transition class labels - used to define what counts as a transition step
trans_classes = ['walk-stairascent', 'walk-stairdescent',
                 'stairascent-walk', 'stairdescent-walk',
                 'stand-walk', 'walk-stand',
                 'walk-rampascent', 'walk-rampdescent',
                 'rampascent-walk', 'rampdescent-walk']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'running on {device}')
print(f'terrain: {ground}')


# simple LSTM model - takes 4 gait phases as input sequence
# output is the predicted class at the 85% phase
class GaitLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GaitLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        # take output at last time step (85% phase)
        return self.fc(self.dropout(out[:, -1, :]))


# load CSV data for each subject
# the CSV has 4 rows per gait step in order: 10%, 35%, 60%, 85%
print('\nloading data...')

subj_data = {}
all_labels = set()

for subject in subjects:
    x_file = os.path.join(csv_dir, f'{subject}_input.csv')
    y_file = os.path.join(csv_dir, f'{subject}_output.csv')

    if not os.path.exists(x_file) or not os.path.exists(y_file):
        print(f'skipping {subject} - csv not found')
        continue

    try:
        X_df = pd.read_csv(x_file)
        out_df = pd.read_csv(y_file)

        out_df['gait_round'] = out_df['gait'].round().astype(int)

        X_arr = X_df.values.astype(np.float32)
        labels_arr = out_df['label'].values
        trial_arr = out_df['trial'].values
        gait_arr = out_df['gait_round'].values

        n_rows = len(X_arr)

        # find gait steps - each step starts at gait=10%
        step_starts = np.where(gait_arr == 10)[0]

        X_seq_list = []
        y_list = []
        trial_list = []
        trans_list = []

        for start in step_starts:
            if start + 3 >= n_rows:
                continue

            # make sure all 4 phases are in correct order
            phases_here = gait_arr[start:start+4]
            if not (phases_here[0] == 10 and phases_here[1] == 35 and
                    phases_here[2] == 60 and phases_here[3] == 85):
                continue

            # all 4 rows should be from the same trial
            trials_here = trial_arr[start:start+4]
            if len(set(trials_here)) != 1:
                continue

            X_step = X_arr[start:start+4]
            lbl_85 = str(labels_arr[start+3])
            tr = int(trials_here[0])

            X_seq_list.append(X_step)
            y_list.append(lbl_85)
            trial_list.append(tr)
            trans_list.append(lbl_85 in trans_classes)
            all_labels.add(lbl_85)

        if not X_seq_list:
            print(f'no valid steps found for {subject}')
            continue

        X_seq = np.stack(X_seq_list)
        y_arr = np.array(y_list)
        tr_arr = np.array(trial_list)
        ts_arr = np.array(trans_list)

        # only keep first 5 trials
        unique_trials = np.unique(tr_arr)[:max_trials]
        mask = np.isin(tr_arr, unique_trials)

        X_seq  = X_seq[mask]
        y_arr  = y_arr[mask]
        tr_arr = tr_arr[mask]
        ts_arr = ts_arr[mask]

        subj_data[subject] = {
            'X_seq': X_seq,
            'y': y_arr,
            'trials': tr_arr,
            'trans': ts_arr
        }

        print(f'  {subject}: {len(y_arr)} steps, '
              f'{ts_arr.sum()} transition, {(~ts_arr).sum()} steady')

    except Exception as e:
        print(f'error loading {subject}: {e}')

# encode string labels to numbers
all_labels = sorted(all_labels)
num_classes = len(all_labels)
label2idx = {l: i for i, l in enumerate(all_labels)}

print(f'\nfound {num_classes} classes: {all_labels}')

for subject, sdata in subj_data.items():
    sdata['y_num'] = np.array([label2idx[l] for l in sdata['y']])


# main loop - per subject leave one trial out cross validation
print('\nrunning cross validation...\n')

results = {
    'LDA':  {'overall': [], 'steady': [], 'transition': []},
    'DBN':  {'overall': [], 'steady': [], 'transition': []},
    'LSTM': {'overall': [], 'steady': [], 'transition': []}
}

valid_subjects = []

for subject in subjects:
    if subject not in subj_data:
        continue

    sdata = subj_data[subject]
    X_seq = sdata['X_seq']
    y_num = sdata['y_num']
    trials = sdata['trials']
    trans_s = sdata['trans']

    n_features = X_seq.shape[2]
    unique_trials = np.unique(trials)

    print(f'{subject} - {len(unique_trials)} trials, {len(y_num)} steps')

    truth_all = []
    trans_all = []
    pred_lda  = []
    pred_dbn  = []
    pred_lstm = []

    for fold, test_trial in enumerate(unique_trials):
        train_m = trials != test_trial
        test_m  = trials == test_trial

        if train_m.sum() < 5 or test_m.sum() < 1:
            continue

        X_train_seq = X_seq[train_m]
        X_test_seq  = X_seq[test_m]
        y_train = y_num[train_m]
        y_test  = y_num[test_m]
        trans_test = trans_s[test_m]

        if len(np.unique(y_train)) < 2:
            continue

        # normalise using 85% phase training stats
        X_train_85 = X_train_seq[:, 3, :]
        mu    = X_train_85.mean(axis=0)
        sigma = X_train_85.std(axis=0)
        sigma[sigma == 0] = 1.0

        X_train_n = (X_train_seq - mu) / sigma
        X_test_n  = (X_test_seq  - mu) / sigma

        # LDA - uses only 85% phase features, no temporal info
        try:
            lda = LinearDiscriminantAnalysis()
            lda.fit(X_train_n[:, 3, :], y_train)
            p_lda = lda.predict(X_test_n[:, 3, :])
        except Exception:
            p_lda = np.full(y_test.shape, np.bincount(y_train).argmax())

        # DBN - 4 phase HMM chain
        try:
            # build transition matrices for each phase pair
            T = []
            for pi in range(3):
                Tp = np.ones((num_classes, num_classes))
                for k in range(len(y_train)):
                    fc = y_num[train_m][k]
                    tc = y_train[k]
                    w = trans_weight if fc != tc else 1
                    Tp[fc, tc] += w
                Tp /= Tp.sum(axis=1, keepdims=True)
                T.append(Tp)

            # train one LDA model per phase
            lda_ph = []
            lda_ok = []
            for pi in range(4):
                try:
                    lp = LinearDiscriminantAnalysis()
                    lp.fit(X_train_n[:, pi, :], y_train)
                    lda_ph.append(lp)
                    lda_ok.append(True)
                except Exception:
                    lda_ph.append(None)
                    lda_ok.append(False)

            n_test = X_test_n.shape[0]

            # initialise belief at 10% phase
            if lda_ok[0]:
                probs = lda_ph[0].predict_proba(X_test_n[:, 0, :])
                belief = np.full((n_test, num_classes), 1e-10)
                for ci, cls in enumerate(lda_ph[0].classes_):
                    belief[:, cls] = probs[:, ci]
                belief = np.maximum(belief, 1e-10)
                belief /= belief.sum(axis=1, keepdims=True)
            else:
                belief = np.ones((n_test, num_classes)) / num_classes

            # propagate through 35%, 60%, 85%
            for pi in range(1, 4):
                tp = belief @ T[pi-1]
                tp /= tp.sum(axis=1, keepdims=True)
                if lda_ok[pi]:
                    probs = lda_ph[pi].predict_proba(X_test_n[:, pi, :])
                    emission = np.full((n_test, num_classes), 1e-10)
                    for ci, cls in enumerate(lda_ph[pi].classes_):
                        emission[:, cls] = probs[:, ci]
                    emission = np.maximum(emission, 1e-10)
                    emission /= emission.sum(axis=1, keepdims=True)
                    belief = emission * tp
                    belief /= belief.sum(axis=1, keepdims=True)
                else:
                    belief = tp

            p_dbn = belief.argmax(axis=1)

        except Exception as e:
            p_dbn = p_lda.copy()

        # LSTM - learns temporal patterns from all 4 phases
        try:
            X_tr_t = torch.FloatTensor(X_train_n.astype(np.float32)).to(device)
            y_tr_t = torch.LongTensor(y_train).to(device)
            X_te_t = torch.FloatTensor(X_test_n.astype(np.float32)).to(device)

            dataset = TensorDataset(X_tr_t, y_tr_t)
            loader = DataLoader(dataset,
                                batch_size=min(batch_size, len(y_train)),
                                shuffle=True)

            model = GaitLSTM(n_features, hidden_size,
                             num_layers, num_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=20, gamma=0.5)

            model.train()
            for epoch in range(epochs):
                for Xb, yb in loader:
                    optimizer.zero_grad()
                    loss = criterion(model(Xb), yb)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

            model.eval()
            with torch.no_grad():
                p_lstm = model(X_te_t).argmax(dim=1).cpu().numpy()

        except Exception as e:
            p_lstm = p_lda.copy()
            print(f'  lstm failed on fold {fold}: {e}')

        truth_all.extend(y_test.tolist())
        trans_all.extend(trans_test.tolist())
        pred_lda.extend(p_lda.tolist())
        pred_dbn.extend(p_dbn.tolist())
        pred_lstm.extend(p_lstm.tolist())

    if not truth_all:
        continue

    truth_all = np.array(truth_all)
    trans_all = np.array(trans_all, dtype=bool)
    pred_lda  = np.array(pred_lda)
    pred_dbn  = np.array(pred_dbn)
    pred_lstm = np.array(pred_lstm)

    s_m = ~trans_all
    t_m =  trans_all

    for name, preds in [('LDA', pred_lda), ('DBN', pred_dbn), ('LSTM', pred_lstm)]:
        ov = accuracy_score(truth_all, preds) * 100
        st = accuracy_score(truth_all[s_m], preds[s_m]) * 100 if s_m.any() else np.nan
        tr = accuracy_score(truth_all[t_m], preds[t_m]) * 100 if t_m.any() else np.nan
        results[name]['overall'].append(ov)
        results[name]['steady'].append(st)
        results[name]['transition'].append(tr)

    valid_subjects.append(subject)

    print(f'  overall:    LDA={results["LDA"]["overall"][-1]:.1f}%  '
          f'DBN={results["DBN"]["overall"][-1]:.1f}%  '
          f'LSTM={results["LSTM"]["overall"][-1]:.1f}%')
    print(f'  transition: LDA={results["LDA"]["transition"][-1]:.1f}%  '
          f'DBN={results["DBN"]["transition"][-1]:.1f}%  '
          f'LSTM={results["LSTM"]["transition"][-1]:.1f}%')
    print()


# compute averages across subjects
print('final results:')
models = ['LDA', 'DBN', 'LSTM']
agg = {}
for m in models:
    ov = np.nanmean(results[m]['overall'])
    st = np.nanmean(results[m]['steady'])
    tr = np.nanmean(results[m]['transition'])
    agg[m] = {'ov': ov, 'st': st, 'tr': tr}
    print(f'  {m}: overall={ov:.2f}%, steady={st:.2f}%, transition={tr:.2f}%')

print(f'\nLSTM vs DBN at transitions: {agg["LSTM"]["tr"] - agg["DBN"]["tr"]:+.2f}%')
print(f'LSTM vs LDA at transitions: {agg["LSTM"]["tr"] - agg["LDA"]["tr"]:+.2f}%')


# plots
colors = {'LDA': '#1f77b4', 'DBN': '#ff7f0e', 'LSTM': '#2ca02c'}

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for idx, (key, title) in enumerate([('ov', 'Overall'), ('st', 'Steady State'), ('tr', 'Transition')]):
    ax = axes[idx]
    vals = [agg[m][key] for m in models]
    bars = ax.bar(models, vals, color=[colors[m] for m in models],
                  edgecolor='black', linewidth=0.5)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 108)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val+1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

fig.suptitle(f'LDA vs DBN vs LSTM — {ground}\n4-phase sequence (10%→35%→60%→85%)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'lstm_vs_dbn_vs_lda.png'), dpi=150, bbox_inches='tight')
plt.close()

# per subject transition plot
fig2, ax2 = plt.subplots(figsize=(12, 5))
x = np.arange(len(valid_subjects))
ax2.plot(x, results['LDA']['transition'], 'bo-', label='LDA', linewidth=1.5, markersize=7)
ax2.plot(x, results['DBN']['transition'], 'r^-', label='DBN', linewidth=1.5, markersize=7)
ax2.plot(x, results['LSTM']['transition'], 'gs-', label='LSTM', linewidth=1.5, markersize=7)
ax2.set_xticks(x)
ax2.set_xticklabels(valid_subjects, rotation=45)
ax2.set_ylabel('Transition Accuracy (%)')
ax2.set_title('Per-Subject Transition Accuracy')
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_ylim(0, 105)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'per_subject_transition.png'), dpi=150, bbox_inches='tight')
plt.close()

print('plots saved')

# save results to text file
txt_out = os.path.join(output_dir, 'lstm_results.txt')
with open(txt_out, 'w') as f:
    f.write(f'LSTM vs DBN vs LDA RESULTS\n')
    f.write(f'ground: {ground}\n')
    f.write(f'subjects: {len(valid_subjects)}\n')
    f.write(f'lstm: hidden={hidden_size}, layers={num_layers}, epochs={epochs}\n\n')

    for m in models:
        f.write(f'{m}: overall={agg[m]["ov"]:.2f}%, '
                f'steady={agg[m]["st"]:.2f}%, '
                f'transition={agg[m]["tr"]:.2f}%\n')
    f.write('\n')

    f.write('per-subject transition accuracy:\n')
    f.write(f'{"subject":<8}  {"LDA":>8}  {"DBN":>8}  {"LSTM":>8}\n')
    for i, subj in enumerate(valid_subjects):
        f.write(f'{subj:<8}  '
                f'{results["LDA"]["transition"][i]:>8.2f}  '
                f'{results["DBN"]["transition"][i]:>8.2f}  '
                f'{results["LSTM"]["transition"][i]:>8.2f}\n')

print(f'results saved to {txt_out}')
print('done!')