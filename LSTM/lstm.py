#!/home/eeiww/ut55iqoh/lstm_env/bin/python
# ========================================================
# LSTM vs DBN vs LDA COMPARISON
#
# Data structure (from CSV):
#   Every 4 consecutive rows = one gait step
#   Row 0: gait 10%  features + label
#   Row 1: gait 35%  features + label
#   Row 2: gait 60%  features + label
#   Row 3: gait 85%  features + label
#
# LSTM input: [n_steps x 4 phases x 48 features]
# Target    : label at 85% (row 3 of each group)
#
# Transition: label at 10% differs from label at 85%
# CV        : Per-subject Leave-One-Trial-Out
# ========================================================

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

# ========================================================
# SETTINGS
# ========================================================

GROUND   = 'ramp'
SUBJECTS = ['ab07', 'ab08', 'ab09', 'ab12', 'ab13', 'ab14',
            'ab17', 'ab18', 'ab19', 'ab20', 'ab21', 'ab23',
            'ab24', 'ab27', 'ab28']

CSV_DIR    = f'/home/eeiww/ut55iqoh/MLMA_CAM21/csv_data/{GROUND}/'
OUTPUT_DIR = f'/home/eeiww/ut55iqoh/MLMA_CAM21/result_lstm/{GROUND}/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LSTM hyperparameters
LSTM_HIDDEN  = 64
LSTM_LAYERS  = 2
LSTM_EPOCHS  = 50
LSTM_LR      = 0.001
LSTM_BATCH   = 32

# DBN settings
TRANS_WEIGHT = 3
MAX_TRIALS   = 5

PHASE_IDX = {10: 0, 35: 1, 60: 2, 85: 3}  # phase value -> sequence index

print('='*55)
print('LSTM vs DBN vs LDA COMPARISON')
print(f'Ground  : {GROUND}')
print(f'Phases  : 10% -> 35% -> 60% -> 85%')
print(f'LSTM    : hidden={LSTM_HIDDEN}, layers={LSTM_LAYERS}, epochs={LSTM_EPOCHS}')
print(f'Device  : {"GPU" if torch.cuda.is_available() else "CPU"}')
print('='*55)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========================================================
# STEP 1: LOAD AND STRUCTURE DATA PER SUBJECT
#
# CSV rows come in groups of 4 (10%,35%,60%,85%)
# We group them into steps:
#   X_seq[step, phase, feature]  shape: [n_steps, 4, 48]
#   y[step]                      label at 85%
#   trial[step]                  trial number
#   trans[step]                  True if 10% label != 85% label
# ========================================================

print('\nSTEP 1: LOADING DATA\n')

subj_data  = {}
all_labels = set()

for subject in SUBJECTS:
    x_file = os.path.join(CSV_DIR, f'{subject}_input.csv')
    y_file = os.path.join(CSV_DIR, f'{subject}_output.csv')

    if not os.path.exists(x_file) or not os.path.exists(y_file):
        print(f'WARNING: CSV not found for {subject}. Skipping.')
        continue

    try:
        X_df   = pd.read_csv(x_file)
        out_df = pd.read_csv(y_file)

        # Round gait to nearest int for phase detection
        out_df['gait_round'] = out_df['gait'].round().astype(int)

        X_arr      = X_df.values.astype(np.float32)
        labels_arr = out_df['label'].values
        trial_arr  = out_df['trial'].values
        gait_arr   = out_df['gait_round'].values

        n_rows     = len(X_arr)
        n_features = X_arr.shape[1]

        # Group rows into steps of 4 phases
        # Each step: rows where gait rounds to 10, 35, 60, 85
        # Data is already in order: 10,35,60,85,10,35,60,85,...
        # So we can reshape directly in groups of 4

        # Verify the pattern holds
        gait_pattern = gait_arr[:8]

        # Find steps by looking for gait=10 rows as step starts
        step_starts = np.where(gait_arr == 10)[0]

        X_seq_list  = []
        y_list      = []
        trial_list  = []
        trans_list  = []

        for start in step_starts:
            # Check we have all 4 phases starting here
            if start + 3 >= n_rows:
                continue

            # Verify phases are 10,35,60,85 in order
            phases_here = gait_arr[start:start+4]
            if not (phases_here[0] == 10 and
                    phases_here[1] == 35 and
                    phases_here[2] == 60 and
                    phases_here[3] == 85):
                continue

            # Check all 4 rows belong to same trial
            trials_here = trial_arr[start:start+4]
            if len(set(trials_here)) != 1:
                continue

            # Build sequence [4 phases x features]
            X_step = X_arr[start:start+4]   # shape [4, 48]

            # Labels
            lbl_10 = str(labels_arr[start])
            lbl_85 = str(labels_arr[start+3])
            tr     = int(trials_here[0])

            X_seq_list.append(X_step)
            y_list.append(lbl_85)          # target = label at 85%
            trial_list.append(tr)
            trans_list.append(lbl_10 != lbl_85)

            all_labels.add(lbl_10)
            all_labels.add(lbl_85)

        if not X_seq_list:
            print(f'WARNING: No valid steps for {subject}.')
            continue

        X_seq  = np.stack(X_seq_list)    # [n_steps, 4, 48]
        y_arr  = np.array(y_list)
        tr_arr = np.array(trial_list)
        ts_arr = np.array(trans_list)

        # Keep only first MAX_TRIALS trials
        unique_trials = np.unique(tr_arr)[:MAX_TRIALS]
        mask = np.isin(tr_arr, unique_trials)

        X_seq  = X_seq[mask]
        y_arr  = y_arr[mask]
        tr_arr = tr_arr[mask]
        ts_arr = ts_arr[mask]

        subj_data[subject] = {
            'X_seq' : X_seq,    # [n_steps, 4, 48]
            'y'     : y_arr,    # string labels
            'trials': tr_arr,
            'trans' : ts_arr
        }

        n_steps = len(y_arr)
        print(f'Loaded: {subject}  ({n_steps} steps | '
              f'steady: {int((~ts_arr).sum())} | '
              f'transition: {int(ts_arr.sum())})')

    except Exception as e:
        print(f'ERROR: {subject} - {e}')
        import traceback
        traceback.print_exc()

# Global label encoding
all_labels  = sorted(all_labels)
num_classes = len(all_labels)
label2idx   = {l: i for i, l in enumerate(all_labels)}

print(f'\nClasses ({num_classes}):')
for i, l in enumerate(all_labels):
    print(f'  {i} -> {l}')

# Convert string labels to numeric
for subject, sdata in subj_data.items():
    sdata['y_num'] = np.array([label2idx[l] for l in sdata['y']])

# ========================================================
# STEP 2: LSTM MODEL
# ========================================================

class GaitLSTM(nn.Module):
    """
    LSTM for gait phase classification.
    Input  : [batch, seq_len=4, num_features]
              seq_len = 4 gait phases (10%, 35%, 60%, 85%)
    Output : [batch, num_classes]
    Uses output at last timestep (85%) for classification.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GaitLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.3 if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]     # last timestep = 85% phase
        return self.fc(self.dropout(last))

# ========================================================
# STEP 3: PER-SUBJECT LEAVE-ONE-TRIAL-OUT CV
# ========================================================

print('\n' + '='*55)
print('STEP 3: PER-SUBJECT LEAVE-ONE-TRIAL-OUT CV')
print('LDA  : 85%% features only (no temporal)')
print('DBN  : 4-phase HMM chain (hand-crafted temporal)')
print('LSTM : 4-phase sequence  (learned temporal)')
print('='*55 + '\n')

results = {
    'LDA' : {'overall': [], 'steady': [], 'transition': []},
    'DBN' : {'overall': [], 'steady': [], 'transition': []},
    'LSTM': {'overall': [], 'steady': [], 'transition': []}
}

valid_subjects = []

for subject in SUBJECTS:
    if subject not in subj_data:
        continue

    sdata   = subj_data[subject]
    X_seq   = sdata['X_seq']    # [n_steps, 4, 48]
    y_num   = sdata['y_num']    # numeric labels
    trials  = sdata['trials']
    trans_s = sdata['trans']

    n_features     = X_seq.shape[2]
    unique_trials  = np.unique(trials)
    num_folds      = len(unique_trials)

    print(f'Subject {subject} | {num_folds} trials | {len(y_num)} steps')

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

        # Split sequences
        X_train_seq = X_seq[train_m]    # [n_train, 4, 48]
        X_test_seq  = X_seq[test_m]     # [n_test,  4, 48]
        y_train     = y_num[train_m]
        y_test      = y_num[test_m]
        trans_test  = trans_s[test_m]

        if len(np.unique(y_train)) < 2:
            continue

        # Extract individual phase features
        # Phase index: 0=10%, 1=35%, 2=60%, 3=85%
        X_train_85 = X_train_seq[:, 3, :]   # 85% features for LDA/SVM
        X_test_85  = X_test_seq[:,  3, :]

        # Normalise using 85% training statistics
        mu    = X_train_85.mean(axis=0)
        sigma = X_train_85.std(axis=0)
        sigma[sigma == 0] = 1.0

        # Normalise all phases using same stats
        X_train_n = (X_train_seq - mu) / sigma   # [n, 4, 48]
        X_test_n  = (X_test_seq  - mu) / sigma

        X_train_85_n = X_train_n[:, 3, :]
        X_test_85_n  = X_test_n[:,  3, :]

        # ----------------------------------------------------
        # LDA at 85% only — no temporal
        # ----------------------------------------------------
        try:
            lda = LinearDiscriminantAnalysis()
            lda.fit(X_train_85_n, y_train)
            p_lda = lda.predict(X_test_85_n)
        except Exception:
            p_lda = np.full(y_test.shape,
                           np.bincount(y_train).argmax())

        # ----------------------------------------------------
        # DBN: 4-phase HMM chain
        # ----------------------------------------------------
        try:
            # Build transition matrices between consecutive phases
            T = []
            for pi in range(3):   # 0->1, 1->2, 2->3
                Tp = np.ones((num_classes, num_classes))
                for k in range(len(y_train)):
                    fc = y_num[train_m][k]   # class at phase pi
                    # Get label at phase pi+1 for same step
                    # Use rounded gait labels from sequence
                    tc = y_train[k]          # simplification: use 85% label
                    w  = TRANS_WEIGHT if fc != tc else 1
                    Tp[fc, tc] += w
                Tp /= Tp.sum(axis=1, keepdims=True)
                T.append(Tp)

            # Train LDA at each phase
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

            # Phase 0 (10%) — initial belief
            if lda_ok[0]:
                probs  = lda_ph[0].predict_proba(X_test_n[:, 0, :])
                belief = np.full((n_test, num_classes), 1e-10)
                for ci, cls in enumerate(lda_ph[0].classes_):
                    belief[:, cls] = probs[:, ci]
                belief = np.maximum(belief, 1e-10)
                belief /= belief.sum(axis=1, keepdims=True)
            else:
                belief = np.ones((n_test, num_classes)) / num_classes

            # Phases 1,2,3 — propagate and update
            for pi in range(1, 4):
                tp  = belief @ T[pi-1]
                tp /= tp.sum(axis=1, keepdims=True)
                if lda_ok[pi]:
                    probs    = lda_ph[pi].predict_proba(X_test_n[:, pi, :])
                    emission = np.full((n_test, num_classes), 1e-10)
                    for ci, cls in enumerate(lda_ph[pi].classes_):
                        emission[:, cls] = probs[:, ci]
                    emission  = np.maximum(emission, 1e-10)
                    emission /= emission.sum(axis=1, keepdims=True)
                    belief    = emission * tp
                    belief   /= belief.sum(axis=1, keepdims=True)
                else:
                    belief = tp

            p_dbn = belief.argmax(axis=1)

        except Exception as e:
            p_dbn = p_lda.copy()

        # ----------------------------------------------------
        # LSTM: learns from full 4-phase sequence
        # ----------------------------------------------------
        try:
            # Input: [n_steps, 4 phases, 48 features]
            X_tr_t = torch.FloatTensor(
                X_train_n.astype(np.float32)).to(device)
            y_tr_t = torch.LongTensor(y_train).to(device)
            X_te_t = torch.FloatTensor(
                X_test_n.astype(np.float32)).to(device)

            dataset = TensorDataset(X_tr_t, y_tr_t)
            loader  = DataLoader(
                dataset,
                batch_size=min(LSTM_BATCH, len(y_train)),
                shuffle=True
            )

            model     = GaitLSTM(n_features, LSTM_HIDDEN,
                                  LSTM_LAYERS, num_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=20, gamma=0.5)

            model.train()
            for epoch in range(LSTM_EPOCHS):
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
            print(f'  LSTM failed fold {fold}: {e}')

        # Accumulate
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

    for name, preds in [('LDA',pred_lda),
                         ('DBN',pred_dbn),
                         ('LSTM',pred_lstm)]:
        ov = accuracy_score(truth_all, preds) * 100
        st = accuracy_score(truth_all[s_m], preds[s_m])*100 \
             if s_m.any() else np.nan
        tr = accuracy_score(truth_all[t_m], preds[t_m])*100 \
             if t_m.any() else np.nan
        results[name]['overall'].append(ov)
        results[name]['steady'].append(st)
        results[name]['transition'].append(tr)

    valid_subjects.append(subject)

    print(f'  Overall    LDA:{results["LDA"]["overall"][-1]:.1f}%  '
          f'DBN:{results["DBN"]["overall"][-1]:.1f}%  '
          f'LSTM:{results["LSTM"]["overall"][-1]:.1f}%')
    print(f'  Steady     LDA:{results["LDA"]["steady"][-1]:.1f}%  '
          f'DBN:{results["DBN"]["steady"][-1]:.1f}%  '
          f'LSTM:{results["LSTM"]["steady"][-1]:.1f}%')
    print(f'  Transition LDA:{results["LDA"]["transition"][-1]:.1f}%  '
          f'DBN:{results["DBN"]["transition"][-1]:.1f}%  '
          f'LSTM:{results["LSTM"]["transition"][-1]:.1f}%')
    print()

# ========================================================
# STEP 4: AGGREGATE RESULTS
# ========================================================

print('='*55)
print('STEP 4: AGGREGATE RESULTS')
print('='*55 + '\n')

models = ['LDA', 'DBN', 'LSTM']

print(f'{"Model":<6}  {"Overall(%)":>10}  '
      f'{"Steady(%)":>10}  {"Transition(%)":>14}')
print('-'*45)

agg = {}
for m in models:
    ov = np.nanmean(results[m]['overall'])
    st = np.nanmean(results[m]['steady'])
    tr = np.nanmean(results[m]['transition'])
    agg[m] = {'ov': ov, 'st': st, 'tr': tr}
    print(f'{m:<6}  {ov:>10.2f}  {st:>10.2f}  {tr:>14.2f}')

print()
print('LSTM advantage over DBN:')
print(f'  Overall    : {agg["LSTM"]["ov"]-agg["DBN"]["ov"]:+.2f}%')
print(f'  Steady     : {agg["LSTM"]["st"]-agg["DBN"]["st"]:+.2f}%')
print(f'  Transition : {agg["LSTM"]["tr"]-agg["DBN"]["tr"]:+.2f}%')

print('\nLSTM advantage over LDA:')
print(f'  Overall    : {agg["LSTM"]["ov"]-agg["LDA"]["ov"]:+.2f}%')
print(f'  Steady     : {agg["LSTM"]["st"]-agg["LDA"]["st"]:+.2f}%')
print(f'  Transition : {agg["LSTM"]["tr"]-agg["LDA"]["tr"]:+.2f}%')

# ========================================================
# STEP 5: VISUALISATION
# ========================================================

print('\nSTEP 5: PLOTTING\n')

colors = {'LDA':'#1f77b4', 'DBN':'#ff7f0e', 'LSTM':'#2ca02c'}

# Plot 1: Bar chart
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for idx, (met, mname, key) in enumerate([
        ('overall',    'Overall',      'ov'),
        ('steady',     'Steady State', 'st'),
        ('transition', 'Transition',   'tr')]):
    ax   = axes[idx]
    vals = [agg[m][key] for m in models]
    bars = ax.bar(models, vals,
                  color=[colors[m] for m in models],
                  edgecolor='black', linewidth=0.5)
    ax.set_title(mname, fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 108)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                val+1, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=9)

fig.suptitle(f'LDA vs DBN vs LSTM — {GROUND}\n'
             f'4-phase sequence (10%→35%→60%→85%)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, 'lstm_vs_dbn_vs_lda.png')
plt.savefig(out1, dpi=150, bbox_inches='tight')
print(f'Saved: {out1}')
plt.close()

# Plot 2: Per-subject transition accuracy
fig2, ax2 = plt.subplots(figsize=(12, 5))
x = np.arange(len(valid_subjects))
ax2.plot(x, results['LDA']['transition'],
         'bo-', label='LDA',  linewidth=1.5, markersize=7)
ax2.plot(x, results['DBN']['transition'],
         'r^-', label='DBN',  linewidth=1.5, markersize=7)
ax2.plot(x, results['LSTM']['transition'],
         'gs-', label='LSTM', linewidth=1.5, markersize=7)
ax2.set_xticks(x)
ax2.set_xticklabels(valid_subjects, rotation=45)
ax2.set_ylabel('Transition Accuracy (%)')
ax2.set_title('Per-Subject Transition: LDA vs DBN vs LSTM')
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_ylim(0, 105)
plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, 'per_subject_transition.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
print(f'Saved: {out2}')
plt.close()

# ========================================================
# STEP 6: SAVE RESULTS
# ========================================================

print('\nSTEP 6: SAVING RESULTS\n')

txt_out = os.path.join(OUTPUT_DIR, 'lstm_results.txt')
with open(txt_out, 'w') as f:
    f.write('='*55 + '\n')
    f.write('LSTM vs DBN vs LDA RESULTS\n')
    f.write(f'Ground   : {GROUND}\n')
    f.write(f'Subjects : {len(valid_subjects)}\n')
    f.write(f'LSTM     : hidden={LSTM_HIDDEN}, '
            f'layers={LSTM_LAYERS}, epochs={LSTM_EPOCHS}\n')
    f.write(f'CV       : Per-subject Leave-One-Trial-Out\n')
    f.write('='*55 + '\n\n')

    f.write(f'{"Model":<6}  {"Overall(%)":>10}  '
            f'{"Steady(%)":>10}  {"Transition(%)":>14}\n')
    f.write('-'*45 + '\n')
    for m in models:
        f.write(f'{m:<6}  {agg[m]["ov"]:>10.2f}  '
                f'{agg[m]["st"]:>10.2f}  {agg[m]["tr"]:>14.2f}\n')
    f.write('\n')

    f.write('LSTM advantage over DBN:\n')
    f.write(f'  Overall    : {agg["LSTM"]["ov"]-agg["DBN"]["ov"]:+.2f}%\n')
    f.write(f'  Steady     : {agg["LSTM"]["st"]-agg["DBN"]["st"]:+.2f}%\n')
    f.write(f'  Transition : {agg["LSTM"]["tr"]-agg["DBN"]["tr"]:+.2f}%\n\n')

    f.write('LSTM advantage over LDA:\n')
    f.write(f'  Overall    : {agg["LSTM"]["ov"]-agg["LDA"]["ov"]:+.2f}%\n')
    f.write(f'  Steady     : {agg["LSTM"]["st"]-agg["LDA"]["st"]:+.2f}%\n')
    f.write(f'  Transition : {agg["LSTM"]["tr"]-agg["LDA"]["tr"]:+.2f}%\n\n')

    f.write('Per-subject transition accuracy:\n')
    f.write(f'{"Subject":<8}  {"LDA":>8}  {"DBN":>8}  {"LSTM":>8}\n')
    f.write('-'*35 + '\n')
    for i, subj in enumerate(valid_subjects):
        f.write(f'{subj:<8}  '
                f'{results["LDA"]["transition"][i]:>8.2f}  '
                f'{results["DBN"]["transition"][i]:>8.2f}  '
                f'{results["LSTM"]["transition"][i]:>8.2f}\n')

print(f'Saved: {txt_out}')
print('\n' + '='*55)
print('LSTM COMPARISON COMPLETE!')
print('='*55)