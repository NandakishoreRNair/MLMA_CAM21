import pandas as pd
import numpy as np

csv_dir = '/home/eeiww/ut55iqoh/MLMA_CAM21/csv_data/levelground/'
subject = 'ab09'

X_df   = pd.read_csv(f'{csv_dir}/{subject}_input.csv')
out_df = pd.read_csv(f'{csv_dir}/{subject}_output.csv')

# Round gait to nearest integer
out_df['gait_round'] = out_df['gait'].round().astype(int)

print('=== SAMPLES PER GAIT PHASE (rounded) ===')
for g in sorted(out_df['gait_round'].unique()):
    n = (out_df['gait_round'] == g).sum()
    print(f'  Gait {g:3d}% : {n} samples')

print()
print('=== SAMPLES PER TRIAL PER PHASE ===')
for tr in sorted(out_df['trial'].unique()):
    tr_mask = out_df['trial'] == tr
    counts  = {}
    for g in [10, 35, 60, 85]:
        counts[g] = ((out_df['gait_round'] == g) & tr_mask).sum()
    print(f'  Trial {tr}: 10%={counts[10]} 35%={counts[35]} '
          f'60%={counts[60]} 85%={counts[85]}')

print()
print('=== DATA ORDER CHECK (first 10 rows) ===')
print(out_df[['label','trial','gait','gait_round']].head(10))

print()
print('=== ROW COUNTS ===')
print(f'Input : {len(X_df)} rows')
print(f'Output: {len(out_df)} rows')
print(f'Match : {len(X_df) == len(out_df)}')