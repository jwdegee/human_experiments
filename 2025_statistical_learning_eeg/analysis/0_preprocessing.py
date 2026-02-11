import os, glob, datetime
from functools import reduce
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
from statsmodels.stats.anova import AnovaRM
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed, Memory
from tqdm import tqdm
from IPython import embed as shell

memory = Memory(os.path.expanduser('cache'), verbose=0)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(style='ticks', font='Arial', font_scale=1, rc={
    'axes.linewidth': 0.25, 
    'axes.labelsize': 7, 
    'axes.titlesize': 7, 
    'xtick.labelsize': 6, 
    'ytick.labelsize': 6, 
    'legend.fontsize': 6, 
    'xtick.major.width': 0.25, 
    'ytick.major.width': 0.25,
    'text.color': 'Black',
    'axes.labelcolor':'Black',
    'xtick.color':'Black',
    'ytick.color':'Black',} )
sns.plotting_context()
sns.set_palette("tab10")

def make_epochs(df, df_meta, locking, start, dur, measure, fs, baseline=False, b_start=-1, b_dur=1):

    # make sure we start with index 0:
    df_meta = df_meta.reset_index(drop=True)

    # locking_inds = np.array(df['time'].searchsorted(df_meta.loc[~df_meta[locking].isna(), locking]).ravel())
    locking_inds = np.array(df['time'].searchsorted(df_meta[locking]).ravel())
    # locking_inds = np.array([find_nearest(np.array(df['time']), t) for t in df_meta[locking]])
    
    # print(locking_inds)

    start_inds = locking_inds + int(start/(1/fs))
    end_inds = start_inds + int(dur/(1/fs)) - 1
    start_inds_b = locking_inds + int(b_start/(1/fs))
    end_inds_b = start_inds_b + int(b_dur/(1/fs))
    
    epochs = []
    for s, e, sb, eb in zip(start_inds, end_inds, start_inds_b, end_inds_b):
        epoch = np.array(df.loc[s:e, measure]) 
        if baseline:
            epoch = epoch - np.array(df.loc[sb:eb,measure]).mean()
        if s < 0:
            epoch = np.concatenate((np.repeat(np.NaN, abs(s)), epoch))
        epochs.append(epoch)
    epochs = pd.DataFrame(epochs)
    epochs.columns = np.arange(start, start+dur, 1/fs).round(5)
    if df_meta[locking].isna().sum() > 0:
        epochs.loc[df_meta[locking].isna(),:] = np.NaN

    return epochs

@memory.cache
def load_data(filename, figs_dir):

    from pupilprep import preprocess_pupil

    subj = os.path.basename(filename).split('_')[0]
    ses = os.path.basename(filename).split('_')[1]

    # load and preprocess pupil data:
    params = {'lp':10, 'hp':0.01, 'order':3,
              'regress_xy':False, 'regress_blinks':True, 'regress_sacs':True}
    df, events, fs = preprocess_pupil.preprocess_pupil(filename=filename, params=params)

    # resample:
    df['time2'] = pd.TimedeltaIndex(df['time'], unit='s')
    df = df.resample(rule='10 ms', on='time2').mean().reset_index()
    fs = 100
    
    # load meta:
    df_meta_all = pd.read_csv(filename.split('.')[0]+'_events.tsv', sep='\t')
    df_meta = df_meta_all.loc[(df_meta_all['phase']==2),:].reset_index(drop=True)
    df_meta['subject_id'] = subj
    df_meta['block_id'] = ses
    df_meta = df_meta.iloc[np.where(df_meta['trial_nr'].diff()!=0)[0]]
    if not df_meta.shape[0] == 600:
        print(filename)
        raise ValueError('A very specific bad thing happened.')

    # add datetime
    timestamps = os.path.basename(filename).split('_')[2:-1]
    timestamps[-1] = timestamps[-1].split('.')[0]
    df_meta['start_time'] = datetime.datetime(*[int(t) for t in timestamps])
    df_meta['morning'] = df_meta['start_time'].dt.time <= datetime.time(13,0,0)

    # add timestamps:
    df_meta['time_stim'] = events.loc[events['description'].str.contains('start_type-stim_trial-.*_phase-2'), 'onset'].values[:df_meta.shape[0]]

    # add blinks:
    df_meta['blinks'] = [events.loc[(events['description']=='blink') & 
            (events['onset']>df_meta['time_stim'].iloc[i]) &
            (events['onset']<(df_meta['time_stim'].iloc[i]+1)), 'duration'].shape[0]
                for i in range(df_meta.shape[0])]
    df_meta['sacs'] = [events.loc[(events['description']=='saccade') & 
            (events['onset']>df_meta['time_stim'].iloc[i]) &
            (events['onset']<(df_meta['time_stim'].iloc[i]+1)), 'duration'].shape[0]
                for i in range(df_meta.shape[0])]
    
    # make epochs:
    columns = ['subject_id', 'block_id', 'trial_nr']
    epochs = make_epochs(df=df, df_meta=df_meta, locking='time_stim', start=-0.5, dur=1.5, measure='pupil_int_lp_clean_psc', fs=fs, 
                    baseline=False, b_start=-1, b_dur=1)
    epochs[columns] = df_meta[columns].values
    epochs_p_stim = epochs.set_index(columns)

    # plot:
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(211)
    plt.plot(df['time'], df['pupil'])
    plt.plot(df['time'], df['pupil_int'])
    plt.plot(df['time'], df['pupil_int_lp'])
    blinks = events.loc[(events['description']=='BAD_blink'), 'onset'].values
    for b in blinks:
        plt.axvspan(b-0.05, b+0.1, color='r', alpha=0.2)
    ax = fig.add_subplot(212)
    plt.plot(df['time'], df['pupil_int_lp_psc'])
    plt.tight_layout()
    sns.despine()
    fig.savefig(os.path.join(figs_dir, 'sessions/{}_{}.pdf'.format(subj, ses)))

    # x = np.array(epochs_p_stim.columns, dtype=float)
    # epochs_p_stim_b = epochs_p_stim - np.atleast_2d(epochs_p_stim.loc[:,(x>-0.1)&(x<0.1)].mean(axis=1).values).T
    # fig = plt.figure(figsize=(2,2))
    # ax = fig.add_subplot(111)
    # plt.plot(x, epochs_p_stim_b.mean(axis=0))
    # plt.tight_layout()
    # sns.despine()
    # fig.savefig(os.path.join(figs_dir, 'sessions/{}_{}_er.pdf'.format(subj, ses)))

    return df_meta, epochs_p_stim

project_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(project_dir, 'data')
figs_dir = os.path.join(project_dir, 'figs')

# convert:
edf_filenames = glob.glob(os.path.join(data_dir, '*.edf'))
for f in edf_filenames:
    if not os.path.exists(f.split('.')[0] + '.asc'):
        print(f)
        os.system('edf2asc {}'.format(f))
    else:
        print('skipping', f)

n_jobs = 5
asc_filenames = glob.glob(os.path.join(data_dir, '*.asc'))
# asc_filenames = [f for f in asc_filenames if '9' == f.split('/')[-1][0]]
print(len(asc_filenames))
res = Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(load_data)(filename, figs_dir) for filename in tqdm(asc_filenames))
plt.close('all')

# unpack:
df = pd.concat([res[i][0] for i in range(len(res))]).reset_index(drop=True)
epochs_p_stim = pd.concat([res[i][1] for i in range(len(res))])

# sort:
df['subject_id'] = df['subject_id'].astype(str)
df['block_id'] = df['block_id'].astype(str)
df = df.sort_values(by=['subject_id', 'block_id']).reset_index(drop=True)
epochs_p_stim = epochs_p_stim.sort_values(by=['subject_id', 'block_id'])

# ind = (df['condition']=='auditory').values
# epochs_p_stim = epochs_p_stim.diff(axis=1)*100
# plt.plot(epochs_p_stim.loc[~ind,:].mean(axis=0))
# plt.plot(epochs_p_stim.loc[ind,:].mean(axis=0))

# baseline the epochs:
x = epochs_p_stim.columns
df['pupil_b'] = np.array(epochs_p_stim.loc[:,(x>-0.25)&(x<=0)].mean(axis=1))
epochs_p_stim = epochs_p_stim - np.atleast_2d(df['pupil_b']).T

# save:
df.to_csv('df.csv')
epochs_p_stim.to_hdf('epochs.hdf', key='pupil')