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
from joblib import Parallel, delayed
from tqdm import tqdm
from IPython import embed as shell

# import utils

def plot_pupil_responses(epochs):

    means = epochs.groupby(['subject_id', 'condition']).mean().groupby(['condition']).mean()
    sems = epochs.groupby(['subject_id', 'condition']).mean().groupby(['condition']).sem()

    fig = plt.figure(figsize=(4,2))
    ax = fig.add_subplot(121)
    x = np.array(means.columns, dtype=float)
    for i in range(3,4):
        plt.fill_between(x, means.iloc[i]-sems.iloc[i], 
                            means.iloc[i]+sems.iloc[i], alpha=0.2)
        plt.plot(x, means.iloc[i], label=means.iloc[i].name)
    plt.xlabel('Time from 1st stim (s)')
    plt.ylabel('Pupil response (% signal change)')
    plt.legend()
    plt.axvspan(0,2, color='grey', alpha=0.2)
    ax = fig.add_subplot(122)
    x = np.array(means.columns, dtype=float)
    for i in range(3):
        plt.fill_between(x, means.iloc[i]-means.iloc[3]-sems.iloc[i], 
                            means.iloc[i]-means.iloc[3]+sems.iloc[i], 
                            color=sns.color_palette()[i+1], alpha=0.2)
        plt.plot(x, means.iloc[i]-means.iloc[3], color=sns.color_palette()[i+1], label=means.iloc[i].name)
    plt.axhline(0, color='black', lw=1)
    plt.xlabel('Time from 1st stim (s)')
    plt.ylabel('Pupil response (% signal change)')
    plt.legend()
    plt.axvspan(0,2, color='grey', alpha=0.2)
    sns.despine(trim=True)
    plt.tight_layout()

    return fig



    # groupby = ['subject_id', 'session_id', 'condition']
    df_res1 = df.groupby(groupby).mean()
    df_res1['choice_abs'] = abs(df_res1['choice']-0.5)
    df_res2 = df.groupby(groupby).apply(utils.sdt, stim_column='stimulus', response_column='response')
    globals().update(locals())
    df_res = reduce(lambda left, right: pd.merge(left,right, on=groupby), [df_res1, df_res2]).reset_index()
    df_res['c_abs'] = abs(df_res['c'])
    df_res['strength'] = 0

    df_res_d = (df_res.loc[df_res['condition']==1,:].set_index(groupby).droplevel(level=1) - 
                df_res.loc[df_res['condition']==0,:].set_index(groupby).droplevel(level=1)).reset_index()
    df_res_o = ((df_res.loc[df_res['condition']==1,:].set_index(groupby).droplevel(level=1) + 
                df_res.loc[df_res['condition']==0,:].set_index(groupby).droplevel(level=1))/2).reset_index()
    

    df_res['iteration'] = iteration
    df_res_o['iteration'] = iteration
    df_res_d['iteration'] = iteration

    return df_res, df_res_d, df_res_o

def add_LLRin(df):

    df['pIn_a'] = df['stimulus'].map({0: 0.1, 1:0.9})
    df['pIn_b'] = df['stimulus'].map({0: 0.9, 1:0.1})
    df['LLRin'] = np.log(df['pIn_a']/df['pIn_b'])

    return df

def l2p(Lin, dir):
    if dir == 'n':
        return 1/(np.exp(Lin)+1)
    elif dir == 'p':
        return np.exp(Lin)/(np.exp(Lin)+1)

def glaze_sim_fast(pIn, LLRin, H, s_type='CPP'):
    
    n_samples = len(LLRin)
    
    # # Apply gain term to LLRs (represents subjective component of generative variance)
    # LLRin = LLRin * B

    # compute posterior belief for each sample:
    scaled_prior = np.zeros(n_samples)
    LPRout = np.zeros(n_samples)
    if H > 0:
        startpoint = 0
        LPRout[0] = LLRin[0]+startpoint+np.log(((1-H)/H)+np.exp(-startpoint))-np.log(((1-H)/H)+np.exp(startpoint))
        for i in range(1,n_samples):
            scaled_prior[i] = (LPRout[i-1] 
                               + np.log(((1 - H) / H) + np.exp(-LPRout[i-1]))
                               - np.log(((1 - H) / H) + np.exp(LPRout[i-1])))
            LPRout[i] = LLRin[i] + scaled_prior[i]
    else:
        pass
    
    # compute surprise:
    surprise_CPP = np.zeros(n_samples)
    surprise_absL = np.zeros(n_samples)
    pL = l2p(LPRout, 'p')
    pR = l2p(LPRout, 'n')
    for i in range(1,n_samples):
        surprise_CPP[i] = (
            (H * ((pIn[1,i] * pL[i-1]) + (pIn[0,i] * pR[i-1]))) / 
            ((H * ((pIn[1,i] * pL[i-1]) + (pIn[0,i] * pR[i-1]))) + ((1 - H) * ((pIn[0,i] * pL[i-1]) + (pIn[1,i] * pR[i-1]))))
            )
    surprise_absL[1:] = abs(LPRout[1:] - LPRout[0:-1])
    
    return LPRout, scaled_prior, surprise_CPP, surprise_absL

def compute_state_changes(df):
    df['state_change'] = ((df['state'].diff() == -2)|(df['state'].diff() == 2)).astype(int)

    df['p_state_dur'] = np.NaN
    changes = df['trial_nr'].iloc[np.concatenate((np.where(df['state_change']==1)[0], np.array([-1])))]
    durs = changes.diff()
    for i in range(changes.shape[0]-1):
        state_start = changes.iloc[i]
        state_end = changes.iloc[i+1]
        df.loc[(df['trial_nr']>=state_start)&
               (df['trial_nr']<=state_end), 'p_state_dur'] = durs.iloc[i]
    
    df['trial_from_state_change'] = np.NaN
    df['trial_to_state_change'] = np.NaN
    df['only_standards_in_state'] = 1
    changes = df['trial_nr'].iloc[np.concatenate((np.array([0]), np.where(df['state_change']==1)[0]))]
    for i in range(changes.shape[0]-1):
        state_start = changes.iloc[i]
        state_end = changes.iloc[i+1]
        state_dur = state_end - state_start
        df.loc[(df['trial_nr']>=state_start)&
               (df['trial_nr']<=state_end), 'trial_to_state_change'] = np.arange(state_dur+1, 0, -1)-1
    changes = df['trial_nr'].iloc[np.concatenate((np.where(df['state_change']==1)[0], np.array([-1])))]
    for i in range(changes.shape[0]-1):
        state_start = changes.iloc[i]
        state_end = changes.iloc[i+1]
        state_dur = state_end - state_start
        df.loc[(df['trial_nr']>=state_start)&
               (df['trial_nr']<=state_end), 'trial_from_state_change'] = np.arange(0, state_dur+1, 1)
        
        # oddballs = df.loc[(df['trial_nr']>=state_start)&(df['trial_nr']<=state_end)&(df['oddball']==1), 'trial_nr']
        # if len(oddballs) > 0:
        #     df.loc[(df['trial_nr']>=state_start)&
        #            (df['trial_nr']>=oddballs.iloc[0])&
        #            (df['trial_nr']<state_end), 'only_standards_in_state'] = 0
    return df

def make_epochs(df, min_n_trials = 5):

    states = df['state'].values
    samples = df['sample'].values
    LLRin = df['LLRin'].values
    pupil = df['pupil_r'].values
    prior = df['prior'].values
    pe = df['pe'].values

    changes0 = np.where(np.diff(states)==-2)[0] + 1
    changes1 = np.where(np.diff(states)==2)[0] + 1

    actuals = []
    priors = []
    pes = []
    pupils = []
    for i, changes in enumerate([changes0, changes1]):
        
        if i == 1:
            continue
        
        if i == 0:
            flip = 1
        if i == 1:
            flip = -1
        for c in changes[min_n_trials:-min_n_trials]:
            if (states[c-min_n_trials:c].std() == 0) & (states[c:c+min_n_trials].std() == 0):
                actuals.append(LLRin[c-min_n_trials:c+min_n_trials]*flip)
                pupils.append(pupil[c-min_n_trials:c+min_n_trials])
                priors.append(prior[c-min_n_trials:c+min_n_trials]*flip)
                # pes.append(pe[c-min_n_trials:c+min_n_trials]*flip)
                pes.append(abs(pe[c-min_n_trials:c+min_n_trials]))
    trans_n = len(actuals)
    actuals = np.vstack(actuals)
    priors = np.vstack(priors)
    pes = np.vstack(pes)
    pupils = np.vstack(pupils)

    # res = pd.DataFrame({'x': np.arange(-min_n_trials, min_n_trials),
    #                     'actual': actuals.mean(axis=0),
    #                     'prior': priors.mean(axis=0),
    #                     'pe': pes.mean(axis=0),
    #                     'pupil': pupils.mean(axis=0),})

    return actuals, priors, pes, pupils

def pupil_timecourses_steady(df, epochs_p_stim, 
                             min_p_state_dur=5, min_trial_from_state_change=8,
                             colors=['#BD89B9', '#84CEEF']):
    fig = plt.figure(figsize=(3.5,3.5))
    plt_nr = 1
    for i, c in enumerate(['visual', 'auditory']):
        for s in [0,1]:
            ax = fig.add_subplot(2,2,plt_nr)
        
            for state, color in zip([-1, 1], colors):

                ind = np.array(
                            (df['state']==state)&
                            (df['condition']==c)&
                            (df['stimulus']==s)&
                            (df['trial_from_state_change']>=min_trial_from_state_change)&
                            (df['p_state_dur']>=min_p_state_dur))
                level1 = epochs_p_stim.loc[ind,:].groupby(['subject_id']).mean()
                x = level1.columns.astype(float)
                ax.fill_between(x, level1.mean(axis=0)-level1.sem(axis=0),
                                        level1.mean(axis=0)+level1.sem(axis=0), 
                                        color=color, alpha=0.2)
                ax.plot(x, level1.mean(axis=0), color=color, label='state {}'.format(state))
            plt.axvspan(0.2, 0.45, color='grey', alpha=0.2)
            ax.set_xlabel('Time from stimulus (s)')
            ax.set_ylabel('Pupil response (% change)')
            ax.axvline(0, color='k', lw=0.5)
            ax.legend()
            plt_nr += 1
    sns.despine()
    plt.tight_layout()
    return fig

def linear_model(df, x='surprise_0.05', y='pupil_r'):
    rs = []
    for (subj), d in df.groupby('subject_id'):
        x_values = np.array(d[x])
        x_values = (x_values-x_values.mean()) / x_values.std()
        y_values = np.array(d[y])
        res = sp.stats.linregress(x_values, y_values)
        rs.append(res[0])
    return np.array(rs)

def time_wise_regression(epochs, x, win=10):
    epochs_rolled = epochs.rolling(win, axis=1, center=True).mean()
    # epochs_rolled = epochs_rolled / epochs_rolled.std(axis=0)
    x_values = np.array(epochs.index.get_level_values(x))
    x_values = (x_values-x_values.mean()) / x_values.std()
    r = epochs_rolled.apply(lambda y: sp.stats.linregress(x_values, y), result_type='expand').rename(index={0:'slope', 
                                                                                1:'intercept', 2:'rvalue', 
                                                                                3:'p-value', 4:'stderr'})
    r.index.names = ['measure']
    return r

def plot_regression_timecourses(epochs, df, surprise_measure, hs=[0.05, 0.01, 0.50, 0.75]):

    # compute coefficients:
    res_hs = []
    for h in hs:
        x = 'surprise_{}_{}'.format(surprise_measure, h)
        if not x in epochs.index.names:
            epochs[x] = df.loc[:, x].values
            epochs = epochs.set_index(x, append=True)
        res = epochs.groupby('subject_id').apply(time_wise_regression, x)
        res['h'] = h
        res = res.set_index('h', append=True)    
        res_hs.append(res)
    res = pd.concat(res_hs)
    
    # plot:
    fig = plt.figure(figsize=(5,2.5))
    ax = fig.add_subplot(121)
    plt.axvline(0, color='black', lw=0.5)
    plt.axhline(0, color='black', lw=0.5)
    x = np.array(res.columns, dtype=float)
    for h, c in zip(hs, ['green', 'blue', 'orange', 'red']):
        plt.fill_between(x, 
                         res.loc[:, 'intercept', h].mean(axis=0)-res.loc[:, 'intercept', h].sem(axis=0),
                         res.loc[:, 'intercept', h].mean(axis=0)+res.loc[:, 'intercept', h].sem(axis=0),
                         color=c, alpha=0.2)
        plt.plot(x, res.loc[:, 'intercept', h].mean(axis=0), color=c, label=h)
    plt.xlabel('Time from stimulus (s)')
    plt.ylabel('Intercept')
    plt.legend()
    ax = fig.add_subplot(122)
    plt.axvline(0, color='black', lw=0.5)
    plt.axhline(0, color='black', lw=0.5)
    for h, c in zip(hs, ['green', 'blue', 'orange', 'red']):
        plt.fill_between(x, 
                         res.loc[:, 'slope', h].mean(axis=0)-res.loc[:, 'slope', h].sem(axis=0),
                         res.loc[:, 'slope', h].mean(axis=0)+res.loc[:, 'slope', h].sem(axis=0),
                         color=c, alpha=0.2)
        plt.plot(x, res.loc[:, 'slope', h].mean(axis=0), color=c, label=h)
    plt.xlabel('Time from stimulus (s)')
    plt.ylabel('Slope')
    plt.legend()
    
    # add bars:
    res1 = res.loc[:,'slope',0.05]-res.loc[:,'slope',0.01].values
    res2 = res.loc[:,'slope',0.05]-res.loc[:,'slope',0.50].values
    res3 = res.loc[:,'slope',0.05]-res.loc[:,'slope',0.75].values
    for res, p, c in zip([res1, res2, res3], [0.2, 0.1, 0], ['blue', 'orange', 'red']):
        import mne
        t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(res.values)
        for i in range(len(clusters)):
            if cluster_pv[i] < 0.05:
                x_start = x[clusters[i][0][0]]
                x_stop = x[clusters[i][0][-1]]
                height = (ax.get_ylim()[1]-ax.get_ylim()[0])/100
                print(x_start, x_stop)
                ax.add_patch(matplotlib.patches.Rectangle((x_start, p),
                                            x_stop-x_start,
                                            height, color=c))
    sns.despine()
    plt.tight_layout()

    return fig

def pupil_timecourses(df, epochs_p_stim, p_state_dur=5):
    fig, axes = plt.subplots(1,2,figsize=(5,2.5))
    for i, c in enumerate(['visual', 'auditory']):
        for t in [0,1,2]:
            ind = np.array((df['condition']==c)&
                        (df['standard']==1)&
                        (df['only_standards_in_state']==1)&
                        (df['trial_from_state_change']==t)&
                        (df['p_state_dur']>=p_state_dur))
            level1 = epochs_p_stim.loc[ind,:].groupby(['subject_id']).mean()
            x = level1.columns.astype(float)
            axes[i].fill_between(x, level1.mean(axis=0)-level1.sem(axis=0),
                                    level1.mean(axis=0)+level1.sem(axis=0), alpha=0.2)
            axes[i].plot(x, level1.mean(axis=0), label='standard after {}'.format(t))
        axes[i].set_xlabel('Time from stimulus (s)')
        axes[i].set_ylabel('Pupil response (% change)')
        axes[i].axvline(0, color='k', lw=0.5)
        axes[i].legend()
    sns.despine()
    plt.tight_layout()
    return fig

def pupil_scalars(df, y='pupil_r', min_p_state_dur=10, max_trial_from_state_change=5):

    import matplotlib.transforms as transforms

    fig, axes = plt.subplots(1,2,figsize=(5,2.5))
    for i, c in enumerate(['visual', 'auditory']):

        ax = axes[i]
        
        ind = np.array((df['condition']==c)&
                    # (df['standard']==1)&
                    # (df['only_standards_in_state']==1)&
                    (df['trial_from_state_change']>=0)&
                    (df['trial_from_state_change']<=max_trial_from_state_change)&
                    (df['p_state_dur']>=p_state_dur))
        level0 = df.loc[ind,:].groupby(['subject_id', 'trial_from_state_change'])[y].mean()
        mean = level0.groupby(['trial_from_state_change']).mean()
        sem = level0.groupby(['trial_from_state_change']).sem()
        ax.errorbar(mean.index, y=mean, yerr=sem, fmt='-o')
        
        # add mean:
        ind = np.array((df['condition']==c)&
                    # (df['standard']==1)&
                    # (df['only_standards_in_state']==1)&
                    (df['trial_from_state_change']>max_trial_from_state_change)&
                    (df['p_state_dur']>=p_state_dur))
        
        # stats:
        level0b = df.loc[ind,:].groupby(['subject_id'])[y].mean()

        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        for (trial), d in level0.groupby('trial_from_state_change'):
            t, p = sp.stats.ttest_rel(d, level0b)
            if p < 0.05:
                ax.text(x=trial, y=0.9, s='{}'.format(round(p,3)),
                             size=6, rotation=45, transform=trans)

        ax.axhline(level0b.mean(), color='r', lw=1, ls='--')
        
        ax.set_xlabel('Stimulus (# from state change)')
        ax.set_ylabel('Pupil response (% change)')
    sns.despine()
    plt.tight_layout()
    return fig

def compute_pe(df, timescale=5):
    
    for i in range(1, timescale+1):
        df['stimulus_{}'.format(i)] = df['stimulus'].shift(i).values 
    
    ind = (df[['stimulus_{}'.format(i) for i in range(1, timescale+1)]].isna().sum(axis=1) > 0)

    df.loc[~ind, 'prior'] = df.loc[~ind, ['stimulus_{}'.format(i) for i in range(1, timescale+1)]].mean(axis=1).round(2)
    df.loc[~ind, 'pe'] = df.loc[~ind, 'stimulus'] - df.loc[~ind, 'prior']
    
    return df

def zscore(df, x):
    df['{}_z'.format(x)] = (df[x]-df[x].mean()) / df[x].std()
    return df

def compute_pupil_scalars(epochs, x1=0.25, x2=0.75):
    x = epochs.columns
    kernel = epochs.loc[:,(x>=x1)&(x<=x2)].mean(axis=0)
    scalars = (epochs * kernel).sum(axis=1) / (kernel*kernel).sum()
    return scalars

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

project_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(project_dir, 'data')
figs_dir = os.path.join(project_dir, 'figs')

# load:
df = pd.read_csv('df.csv')
epochs_p_stim  = pd.read_hdf('epochs.hdf', key='pupil')
df['subject_id'] = df['subject_id'].astype(str)
df['block_id'] = df['block_id'].astype(str)
df = df.sort_values(by=['subject_id', 'block_id']).reset_index(drop=True)
epochs_p_stim = epochs_p_stim.sort_values(by=['subject_id', 'block_id'])

# add state changes:
df = df.groupby(['subject_id', 'block_id'], group_keys=False).apply(compute_state_changes)

# add pupil responses:
x = np.array(epochs_p_stim.columns, dtype=float)
df['pupil_r'] = epochs_p_stim.loc[:,(x>0.25)&(x<0.75)].mean(axis=1).values

# plot evoked pupil responses:
df['blink'] = df['blinks']>0
ind = np.array(df['block_id'].astype(int)>=5)
means_low = epochs_p_stim.loc[np.array(df['trial_from_state_change']>15),:].groupby(['subject_id']).mean()
means_high = epochs_p_stim.loc[np.array(df['trial_from_state_change']<5),:].groupby(['subject_id']).mean()
sems_low = epochs_p_stim.loc[np.array(df['trial_from_state_change']>15),:].groupby(['subject_id']).sem()
sems_high = epochs_p_stim.loc[np.array(df['trial_from_state_change']<5),:].groupby(['subject_id']).sem()
x = np.array(means_low.columns, dtype=float)
fig = plt.figure(figsize=(6,2))
for i in range(3):
    ax = fig.add_subplot(1,3,i+1)
    plt.fill_between(x, means_low.iloc[i]-sems_low.iloc[i], means_low.iloc[i]+sems_low.iloc[i], alpha=0.2)
    plt.plot(x, means_low.iloc[i])
    plt.fill_between(x, means_high.iloc[i]-sems_high.iloc[i], means_high.iloc[i]+sems_high.iloc[i], alpha=0.2)
    plt.plot(x, means_high.iloc[i])
sns.despine()
plt.tight_layout()
fig.savefig(os.path.join(figs_dir, 'pupil_timecourses.pdf'))

# print some stuff:
print(df.loc[df['trial_from_state_change']<5,:].groupby(['subject_id'])['pupil_r'].mean())
print(df.loc[df['trial_from_state_change']>15,:].groupby(['subject_id'])['pupil_r'].mean())
plt.plot(df.groupby(['trial_from_state_change'])['pupil_r'].mean().iloc[0:20])


# # add glaze:
# # df = add_LLRin(df)
# dfs = []
# for (s,b), d in df.groupby(['subject_id', 'block_id'], group_keys=False):
#     for h in [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75]:
#         LPRout, scaled_prior, surprise_CPP, surprise_absL = glaze_sim_fast(pIn=np.vstack((d['pIn_a'], d['pIn_b'])),
#                                                         LLRin=d['LLRin'].values, H=h,
#                                                         s_type='absL')
#         d['LPRout_{}'.format(h)] = LPRout
#         d['surprise_CPP_{}'.format(h)] = surprise_CPP
#         d['surprise_absL_{}'.format(h)] = surprise_absL

#     dfs.append(d)
# df = pd.concat(dfs)

# # add pupil:
# x = np.array(epochs_p_stim.columns, dtype=float)
# df['pupil_b'] = epochs_p_stim.loc[:,(x>-0.25)&(x<0)].mean(axis=1).values
# # df['pupil_r'] = epochs_p_stim.loc[:,(x>0.3)&(x<0.6)].mean(axis=1).values - df['pupil_b']
# epochs_p_stim = epochs_p_stim.diff(axis=1)*100
# # epochs_p_stim = epochs_p_stim - np.atleast_2d(df['pupil_b']).T
# even_trial_ind = np.array(df['trial_nr']%2==0)
# df.loc[even_trial_ind,'pupil_r'] = epochs_p_stim.loc[even_trial_ind,(x>0.28)&(x<0.42)].mean(axis=1).values
# odd_trial_ind = np.array(df['trial_nr']%2==1)
# df.loc[odd_trial_ind,'pupil_r'] = epochs_p_stim.loc[odd_trial_ind,(x>0.28)&(x<0.44)].mean(axis=1).values

# # add blink:
# # df['blink'] = (df['blinks']>0.5).astype(int)
# df['blink'] = (df['blinks']>0).astype(int)
# print(df.groupby(['subject_id'])['blink'].mean())

# # exclude subjects and trials:
# exclude = np.array((
#                     # (df['subject_id']=='4') | # too much blinking
#                     # (df['subject_id']=='9') | # too much blinking
#                     (df['subject_id']=='14') | # not complete
#                     (df['subject_id']=='20') | # not complete
#                     # (df['pupil_r']<-75) | 
#                     # (df['pupil_r']>75)
#                     (df['blink'] == 1)
#                     ))
# df = df.loc[~exclude,:].reset_index()
# epochs_p_stim = epochs_p_stim.loc[~exclude,:]
# print('# subjects = {}'.format(len(df['subject_id'].unique())))

# fig = plt.figure()
# plt.plot(df.loc[df['condition']=='visual',:].groupby(['trial_nr'], group_keys=False)['pupil_b'].mean())
# plt.plot(df.loc[df['condition']=='auditory',:].groupby(['trial_nr'], group_keys=False)['pupil_b'].mean())
# plt.axvspan(0, 25, color='red', alpha=0.1)
# fig.savefig(os.path.join(figs_dir, 'pupil_baselines_across_trials.pdf'))

# # exclude first 25 trials:
# exclude = np.array((
#                     (df['trial_nr']<25)
#                     ))
# df = df.loc[~exclude,:].reset_index()
# epochs_p_stim = epochs_p_stim.loc[~exclude,:]
# print('# subjects = {}'.format(len(df['subject_id'].unique())))

# # add pupil proj:
# epochs_p_stim['condition'] = df['condition'].values
# epochs_p_stim = epochs_p_stim.set_index(['condition'], append=True)

# scalars = epochs_p_stim.groupby(['subject_id', 'condition'], group_keys=False).apply(compute_pupil_scalars, x1=0, x2=1).reset_index()
# # scalars = compute_pupil_scalars(epochs_p_stim, x1=0.25, x2=0.75).reset_index()
# scalars.columns = ['subject_id', 'block_id', 'trial_nr', 'condition', 'pupil_proj']
# df = df.merge(scalars, on=['subject_id', 'block_id', 'trial_nr', 'condition'])


# print()
# ind = (df['condition']=='auditory')&(df['trial_from_state_change']>=5)
# aovrm = AnovaRM(df.loc[ind,:], 'pupil_proj', 'subject_id', within=['stimulus', 'state'], aggregate_func='mean').fit()
# print(aovrm.summary())

# print()
# ind = (df['condition']=='visual')&(df['trial_from_state_change']>=5)
# aovrm = AnovaRM(df.loc[ind,:], 'pupil_proj', 'subject_id', within=['stimulus', 'state'], aggregate_func='mean').fit()
# print(aovrm.summary())

# group = 'auditory'
# min_trial_from_state_change = 5
# min_p_state_dur = 0
# ind = (
#             (df['condition'] == group) &
#             (df['trial_from_state_change'] >= min_trial_from_state_change)
#             # (df['p_state_dur'] >= min_p_state_dur)
#         )
# model = AnovaRM(
#     df.loc[ind,:],
#     depvar='pupil_proj',
#     subject='subject_id',
#     within=['stimulus', 'state'],
#     aggregate_func='mean'
# )
# results = model.fit()
# print(results.summary())


# # correct phasic pupil responses:
# def correct_phasic_pupil_responses(df, trial_cutoff=-1):
#     import statsmodels.formula.api as smf
#     model = smf.ols(formula='pupil_r ~ pupil_b', data=df.loc[df['trial_nr']>trial_cutoff,:]).fit()
#     df.loc[df['trial_nr']>trial_cutoff, 'pupil_r_c'] = model.resid + df.loc[df['trial_nr']>trial_cutoff, 'pupil_r'].mean()
#     model = smf.ols(formula='pupil_proj ~ pupil_b', data=df.loc[df['trial_nr']>trial_cutoff,:]).fit()
#     df.loc[df['trial_nr']>trial_cutoff, 'pupil_proj_c'] = model.resid + df.loc[df['trial_nr']>trial_cutoff, 'pupil_proj'].mean()
#     return df
# df = df.groupby(['subject_id', 'block_id'], group_keys=False).apply(correct_phasic_pupil_responses)
# # df = df.groupby(['subject_id', 'condition'], group_keys=False).apply(correct_phasic_pupil_responses)

# # plot pupil time courses across all trials:
# for condition in ['visual', 'auditory']:
#     for trials in ['odd', 'even']:
        
#         if trials == 'odd':
#             trial_ind = np.array(df['trial_nr']%2==1)
#         else:
#             trial_ind = np.array(df['trial_nr']%2==0)
        
#         ind = np.array((df['condition']==condition)&(df['standard'])&(df['trial_from_state_change']>=5))&trial_ind
#         res1 = epochs_p_stim.loc[ind,:].groupby('subject_id').mean()
#         ind = np.array((df['condition']==condition)&(df['oddball'])&(df['trial_from_state_change']>=5))&trial_ind
#         res2 = epochs_p_stim.loc[ind,:].groupby('subject_id').mean()

#         colors=['black', 'red']
#         x = res1.columns.astype(float)
#         fig = plt.figure(figsize=(2.5,2))
#         ax = fig.add_subplot(111)
#         plt.axvline(0, color='black', lw=0.5)
#         ax.fill_between(x, res1.mean(axis=0)-res1.sem(axis=0),
#                                 res1.mean(axis=0)+res1.sem(axis=0), 
#                                 color=colors[0], alpha=0.2)
#         ax.plot(x, res1.mean(axis=0), color=colors[0], label='standard {}'.format(s))
#         ax.fill_between(x, res2.mean(axis=0)-res2.sem(axis=0),
#                                 res2.mean(axis=0)+res2.sem(axis=0), 
#                                 color=colors[1], alpha=0.2)
#         ax.plot(x, res2.mean(axis=0), color=colors[1], label='standard {}'.format(s))
#         plt.xlabel('Time (s)')
#         plt.ylabel('Pupil response (% signal change / s)')

#         # add bars:
#         import mne
#         t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(res2.values-res1.values)
#         print(cluster_pv)
#         for i in range(len(clusters)):
#             if cluster_pv[i] < 0.05:
#                 x_start = x[clusters[i][0][0]]
#                 x_stop = x[clusters[i][0][-1]]
#                 height = (ax.get_ylim()[1]-ax.get_ylim()[0])/100
#                 print(x_start, x_stop)
#                 ax.add_patch(matplotlib.patches.Rectangle((x_start, 0),
#                                             x_stop-x_start,
#                                             height, color='black'))
#         sns.despine()
#         plt.tight_layout()
#         fig.savefig(os.path.join(figs_dir, 'pupil_timecourses_all_trials_{}_{}.pdf'.format(condition, trials)))



# df = df.loc[df['condition']=='auditory',['subject_id', 'block_id', 'trial_nr', 'state', 'trial_from_state_change', 'stimulus', 'pupil_proj', 'pupil_r_c']].reset_index(drop=True)
# df = df.rename({'trial_nr':'trial_id'}, axis=1)
# df.to_csv('data.csv')

# shell()

# # plot steady state time courses stimulus x state interaction:
# fig = pupil_timecourses_steady(df, epochs_p_stim, 
#                              min_p_state_dur=5, min_trial_from_state_change=5,
#                              colors=['#BD89B9', '#84CEEF'])
# fig.savefig(os.path.join(figs_dir, 'pupil_timecourses_steady.pdf'))

# # heatmap:
# trial_from_state_changes = np.arange(0,10,1)
# p_state_durs = np.arange(0,10,1)
# for condition in ['visual', 'auditory']:
#     f_values = np.zeros((len(trial_from_state_changes), len(p_state_durs)))
#     for i, trial_from_state_change in enumerate(trial_from_state_changes):
#         for j, p_state_dur in enumerate(p_state_durs):
#             data = df.loc[(df['condition'] == condition)&
#                           (df['trial_from_state_change']>=trial_from_state_change)&
#                           (df['p_state_dur']>=p_state_dur),:]
#             aovrm = AnovaRM(data, 'pupil_proj', 'subject_id', within=['stimulus', 'state'], aggregate_func='mean').fit()
#             f_values[i,j] = aovrm.anova_table.loc['stimulus:state','F Value']
#     fig = plt.figure(figsize=(2.5,2))
#     ax = fig.add_subplot(111)
#     plt.pcolormesh(trial_from_state_changes, p_state_durs, f_values, cmap='summer', snap=True)
#     plt.colorbar()
#     plt.xlabel("Prev. state duration")
#     plt.ylabel("Trial from state change")
#     ax.set_aspect('equal')
#     plt.tight_layout()
#     fig.savefig(os.path.join(figs_dir, 'heatmap_fvalues_{}.pdf'.format(condition)))

# # time-wise regression to suprise:
# for condition in ['visual', 'auditory']:

#     ind = ((df['condition']==condition)).values
#     fig = plot_regression_timecourses(epochs_p_stim.loc[ind,:].copy(), 
#                                 df.loc[ind,:].copy(), 
#                                 surprise_measure='CPP')
#     fig.savefig(os.path.join(figs_dir, 'pupil_timecourses_regression_{}.pdf'.format(condition)))

# # plot correlation to normative surprise:
# for surprise_measure in ['CPP', 'absL']:
#     y = 'pupil_proj'

#     hs = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75]
#     res = [] 
#     for h in hs:
#         r = pd.DataFrame({'slope': linear_model(df.loc[df['condition']=='auditory'], 
#                                                     x='surprise_{}_{}'.format(surprise_measure, h), 
#                                                     y=y)})
#         r['h'] = h
#         res.append(r)
#     res = pd.concat(res)
#     res_mean = res.groupby('h').mean()
#     res_sem = res.groupby('h').sem()
#     h_subj = np.array(hs)[np.where(res_mean['slope']==max(res_mean['slope']))[0][0]]

#     for h in hs:
#         if not h == 0.05:
#             print(sp.stats.wilcoxon(res.loc[res['h']==h, 'slope'], res.loc[res['h']==0.05, 'slope']))

#     fig = plt.figure(figsize=(4, 2))
#     ax = fig.add_subplot(121)
#     # plt.errorbar(hs, ts, fmt='-o')
#     ax.set_xticks(ticks=np.log10(hs))
#     ax.set_xticklabels(labels=hs)
#     plt.xticks(rotation=45)
#     # ax.set_xscale('log')
#     plt.errorbar(np.log10(hs), res_mean['slope'], yerr=res_sem['slope'], fmt='-o')
#     plt.axvline(np.log10(h_subj), color='r', lw=0.5, ls='--')
#     plt.xlabel('Subjective H')
#     plt.ylabel('Slope')
#     ax = fig.add_subplot(122)
#     df['surprise_bin'] = df.groupby(['subject_id'], group_keys=False)['surprise_{}_{}'.format(surprise_measure, h_subj)].apply(pd.cut, 4, labels=False)
#     res = df.loc[df['condition']=='auditory',:].groupby(['subject_id', 'surprise_bin'])[y].mean().reset_index()
#     sns.pointplot(x='surprise_bin', y=y, units='subject_id', data=res, errorbar='se', errwidth=1, ax=ax)
#     sns.despine()
#     plt.tight_layout()
#     fig.savefig(os.path.join(figs_dir, 'fit_hazard_rate_{}.pdf'.format(surprise_measure)))

# # pupil after state change:
# for c in ['visual', 'auditory']:
#     for y in ['pupil_proj', 'pupil_proj_c', 'pupil_b', 'surprise_CPP_0.05']:

#         steady_states = df.loc[(df['condition']==c)&
#                             (df['trial_from_state_change']>5)&
#                             (df['only_standards_in_state']==1)].groupby(['subject_id'], group_keys=False)[y].mean()

#         fig = plt.figure(figsize=(2,2))
#         ax = fig.add_subplot(111)
#         sns.pointplot(x='trial_from_state_change', y=y, 
#                     units='subject_id', errorbar='se', join=False,
#                     data=df.loc[(df['condition']==c)&
#                                 (df['trial_from_state_change']<=5)&
#                                 (df['only_standards_in_state']==1)],
#                     ax=ax)
#         plt.axhline(steady_states.mean())
#         p_values = []
#         for s in [0,1,2,3,4,5]:
#             values = df.loc[(df['condition']==c)&
#                             (df['trial_from_state_change']==s)&
#                             (df['only_standards_in_state']==1)].groupby(['subject_id'], group_keys=False)[y].mean()
#             t,p = sp.stats.ttest_rel(values, steady_states)
#             plt.text(x=s, y=2, s=str(round(p,3)), size=5)

#         # curve fit:
#         values = df.loc[(df['condition']==c)&
#                             (df['trial_from_state_change']<=5)&
#                             (df['only_standards_in_state']==1)].groupby(['subject_id', 'trial_from_state_change'], group_keys=False)[y].mean().reset_index()

#         from scipy.optimize import curve_fit
#         def fitFunc(t, a, tau, c):
#             return a*np.exp(-t / tau) + c
#         popt, fitCovariances = curve_fit(fitFunc, values['trial_from_state_change'], values[y])
#         A_fit, tau_fit, B_fit = popt
#         x = np.linspace(0,6,50)
#         fitted = fitFunc(x, *popt)
#         plt.plot(x, fitted, color='r', label=f'Fit: tau = {tau_fit:.2f}')
#         plt.legend()
#         sns.despine()
#         plt.tight_layout()
#         fig.savefig(os.path.join(figs_dir, 'pupil_after_state_change_{}_{}.pdf'.format(c,y)))

# # steady state interaction:
# for c in ['visual', 'auditory']:
#     for y in ['pupil_proj', 'pupil_proj_c', 'pupil_r', 'pupil_r_c']:

#         fig = plt.figure(figsize=(2,2))
#         ax = fig.add_subplot(111)
#         sns.pointplot(x='state', y=y, hue='stimulus',
#                     units='subject_id', errorbar='se', join=True,
#                     palette=['#BD89B9', '#84CEEF'],
#                     data=df.loc[(df['condition']==c)&
#                                 (df['trial_from_state_change']>5)],
#                     ax=ax)
#         sns.despine()
#         plt.tight_layout()
#         fig.savefig(os.path.join(figs_dir, 'pupil_steady_state_{}_{}.pdf'.format(c,y)))
