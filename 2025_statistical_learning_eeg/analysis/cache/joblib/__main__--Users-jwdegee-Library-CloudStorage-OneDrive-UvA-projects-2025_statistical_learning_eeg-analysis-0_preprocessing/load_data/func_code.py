# first line: 66
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
