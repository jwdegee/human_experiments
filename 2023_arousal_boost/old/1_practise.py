import shutil
import glob
import os.path as op
import time
import datetime
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

import psychtoolbox as ptb
from psychopy.visual import TextStim, Rect, Circle, GratingStim, NoiseStim, filters
from psychopy import sound
from psychopy import event

from exptools2.core import Trial, Session, PylinkEyetrackerSession

class DetectionTrial(Trial):
    
    def __init__(self, session, trial_nr, phase_durations, phase_names,
                 parameters, timing, load_next_during_phase, 
                 verbose):
        """ Initializes a DetectionTrial object. """
        super().__init__(session, trial_nr, phase_durations, phase_names,
                         parameters, timing, load_next_during_phase, verbose)
        
        self.stimulus = parameters['stimulus']
        self.contrast = parameters['contrast']
        self.condition = parameters['condition']

        self.sound_played_boost = False
        self.mySound = sound.Sound('noise.wav')

        self.fixation = Rect(
                win=self.session.win, units='pix', autoDraw=False,
                size=30, lineWidth =2, lineColor='black', fillColor=None,
        )
        
        self.X = 1024
        sf = 0.02 

        # gaussian annulus mask:  
        center = self.X / 2
        d_0 = center / 2
        c = 128
        x,y  = np.meshgrid(np.arange(0, self.X), np.arange(0, self.X))
        d = ((x - center)**2 + (y - center)**2)**0.5
        fx = np.exp(- ((d - d_0)**2)/c**2 )
        fx_final = (2 * fx) - 1
        fx2 = np.exp(- ((d - d_0)**2)/(1.5*c)**2 )
        fx2_final = (2 * fx2) - 1

        gabor_tex = (
            filters.makeGrating(res=self.X, cycles=int(self.X * sf)) #*
            # filters.makeMask(matrixSize=self.X, shape="gauss", range=[0, 1])
        )
        self.grating = GratingStim(
            win=self.session.win, tex=gabor_tex, mask=fx_final, units='pix', 
            size=(self.X, self.X), contrast=1, opacity=self.contrast
        )
        
        noiseTexture = np.random.random([self.X,self.X])*2.-1.
        self.noise = GratingStim(
            win=self.session.win, tex=noiseTexture, mask=fx2_final, units='pix',
            size=(self.X, self.X), contrast=1, opacity=0.2
        )

    def draw(self):

        if self.phase == 0:  # baseline

            # update:
            noiseTexture = np.random.random([self.X,self.X])*2.-1.
            self.noise.tex = noiseTexture
            
            # draw:
            self.fixation.draw()
            self.noise.draw()
            
        elif self.phase == 1: # decision interval
            
            # update:
            self.fixation.ori = 45
            noiseTexture = np.random.random([self.X,self.X])*2.-1.
            self.noise.tex = noiseTexture
            if (self.condition == 'boost') and not self.sound_played_boost:
                now = ptb.GetSecs()
                self.mySound.play(when=now+2.5)  # play in EXACTLY 0.5s
                self.sound_played_boost = True

            # draw:
            self.fixation.draw()
            self.noise.draw()
            if self.stimulus == 'present':
                self.grating.draw()
                
        elif self.phase == 2: # ITI

            # update:
            self.fixation.ori = 0

            # draw:
            self.fixation.draw()
            
    def get_events(self):
        events = super().get_events()
        
        if (self.phase == 0) and (self.trial_nr == 0):
            for key, t in events:
                if key in ['space']:
                    self.stop_phase()
        
        if self.phase == 1:
            for key, t in events:
                if key in ['z','m']:
                    if ((int(self.session.output_str) % 2) == 0) & (key == 'm') and (self.parameters['stimulus']=='present'):
                        self.parameters['correct'] = 1
                        print('correct!')
                    elif ((int(self.session.output_str) % 2) == 0) & (key == 'z') and (self.parameters['stimulus']=='absent'):
                        self.parameters['correct'] = 1
                        print('correct!')
                    elif ((int(self.session.output_str) % 2) == 1) & (key == 'z') and (self.parameters['stimulus']=='present'):
                        self.parameters['correct'] = 1
                        print('correct!')
                    elif ((int(self.session.output_str) % 2) == 1) & (key == 'm') and (self.parameters['stimulus']=='absent'):
                        self.parameters['correct'] = 1
                        print('correct!')
                    else:
                        self.parameters['correct'] = 0
                        print('error!')
                    self.stop_phase()


class GaborSession(Session):

    def __init__(self, output_str, output_dir=None, settings_file=None, task='yes_no', contrast=0.01, n_trials=10):
        super().__init__(output_str=output_str, output_dir=output_dir, settings_file=settings_file)  # initialize parent class!
        self.task = task
        self.contrast = contrast
        self.n_trials = n_trials  # just an example argument
        self.trials = []  # will be filled with Trials later
        
    def create_trials(self):
        """ Creates trials (ideally before running your session!) """
        
        if self.task == 'yes_no':
            stimuli = ['present', 'absent']
        elif self.task == '2afc':
            stimuli = ['cw', 'ccw']
        conditions = ['normal']

        n_trials_per_strata = int(self.n_trials / len(stimuli) / len(conditions))
        trial_parameters = []
        for stim in stimuli:
            for cond in conditions:
                for t in range(n_trials_per_strata):
                    trial_parameters.append({'task': self.task, 'stimulus': stim, 'contrast':self.contrast, 'condition': cond})
        random.shuffle(trial_parameters)

        for i, parameters in enumerate(trial_parameters):
            trial = DetectionTrial(
                session=self,
                trial_nr=i,
                phase_durations=(1, 3, np.random.uniform(1,2,1)),
                timing='seconds',
                phase_names=('baseline', 'decision', 'iti'),
                parameters=parameters,
                load_next_during_phase=None,
                verbose=True,
            )
            self.trials.append(trial)

    def run(self):
        self.create_trials()
        # self.calibrate_eyetracker()
        self.start_experiment()
        # self.start_recording_eyetracker()

        for trial in self.trials:
            trial.run()
     
        self.close()

if __name__ == '__main__':
    subject_nr = input('Subject #: ')
    block_nr = input('Block #: ')
    dt = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    settings = op.join(op.dirname(__file__), 'settings.yml')
    contrast = pd.read_csv(op.join('subjects', '{}_contrast.tsv'.format(subject_nr)), sep='\t')['contrast'].iloc[-1]
    print(contrast)
    my_sess = GaborSession(output_str='{}_{}_{}'.format(subject_nr, block_nr, dt), output_dir='data/1_practise', 
                            settings_file=settings, task='yes_no', contrast=contrast, n_trials=40)
    my_sess.run()
    time.sleep(1) # Sleep for 1 second

    # analyze:
    filename = glob.glob(op.join('data', '1_practise', '{}_{}_{}*events.tsv'.format(subject_nr, block_nr, dt)))[0]
    df = pd.read_csv(filename, sep='\t')
    print(df.head())
 
    mean_rt = df.loc[df['event_type']=='decision', 'duration'].mean()
    
    df = df.loc[~df['response'].isna(),:].reset_index()
    if (int(subject_nr) % 2) == 0: # even
        df['answer'] = df['response'].map({'z': 'absent', 'm':'present'})
    elif (int(subject_nr) % 2) == 1: # odd
        df['answer'] = df['response'].map({'z': 'present', 'm':'absent'})
    f_correct = (df['answer']==df['stimulus']).mean() * 100
    f_yes = (df['answer']=='present').mean() * 100
    
    print('% correct = {}'.format(f_correct))
    print('% yes = {}'.format(f_yes))
    print('RT = {}'.format(mean_rt))