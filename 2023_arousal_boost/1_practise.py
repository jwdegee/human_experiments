# add choice triggers
from psychopy import prefs
prefs.hardware['audioLib'] = ['PTB']

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
from psychopy.visual import TextStim, Circle, Rect, GratingStim, NoiseStim, filters
from psychopy import sound
from psychopy import event
from psychopy import parallel

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
        self.parameters['correct'] = -1

        self.sound_played_boost = False
        self.mySound = sound.Sound('noise.wav')

        self.fixation = Rect(
                win=self.session.win, units='pix', autoDraw=False,
                size=30, lineWidth =2, lineColor='black', fillColor=None,
        )
        
        if self.session.block_type == 'n':
            perc_signal = 50
        elif self.session.block_type == 'r':
            perc_signal = 40
        elif self.session.block_type == 'f':
            perc_signal = 60
        if (int(self.session.subject_nr) % 2) == 0:
            text_string1 = "Press z for 'no' and m for 'yes'.\n\n{}% of trials will contain a signal.".format(perc_signal)
        elif (int(self.session.subject_nr) % 2) == 1:
            text_string1 = "Press z for 'yes' and m for 'no'.\n\n{}% of trials will contain a signal.".format(perc_signal)
            
        text_string2 = "Please sit still throughout the experiment, and try to relax and minimize blinking.\n\nPress spacebar to start."

        self.intro_text1 = TextStim(win=self.session.win, text=text_string1, pos=(0.0, 3), color=(1, 0, 0), height=1)
        self.intro_text2 = TextStim(win=self.session.win, text=text_string2, pos=(0.0, -3), height=0.5)

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

        self.trigger_intro = False
        self.trigger_baseline = False
        self.trigger_stimulus = False
        self.trigger_iti = False

    def draw(self):


        # play triggers:

        if self.trigger_intro == False:
            if self.phase == 0:
                # trigger:
                parallel.setData(self.session.p_intro)
                time.sleep(self.session.p_width)
                parallel.setData(0)
                self.trigger_intro = True

        if self.trigger_baseline == False:
            if self.phase == 1:
                if self.condition == 'normal':
                    trigger_value = self.session.p_baseline_normal
                elif self.condition == 'boost':
                    trigger_value = self.session.p_baseline_boost
                # trigger:
                parallel.setData(trigger_value)
                time.sleep(self.session.p_width)
                parallel.setData(0)
                self.trigger_baseline = True

        if self.trigger_stimulus == False:
            if self.phase == 2: 
                if self.stimulus == 'absent':
                    trigger_value = self.session.p_stimulus_absent
                elif self.stimulus == 'present':
                    trigger_value = self.session.p_stimulus_present
                # trigger:
                parallel.setData(trigger_value)
                time.sleep(self.session.p_width)
                parallel.setData(0)
                self.trigger_stimulus = True
        
        if self.trigger_iti == False:
            if self.phase == 3:
                # trigger:
                parallel.setData(self.session.p_iti)
                time.sleep(self.session.p_width)
                parallel.setData(0)
                self.trigger_iti = True

        if (self.phase == 0) & (self.trial_nr == 0):  # intro
            self.intro_text1.draw()
            self.intro_text2.draw()

        if (self.phase == 0) & (self.trial_nr > 0):  # intro
            self.fixation.draw()

        if self.phase == 1:  # baseline

            # update:
            noiseTexture = np.random.random([self.X,self.X])*2.-1.
            self.noise.tex = noiseTexture
            if (self.condition == 'boost') and not self.sound_played_boost:
                print('play sound')
                # now = ptb.GetSecs()
                # self.mySound.play(when=now+4)  # play in EXACTLY 4s
                self.mySound.play()
                self.sound_played_boost = True

            # draw:
            self.fixation.draw()
            self.noise.draw()

        elif self.phase == 2: # decision interval
            
            # update:
            self.fixation.ori = 45
            noiseTexture = np.random.random([self.X,self.X])*2.-1.
            self.noise.tex = noiseTexture

            # draw:
            self.fixation.draw()
            self.noise.draw()
            if self.stimulus == 'present':
                self.grating.draw()
                  
        elif self.phase == 3: # ITI

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
        
        if self.phase == 2:
            for key, t in events:
                if key in ['z','m']:
                    if ((int(self.session.subject_nr) % 2) == 0) & (key == 'm') and (self.parameters['stimulus']=='present'):
                        self.parameters['correct'] = 1
                        print('correct!')
                    elif ((int(self.session.subject_nr) % 2) == 0) & (key == 'z') and (self.parameters['stimulus']=='absent'):
                        self.parameters['correct'] = 1
                        print('correct!')
                    elif ((int(self.session.subject_nr) % 2) == 1) & (key == 'z') and (self.parameters['stimulus']=='present'):
                        self.parameters['correct'] = 1
                        print('correct!')
                    elif ((int(self.session.subject_nr) % 2) == 1) & (key == 'm') and (self.parameters['stimulus']=='absent'):
                        self.parameters['correct'] = 1
                        print('correct!')
                    else:
                        self.parameters['correct'] = 0
                        print('error!')
                    self.stop_phase()

class GaborSession(PylinkEyetrackerSession):
# class GaborSession(Session):

    def __init__(self, output_str, output_dir=None, settings_file=None, task='yes_no', n_trials=10, contrast=0.01, block_type='rare', awake=0):
        super().__init__(output_str=output_str, output_dir=output_dir, settings_file=settings_file)  # initialize parent class!
        self.task = task
        self.contrast = contrast
        self.n_trials = n_trials  # just an example argument
        self.block_type = block_type
        self.awake = awake
        self.trials = []  # will be filled with Trials later
        self.subject_nr = int(output_str.split('_')[0])
        print(self.subject_nr)


        # pulses:
        parallel.setPortAddress(0x3010)
        self.p_width = 3/float(1000)

        self.p_session_start = 128
        self.p_session_end = 129

        self.p_intro = 2

        self.p_baseline_normal = 4
        self.p_baseline_boost = 5

        self.p_stimulus_absent = 8
        self.p_stimulus_present = 9
        
        self.p_iti = 16

        self.p_choice_left_yes = 32
        self.p_choice_right_no = 33
        self.p_choice_right_yes = 34
        self.p_choice_left_no = 35
        
    def create_trials(self):
        """ Creates trials (ideally before running your session!) """
        
        if self.task == 'yes_no':
            
            if self.block_type == 'n':
                stimuli = ['present', 'absent']
                print()
                print('this is a normal block!!')

            elif self.block_type == 'r':
                stimuli = ['present', 'present', 'absent', 'absent', 'absent']
                print()
                print('this is a rare block!!')

            elif self.block_type == 'f':
                stimuli = ['present', 'present', 'present', 'absent', 'absent']
                print()
                print('this is a frequent block!!')

        elif self.task == '2afc':
            stimuli = ['cw', 'ccw']
        # conditions = ['boost', 'normal', 'normal', 'normal']
        conditions = ['normal']

        n_trials_per_strata = int(self.n_trials / len(stimuli) / len(conditions))
        trial_parameters = []
        for stim in stimuli:
            for cond in conditions:
                for t in range(n_trials_per_strata):
                    trial_parameters.append({'task': self.task, 'stimulus': stim, 'contrast':self.contrast, 'condition': cond, 'block_type':self.block_type,'awake':self.awake})
        random.shuffle(trial_parameters)

        for i, parameters in enumerate(trial_parameters):
            if i == 0:
                intro_dur = 10
            else:
                intro_dur = 0.1
            trial = DetectionTrial(
                session=self,
                trial_nr=i,
                phase_durations=(intro_dur, 1, 5, np.random.uniform(1.5,2,1)),
                timing='seconds',
                phase_names=('intro', 'baseline', 'decision', 'iti'),
                parameters=parameters,
                load_next_during_phase=None,
                verbose=True,
            )
            self.trials.append(trial)
        
    def run(self):
        self.create_trials()
        self.calibrate_eyetracker()

        # trigger:
        parallel.setData(self.p_session_start)
        time.sleep(self.p_width)
        parallel.setData(0)

        self.start_experiment()
        self.start_recording_eyetracker()
        for trial in self.trials:
            trial.run()

        # trigger:
        parallel.setData(self.p_session_end)
        time.sleep(self.p_width)
        parallel.setData(0)

        self.close()

if __name__ == '__main__':
    subject_nr = input('Subject #: ')
    block_nr = input('Block #: ')
    block_type = 'n'
    awake = '1'
    dt = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    settings = op.join(op.dirname(__file__), 'settings.yml')
    contrast = pd.read_csv(op.join('subjects', '{}_contrast.tsv'.format(subject_nr)), sep='\t')['contrast'].iloc[-1]
    print(contrast)
    my_sess = GaborSession(output_str='{}_{}_{}'.format(subject_nr, block_nr, dt), output_dir='data/1_practise', 
                            settings_file=settings, task='yes_no', n_trials=40, contrast=contrast, block_type=block_type, awake=awake)
    my_sess.run()
    time.sleep(1) # Sleep for 1 second

    # analyze:
    filename = glob.glob(op.join('data', '1_practise', '{}_{}_{}*events.tsv'.format(subject_nr, block_nr, dt)))[0]

    df = pd.read_table(filename)
    mean_rt = df.loc[df['event_type']=='decision', 'duration'].mean()

    df = pd.read_table(filename)
    df = df.loc[~df['response'].isna(),:].reset_index()

    if (int(subject_nr) % 2) == 0: # even
        df['answer'] = df['response'].map({'z': 'absent', 'm':'present'})
    elif (int(subject_nr) % 2) == 1: # odd
        df['answer'] = df['response'].map({'z': 'present', 'm':'absent'})
    f_yes = (df['answer']=='present').mean() * 100


    df = pd.read_table(filename)
    df = df.loc[df['event_type']=='iti',:]
    f_correct = (df['correct']==1).mean() * 100

    print('% correct = {}'.format(f_correct))
    print('% yes = {}'.format(f_yes))
    print('RT = {}'.format(mean_rt))


   