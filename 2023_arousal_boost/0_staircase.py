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
from psychopy.visual import TextStim, Circle, GratingStim, NoiseStim, filters
from psychopy import sound
from psychopy import event, data

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
        self.correct = 0

        self.fixation = Circle(
                win=self.session.win, units='pix', autoDraw=True,
                radius=5, color=(0.5,0.5,0.5),
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
            self.fixation.color = (0.5,0.5,0.5)

        elif self.phase == 1: # decision interval
            
            # update:
            self.fixation.color = (0.4,0.7,0.4)
            noiseTexture = np.random.random([self.X,self.X])*2.-1.
            self.noise.tex = noiseTexture
            if self.stimulus == 'cw':
                self.grating.ori = 45
            elif self.stimulus == 'ccw':
                self.grating.ori = -45

            # draw:
            self.noise.draw()
            if not self.stimulus == 'absent':
                self.grating.draw()
            
            # event:
            if (self.last_resp == 'z'):
                if self.stimulus == 'cw':
                    self.correct = 0
                elif self.stimulus == 'ccw':
                    self.correct = 1
                self.stop_phase()
            elif (self.last_resp == 'm'):
                if self.stimulus == 'cw':
                    self.correct = 1
                elif self.stimulus == 'ccw':
                    self.correct = 0
                self.stop_phase()
        
        elif self.phase == 2: # ITI

            # update:
            self.fixation.color = (0.5,0.5,0.5)

class GaborSession(Session):

    def __init__(self, output_str, output_dir=None, settings_file=None, n_trials=10, task='yes_no'):
        super().__init__(output_str=output_str, output_dir=output_dir, settings_file=settings_file)  # initialize parent class!
        self.task = task
        self.n_trials = n_trials  # just an example argument
        
    def run(self):

        self.start_experiment()
        
        # staircase:
        staircase = data.QuestHandler(startVal=0.0075, startValSd=0.01,
            pThreshold=0.75, grain=0.001, gamma=0.5, delta=0.01,
            nTrials=self.n_trials, minVal=0.001, maxVal=0.02)

        for i in range(self.n_trials):
            if self.task == 'yes_no':
                stimuli = ['present', 'absent']
            elif self.task == '2afc':
                stimuli = ['cw', 'ccw']
            stim = np.random.choice(stimuli)
            contrast = staircase._nextIntensity
            print(contrast)
            parameters = {'task': self.task, 'stimulus': stim, 'contrast':contrast}
            trial = DetectionTrial(
                session=self,
                trial_nr=i,
                phase_durations=(1, 3, np.random.uniform(1,2,1)),
                timing='seconds',
                phase_names=('baseline', 'decision', 'iti'),
                parameters=parameters,
                load_next_during_phase=None,
                verbose=True,)
            trial.run()

            # update staircase:
            staircase.addResponse(trial.correct)
            staircase.calculateNextIntensity()

        self.close()

if __name__ == '__main__':
    subject_nr = input('Subject #: ')
    block_nr = input('Block #: ')
    dt = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    settings = op.join(op.dirname(__file__), 'settings.yml')
    my_sess = GaborSession(output_str='{}_{}_{}'.format(subject_nr, block_nr, dt), output_dir='data/0_staircase', 
                            settings_file=settings, n_trials=40, task='2afc')
    my_sess.run()
    time.sleep(1) # Sleep for 1 second

    # analyze:
    filename = glob.glob(op.join('data', '0_staircase', '{}_{}_{}*events.tsv'.format(subject_nr, block_nr, dt)))[0]
    df = pd.read_csv(filename, sep='\t')
    print(df.head())
    fig = plt.figure()
    plt.plot(df['trial_nr'], df['contrast'])
    plt.title('{}'.format(df['contrast'].iloc[-1]))
    plt.tight_layout()
    fig.savefig(filename.replace('events.tsv', 'contrast.pdf'))