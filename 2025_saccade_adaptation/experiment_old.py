import os.path as op
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
import math

import datetime

from psychopy import prefs
prefs.hardware['audioLib'] = ['sounddevice']
from psychopy import sound, core
print(sound.Sound)
from psychopy.visual import TextStim
from psychopy.visual import GratingStim


from exptools2.core import PylinkEyetrackerSession
from exptools2.core import Trial

class TestTrial(Trial):
    """ Simple trial with text (trial x) and fixation. """
    def __init__(self, session, trial_nr, phase_durations, parameters, **kwargs):
        super().__init__(session, trial_nr, phase_durations, parameters=parameters, **kwargs)

        self.parameters = parameters

        # clock:
        self.trialClock = core.Clock()

        # fixation:
        fixation_color = 'black'
        self.fixation = GratingStim(self.session.win,
                                    pos=(self.parameters['fixation_x'], self.parameters['fixation_y']),
                                    tex='sin',
                                    mask='circle',
                                    size=20,
                                    texRes=201,
                                    color=fixation_color,
                                    sf=0,
                                    units='pix')
          
        # intro text:
        text_string1 = "You will see or hear a sequence of stimuli"
        text_string2 = "Please fixate on the black fixation mark, try to relax and try to minimize blinking.\n\nPress spacebar to start."
        self.intro_text1 = TextStim(win=self.session.win, text=text_string1, pos=(0, 200), color=(1, 0, 0), height=100, wrapWidth=2000)
        self.intro_text2 = TextStim(win=self.session.win, text=text_string2, pos=(0, -200), height=50, wrapWidth=2000)

    def draw(self):
        """ Draws stimuli """
        
        t = self.trialClock.getTime()
        
        if (self.phase == 0) & (self.trial_nr == 0):  # intro
            self.intro_text1.draw()
            self.intro_text2.draw()
            # self.fixation.draw()
        elif (self.phase == 0) & (self.trial_nr != 0):  # intro
            self.fixation.draw()
        if self.phase == 1: # saccade
            self.fixation.draw()
            self.phase1_time = self.trialClock.getTime()

    def get_events(self):
        events = super().get_events()

        if (self.phase == 0) and (self.trial_nr == 0):
            for key, t in events:
                if key in ['space']:
                    self.stop_phase()

        if (self.phase == 1):
            

            
            sample = self.session.tracker.getNewestSample()
            if sample != None:
                
                if sample.isRightSample():
                    gazePos = sample.getRightEye().getGaze()
                elif sample.isLeftSample():
                    gazePos = sample.getLeftEye().getGaze()

            # self.session.detect_saccade.add_data()

            target = (self.parameters['fixation_x'], self.parameters['fixation_y'])
            target = [target[0]+(self.session.win.size[0]/2),
                        -1*(target[1]-(self.session.win.size[1]/2))]

            for key, t in events:
                if key in ['t']:
                    print('gaze = {}'.format(gazePos))
                    print('target = {}'.format(target))
                    self.stop_phase()

            #print('gaze = {}'.format(gazePos))
            #print('target = {}'.format(target))

            distance = math.dist(gazePos, target)
            # print(distance)
            if distance < 150:

                self.stop_phase()


class TestEyetrackerSession(PylinkEyetrackerSession):
    """ Simple session with x trials. """

    def __init__(self, output_str, output_dir=None, settings_file=None, n_trials=10, eyetracker_on=True):
        """ Initializes TestSession object. """
        self.n_trials = n_trials
        self.subject_id = int(output_str.split('_')[0])
        self.block_id = int(output_str.split('_')[1])
        
        # if self.subject_id%2 == 0:
        #     if self.block_id%2 == 0:
        #         self.condition = 'visual'
        #     else:
        #         self.condition = 'auditory'
        # else:
        #     if self.block_id%2 == 1:
        #         self.condition = 'visual'
        #     else:
        #         self.condition = 'auditory'
        self.condition = 'auditory'

        self.detect_saccade = {}

        # order = pd.read_csv('order.csv')
        # order = order.loc[order['subject_id']==self.subject_id, 'order'].values[0]
        # self.condition = order[self.block_id]        
                
        super().__init__(output_str, output_dir=output_dir,
                         settings_file=settings_file, eyetracker_on=eyetracker_on)

    def create_trials(self, timing='seconds'):
       
        def xy_on_circle(r, t, a, b):
            return (r * math.cos(t) + a, r * math.sin(t) + b)
        thetas = np.arange(0,self.n_trials*2,0.5) * np.pi
        positions = [xy_on_circle(500, t, 0, 0) for t in thetas]
        
        print(positions)
        self.trials = []
        for trial_nr in range(self.n_trials):
            
            parameters = {'condition':self.condition,
                        'fixation_x':positions[trial_nr][0],
                        'fixation_y':positions[trial_nr][1],}
            
            if trial_nr == 0:
                durations=(30, 10)
            elif trial_nr == self.n_trials-1:
                durations=(0, 10)
            else:
                durations=(0, 10)

            self.trials.append(
                TestTrial(session=self,
                          trial_nr=trial_nr,
                          phase_durations=durations,
                          parameters=parameters,
                          verbose=False,
                          timing=timing))

    def run(self):
        """ Runs experiment. """

        if self.eyetracker_on:
            self.calibrate_eyetracker()
        self.start_experiment()
        if self.eyetracker_on:
            self.start_recording_eyetracker()
        for trial in self.trials:
            trial.run()

        self.close()  # contains tracker.stopRecording()

if __name__ == '__main__':
    # subject_nr = input('Subject #: ')
    # block_nr = input('Block #: ')
    subject_nr = '1'
    block_nr = '1'
    dt = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    settings = op.join(op.dirname(__file__), 'settings.yml')
    session = TestEyetrackerSession(output_str='{}_{}_{}'.format(subject_nr, block_nr, dt),
                                    output_dir='data/',
                                    eyetracker_on=True, n_trials=25, settings_file=settings)
    session.create_trials()
    session.run()