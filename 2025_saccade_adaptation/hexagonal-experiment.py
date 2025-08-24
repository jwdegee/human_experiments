import os.path as op
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
import math

import datetime

from psychopy import prefs
# prefs.hardware['audioLib'] = ['sounddevice']
from psychopy import sound, core
from psychopy.visual import TextStim
from psychopy.visual import GratingStim, Circle

from exptools2.core import PylinkEyetrackerSession
from exptools2.core import Trial

from online_sac_detect_module import online_sac_detect

screen_distance = 50
fixation_trial0_phase0 = (0,0)

def angle2pixels(screen_distance, angle): #distance in cm, angle in dva
    pixels = math.tan(angle/180*math.pi)*screen_distance/0.018125
    return(pixels)

def target_coord_gen(post_target,prev_target,shift):
    target = prev_target + (post_target - prev_target)*shift
    return(target[0],target[1])

def post_target_coord_gen(base_len, shift):
    state_len = angle2pixels(50,base_len*shift)
    X = math.acos(math.pi/6)*state_len
    Y = math.asin(math.pi/6)*state_len    
    correct_term = ((2*Y)+state_len)/2
    post_target_coord = ([(0,-correct_term),
                        (X,Y-correct_term),
                        (X,Y+state_len-correct_term),
                        (0,(2*Y)+state_len-correct_term),
                        (-X,Y+state_len-correct_term),
                        (-X,Y-correct_term)])
    return(post_target_coord)

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
                                    pos=(0,0),
                                    tex='sin',
                                    mask='circle',
                                    size=20,
                                    texRes=201,
                                    color=fixation_color,
                                    sf=0,
                                    units='pix')

        self.boundary = Circle(self.session.win,
                                    pos=(0,0),
                                    size = angle2pixels(screen_distance, 12)*2,
                                    lineWidth = 4,
                                    lineColor = 'black',
                                    color=None,
                                    units='pix')

        self.roi = Circle(self.session.win,
                                    pos=(0,0),
                                    size = angle2pixels(screen_distance, 1.5)*2,
                                    lineWidth = 4,
                                    lineColor = 'black',
                                    color=None,
                                    units='pix')


        # intro text:
        text_string1 = "You will see or hear a sequence of stimuli"
        text_string2 = "Please fixate on the black fixation mark, try to relax and try to minimize blinking.\n\nPress spacebar to start."
        self.intro_text1 = TextStim(win=self.session.win, text=text_string1, pos=(0, 200), color=(1, 0, 0), height=100, wrapWidth=2000)
        self.intro_text2 = TextStim(win=self.session.win, text=text_string2, pos=(0, -200), height=50, wrapWidth=2000)

    def draw(self):
        """ Draws stimuli """

        # 0: intro
        # 1: 200 ms delay; red fixation
        # 2: 50-350ms delay: black fixation
        # 3: saccade to target
        # 4: saccade to post

        t = self.trialClock.getTime()
        # self.roi.draw()
        if (self.phase == 0) & (self.trial_nr == 0):  # intro
            self.intro_text1.draw()
            self.intro_text2.draw()
            self.phase0_time = self.trialClock.getTime()
        elif (self.phase == 0) & (self.trial_nr != 0):  
            self.fixation.color = 'red'
            self.fixation.pos = (self.parameters['prev_target_post_x'], 
                                                self.parameters['prev_target_post_y'])
            self.fixation.draw()
            self.phase0_time = self.trialClock.getTime()
        elif self.phase == 1: # 200 ms delay; red fixation
            self.fixation.color = 'red'
            self.fixation.pos = (self.parameters['prev_target_post_x'], 
                                                self.parameters['prev_target_post_y'])
            self.fixation.draw()
            self.phase1_time = self.trialClock.getTime()
        elif self.phase == 2: # 50-350ms delay: black fixation
            self.fixation.color = 'black'
            self.fixation.pos = (self.parameters['prev_target_post_x'], 
                                                self.parameters['prev_target_post_y'])
            self.fixation.draw()            
            self.phase1_time = self.trialClock.getTime()

            # self.roi.pos = (self.parameters['prev_target_post_x'], 
            #                                     self.parameters['prev_target_post_y'])
            # self.roi.draw()  

        elif self.phase == 3: # saccade to target
            self.fixation.pos = (self.parameters['target_x'], self.parameters['target_y'])
            self.fixation.color = 'black'
            self.fixation.draw()
            self.phase3_time = self.trialClock.getTime()

            # self.roi.pos = (self.parameters['prev_target_post_x'], 
            #                                     self.parameters['prev_target_post_y'])
            # self.roi.draw()  

        elif self.phase == 4: # saccade to post target
            self.fixation.pos = (self.parameters['target_post_x'], self.parameters['target_post_y'])
            self.fixation.color = 'black'
            self.fixation.draw()
            self.phase4_time = self.trialClock.getTime()

    def get_events(self):
        events = super().get_events()

        sample = self.session.tracker.getNewestSample()
        if sample != None:
            
            if sample.isRightSample():
                gazePos = sample.getRightEye().getGaze()
            elif sample.isLeftSample():
                gazePos = sample.getLeftEye().getGaze()
        
        # add data:
        # print('before:', self.trialClock.getTime())
        self.session.detect_saccade.add_data(gazePos[0], gazePos[1], sample.getTime())
        # print('after:', self.trialClock.getTime())

        # next phase?
        if (self.phase == 0) and (self.trial_nr == 0):
            for key, t in events:
                if key in ['space']:
                    self.stop_phase()

        if (self.phase == 1):
            target = (self.parameters['prev_target_post_x'], self.parameters['prev_target_post_y'])
            target = [target[0]+(self.session.win.size[0]/2),
                        -1*(target[1]-(self.session.win.size[1]/2))]
            distance = math.dist(gazePos, target)
            if (distance < round(angle2pixels(screen_distance, 1.5))) & ((self.phase1_time-self.phase0_time) > 0.2):
                self.stop_phase()

        if (self.phase == 3):
            res_here, run_time_here = self.session.detect_saccade.run_detection()
            target = (self.parameters['prev_target_post_x'], self.parameters['prev_target_post_y'])
            target = [target[0]+(self.session.win.size[0]/2),
                        -1*(target[1]-(self.session.win.size[1]/2))]
            distance = math.dist(gazePos, target)
            print(res_here.sac_detected)
            if (bool(res_here.sac_detected)) & (distance > round(angle2pixels(screen_distance, 2))):
                self.session.detect_saccade.reset_data()    
                self.stop_phase()

        if (self.phase == 4):
            target = (self.parameters['target_post_x'], self.parameters['target_post_y'])
            target = [target[0]+(self.session.win.size[0]/2),
                        -1*(target[1]-(self.session.win.size[1]/2))]
            distance = math.dist(gazePos, target)
            if distance < round(angle2pixels(screen_distance, 1.5)):
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

        self.detect_saccade = online_sac_detect()
        self.detect_saccade.set_parameters(thres_fac=10, above_thres_needed=3, 
                                            restrict_dir_min=0, restrict_dir_max=0,
                                            samp_rate=1000, anchor_vel_thres=20, print_results=0, 
                                            print_parameters=False)
        self.detect_saccade.return_data() # returns current data (should be empty after loading)
        self.detect_saccade.get_parameters() # returns current parameters



        # order = pd.read_csv('order.csv')
        # order = order.loc[order['subject_id']==self.subject_id, 'order'].values[0]
        # self.condition = order[self.block_id]        
        
        print(self.n_trials)

        super().__init__(output_str, output_dir=output_dir,
                         settings_file=settings_file, eyetracker_on=eyetracker_on)

    def create_trials(self, timing='seconds'):

        base_len = 8
        alpha = 0.75
        state = 0 #or 1

        if self.block_id%2 == 0:
            shifts = np.concatenate((np.ones(48), np.repeat(0.75,96), np.repeat(1/0.75,96), np.ones(48))) 
        else:
            shifts = np.concatenate((np.ones(48), np.repeat(1/0.75,96), np.repeat(0.75,96), np.ones(48))) 

        targets = []
        post_targets = []
               
        for n in range(self.n_trials):
            if n%6 == 0:
                states_post_target_coord = post_target_coord_gen(base_len, shifts[n])

            post_targets_x = int(states_post_target_coord[n%6][0])
            post_targets_y = int(states_post_target_coord[n%6][1])
            post_targets.append((post_targets_x,post_targets_y))
            print('post_target',post_targets)
            if n == 0:
                post_target = np.array(post_targets[n])
                prev_target = np.array(fixation_trial0_phase0)
                target_x, target_y = target_coord_gen(post_target, prev_target, shifts[n])
            else:
                post_target = np.array(post_targets[n])
                prev_target= np.array(post_targets[n-1])
                target_x, target_y = target_coord_gen(post_target, prev_target, shifts[n])

            targets.append((target_x,target_y))
            print('target',targets)

        self.trials = []
        for trial_nr in range(self.n_trials):
            
            if trial_nr == 0:
                parameters = {'condition':self.condition,
                            'target_x':targets[trial_nr][0],
                            'target_y':targets[trial_nr][1],
                            'target_post_x':post_targets[trial_nr][0],
                            'target_post_y':post_targets[trial_nr][1],
                            'prev_target_post_x':0,
                            'prev_target_post_y':0,}
            else:
                parameters = {'condition':self.condition,
                            'target_x':targets[trial_nr][0],
                            'target_y':targets[trial_nr][1],
                            'target_post_x':post_targets[trial_nr][0],
                            'target_post_y':post_targets[trial_nr][1],
                            'prev_target_post_x':post_targets[trial_nr-1][0],
                            'prev_target_post_y':post_targets[trial_nr-1][1],}
            
            default_time = 5
            if trial_nr == 0:
                durations = (30, default_time, np.random.uniform(0.05, 0.350), default_time, default_time,)
            else:
                durations = (0.05, default_time, np.random.uniform(0.05, 0.350), default_time, default_time,)
            
            # 0: intro
            # 1: 200 ms delay; red fixation
            # 2: 50-350ms delay: black fixation
            # 3: saccade to target
            # 4: saccade to post target

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
    subject_nr = input('Subject #: ')
    block_nr = input('Block #: ')
    # subject_nr = '101'
    # block_nr = '1'
    dt = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    settings = op.join(op.dirname(__file__), 'settings.yml')
    session = TestEyetrackerSession(output_str='{}_{}_{}'.format(subject_nr, block_nr, dt),
                                    output_dir='data/',
                                    eyetracker_on=True, n_trials=288, settings_file=settings)
    session.create_trials()
    session.run()