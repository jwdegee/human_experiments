import os.path as op
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
import math
import random
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
fixation_trial0_phase0 = (0,500)
mu = 0.15
sigma = 0.15
hazard = 0.04


def angle2pixels(screen_distance, angle): #distance in cm, angle in dva
    pixels = math.tan(angle/180*math.pi)*screen_distance/0.018125
    return(pixels)

def target_in_circle(max_dist_from_origin, saccade_distance, shift, prev_target):  #outer radius, inner radius, and over/undershoot factor
    
    prev_target = np.array(prev_target)
    thetas = np.arange(0,2,0.01) * np.pi
    positions = [xy_on_circle(angle2pixels(screen_distance, saccade_distance), t, prev_target[0], prev_target[1]) for t in thetas]
    distance_from_origin = [math.dist((0,0), p) for p in positions]
    valid_coords = [i for i, j in zip(positions, distance_from_origin) if j < angle2pixels(screen_distance, max_dist_from_origin)]
    try:
        target = np.array(random.choice(valid_coords))
        post_target = target + (target - prev_target)*shift
    except:
        from IPython import embed
        embed()

    # satis = 0
    # prev_target = np.array(prev_target)
    # while satis == 0:
    #     target = np.array([np.random.uniform(-max_dist_from_origin,max_dist_from_origin),
    #                         np.random.uniform(-max_dist_from_origin,max_dist_from_origin)])
    #     post_target = target + (target - prev_target)*shift
    #     if (math.dist((0,0), post_target) > max_dist_from_origin):
    #         continue
    #     if (math.dist(prev_target, target) < min_dist_from_prev_target):
    #         continue
    #     if (math.dist((0,0), post_target) < min_dist_from_origin):
    #         continue
    #     satis = 1
    return(target[0],target[1],post_target[0],post_target[1])

def xy_on_circle(r, t, a, b):
    return (int(r * math.cos(t)) + a, int(r * math.sin(t) + b))



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
                                    pos=fixation_trial0_phase0,
                                    size = angle2pixels(screen_distance, 1.5)*2,
                                    lineWidth = 4,
                                    lineColor = 'black',
                                    color=None,
                                    units='pix')


        # intro text:
        text_string1 = "Welcome to the experiment!"
        text_string2 = "Please make an eye movement to the next target when it appears and try to minimize blinking.\n\nPress spacebar to start."
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
            # print(res_here.sac_detected)
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
        self.detect_saccade = online_sac_detect()
        self.detect_saccade.set_parameters(thres_fac=10, above_thres_needed=3, 
                                            restrict_dir_min=0, restrict_dir_max=0,
                                            samp_rate=1000, anchor_vel_thres=20, print_results=0, 
                                            print_parameters=False)
        self.detect_saccade.return_data() # returns current data (should be empty after loading)
        self.detect_saccade.get_parameters() # returns current parameters
        super().__init__(output_str, output_dir=output_dir,
                         settings_file=settings_file, eyetracker_on=eyetracker_on)

    def create_trials(self, timing='seconds'):
        
        conditions = ['O', 'C', 'U', 'C', 'O', 'C', 'U', 'C', 'O', 'C', 'U', 'C']
        condition = conditions[self.block_id-1]

        if condition == 'O':
            shifts = np.concatenate((np.zeros(48), np.repeat(0.25,96), np.repeat(-0.25,96), np.zeros(48)))
            states = np.zeros(self.n_trials)
        elif condition == 'U':
            shifts = np.concatenate((np.zeros(48), np.repeat(-0.25,96), np.repeat(0.25,96), np.zeros(48))) 
            states = np.zeros(self.n_trials)
        elif condition == 'C':
            state = int(np.random.rand(1)[0]>0.5)
            shifts = np.zeros(self.n_trials)
            states = np.zeros(self.n_trials)
            for trial_nr in range(self.n_trials):
                if np.random.rand(1)[0] < hazard:
                    if state == 0:
                        state = 1
                    elif state == 1:
                        state = 0
                if state == 0:
                    shifts[trial_nr] = np.random.normal(-mu,sigma,1)[0]
                elif state == 1:
                    shifts[trial_nr] = np.random.normal(mu,sigma,1)[0]
                states[trial_nr] = state

        targets = []
        post_targets = []
        for trial_nr in range(self.n_trials):
            
            if trial_nr == 0:
                (targets_x, targets_y,
                post_targets_x,post_targets_y) = target_in_circle(max_dist_from_origin=10, 
                                                                    saccade_distance=10, 
                                                                    shift=shifts[trial_nr], 
                                                                    prev_target=fixation_trial0_phase0)
            else:
                (targets_x, targets_y,
                post_targets_x,post_targets_y) = target_in_circle(max_dist_from_origin=10, 
                                                                    saccade_distance=10, 
                                                                    shift=shifts[trial_nr], 
                                                                    prev_target=post_targets[trial_nr-1])
            targets.append((targets_x,targets_y))
            post_targets.append((post_targets_x,post_targets_y))


        self.trials = []
        for trial_nr in range(self.n_trials):
            
            if trial_nr == 0:
                parameters = {'condition':condition,
                            'target_x':targets[trial_nr][0],
                            'target_y':targets[trial_nr][1],
                            'target_post_x':post_targets[trial_nr][0],
                            'target_post_y':post_targets[trial_nr][1],
                            'prev_target_post_x':fixation_trial0_phase0[0],
                            'prev_target_post_y':fixation_trial0_phase0[0],
                            'shift':shifts[trial_nr],
                            'state':states[trial_nr]}
            else:
                parameters = {'condition':condition,
                            'target_x':targets[trial_nr][0],
                            'target_y':targets[trial_nr][1],
                            'target_post_x':post_targets[trial_nr][0],
                            'target_post_y':post_targets[trial_nr][1],
                            'prev_target_post_x':post_targets[trial_nr-1][0],
                            'prev_target_post_y':post_targets[trial_nr-1][1],
                            'shift':shifts[trial_nr],
                            'state':states[trial_nr]}
            
            default_time = 5
            if trial_nr == 0:
                durations = (30, default_time, np.random.uniform(0.05, 0.350), default_time, default_time,)
            else:
                durations = (0.3, default_time, np.random.uniform(0.05, 0.350), default_time, default_time,)
            
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