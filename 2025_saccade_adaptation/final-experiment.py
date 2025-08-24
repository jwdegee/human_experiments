import os.path as op
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
import math
import random
import datetime

from psychopy import prefs
prefs.hardware['audioLib'] = ['sounddevice']
from psychopy import sound, core
print(sound.Sound)
from psychopy.visual import TextStim
from psychopy.visual import GratingStim

from exptools2.core import PylinkEyetrackerSession
from exptools2.core import Trial

from online_sac_detect_module import online_sac_detect

def angle2pixels(distance, angle): #distance in cm, angle in dva
    pixels = tan(angle)*distance/0.018
    return(pixels)


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
            self.phase0_time = self.trialClock.getTime()
            #print(self.phase0_time)
        elif (self.phase == 0) & (self.trial_nr != 0):  # intro
            self.fixation.pos = (self.parameters['target_x'], self.parameters['target_y'])
            self.fixation.draw()
        elif self.phase == 1: # saccade
            self.fixation.pos = (self.parameters['target_x'], self.parameters['target_y'])
            self.fixation.draw()
            self.phase1_time = self.trialClock.getTime()
        elif self.phase == 2: # saccade
            self.fixation.pos = (self.parameters['target_post_x'], self.parameters['target_post_y'])
            self.fixation.draw()
            self.phase2_time = self.trialClock.getTime()
        elif self.phase == 3: # 200 ms delay
            self.fixation.pos = (self.parameters['target_post_x'], self.parameters['target_post_y'])
            self.fixation.draw()
            self.phase3_time = self.trialClock.getTime()
        elif self.phase == 4: # isi
            self.fixation.pos = (self.parameters['target_post_x'], self.parameters['target_post_y'])
            self.fixation.color = 'green'
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
        self.session.detect_saccade.add_data(gazePos[0], gazePos[1], sample.getTime())

        # next phase?
        if (self.phase == 0) and (self.trial_nr == 0):
            for key, t in events:
                if key in ['space']:
                    self.stop_phase()

        # if (self.phase == 1):
        #     res_here, run_time_here = self.session.detect_saccade.run_detection()
        #     if res_here.sac_detected:
        #         self.session.detect_saccade.reset_data()    
        #         self.stop_phase()

        if (self.phase == 2):
            target = (self.parameters['target_post_x'], self.parameters['target_post_y'])
            target = [target[0]+(self.session.win.size[0]/2),
                        -1*(target[1]-(self.session.win.size[1]/2))]
            distance = math.dist(gazePos, target)
            if distance < 45:
                self.stop_phase()

        if (self.phase == 3):
            target = (self.parameters['target_post_x'], self.parameters['target_post_y'])
            target = [target[0]+(self.session.win.size[0]/2),
                        -1*(target[1]-(self.session.win.size[1]/2))]
            distance = math.dist(gazePos, target)
            if (distance < 45) & ((self.phase3_time-self.phase2_time) > 0.2):
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
        self.detect_saccade.set_parameters(thres_fac=60, above_thres_needed=3, 
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
        print('start trials')

        def target_in_circle(r,min_r,alpha,prev_target):  #outer radius, inner radius, and over/undershoot factor
            satis = 0
            prev_target = np.array(prev_target)
            while satis == 0:
                poss_target = np.array([np.random.uniform(-r,r),np.random.uniform(-r,r)])
                poss_post_target = poss_target + (poss_target - prev_target)*alpha
                
                if (math.dist((0,0),poss_post_target) >  r) |  (math.dist((0,0),poss_post_target) <  min_r):

                    continue
                satis = 1
                print('Im working!')
            return(poss_target[0],poss_target[1],poss_post_target[0],poss_post_target[1])

        fixation_trial0_phase0 = (0,0)
        targets = []
        post_targets = []
        alphas = np.array([0.5,-0.5])
        print(alphas[1])
        state = 1
        hazard_rate = 0.5
        odd_ball = 0.5

        targets_x, targets_y,post_targets_x,post_targets_y = target_in_circle(525, 174, 0, fixation_trial0_phase0)
        target = (targets_x,targets_y)
        post_target = (post_targets_x,post_targets_y)
        targets.append(target)
        post_targets.append(post_target)
        print(targets, post_targets)
        for n in range(self.n_trials):
            #print(state)
            if state == 1:
                trial_type = random.choices(['short','long'], weights = (odd_ball, 1-odd_ball))[0]
                print(trial_type)
                if trial_type == 'short':
                    print('im short 1')
                    targets_x, targets_y,post_targets_x,post_targets_y = target_in_circle(525, 174, alphas[1], post_targets[n-1])
                    
                if trial_type == 'long':
                    print('im long 1')
                    targets_x, targets_y,post_targets_x,post_targets_y = target_in_circle(525, 174, alphas[0], post_targets[n-1])
                switch = random.choices(['switch','dont'], weights = (hazard_rate, 1-hazard_rate))[0]
                if switch == 'switch':
                    state = 2
            if state == 2:
                trial_type = random.choices(['short','long'], weights = (1-odd_ball, odd_ball))[0]
                if trial_type == 'short':
                    print('im short 2')
                    targets_x, targets_y,post_targets_x,post_targets_y = target_in_circle(525, 174, alphas[1], post_targets[n-1])
                if trial_type == 'long':
                    print('im long 2')
                    targets_x, targets_y,post_targets_x,post_targets_y = target_in_circle(525, 174, alphas[0], post_targets[n-1])
                switch = random.choices(['switch','dont'], weights = (hazard_rate, 1-hazard_rate))[0]
                if switch == 'switch':
                    state = 1
            
            target = (targets_x,targets_y)
            post_target = (post_targets_x,post_targets_y)
            targets.append(target)
            post_targets.append(post_target)

        # for alpha in alphas:

        #     for n in range(5):
        #         if (n == 0) & (alpha == alphas[0]):
        #             targets_x, targets_y,post_targets_x,post_targets_y = target_in_circle(525, 174, alpha, fixation_trial0_phase0)
        #             print('I shouldnt be here!')
        #         else:
        #             targets_x, targets_y,post_targets_x,post_targets_y = target_in_circle(525, 174, alpha, post_targets[n-1])
        #         target = (targets_x,targets_y)
        #         post_target = (post_targets_x,post_targets_y)
        #         targets.append(target)
        #         post_targets.append(post_target)

        # for n in range(5):
        #     targets_x, targets_y,post_targets_x,post_targets_y = target_in_circle(525, 174, -0.5, post_targets[n+9])
        #     target = (targets_x,targets_y)
        #     post_target = (post_targets_x,post_targets_y)
        #     targets.append(target)
        #     post_targets.append(post_target)

        print('targets:', targets)
        print('post targets:', post_targets)


        # plt.scatter(np.array(post_targets)[0,0], np.array(post_targets)[0,1])
        # plt.scatter(np.array(targets)[0,0], np.array(targets)[0,1])
        # plt.scatter(np.array(post_targets)[1,0], np.array(post_targets)[1,1])
        # plt.scatter(np.array(targets)[1,0], np.array(targets)[1,1])
        # plt.scatter(np.array(post_targets)[2,0], np.array(post_targets)[2,1])
        # plt.scatter(np.array(targets)[2,0], np.array(targets)[2,1])
        # plt.scatter(np.array(post_targets)[3,0], np.array(post_targets)[3,1])
        # plt.scatter(np.array(targets)[3,0], np.array(targets)[3,1])

        # print(positions)
        self.trials = []
        for trial_nr in range(self.n_trials):
            
            parameters = {'condition':self.condition,
                        'target_x':targets[trial_nr][0],
                        'target_y':targets[trial_nr][1],
                        'target_post_x':post_targets[trial_nr][0],
                        'target_post_y':post_targets[trial_nr][1],}
            
            default_time = 1
            if trial_nr == 0:
                durations = (30, default_time, default_time, default_time, np.random.uniform(0.5,1))
            else:
                durations = (0, default_time, default_time, default_time, np.random.uniform(0.5,1))
            
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
                                    eyetracker_on=True, n_trials=16, settings_file=settings)
    session.create_trials()
    session.run()