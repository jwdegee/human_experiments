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

from IPython import embed

def make_evidence(p):
    return [np.random.uniform(0,1)*p + np.random.uniform(-1,0)*(1-p) for _ in range(8)]
 
def evidence_to_ori(evidence):
    ori = [(e + 1)/2*45 for e in evidence]
    ori = [o + (90*int(np.random.rand(1)>0.5)) for o in ori]
        
    return ori

class DetectionTrial(Trial): #Creation of the patterns for one trial
    
    def __init__(self, session, trial_nr, phase_durations, phase_names,
                 parameters, timing, load_next_during_phase, 
                 verbose):
        """ Initializes a DetectionTrial object. """
        super().__init__(session, trial_nr, phase_durations, phase_names,
                         parameters, timing, load_next_during_phase, verbose)
        
        self.stimulus = parameters['stimulus']
        self.difficulty = parameters['difficulty']
        self.condition = parameters['condition']
        self.orientations = parameters['orientation']
        self.parameters['correct'] = -1
        self.accessory_stimulus_played = False
        self.accessory_stimulus = sound.Sound('short_noise.wav') #define the sound I want to use (here a white noise)

        # intro text:
        perc_signal = 50
        if (int(self.session.output_str) % 2) == 0:
            text_string1 = "Press z for 'diagonal' and m for 'cardinal'.\n\n{}% of trials will be diagonal.".format(perc_signal)
        elif (int(self.session.output_str) % 2) == 1:
            text_string1 = "Press z for 'cardinal' and m for 'diagonal'.\n\n{}% of trials will be diagonal.".format(perc_signal)
        text_string2 = "Please sit still throughout the experiment, and try to relax and minimize blinking.\n\nPress spacebar to start."
        self.intro_text1 = TextStim(win=self.session.win, text=text_string1, pos=(0.0, 3), color=(1, 0, 0), height=1)
        self.intro_text2 = TextStim(win=self.session.win, text=text_string2, pos=(0.0, -3), height=0.5)

        #point of fixation
        self.fixation = Rect( 
                win=self.session.win, units='pix', autoDraw=False,
                size=40, lineWidth=5, lineColor=(0,0,0), fillColor=None, colorSpace='rgb255'
        )

        # #edges (circle)
        # self.circle = Circle( 
        #         win=self.session.win, units='pix', autoDraw=False,
        #         size=800, edges=100, lineColor='black', 
        # )

        #grating (parallel lines)

        width = 800
        height = 800
        sf = 0.02
        # center_x = 20
        # center_y = 20
        # d_0 = 200
        # c = 80

        # # Create a grid to calculate distances
        # x,y  = np.meshgrid(np.arange(0, height), np.arange(0, width))
        # d = ((x - center_x)**2 + (y - center_y)**2)**0.5

        # # Calculate the Gaussian function
        # fx = np.exp(- ((d - d_0)**2)/c**2 )

        # # Normalize between -1 and 1
        # fx_final = (2 * fx) - 1


        ##
        ##gabor_tex = (filters.makeGrating(res=self.X, cycles=int(self.X * sf)) * filters.makeMask(matrixSize=self.X, shape="circle", range=[0, 1]))
        self.grating = GratingStim( 
            win=self.session.win, tex='sin', mask='gauss', units='pix', 
            size=[width,height], contrast=0.5, opacity=1, sf=sf
        )
        
        # Create sound objects for the correct and incorrect feedback sounds
        self.correct_feedback = sound.Sound('test_correct.wav', stereo=False)
        self.incorrect_feedback = sound.Sound('test_incorrect.wav', stereo=False)
        self.feedback_played = False


# Gratingstim(win, tex, mask, units, size, sf, pose, ori, phase?)
    def draw(self): #Draw the pattern through phases via 

        if (self.phase == 0) & (self.trial_nr == 0):  # intro
            self.intro_text1.draw()
            self.intro_text2.draw()
        else:
            self.fixation.draw()
            # self.circle.draw()

        if self.phase == 1:  # baseline
            self.fixation.draw()
            if (self.condition == 'AS_0'):
                if not self.accessory_stimulus_played:
                    self.accessory_stimulus.play()
                    self.accessory_stimulus_played = True

        elif (self.phase >= 2) and (self.phase <= 9): # decision interval = à changer +++++
            self.grating.ori = self.orientations[min(self.phase-2,7)]
            self.grating.draw()

            if (self.condition == 'AS_1') & (self.phase == 2):
                if not self.accessory_stimulus_played:
                    self.accessory_stimulus.play()
                    self.accessory_stimulus_played = True
            elif (self.condition == 'AS_2') & (self.phase == 6):
                if not self.accessory_stimulus_played:
                    self.accessory_stimulus.play()
                    self.accessory_stimulus_played = True
            
            if self.phase == 3:
                self.accessory_stimulus.stop()
            if self.phase == 7:
                self.accessory_stimulus.stop()
            self.fixation.draw()

        elif self.phase == 10: #report
            self.fixation.ori = 45
            self.fixation.draw()
     
        elif self.phase == 11: #rdelay
            self.fixation.ori = 45
            self.fixation.draw()

        elif self.phase == 12: # Feedback/ ITI
            self.fixation.ori = 0
            self.fixation.draw()

            #After the participant responds, check whether their response was correct
            if self.feedback_played == False:
                if self.parameters['correct'] == 1:
                # Play the correct feedback sound
                    self.correct_feedback.play()
                    self.feedback_played = True
                else:
                # Play the incorrect feedback sound
                    self.incorrect_feedback.play()
                    self.feedback_played = True

    def get_events(self):
        events = super().get_events()

        if (self.phase == 0) and (self.trial_nr == 0):
            for key, t in events:
                if key in ['space']:
                    self.stop_phase()

        if self.phase == 10:
            for key, t in events:
                if key in ['z','m']:
                    if ((int(self.session.output_str) % 2) == 0) and (key == 'z') and (np.mean(self.parameters['DV']) > 0):
                        self.parameters['correct'] = 1
                    elif ((int(self.session.output_str) % 2) == 0) and (key == 'm') and (np.mean(self.parameters['DV']) < 0):
                        self.parameters['correct'] = 1
                    
                    elif ((int(self.session.output_str) % 2) == 1) and (key == 'z') and (np.mean(self.parameters['DV']) < 0):
                        self.parameters['correct'] = 1
                    elif ((int(self.session.output_str) % 2) == 1) and (key == 'm') and (np.mean(self.parameters['DV']) > 0):
                        self.parameters['correct'] = 1
                    
                    else:
                        self.parameters['correct']=0

                    self.stop_phase()

class GaborSession(PylinkEyetrackerSession): #Run session (or self) with multiple trials

    # initialize parent class that will contain several trials!
    def __init__(self, output_str, output_dir=None, settings_file=None, task='diagonal_cardinal', n_trials=1, difficulty=0.5, awake=0):
        super().__init__(output_str, output_dir=None, settings_file=settings_file)  
        self.task = task
        self.difficulty = difficulty
        self.n_trials = n_trials  # just an example argument
        self.awake = awake
        self.trials = []  # will be filled with Trials later

    def create_trials(self):
        """ Creates trials (ideally before running your session!) """ 
        
        #Defining the stimuli wrt the task + conditions 
        if self.task == 'diagonal_cardinal':
            stimuli = ['diagonal', 'cardinal']
        conditions = ['normal', 'normal', 'AS_0', 'AS_1', 'AS_2']

        #creation of a dict with all the trial parameters (task, stimuli, contrast of each participant, condition) = all the combination possible?
        n_trials_per_strata = int(self.n_trials / len(stimuli) / len(conditions)) #number of trials per ?
        trial_parameters = [] 
        for stim in stimuli: 
            for cond in conditions:
                for t in range(n_trials_per_strata):
                        if stim == 'cardinal':
                            evidence = make_evidence(1-self.difficulty)
                        elif stim == 'diagonal':
                            evidence = make_evidence(self.difficulty)
                        ori = evidence_to_ori(evidence)
                        trial_parameters.append({'task': self.task, 'stimulus': stim, 'condition': cond, 'awake':self.awake, 'difficulty':self.difficulty, 'orientation': ori, 'DV': evidence})
        random.shuffle(trial_parameters)

        #creation of a trial with the parameters of the dict, via the previous class "detectiontrial()"
        frame_rate = 60
        for i, parameters in enumerate(trial_parameters):

            if i == 0:
                phase_durations=[30,  1, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 5, 0.25, np.random.uniform(2.5,3.5,1)]
            else:
                phase_durations=[0.1, 1, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 5, 0.25, np.random.uniform(2.5,3.5,1)]

            trial = DetectionTrial(
                session=self,
                trial_nr=i,
                phase_durations=phase_durations,
                #phase_durations=[int(d*frame_rate) for d in phase_durations],
                
                timing='seconds',
                phase_names=('intro', 'baseline', 'stim1', 'stim2', 'stim3', 'stim4', 'stim5', 'stim6', 'stim7', 'stim8', 'decision', 'delay', 'iti'),
                parameters=parameters,
                load_next_during_phase=None,
                verbose=True,
            )
            self.trials.append(trial) #increment the list of trials with new trials

    #Run a session with several trials that are created via "create_trials"
    def run(self):
        self.create_trials() #creation of the trials via def created before
        # self.start_experiment()
        
        self.calibrate_eyetracker()
        self.start_experiment()
        self.start_recording_eyetracker()

        for trial in self.trials:
            trial.run()
     
        self.close()

if __name__ == '__main__':
    subject_nr = input('Subject #: ')
    block_nr = input('Block #: ')
    awake = 1 #input('How awake are you [1-7]? ')
    dt = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    settings = op.join(op.dirname(__file__), 'settings.yml')
    difficulty = pd.read_csv(op.join('subjects', '{}_difficulty.tsv'.format(subject_nr)), sep='\t')['difficulty'].iloc[-1]
    print(difficulty)
    my_sess = GaborSession(str(subject_nr), '~/logs', settings_file=settings, task='diagonal_cardinal', n_trials=64, difficulty=difficulty, awake=awake) #Set and run a session
    my_sess.run()

    # move:
    time.sleep(1) # Sleep for 1 second
    filenames = glob.glob(op.join('logs', '*'))
    for f in filenames:
        basename = op.basename(f)
        basename = basename.replace(str(subject_nr), '{}_{}_{}'.format(subject_nr, block_nr, dt))
        shutil.copy(f, op.join('data', '1_averaging', basename))
    
    # analyze + data acquisition (file with the data)
    filename = glob.glob(op.join('data', '1_averaging', '{}_{}_{}*events.tsv'.format(subject_nr, block_nr, dt)))[0]
    df = pd.read_csv(filename, sep='\t')
    print(df.head())
    
    df = df.loc[df['event_type']=='iti',:]
    f_correct = (df['correct']==1).mean() * 100
    # f_yes = (df['answer']=='present').mean() * 100
    
    print('% correct = {}'.format(f_correct))
    # print('% yes = {}'.format(f_yes))
    # print('RT = {}'.format(mean_rt))