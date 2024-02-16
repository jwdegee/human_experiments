# todo:
# - sound only in one ear
# - check with decibel meter

import shutil
import glob
import os.path as op
import time
import datetime
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# import psychtoolbox as ptb

from psychopy import prefs
prefs.hardware['audioLib'] = ['pygame']

from psychopy.visual import TextStim, Rect, Circle, GratingStim, NoiseStim, filters
from psychopy import sound
from psychopy import event, data

from exptools2.core import Trial, Session, PylinkEyetrackerSession

from IPython import embed

def make_evidence(p):
    return np.array([np.random.uniform(0,1)*p + np.random.uniform(-1,0)*(1-p) for _ in range(8)])
 
# def evidence_to_ori(evidence):
#     ori = [np.random.choice([-1,1], p=[0.5, 0.5])*(e + 1)/2*45 for e in evidence]
#     ori = [o + (90*int(np.random.rand(1)>0.5)) for o in ori]
#     return ori

def evidence_to_ori(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = (value - leftMin) / leftSpan
    return rightMin + (valueScaled * rightSpan)

from IPython import embed as shell
# shell()

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

        # intro text:
        perc_signal = 50
        if (int(self.session.subject_nr) % 2) == 0:
            text_string1 = "Press z for 'diagonal' and m for 'cardinal'.\n\n{}% of trials will be diagonal.".format(perc_signal)
        elif (int(self.session.subject_nr) % 2) == 1:
            text_string1 = "Press z for 'cardinal' and m for 'diagonal'.\n\n{}% of trials will be diagonal.".format(perc_signal)
        text_string2 = "Please sit still throughout the experiment, and try to relax and minimize blinking.\n\nPress spacebar to start."
        self.intro_text1 = TextStim(win=self.session.win, text=text_string1, pos=(0.0, 3), color=(1, 0, 0), height=1)
        self.intro_text2 = TextStim(win=self.session.win, text=text_string2, pos=(0.0, -3), height=0.5)

        #point of fixation
        self.fixation = Circle(
                win=self.session.win, units='pix', autoDraw=True,
                radius=5, fillColor='black',
        )

        # #edges (circle)
        # self.circle = Circle( 
        #         win=self.session.win, units='pix', autoDraw=False,
        #         size=800, edges=100, lineColor='black', 
        # )

        #grating (parallel lines)
        size = 800
        sf = 0.02
        self.grating = GratingStim( 
            win=self.session.win, tex='sin', mask='gauss', units='pix', 
            size=[size,size], contrast=0.5, opacity=1, sf=sf
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

        elif (self.phase == 2) | (self.phase == 4) | (self.phase == 6) | (self.phase == 8) | (self.phase == 10) | (self.phase == 12) | (self.phase == 14) | (self.phase == 16):
            self.grating.ori = self.orientations[min(self.phase-2,7)]
            self.grating.draw()
            self.fixation.draw()

        elif (self.phase == 3) | (self.phase == 5) | (self.phase == 7) | (self.phase == 9) | (self.phase == 11) | (self.phase == 13) | (self.phase == 15) | (self.phase == 17):
            self.fixation.draw()

        elif self.phase == 18: #report
            self.fixation.fillColor = 'blue'
            self.fixation.draw()
     
        elif self.phase == 19: #rdelay
            self.fixation.fillColor = 'black'
            self.fixation.draw()

        elif self.phase == 20: # Feedback
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
        
        elif self.phase == 13: # Attribution response
            self.fixation.draw()

        elif self.phase == 14: # ITI
            self.fixation.draw()

    def get_events(self):
        events = super().get_events()

        if (self.phase == 0) and (self.trial_nr == 0):
            for key, t in events:
                if key in ['space']:
                    self.stop_phase()

        if self.phase == 10:
            for key, t in events:
                if key in ['z','m']:
                    if ((int(self.session.subject_nr) % 2) == 0) and (key == 'z') and (np.mean(self.parameters['DV']) > 0): # even: z for diag
                        self.parameters['correct'] = 1
                    elif ((int(self.session.subject_nr) % 2) == 0) and (key == 'm') and (np.mean(self.parameters['DV']) < 0): # even: m for card
                        self.parameters['correct'] = 1
                    elif ((int(self.session.subject_nr) % 2) == 1) and (key == 'z') and (np.mean(self.parameters['DV']) < 0): # odd: z for card
                        self.parameters['correct'] = 1
                    elif ((int(self.session.subject_nr) % 2) == 1) and (key == 'm') and (np.mean(self.parameters['DV']) > 0): # odd: m for diag
                        self.parameters['correct'] = 1
                    else:
                        self.parameters['correct']=0

                    self.stop_phase()

class GaborSession(PylinkEyetrackerSession): #Run session (or self) with multiple trials

    # initialize parent class that will contain several trials!
    def __init__(self, output_str, output_dir=None, settings_file=None, task='diagonal_cardinal', n_trials=1, awake=0, eyetracker_on=True):
        super().__init__(output_str=output_str, output_dir=output_dir, settings_file=settings_file, eyetracker_on=eyetracker_on)
        self.task = task
        self.n_trials = n_trials  # just an example argument
        self.awake = awake
        self.subject_nr = int(output_str.split('_')[0])
        print(self.subject_nr)
        self.trials = []  # will be filled with Trials later

    #Run a session with several trials that are created via "create_trials"
    def run(self):
        # self.create_trials() #creation of the trials via def created before
        # self.start_experiment()
        
        if self.eyetracker_on:
            self.calibrate_eyetracker()
        self.start_experiment()
        if self.eyetracker_on:
            self.start_recording_eyetracker()

        # staircase:
        staircase = data.QuestHandler(startVal=0.65, startValSd=0.15,
            pThreshold=0.75, grain=0.01, gamma=0.5, delta=0.01,
            nTrials=self.n_trials, minVal=0.5, maxVal=1)
        

        #Defining the stimuli wrt the task + conditions 
        stimuli = ['diagonal', 'cardinal']
        conditions = ['normal', 'impossible']

        #creation of a dict with all the trial parameters (task, stimuli, contrast of each participant, condition) = all the combination possible?
        n_trials_per_strata = int(self.n_trials / len(stimuli) / len(conditions)) #number of trials per ?
        trial_parameters = [] 
        for stim in stimuli: 
            for cond in conditions:
                for t in range(n_trials_per_strata):
                    trial_parameters.append({'stimulus': stim, 'condition': cond})
        random.shuffle(trial_parameters)

        for i in range(self.n_trials):
            stim = trial_parameters[i]['stimulus']
            cond = trial_parameters[i]['condition']
            if cond == 'impossible':
                difficulty = 0.5
                evidences = [make_evidence(difficulty) for _ in range(500)]
                evidences_mean = np.array([np.mean(e) for e in evidences])
                loc = np.where(np.abs(evidences_mean)<0.01)[0][0]
                evidence = evidences[loc]
                difficulty_actual = abs(np.mean(evidence))+0.5
                print(np.mean(evidence))
                print(difficulty_actual)
            else:
                difficulty = staircase._nextIntensity
                if stim == 'cardinal':
                    evidence = make_evidence(1-difficulty)
                    difficulty_actual = abs(np.mean(evidence)-0.5)
                elif stim == 'diagonal':
                    evidence = make_evidence(difficulty)
                    difficulty_actual = np.mean(evidence)+0.5
                if difficulty_actual > 1:
                    difficulty_actual = 1
                if difficulty_actual < 0:
                    difficulty_actual = 0
                print()
                print(difficulty)
                print(difficulty_actual)

            # ori = evidence_to_ori(evidence)
            ori = evidence_to_ori(evidence, 0, 1, -45, 45)
            print(ori)

            parameters = {'task': self.task, 'awake':self.awake, 'stimulus': stim, 'condition': cond, 
                            'difficulty':difficulty, 'difficulty_actual':difficulty_actual, 'orientation': ori, 'DV': evidence}

            if i == 0:
                phase_durations=[30,  1, 0.2, 0.05, 0.2, 0.05, 0.2, 0.05, 0.2, 0.05, 0.2, 0.05, 0.2, 0.05, 0.2, 0.05, 0.2, 0.05, 5, 0.25, 3, 5, np.random.uniform(1.5,2.5,1)]
                # phase_durations=[30,  1, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 5, 0.25, np.random.uniform(1.5,2.5,1)]
            else:
                phase_durations=[0.1, 1, 0.2, 0.05, 0.2, 0.05, 0.2, 0.05, 0.2, 0.05, 0.2, 0.05, 0.2, 0.05, 0.2, 0.05, 0.2, 0.05, 5, 0.25, 3, 5, np.random.uniform(1.5,2.5,1)]
                # phase_durations=[0.1, 1, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 5, 0.25, np.random.uniform(1.5,2.5,1)]


            trial = DetectionTrial(
                session=self,
                trial_nr=i,
                phase_durations=phase_durations,
                timing='seconds',
                phase_names=('intro', 'baseline', 'stim1', 'stim2', 'stim3', 'stim4', 'stim5', 'stim6', 'stim7', 'stim8', 'decision', 'delay', 'iti'),
                parameters=parameters,
                load_next_during_phase=None,
                verbose=True,
            )

            trial.run()

            # update staircase:
            correct = trial.parameters['correct']
            print(correct)
            if correct == -1:
                pass
            else:
                if cond == 'normal':
                    staircase.addResponse(correct, intensity=difficulty_actual)
                    staircase.calculateNextIntensity()
     
        self.close()

if __name__ == '__main__':
    subject_nr = input('Subject #: ')
    block_nr = input('Block #: ')
    awake = 1 #input('How awake are you [1-7]? ')
    dt = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    settings = op.join(op.dirname(__file__), 'settings.yml')
    my_sess = GaborSession(output_str='{}_{}_{}'.format(subject_nr, block_nr, dt), 
                           output_dir='data/0_staircase', 
                           settings_file=settings, 
                           task='diagonal_cardinal', 
                           n_trials=100, awake=awake,
                           eyetracker_on=False) #Set and run a session
    my_sess.run()
    time.sleep(1) # Sleep for 1 second
   
    # analyze + data acquisition (file with the data)
    filename = glob.glob(op.join('data', '0_staircase', '{}_{}_{}*events.tsv'.format(subject_nr, block_nr, dt)))[0]
    df = pd.read_csv(filename, sep='\t')
    print(df.head())
    
    df = df.loc[(df['event_type']=='iti')&(df['trial_nr']>20),:]
    f_correct = (df.loc[df['condition']=='impossible', 'correct']==1).mean() * 100
    print('% correct = {}'.format(f_correct))
    f_correct = (df.loc[df['condition']=='normal', 'correct']==1).mean() * 100
    print('% correct = {}'.format(f_correct))

    # analyze:
    filename = glob.glob(op.join('data', '0_staircase', '{}_{}_{}*events.tsv'.format(subject_nr, block_nr, dt)))[0]
    df = pd.read_csv(filename, sep='\t')
    print(df.head())
    fig = plt.figure()
    plt.plot(df['trial_nr'], df['difficulty'])
    plt.title('{}'.format(df['difficulty'].iloc[-1]))
    plt.tight_layout()
    fig.savefig(filename.replace('events.tsv', 'difficulty.pdf'))