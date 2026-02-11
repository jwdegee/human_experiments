import os.path as op
import random
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
import datetime

from psychopy import prefs
prefs.hardware['audioLib'] = ['PTB']
prefs.hardware['audioDevice'] = 'Headphones (Realtek(R) Audio)'
from psychopy import sound, core
from psychopy.visual import TextStim
from psychopy.visual import GratingStim

from exptools2.core import PylinkEyetrackerSession
from exptools2.core import Trial

import serial #Import the serial library
port = serial.Serial('COM3', 115200)

def send_trigger(serialport: serial.Serial, signal_byte: int):  
    signal = bytes([signal_byte]) 
    serialport.write(signal)

def into_logspaced_freqs(values, values_min, values_max, min_f, nr_octaves):
    
    # express into octaves:
    span = values_max - values_min
    values_scaled = (values-values_min) / span * 100
    all_freqs = np.logspace(start=np.log10(min_f),
                            stop=np.log10(min_f*(2**nr_octaves)),
                            num=10000)
    freqs = np.percentile(all_freqs,values_scaled)
    return freqs

def into_ories(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = (value - leftMin) / leftSpan
    return rightMin + (valueScaled * rightSpan)

#def make_samples(H, mu, sigma, cutoff, n_samples):

#    state = int(np.random.rand(1)[0]>0.5)
#    samples = np.zeros(n_samples)
#    states = np.zeros(n_samples)
#    for i in range(n_samples):

        # check if state flipped:
#        if np.random.rand(1)[0] < H:
#            if state == 0:
#                state = 1
#            elif state == 1:
#                state = 0
        
        # draw sample:
#        sample = np.random.normal(mu[state], sigma[state], 1)[0]
        
        # threshold:
#        if sample < cutoff[0]:
#            sample = cutoff[0]
#        elif sample > cutoff[1]:
#            sample = cutoff[1]

        # add
  #      samples[i] = sample
  #      states[i] = state
 #   states[states==0] = -1

  #  LLRin = np.log(sp.stats.norm.pdf(samples,mu[0]*-1,sigma[0])/
 #               sp.stats.norm.pdf(samples,mu[1]*-1,sigma[1]))
#
 #   return states, samples, LLRin

def make_samples(H, mu, p, n_samples):

     state = int(np.random.rand(1)[0]>0.5)
     samples = np.zeros(n_samples)
     states = np.zeros(n_samples)
     for i in range(n_samples):

         # check if state flipped:
         if np.random.rand(1)[0] < H:
             if state == 0:
                 state = 1
             elif state == 1:
                 state = 0
        
         # draw sample:
         if np.random.rand(1)[0] > p:
             if state == 0:
                 sample = mu[1] # oddball
             elif state == 1:
                 sample = mu[0] # oddball
         else:
             if state == 0:
                 sample = mu[0] # standard
             elif state == 1:
                 sample = mu[1] # standard
        
         # add
         samples[i] = sample
         states[i] = state
     states[states==0] = -1

     return states, samples

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
                                    size=0.2,
                                    texRes=9,
                                    color=fixation_color,
                                    sf=0)

        # gabor:
        if 'ori' in self.parameters:
            size = 800
            sf = 0.02
            self.grating = GratingStim( 
                win=self.session.win, tex='sin', mask='gauss', units='pix', 
                size=[size,size], contrast=1, opacity=1, sf=sf, texRes=128,
            )
            self.grating.ori = self.parameters['ori']
        
        # sound:
        if 'freq' in self.parameters:
            self.sound_played = False
            self.snd = sound.Sound(value=self.parameters['freq'], hamming=True, secs=0.5, octave=4, stereo=True)
            self.snd.setVolume(self.parameters['volume'])
            
        # intro text:
        text_string1 = "You will see or hear a sequence of stimuli"
        text_string2 = "Please fixate on the black fixation mark, try to relax and try to minimize blinking.\n\nPress spacebar to start."
        self.intro_text1 = TextStim(win=self.session.win, text=text_string1, pos=(0.0, 3), color=(1, 0, 0), height=1)
        self.intro_text2 = TextStim(win=self.session.win, text=text_string2, pos=(0.0, -3), height=0.5)

        self.stimulus_trigger_played = False

    def draw(self):
        """ Draws stimuli """
        
        t = self.trialClock.getTime()
        
        if (self.phase == 0) & (self.trial_nr == 0):  # intro
            self.intro_text1.draw()
            self.intro_text2.draw()
            self.fixation.draw()
        elif (self.phase == 0) & (self.trial_nr != 0):  # intro
            self.fixation.draw()
        
        if self.phase == 1: # delay
            self.fixation.draw()
            self.phase1_time = self.trialClock.getTime()

        if self.phase == 2: # stimulus
            
            t = self.trialClock.getTime() - self.phase1_time
            
            if 'freq' in self.parameters:
                if not self.sound_played:
                    self.snd.play()
                    self.sound_played = True
                if not self.stimulus_trigger_played:
                    send_trigger(port, 2)
                    self.stimulus_trigger_played  = True


            if 'ori' in self.parameters:
                if t < 0.1:
                    contrast = t / 0.1 # ramp from 0 at the beginning, 1.0 at the end
                elif t > 0.4:
                    contrast = 1.0 - ((t-0.4) / 0.1) # ramp down from 1 to 0
                else:
                    contrast = 1.0 # must be in the middle 2s period
                contrast = max((contrast, 0))         
                print(contrast)
                self.grating.contrast = contrast
                self.grating.draw()
                # self.grating.setPhase(0.5*t)
                
            self.fixation.draw()

    def get_events(self):
        events = super().get_events()

        if (self.phase == 0) and (self.trial_nr == 0):
            for key, t in events:
                if key in ['space']:
                    self.stop_phase()

class TestEyetrackerSession(PylinkEyetrackerSession):
    """ Simple session with x trials. """

    def __init__(self, output_str, output_dir=None, settings_file=None, n_trials=10, eyetracker_on=True):
        """ Initializes TestSession object. """
        self.n_trials = n_trials
        self.subject_id = int(output_str.split('_')[0])
        self.block_id = int(output_str.split('_')[1])
                
        super().__init__(output_str, output_dir=output_dir,
                         settings_file=settings_file, eyetracker_on=eyetracker_on)

    def create_trials(self, timing='seconds'):

        # conditions = ['control', 'surprise', 'control', 'surprise', 'control', 'surprise']
        condition = 'surprise'

        # draw states:
        H = 0.05
        if condition == 'control':
            frequencies = [500, 650, 800, 1000, 1300, 1600, 2000, 2600, 3300, 4000]
            freqs = []
            for f in frequencies:
                for _ in range(int(self.n_trials / len(frequencies))):
                    freqs.append(f)
            random.shuffle(freqs)
            states = np.ones(self.n_trials)
        elif condition == 'surprise':
        #elif self.condition == 'auditory':
            states, freqs = make_samples(H=H, 
                                            mu=[1000,2000], 
                                            p=0.9,
                                            n_samples=self.n_trials)

        # map onto frequencies:
        #freqs = into_logspaced_freqs(freqs, -cutoff, cutoff, 500, 3)
        #print(min(freqs))
        #print(max(freqs))

        # compute volumes (correcting for equal-loudness contour):
        from iso226 import iso226_spl_itpl
        contour_interpolated = iso226_spl_itpl(L_N=40, hfe=False, k=3)
        contour_inverted = 1 / contour_interpolated(freqs)
        volumes = contour_inverted / contour_inverted.max()

        print(states)

        self.trials = []
        for trial_nr in range(self.n_trials):
            parameters = {
                            'condition':condition,
                            'hazard':H,
                            'state':states[trial_nr],
                            'freq':freqs[trial_nr],
                            'volume':volumes[trial_nr]}
            
            if trial_nr == 0:
                durations=(30, 1, 0.5)
            elif trial_nr == self.n_trials-1:
                durations=(0, 1, 5)
            else:
                durations=(0, 1, 0.5)

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
        port.close()

if __name__ == '__main__':
    subject_nr = input('Subject #: ')
    block_nr = input('Block #: ')
    dt = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    settings = op.join(op.dirname(__file__), 'settings.yml')
    session = TestEyetrackerSession(output_str='{}_{}_{}'.format(subject_nr, block_nr, dt),
                                    output_dir='data/',
                                    eyetracker_on=True, n_trials=200, settings_file=settings)
    session.create_trials()
    session.run()