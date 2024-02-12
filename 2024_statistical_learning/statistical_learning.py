import numpy as np
import scipy as sp
from scipy import stats

from psychopy import prefs
prefs.hardware['audioLib'] = ['pygame']
from psychopy import sound
print(sound.Sound)
from psychopy.visual import TextStim
from psychopy.visual import GratingStim


from exptools2.core import PylinkEyetrackerSession
from exptools2.core import Trial

def into_logspaced_freqs(values, values_min, values_max, min_f, nr_octaves):
    
    # express into octaves:
    span = values_max - values_min
    values_scaled = (values-values_min) / span * 100
    all_freqs = np.logspace(start=np.log10(min_f),
                            stop=np.log10(min_f*(2**nr_octaves)),
                            num=10000)
    freqs = np.percentile(all_freqs,values_scaled)
    return freqs

def make_samples(H, mu, sigma, cutoff, n_samples):

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
        sample = np.random.normal(mu[state], sigma[state], 1)[0]
        
        # threshold:
        if sample < cutoff[0]:
            sample = cutoff[0]
        elif sample > cutoff[1]:
            sample = cutoff[1]

        # add
        samples[i] = sample
        states[i] = state
    states[states==0] = -1

    LLRin = np.log(sp.stats.norm.pdf(samples,mu[0]*-1,sigma[0])/
                sp.stats.norm.pdf(samples,mu[1]*-1,sigma[1]))

    return states, samples, LLRin

class TestTrial(Trial):
    """ Simple trial with text (trial x) and fixation. """
    def __init__(self, session, trial_nr, phase_durations, parameters, **kwargs):
        super().__init__(session, trial_nr, phase_durations, parameters=parameters, **kwargs)

        self.sound_played = False
        self.snd = sound.Sound(value=parameters['freq'], 
                volume=parameters['volume'], 
                secs=0.5, octave=4, stereo=-1) 
                # loops=0, sampleRate=44100, blockSize=128, 
                # preBuffer=-1, hamming=True, startTime=0, 
                # stopTime=-1, name='', autoLog=True)
        if parameters['state'] == -1:
            fixation_color = 'green'
        elif parameters['state'] == 1:
            fixation_color = 'red'
        
        self.fixation = GratingStim(self.session.win,
                                    pos=(0,0),
                                    tex='sin',
                                    mask='circle',
                                    size=0.5,
                                    texRes=9,
                                    color=fixation_color,
                                    sf=0)

    def draw(self):
        """ Draws stimuli """
        if self.phase == 0:
            self.fixation.draw()
        else:
            self.fixation.draw()
            if not self.sound_played:
                self.snd.play()
                self.sound_played = True

class TestEyetrackerSession(PylinkEyetrackerSession):
    """ Simple session with x trials. """

    def __init__(self, output_str, output_dir=None, settings_file=None, n_trials=10, eyetracker_on=True):
        """ Initializes TestSession object. """
        self.n_trials = n_trials
        super().__init__(output_str, output_dir=output_dir,
                         settings_file=settings_file, eyetracker_on=eyetracker_on)

    def create_trials(self, durations=(0.1, 1), timing='seconds'):
        
        # draw states:
        mu = 20          # means of generative distributions (polar angles relative to downward vertical midline of zero; + is left of midline)
        sigma = 20        # standard deviations
        cutoff = sp.stats.norm.ppf(0.99,loc=mu,scale=sigma)
        H = 0.10 # hazard rate of distribution switches
        states, samples, LLRin = make_samples(H=H, 
                                              mu=[-mu,mu], 
                                              sigma=[sigma,sigma],
                                              cutoff=[-cutoff,cutoff], 
                                              n_samples=self.n_trials)
        print(states)

        # map onto frequencies:
        freqs = into_logspaced_freqs(samples, -cutoff, cutoff, 500, 3)

        # compute volumes (correcting for equal-loudness contour):
        from iso226 import *
        contour_interpolated = iso226_spl_itpl(L_N=40, hfe=False, k=3)
        contour_inverted = 1 / contour_interpolated(freqs)
        volumes = contour_inverted / contour_inverted.max()

        self.trials = []
        for trial_nr in range(self.n_trials):

            parameters = {'state':states[trial_nr],
                            'sample':samples[trial_nr],
                            'LLRin':LLRin[trial_nr],
                            'freq':freqs[trial_nr],
                            'volume':volumes[trial_nr],}
            
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

    settings_file = 'settings.yml'
    session = TestEyetrackerSession('sub-01', eyetracker_on=False, n_trials=10, settings_file=settings_file)
    session.create_trials()
    session.run()
