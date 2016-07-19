# Note these are just sample tests
# Not very comprehensive tests

from unittest import TestCase

from hidden_markov import hmm 
import numpy as np

states = ('s', 't')

#list of possible observations
possible_observation = ('A','B' )

# The observations that we observe and feed to the model
observations = ('A', 'B','B','A')
obs4 = ('B', 'A','B')

# Tuple of observations
observation_tuple = []
observation_tuple.extend( [observations,obs4] )
quantities_observations = [10, 20]

# Numpy arrays of the data
start_probability = np.matrix( '0.5 0.5')
transition_probability = np.matrix('0.6 0.4 ;  0.3 0.7 ')
emission_probability = np.matrix( '0.3 0.7 ; 0.4 0.6 ' )


class TestHmm(TestCase):

########## Training HMM ####################
    def test_forward(self):

        #Declare Class object
        test = hmm(states,possible_observation,start_probability,transition_probability,emission_probability)

        # Forward algorithm
        forw_prob = (test.forward_algo(observations))
        forw_prob = round(forw_prob, 5)
        self.assertEqual(0.05153, forw_prob)

    def test_viterbi(self):
        #Declare Class object
        test = hmm(states,possible_observation,start_probability,transition_probability,emission_probability)
        # Viterbi algorithm
        vit_out = (test.viterbi(observations))
        self.assertEqual(['t','t','t','t'] , vit_out)

    def test_log(self):
        #Declare Class object
        test = hmm(states,possible_observation,start_probability,transition_probability,emission_probability)
        # Log prob function
        prob = test.log_prob(observation_tuple, quantities_observations)
        prob = round(prob, 3)
        self.assertEqual(-67.920, prob)

    def test_train_hmm(self):
        #Declare Class object
        test = hmm(states,possible_observation,start_probability,transition_probability,emission_probability)
        # Baum welch Algorithm
        num_iter=1000
        e,t,s = test.train_hmm(observation_tuple,num_iter,quantities_observations)


