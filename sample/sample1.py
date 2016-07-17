from hidden_markov import hmm 
from hidden_markov import hmm_scaling

import numpy as np

########## Training HMM ####################

def test():
    states = ('s1', 's2')

    #list of possible observations
    possible_observation = ('R','W', 'B')


    # The observations that we observe and feed to the model
    observations = ('R', 'W','B','B')
    obs4 = ('R', 'R','W','B')
    obs3 = ('R', 'B','W','B')
    obs2 = ('R', 'W','B','R')


    # Numpy arrays of the data
    start_probability = np.matrix( '0.8 0.2 ')
    transition_probability = np.matrix('0.6 0.4 ;  0.3 0.7 ')
    emission_probability = np.matrix( '0.3 0.4 0.3 ; 0.4  0.3  0.3 ' )


    # states = ('Healthy', 'Fever')
    #
    # #list of possible observations
    # possible_observation = ('normal','cold', 'dizzy')
    #
    # state_map = { 0 :'Healthy',1: 'Fever' }
    #
    # # The observations that we observe and feed to the model
    # observations = ('normal', 'cold','dizzy')
    #
    # # Numpy arrays of the data
    # start_probability = np.matrix( '0.6 0.4 ')
    # transition_probability = np.matrix('0.7 0.3 ;  0.4 0.6 ')
    # emission_probability = np.matrix( '0.5 0.4 0.1 ; 0.1  0.3  0.6 ' )

    test = hmm(states,possible_observation,start_probability,transition_probability,emission_probability)
    print (test.viterbi(observations))
    print (test.forward_algo(observations))

    print ("")

    # start_prob,em_prob,trans_prob=start_probability,emission_probability,transition_probability
    forward1 = test.alpha_cal(observations)
    print ("probability of sequence with original parameters : %f"%( np.sum(forward1[:,3])))

    num_iter=4
    print ("applied Baum welch on")
    print (observations)
    e,t,s = test.train_hmm(observations,num_iter)
    forward1 = test.alpha_cal(observations)
    print("parameters emission,transition and start")
    print(e)
    print(t)
    print(s)
    print ("probability of sequence after %d iterations : %f"%(num_iter,np.sum(forward1[:,3])))

########## Training HMM ####################
def test_scale():

    states = ('s', 't')

    #list of possible observations
    possible_observation = ('A','B' )

    state_map = { 0 :'s', 1: 't' }

    # The observations that we observe and feed to the model
    observations = ('A', 'B','B','A')
    obs4 = ('B', 'A','B')

    #  obs3 = ('R', 'W','W','W')
    #  obs2 = ('W', 'W','R','R')

    observation_tuple = []
    observation_tuple.extend( [observations,obs4] )
    quantities_observations = [10, 20]

    # Numpy arrays of the data
    start_probability = np.matrix( '0.5 0.5 ')
    transition_probability = np.matrix('0.6 0.4 ;  0.3 0.7 ')
    emission_probability = np.matrix( '0.3 0.7 ; 0.4 0.6 ' )

    test = hmm_scaling(states,possible_observation,start_probability,transition_probability,emission_probability)

    # start_prob,em_prob,trans_prob=start_probability,emission_probability,transition_probability
    prob = test.log_prob(observation_tuple, quantities_observations)
    print ("probability of sequence with original parameters : %f"%(prob))
    print ("")

    num_iter=1000

    print ("applied Baum welch on")
    print (observation_tuple)

    e,t,s = test.train_hmm(observation_tuple,num_iter,quantities_observations)
    print("parameters emission,transition and start")
    print(e)
    print("")
    print(t)
    print("")
    print(s)

    prob = test.log_prob(observation_tuple, quantities_observations)
    print ("probability of sequence after %d iterations : %f"%(num_iter,prob))
