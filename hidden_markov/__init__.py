"""
This package implements the hidden markov model. The implementation assumes the 1st order markov assumption, i.e. the current state depends solely on the previous state.  Hidden markov models primarily use three tasks::

    * Forward Algorithm    : Finding the probabiliity of an observation sequence.
    * Viterbi Algorithm    : Finding the most probable sequence of hidden states, given the observation sequence.
    * Baum-welch Algorithm : Finding the best set of model parameters, given a set of observation sequences.

Features:
--------
    * The package implements a scaled version of the Viterbi and the Baum-Welch algorithm. The HMM is prone to underflow error, and hence log-probabilities and scaled probabilities are used to prevent this. However the forward algorithm does not use any scaling, and the absolute probability is reported, hence use this function with caution and enusure that underflow error does not occur
    * The package uses numpy matrices for computations, in order to provide a decrease in runtime. 

"""
from .hmm_class import hmm
