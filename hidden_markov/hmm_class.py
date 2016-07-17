import numpy as np

class hmm:


    def __init__(self, states, observations, start_prob , trans_prob,  em_prob):

        # start, em and trans_prob
        self.states = states
        self.observations = observations
        self.start_prob = start_prob
        self.trans_prob = trans_prob
        self.em_prob = em_prob

        self.generate_obs_map()
        self.generate_state_map()

        # Raise error if it is wrong data-type
        if(type(self.em_prob) != np.matrixlib.defmatrix.matrix):
            raise TypeError("Emission probability is not a numpy Matrix")

        if(type(self.trans_prob) != np.matrixlib.defmatrix.matrix):
            raise TypeError("Transition probability is not a numpy Matrix")

        if(type(self.start_prob) != np.matrixlib.defmatrix.matrix):
            raise TypeError("Start probability is not a numpy Matrix")

        if( type(self.states) is not list and type(self.states) is not tuple):
            raise TypeError("States is not a list/tuple")

        if(type(self.observations) is not list and type(self.observations) is not tuple):
            raise TypeError("Observations is not a list/tuple")

        # Convert everything to lists
        self.states=list(self.states)
        self.observations=list(self.observations)

        # Dimension Check
        s_len = len(states)
        o_len = len(observations)

        if( (s_len,o_len)!= self.em_prob.shape ):
            print("Input has incorrect dimensions, Correct dimensions is (%d,%d)" % (s_len,o_len))
            raise ValueError("Emission probability has incorrect dimensions")

        if( (s_len,s_len)!= self.trans_prob.shape ):
            print("Input has incorrect dimensions, Correct dimensions is (%d,%d)" % (s_len,s_len))
            raise ValueError("Transition probability has incorrect dimensions")

        if( s_len!= (self.start_prob).shape[1]):
            print("Input has incorrect dimensions, Correct dimensions is %d" % s_len)
            raise ValueError("Start probability has incorrect dimensions")

        # No negative numbers
        if(not( (self.start_prob>=0).all() )):
            raise ValueError("Negative probabilities are not allowed")

        if(not( (self.em_prob>=0).all() )):
            raise ValueError("Negative probabilities are not allowed")

        if(not( (self.trans_prob>=0).all() )):
            raise ValueError("Negative probabilities are not allowed")

        # Summation of probabilities is 1
        # create a list of 1's
        tmp2 = [ 1 for i in range(s_len) ]

        # find summation of emission prob
        summation = np.sum(em_prob,axis=1)
        tmp1 = list (np.squeeze(np.asarray(summation)))

        #Compare
        if(tmp1 != tmp2):
            raise ValueError("Probabilities entered for emission matrix are invalid")

        # find summation of transition prob
        summation = np.sum(trans_prob,axis=1)
        tmp1 = list (np.squeeze(np.asarray(summation)))

        #Compare
        if(tmp1 != tmp2):
            raise ValueError("Probabilities entered for transition matrix are invalid")

        summation = np.sum(start_prob,axis=1)
        if (summation[0,0]!=1):
            raise ValueError("Probabilities entered for start state are invalid")

    # ================ Generate state_map ===================

    def generate_state_map(self):
        self.state_map = {}
        for i,o in enumerate(self.states):
            self.state_map[i] = o

    # ================ Generate Obs_map ===================

    def generate_obs_map(self):
        self.obs_map = {}
        for i,o in enumerate(self.observations):
            self.obs_map[o] = i

    # ================Viterbi ===================
    """
    Function returns the most likely path, and its associated probability
    Function makes assumption that order of states is same in state,start_prob,em_prob,trans_prob
    start_prob,em_prob,trans_prob are numpy objects
    """

    def viterbi(self,observations):
        # Find total states,observations
        total_stages = len(observations)
        num_states = len(self.states)

        # initialize data
        # Path stores the state sequence giving maximum probability
        old_path = np.zeros( (total_stages, num_states) )
        new_path = np.zeros( (total_stages, num_states) )

        # Find initial delta
        # Map observation to an index
        # delta[s] stores the probability of most probable path ending in state 's'
        ob_ind = self.obs_map[ observations[0] ]
        delta = np.multiply ( np.transpose(self.em_prob[:,ob_ind]) , self.start_prob )

        # initialize path
        old_path[0,:] = [i for i in range(num_states) ]

        # Find delta[t][x] for each state 'x' at the iteration 't'
        # delta[t][x] can be found using delta[t-1][x] and taking the maximum possible path
        for curr_t in range(1,total_stages):

            # Map observation to an index
            ob_ind = self.obs_map[ observations[curr_t] ]
            # Find temp and take max along each row to get delta
            temp  =  np.multiply (np.multiply(delta , self.trans_prob.transpose()) , self.em_prob[:, ob_ind] )

            # Update delta
            delta = temp.max(axis = 1).transpose()

            # Find state which is most probable using argax
            # Convert to a list for easier processing
            max_temp = temp.argmax(axis=1).transpose()
            max_temp = np.ravel(max_temp).tolist()

            # Update path
            for s in range(num_states):
                new_path[:curr_t,s] = old_path[0:curr_t, max_temp[s] ]

            new_path[curr_t,:] = [i for i in range(num_states) ]
            old_path = new_path.copy()


        # Find the state in last stage, giving maximum probability
        final_max = np.argmax(np.ravel(delta))
        best_path = old_path[:,final_max].tolist()
        best_path_map = [ self.state_map[i] for i in best_path]

        return best_path_map, delta[0,final_max]

    # =========== Forward =======================

    """
    Function returns the probability of an observation sequence
    Function makes assumption that order of states is same in state,start_prob,em_prob,trans_prob
    start_prob,em_prob,trans_prob are numpy objects
    """


    def forward_algo(self,observations):
        # Store total number of observations
        total_stages = len(observations)

        # Alpha[i] stores the probability of reaching state 'i' in stage 'j' where 'j' is the iteration number

        # Inittialize Alpha
        ob_ind = self.obs_map[ observations[0] ]
        alpha = np.multiply ( np.transpose(self.em_prob[:,ob_ind]) , self.start_prob )

        # Iteratively find alpha(using knowledge of alpha in the previous stage)
        for curr_t in range(1,total_stages):
            ob_ind = self.obs_map[observations[curr_t]]
            alpha = np.dot( alpha , self.trans_prob)
            alpha = np.multiply( alpha , np.transpose( self.em_prob[:,ob_ind] ))

        # Sum the alpha's over the last stage
        total_prob = alpha.sum()
        return ( total_prob )


    # ============Forward-Backward=================

    """
    Function trains start,emission and transition probabilities for a given obervation sequence
    Uses the forward-backward method(principle of expectation maximization_
    """


    def alpha_cal(self,observations):
    # Calculate alpha matrix and return it

        num_states = self.em_prob.shape[0]
        total_stages = len(observations)

        # Initialize values
        ob_ind = self.obs_map[ observations[0] ]
        alpha = np.asmatrix(np.zeros((num_states,total_stages)))

        # Handle alpha base case
        alpha[:,0] = np.multiply ( np.transpose(self.em_prob[:,ob_ind]) , self.start_prob ).transpose()

        # Iteratively calculate alpha(t) for all 't'
        for curr_t in range(1,total_stages):
            ob_ind = self.obs_map[observations[curr_t]]
            alpha[:,curr_t] = np.dot( alpha[:,curr_t-1].transpose() , self.trans_prob).transpose()
            alpha[:,curr_t] = np.multiply( alpha[:,curr_t].transpose() , np.transpose( self.em_prob[:,ob_ind] )).transpose()

        # return the computed alpha
        return alpha


    def beta_cal(self,observations):
    # Calculate Beta maxtrix

        num_states = self.em_prob.shape[0]
        total_stages = len(observations)

        # Initialize values
        ob_ind = self.obs_map[ observations[total_stages-1] ]
        beta = np.asmatrix(np.zeros((num_states,total_stages)))

        # Handle beta base case
        beta[:,total_stages-1] = 1

        # Iteratively calculate beta(t) for all 't'
        for curr_t in range(total_stages-1,0,-1):
            ob_ind = self.obs_map[observations[curr_t]]
            beta[:,curr_t-1] = np.multiply( beta[:,curr_t] , self.em_prob[:,ob_ind] )
            beta[:,curr_t-1] = np.dot( self.trans_prob, beta[:,curr_t-1] )

        # return the computed beta
        return beta


    def forward_backward(self,observations):
        num_states = self.em_prob.shape[0]
        num_obs = len(observations)

        # Find alpha and beta values
        alpha = self.alpha_cal(observations)
        beta = self.beta_cal(observations)

        # Calculate sum [alpha(num_obs)]
        # i.e calculate the last row of alpha
        prob_obs_seq = np.sum(alpha[:,num_obs-1])

        # Calculate delta1
        # Delta is simply product of alpha and beta
        delta1 = np.multiply(alpha,beta)/ prob_obs_seq

        return delta1


    def train_emission(self,observations):
        # Initialize matrix
        new_em_prob = np.asmatrix(np.zeros(self.em_prob.shape))

        # Indexing position of unique observations in the observation sequence
        selectCols=[]
        for i in range(self.em_prob.shape[1]):
            selectCols.append([])
        for i in range(len(observations)):
            selectCols[ self.obs_map[observations[i]] ].append(i)

        # Calculate delta matrix
        delta = self.forward_backward(observations)

        # Sum the rowise of delta matrix, which gives probability of a particular state
        totalProb = np.sum(delta,axis=1)

        # Re-estimate emission matrix
        for i in range(self.em_prob.shape[0]):
            for j in range(self.em_prob.shape[1]):
                new_em_prob[i,j] = np.sum(delta[i,selectCols[j]])/totalProb[i]
        return new_em_prob


    def train_transition(self,observations):
        # Initialize transition matrix
        new_trans_prob = np.asmatrix(np.zeros(self.trans_prob.shape))

        # Find alpha and beta
        alpha = self.alpha_cal(observations)
        beta = self.beta_cal(observations)

        # calculate transition matrix values
        for t in range(len(observations)-1):
            temp1 = np.multiply(alpha[:,t],beta[:,t+1].transpose())
            temp1 = np.multiply(self.trans_prob,temp1)
            new_trans_prob = new_trans_prob + np.multiply(temp1,self.em_prob[:,self.obs_map[observations[t+1]]].transpose())

        # Normalize values so that sum of probabilities is 1
        for i in range(self.trans_prob.shape[0]):
            new_trans_prob[i,:] = new_trans_prob[i,:]/np.sum(new_trans_prob[i,:])

        return new_trans_prob

    def train_start_prob(self,observations):
        delta = self.forward_backward(observations)
        return delta[:,0].transpose()

    def train_hmm(self,observations,iterations):

        eps = 0.001

        emProbNew = np.asmatrix(np.zeros((self.em_prob.shape)))
        transProbNew = np.asmatrix(np.zeros((self.trans_prob.shape)))
        startProbNew = np.asmatrix(np.zeros((self.start_prob.shape)))
        prob = - float('inf')

        # Train the model 'iteration' number of times
        # store em_prob and trans_prob copies since you should use same values for one loop
        for i in range(iterations):

            emProbNew= self.train_emission(observations)
            transProbNew = self.train_transition(observations)
            startProbNew = self.train_start_prob(observations)

            self.em_prob,self.trans_prob = emProbNew,transProbNew
            self.start_prob = startProbNew

            if(self.forward_algo(observations) - prob)>eps:
                prob = self.forward_algo(observations)
            else:
                break

        return self.em_prob, self.trans_prob , self.start_prob

    def randomize(observations):
    # Generate random transition,start and emission probabilities

        # Store observations and states
        num_obs = len(self.observations)
        num_states = len(states)

        # Generate a random list with sum of numbers = 1
        a = np.random.random(num_states)
        a /= a.sum()
        # Initialize start_prob
        self.start_prob = a

        # Initialize transition matrix
        # Fill each row with a list that sums upto 1
        self.trans_prob = np.asmatrix(np.zeros((num_states,num_states)))
        for i in range(num_states):
            a = np.random.random(num_states)
            a /= a.sum()
            self.trans_prob[i,:] = a

        # Initialize emission matrix
        # Fill each row with a list that sums upto 1
        self.em_prob = np.asmatrix(np.zeros((num_states,num_obs)))
        for i in range(num_states):
            a = np.random.random(num_obs)
            a /= a.sum()
            self.em_prob[i,:] = a

        return self.start_prob, self.trans_prob, self.em_prob


# ============================================================

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

