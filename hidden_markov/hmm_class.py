import numpy as np

class hmm:

    """ Stores a hidden markov model object, and the model parameters.

    **Implemented Algorithms** :

        * Viterbi Algorithm
        * Forward Algorithm
        * Baum-Welch Algorithm

    """

    def __init__(self, states, observations, start_prob , trans_prob,  em_prob):
        """ Initialize The hmm class object.

        **Arguments**:

        :param states: The set of hidden states
        :type states: A list or tuple

        :param observations: The set unique of possible observations
        :type observations: A list or tuple

        :param start_prob: The start probabilities of various states, given in same order as 'states' variable. **start_prob[i] = probability( start at states[i] )**.
        :type start_prob: Numpy matrix, dimension = [ length(states) X 1 ]

        :param trans_prob: The transition probabilities, with ordering same as 'states' variable . **trans_prob[i,j] = probability(states[i] -> states[j])**.
        :type trans_prob: Numpy matrix, dimension = [ len(states) X len(states) ]  

        :param em_prob: The emission probabilities, with ordering same as 'states' variable and 'observations' variable. **em_prob[i,j] = probability(states[i],observations[j])**.
        :type em_prob: Numpy matrix, dimension = [ len(states) X len(observations) ]      

        **Example**:

        >>> states = ('s', 't')
        >>> possible_observation = ('A','B' )
        >>> # Numpy arrays of the data
        >>> start_probability = np.matrix( '0.5 0.5 ')
        >>> transition_probability = np.matrix('0.6 0.4 ;  0.3 0.7 ')
        >>> emission_probability = np.matrix( '0.3 0.7 ; 0.4 0.6 ' )
        >>> test = hmm(states,possible_observation,start_probability,transition_probability,emission_probability)

        """

        # start, em and trans_prob 
        self.states = states
        self.observations = observations
        self.start_prob = start_prob
        self.trans_prob = trans_prob
        self.em_prob = em_prob

        self._generate_obs_map()
        self._generate_state_map()

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

    def _generate_state_map(self):
        self.state_map = {}
        for i,o in enumerate(self.states):
            self.state_map[i] = o

    # ================ Generate Obs_map ===================

    def _generate_obs_map(self):
        self.obs_map = {}
        for i,o in enumerate(self.observations):
            self.obs_map[o] = i


    # ================ Forward algo===================

    def forward_algo(self,observations):
        """ Finds the probability of an observation sequence for given model parameters

        **Arguments**:

        :param observations: The observation sequence, where each element belongs to 'observations' variable declared with __init__ object. 
        :type observations: A list or tuple

        :return: The probability of occurence of the observation sequence
        :rtype: float 

        **Example**:

        >>> states = ('s', 't')
        >>> possible_observation = ('A','B' )
        >>> # Numpy arrays of the data
        >>> start_probability = np.matrix( '0.5 0.5 ')
        >>> transition_probability = np.matrix('0.6 0.4 ;  0.3 0.7 ')
        >>> emission_probability = np.matrix( '0.3 0.7 ; 0.4 0.6 ' )
        >>> # Initialize class object
        >>> test = hmm(states,possible_observation,start_probability,transition_probability,emission_probability)
        >>> observations = ('A', 'B','B','A')
        >>> print(test.forward_algo(observations))

        .. note::
            No scaling applied here and hence this routine is susceptible to underflow errors. Use :func:`hmm.log_prob` instead.

        """

        # Store total number of observations total_stages = len(observations) 
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

    # ================Viterbi ===================

    def viterbi(self,observations):
        """ The probability of occurence of the observation sequence

        **Arguments**:

        :param observations: The observation sequence, where each element belongs to 'observations' variable declared with __init__ object. 
        :type observations: A list or tuple

        :return: Returns a list of hidden states. 
        :rtype: list of states           
            
        **Features**:

        Scaling applied here. This ensures that no underflow error occurs.


        **Example**:

        >>> states = ('s', 't')
        >>> possible_observation = ('A','B' )
        >>> # Numpy arrays of the data
        >>> start_probability = np.matrix( '0.5 0.5 ')
        >>> transition_probability = np.matrix('0.6 0.4 ;  0.3 0.7 ')
        >>> emission_probability = np.matrix( '0.3 0.7 ; 0.4 0.6 ' )
        >>> # Initialize class object
        >>> test = hmm(states,possible_observation,start_probability,transition_probability,emission_probability)
        >>> observations = ('A', 'B','B','A')
        >>> print(test.viterbi(observations))

        """

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

        # Scale delta
        delta = delta /np.sum(delta)

        # initialize path
        old_path[0,:] = [i for i in range(num_states) ]

        # Find delta[t][x] for each state 'x' at the iteration 't'
        # delta[t][x] can be found using delta[t-1][x] and taking the maximum possible path
        for curr_t in range(1,total_stages):

            # Map observation to an index
            ob_ind = self.obs_map[ observations[curr_t] ]
            # Find temp and take max along each row to get delta
            temp  =  np.multiply (np.multiply(delta , self.trans_prob.transpose()) , self.em_prob[:, ob_ind] )

            # Update delta and scale it
            delta = temp.max(axis = 1).transpose()
            delta = delta /np.sum(delta)

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

        return best_path_map

    # ================ Baum Welch ===================

    def train_hmm(self,observation_list, iterations, quantities):
        """ Runs the Baum Welch Algorithm and finds the new model parameters

        **Arguments**:

        :param observation_list: A nested list, or a list of lists  
        :type observation_list: Contains a list  multiple observation sequences.

        :param iterations: Maximum number of iterations for the algorithm 
        :type iterations: An integer 

        :param quantities: Number of times, each corresponding item in  'observation_list' occurs.
        :type quantities: A list of integers

        :return: Returns the emission, transition and start probabilites as numpy matrices
        :rtype: Three numpy matices 
            
        **Features**:

        Scaling applied here. This ensures that no underflow error occurs.
            
        **Example**:

        >>> states = ('s', 't')
        >>> possible_observation = ('A','B' )
        >>> # Numpy arrays of the data
        >>> start_probability = np.matrix( '0.5 0.5 ')
        >>> transition_probability = np.matrix('0.6 0.4 ;  0.3 0.7 ')
        >>> emission_probability = np.matrix( '0.3 0.7 ; 0.4 0.6 ' )
        >>> # Initialize class object
        >>> test = hmm(states,possible_observation,start_probability,transition_probability,emission_probability)
        >>> 
        >>> observations = ('A', 'B','B','A')
        >>> obs4 = ('B', 'A','B')
        >>> observation_tuple = []
        >>> observation_tuple.extend( [observations,obs4] )
        >>> quantities_observations = [10, 20]
        >>> num_iter=1000
        >>> e,t,s = test.train_hmm(observation_tuple,num_iter,quantities_observations)
        >>> # e,t,s contain new emission transition and start probabilities

        """

        obs_size = len(observation_list)
        prob = float('inf')
        q = quantities

        # Train the model 'iteration' number of times
        # store em_prob and trans_prob copies since you should use same values for one loop
        for i in range(iterations):

            emProbNew = np.asmatrix(np.zeros((self.em_prob.shape)))
            transProbNew = np.asmatrix(np.zeros((self.trans_prob.shape)))
            startProbNew = np.asmatrix(np.zeros((self.start_prob.shape)))

            for j in range(obs_size):

                # re-assing values based on weight
                emProbNew= emProbNew + q[j] * self._train_emission(observation_list[j])
                transProbNew = transProbNew + q[j] * self._train_transition(observation_list[j])
                startProbNew = startProbNew + q[j] * self._train_start_prob(observation_list[j])


            # Normalizing
            em_norm = emProbNew.sum(axis = 1)
            trans_norm = transProbNew.sum(axis = 1)
            start_norm = startProbNew.sum(axis = 1)

            emProbNew = emProbNew/ em_norm.transpose()
            startProbNew = startProbNew/ start_norm.transpose()
            transProbNew = transProbNew/ trans_norm.transpose()


            self.em_prob,self.trans_prob = emProbNew,transProbNew
            self.start_prob = startProbNew

            if prob -  self.log_prob(observation_list,quantities)>0.0000001:
                prob = self.log_prob(observation_list,quantities)
            else:
                return self.em_prob, self.trans_prob , self.start_prob


        return self.em_prob, self.trans_prob , self.start_prob

    def _alpha_cal(self,observations):
        # Calculate alpha matrix and return it
        num_states = self.em_prob.shape[0]
        total_stages = len(observations)

        # Initialize values
        ob_ind = self.obs_map[ observations[0] ]
        alpha = np.asmatrix(np.zeros((num_states,total_stages)))
        c_scale = np.asmatrix(np.zeros((total_stages,1)))

        # Handle alpha base case
        alpha[:,0] = np.multiply ( np.transpose(self.em_prob[:,ob_ind]) , self.start_prob ).transpose()
        # store scaling factors, scale alpha
        c_scale[0,0] = 1/np.sum(alpha[:,0])
        alpha[:,0] = alpha[:,0] * c_scale[0]
        # Iteratively calculate alpha(t) for all 't'
        for curr_t in range(1,total_stages):
            ob_ind = self.obs_map[observations[curr_t]]
            alpha[:,curr_t] = np.dot( alpha[:,curr_t-1].transpose() , self.trans_prob).transpose()
            alpha[:,curr_t] = np.multiply( alpha[:,curr_t].transpose() , np.transpose( self.em_prob[:,ob_ind] )).transpose()

            # Store scaling factors, scale alpha
            c_scale[curr_t] = 1/np.sum(alpha[:,curr_t])
            alpha[:,curr_t] = alpha[:,curr_t] * c_scale[curr_t]

        # return the computed alpha
        return (alpha,c_scale)

    def _beta_cal(self,observations,c_scale):
        # Calculate Beta maxtrix
        num_states = self.em_prob.shape[0]
        total_stages = len(observations)

        # Initialize values
        ob_ind = self.obs_map[ observations[total_stages-1] ]
        beta = np.asmatrix(np.zeros((num_states,total_stages)))

        # Handle beta base case
        beta[:,total_stages-1] = c_scale[total_stages-1]

        # Iteratively calculate beta(t) for all 't'
        for curr_t in range(total_stages-1,0,-1):
            ob_ind = self.obs_map[observations[curr_t]]
            beta[:,curr_t-1] = np.multiply( beta[:,curr_t] , self.em_prob[:,ob_ind] )
            beta[:,curr_t-1] = np.dot( self.trans_prob, beta[:,curr_t-1] )
            beta[:,curr_t-1] = np.multiply( beta[:,curr_t-1] , c_scale[curr_t -1 ] )

        # return the computed beta
        return beta

    def _forward_backward(self,observations):
        num_states = self.em_prob.shape[0]
        num_obs = len(observations)

        # Find alpha and beta values
        alpha, c = self._alpha_cal(observations)
        beta = self._beta_cal(observations,c)

        # Calculate sum [alpha(num_obs)]
        # i.e calculate the last row of alpha
        prob_obs_seq = np.sum(alpha[:,num_obs-1])

        # Calculate delta1
        # Delta is simply product of alpha and beta
        delta1 = np.multiply(alpha,beta)/ prob_obs_seq 
        delta1 = delta1/c.transpose()

        return delta1

    def _train_emission(self,observations):
        # Initialize matrix
        new_em_prob = np.asmatrix(np.zeros(self.em_prob.shape))

        # Indexing position of unique observations in the observation sequence    
        selectCols=[]
        for i in range(self.em_prob.shape[1]):
            selectCols.append([])
        for i in range(len(observations)):
            selectCols[ self.obs_map[observations[i]] ].append(i)

        # Calculate delta matrix
        delta = self._forward_backward(observations)

        # Sum the rowise of delta matrix, which gives probability of a particular state
        totalProb = np.sum(delta,axis=1)

        # Re-estimate emission matrix
        for i in range(self.em_prob.shape[0]):
            for j in range(self.em_prob.shape[1]):
                new_em_prob[i,j] = np.sum(delta[i,selectCols[j]])/totalProb[i]
        return new_em_prob

    def _train_transition(self,observations):
        # Initialize transition matrix
        new_trans_prob = np.asmatrix(np.zeros(self.trans_prob.shape))

        # Find alpha and beta
        alpha,c = self._alpha_cal(observations)
        beta = self._beta_cal(observations,c)

        # calculate transition matrix values
        for t in range(len(observations)-1):
            temp1 = np.multiply(alpha[:,t],beta[:,t+1].transpose())
            temp1 = np.multiply(self.trans_prob,temp1)
            new_trans_prob = new_trans_prob + np.multiply(temp1,self.em_prob[:,self.obs_map[observations[t+1]]].transpose())

        # Normalize values so that sum of probabilities is 1
        for i in range(self.trans_prob.shape[0]):
            new_trans_prob[i,:] = new_trans_prob[i,:]/np.sum(new_trans_prob[i,:])

        return new_trans_prob

    def _train_start_prob(self,observations):
        delta = self._forward_backward(observations)
        norm = sum(delta[:,0])
        return delta[:,0].transpose()/norm


    def _randomize():
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

    def log_prob(self,observations_list, quantities): 
        """ Finds Weighted log probability of a list of observation sequences

        **Arguments**:

        :param observation_list: A nested list, or a list of lists  
        :type observation_list: Contains a list  multiple observation sequences.

        :param quantities: Number of times, each corresponding item in  'observation_list' occurs.
        :type quantities: A list of integers

        :return: Weighted log probability of multiple observations. 
        :rtype: float
            
        **Features**:

        Scaling applied here. This ensures that no underflow error occurs.

        **Example**:

        >>> states = ('s', 't')
        >>> possible_observation = ('A','B' )
        >>> # Numpy arrays of the data
        >>> start_probability = np.matrix( '0.5 0.5 ')
        >>> transition_probability = np.matrix('0.6 0.4 ;  0.3 0.7 ')
        >>> emission_probability = np.matrix( '0.3 0.7 ; 0.4 0.6 ' )
        >>> # Initialize class object
        >>> test = hmm(states,possible_observation,start_probability,transition_probability,emission_probability)
        >>> observations = ('A', 'B','B','A')
        >>> obs4 = ('B', 'A','B')
        >>> observation_tuple = []
        >>> observation_tuple.extend( [observations,obs4] )
        >>> quantities_observations = [10, 20]
        >>>
        >>> prob = test.log_prob(observation_tuple, quantities_observations)

        """

        prob = 0
        for q,obs in enumerate(observations_list):
            temp,c_scale = self._alpha_cal(obs)
            prob = prob +  -1 *  quantities[q] * np.sum(np.log(c_scale))
        return prob

# ============================================================

