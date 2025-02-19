import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

class SpikingMatrix:
    def __init__(self, input_size, output_size, hidden_size):
        '''Initialize parameters for matrix'''
        # Initialize membrane potential at 0
        self.input_neurons_pot = tf.Variable(tf.zeros(input_size, dtype=tf.float32))
        self.hidden_neurons_pot = tf.Variable(tf.zeros(hidden_size, dtype=tf.float32)) 
        self.output_neurons_pot = tf.Variable(tf.zeros(output_size, dtype=tf.float32)) 
        
        # Set spike threshold at 1
        self.spike_threshold = 1

        # Initialize connection probability in range [-1, 1]
        self.input_hidden_prob = tf.Variable(tf.random.uniform([hidden_size, input_size], -1, 1.01, dtype=tf.float32))
        self.hidden_hidden_prob = tf.Variable(tf.random.uniform([hidden_size, hidden_size], -1, 1.01, dtype=tf.float32))
        self.hidden_output_prob = tf.Variable(tf.random.uniform([output_size, hidden_size], -1, 1.01, dtype=tf.float32))

        self.input_hidden_weights = tf.Variable(tf.random.uniform([hidden_size, input_size], 0.01, 1, dtype=tf.float32))
        self.hidden_hidden_weights = tf.Variable(tf.random.uniform([hidden_size, hidden_size], 0.01, 1, dtype=tf.float32))
        self.hidden_output_weights = tf.Variable(tf.random.uniform([output_size, hidden_size], 0.01, 1, dtype=tf.float32))

        # A list to keep track of previously spiked neurons in all 3 layers
        self.active_neurons = [[], [None], []]

    def get_active_input_hidden(self, spiked_input_neurons):
        """Gather the probilities and weights according to spiked input neurons.\n
        The first dimension m is the hidden neurons receiving the signal.
        The second dimension n is the spiked input neurons.
        
        Input: 
            spiked_input_neurons. Shape: [m, 1]
        
        Output:
            gathered_prob. Shape: [m, n]
            gathered_weights. Shape: [m, n]
        """
        gathered_prob = tf.squeeze(tf.gather(self.input_hidden_prob, spiked_input_neurons, axis=1))
        gathered_weights = tf.squeeze(tf.gather(self.input_hidden_weights, spiked_input_neurons, axis=1))
        return gathered_prob, gathered_weights
               
    def get_active_hidden_hidden(self, spiked_hidden_neurons):
        """Gather the probilities and weights according to spiked hidden neurons.
        The first dimension m is the hidden neurons receiving the signal.
        The second dimension n is the spiked hidden neurons.
        
        Input: 
            spiked_input_neurons. Shape: [m, 1]
        
        Output:
            gathered_prob. Shape: [m, n]
            gathered_weights. Shape: [m, n]
        """
        gathered_prob = tf.squeeze(tf.gather(self.hidden_hidden_prob, spiked_hidden_neurons, axis=1))
        gathered_weights = tf.squeeze(tf.gather(self.hidden_hidden_weights, spiked_hidden_neurons, axis=1))
        return gathered_prob, gathered_weights
    
    def get_active_hidden_output(self, spiked_hidden_neurons):
        """Gather the probilities and weights according to spiked hidden neurons.
        The first dimension m is the output neurons receiving the signal.
        The second dimension n is the spiked hidden neurons.
        
        Input: 
            spiked_input_neurons. Shape: [m, 1]
        
        Output:
            gathered_prob. Shape: [m, n]
            gathered_weights. Shape: [m, n]
        """
        gathered_prob = tf.squeeze(tf.gather(self.hidden_output_prob, spiked_hidden_neurons, axis=1))
        gathered_weights = tf.squeeze(tf.gather(self.hidden_output_weights, spiked_hidden_neurons, axis=1))
        return gathered_prob, gathered_weights
    
    def choose_active_weights(self, active_prob, active_weights):
        """Choose the connection and corresponding weights probabilisticly.\n
        Given the probabilities tensor and weights tensor, return a sparse weights tensor of the same shape with unchosen weights reduced to zero
        
        Input:
            active_prob. Shape: [m, n]
            active_weights. Shape: [m, n]

        Output:
            chosen_weights: Shape: [m, n] 
            (If n is 0, then expand chosen_weights to have shape [m, 1])
        """
        condition = tf.logical_or(tf.abs(active_prob) > 0.8, 
                                  tf.abs(active_prob) > tf.random.uniform(tf.shape(active_prob), minval=0.0, maxval=1.0))
        chosen_weights = tf.where(condition, active_weights * tf.sign(active_prob), tf.zeros(active_weights.shape, dtype=tf.float32))
        if len(chosen_weights.shape) == 1:
            chosen_weights = tf.expand_dims(chosen_weights, axis=1)

        return chosen_weights
    
    def calculate_current(self, chosen_weights):
        """Calculate the combined current by summing weights contribution by the second axis.\n
        
        Input:
            chosen_weights. Shape: [m, n] 
            (n>=1)
            
        Output:
            delta_pot = tf.reduce_mean(chosen_weights, axis =1). Shape: [m, 1]
        """
        return tf.reduce_mean(chosen_weights, axis=1)
    
    def update_input_pot(self, delta_pot):
        self.input_neurons_pot.assign_add(tf.reshape(delta_pot,[-1]))
        spiked_input_neurons = tf.where(self.input_neurons_pot > self.spike_threshold)
        self.input_neurons_pot.assign(tf.tensor_scatter_nd_update(self.input_neurons_pot, spiked_input_neurons, tf.reshape(tf.zeros_like(spiked_input_neurons, dtype=tf.float32), [-1])))
        if spiked_input_neurons.shape == [0,1]:
            return None
        return spiked_input_neurons
    
    def update_hidden_pot(self, delta_pot):
        self.hidden_neurons_pot.assign_add(tf.reshape(delta_pot,[-1]))
        spiked_hidden_neurons = tf.where(self.hidden_neurons_pot > self.spike_threshold)
        self.hidden_neurons_pot.assign(tf.tensor_scatter_nd_update(self.hidden_neurons_pot, spiked_hidden_neurons, tf.reshape(tf.zeros_like(spiked_hidden_neurons, dtype=tf.float32), [-1])))

        if spiked_hidden_neurons.shape == [0,1]:
            print("Dead network")
            return None
        return spiked_hidden_neurons        
    
    def update_output_pot(self, delta_pot):
        self.output_neurons_pot.assign_add(tf.reshape(delta_pot,[-1]))
        spiked_output_neurons = tf.where(self.output_neurons_pot > self.spike_threshold)
        self.output_neurons_pot.assign(tf.tensor_scatter_nd_update(self.output_neurons_pot, spiked_output_neurons, tf.reshape(tf.zeros_like(spiked_output_neurons, dtype=tf.float32), [-1])))

        if spiked_output_neurons.shape == [0,1]:
            return None
        
        return spiked_output_neurons        
    
    def update_active_neurons(self, spiked_input_neurons, spiked_hidden_neurons, spiked_output_neurons):
        """Update the list active_neurons with the most recent spiked neurons"""
        self.active_neurons[0].append(spiked_input_neurons)
        self.active_neurons[1].append(spiked_hidden_neurons)
        self.active_neurons[2].append(spiked_output_neurons)
    
    def reset_memory(self):
        """Clear active_neurons list"""
        self.active_neurons = [[], [self.active_neurons[1][-1]], []]

    def forward(self, input_indices, input_times, output_length):
        input_current = tf.zeros_like(self.input_neurons_pot, dtype=tf.float32)
        input_current = tf.tensor_scatter_nd_update(input_current, input_indices, tf.reshape(tf.ones_like(input_indices, dtype=tf.float32) * 2, [-1]))

        for t in range(output_length):
            if t in input_times:
                spiked_input_neurons = self.update_input_pot(input_current)
                input_hidden_prob, input_hidden_weights = self.get_active_input_hidden(spiked_input_neurons)
                hidden_delta_pot = self.calculate_current(self.choose_active_weights(input_hidden_prob, input_hidden_weights))
            else:
                spiked_input_neurons = None
                hidden_delta_pot = tf.zeros_like(self.hidden_neurons_pot, dtype=tf.float32)

            if self.active_neurons[1][-1] != None:
                hidden_hidden_prob, hidden_hidden_weights = self.get_active_hidden_hidden(self.active_neurons[1][-1])     
                hidden_delta_pot += self.calculate_current(self.choose_active_weights(hidden_hidden_prob, hidden_hidden_weights))

                hidden_output_prob, hidden_output_weights = self.get_active_hidden_output(self.active_neurons[1][-1])
                output_delta_pot = self.calculate_current(self.choose_active_weights(hidden_output_prob, hidden_output_weights))     
                spiked_output_neurons = self.update_output_pot(output_delta_pot)
            else:
                spiked_output_neurons = None     

            spiked_hidden_neurons = self.update_hidden_pot(hidden_delta_pot)            
            self.update_active_neurons(spiked_input_neurons, spiked_hidden_neurons, spiked_output_neurons)
            

        return self.active_neurons


# Test the class
if __name__ == "__main__":
    matrix = SpikingMatrix(input_size=20, output_size=10, hidden_size=1000)

    input_indices = tf.constant([[0], [1], [2], [3], [4], [5], [10], [15]], dtype=tf.int32)
    input_times = [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17, 21]
    output_length = 1000

    input_layer, hidden_layer, output_layer = matrix.forward(input_indices, input_times, output_length)
    matrix.reset_memory()

    # Print the output of each layer if needed
