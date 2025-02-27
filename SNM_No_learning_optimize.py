import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

@tf.function
def get_active_by_mask(spiked_indices, prob_matrix, weight_matrix):
    one_hot_mask = tf.reduce_sum(tf.one_hot(tf.squeeze(spiked_indices, axis=-1),
                                            depth=prob_matrix.shape[1]), axis=0)
    # Create a boolean mask where nonzero values indicate spiked neurons
    mask_bool = tf.cast(one_hot_mask, tf.bool)
    # Use boolean_mask along the appropriate axis (here, axis=1)
    active_prob = tf.boolean_mask(prob_matrix, mask_bool, axis=1)
    active_weights = tf.boolean_mask(weight_matrix, mask_bool, axis=1)
    return active_prob, active_weights

@tf.function
def generate_random_pool(num_time, pool_size_row, pool_size_col):
    """
    """
    return tf.random.uniform([num_time, pool_size_row, pool_size_col], minval=0, maxval=1, dtype=tf.float32)

@tf.function
def choose_active_weights(active_prob, active_weights, random_pool):
    """
    """
    condition = tf.cast(tf.abs(active_prob) > random_pool, dtype=tf.float32)
    chosen_weights = tf.sparse.from_dense(active_weights * condition * tf.sign(active_prob))
    return chosen_weights

def calculate_current(chosen_weights):
    """
    """
    row_sum = tf.sparse.reduce_sum(chosen_weights, axis=1)
    nonzero_counts = tf.math.bincount(chosen_weights.indices[:, 0], minlength=chosen_weights.dense_shape[0], dtype=tf.float32)
    nonzero_counts = tf.where(nonzero_counts > 0, nonzero_counts, 1.0)
    delta_potential = tf.sparse.from_dense(row_sum / nonzero_counts)
    return delta_potential


class SpikingMatrix:
    def __init__(self, input_size, output_size, hidden_size):
        '''Initialize parameters for matrix'''
        # Initialize membrane potential at 0
        self.hid_neu_pot = tf.Variable(tf.zeros(hidden_size, dtype=tf.float32)) 
        self.out_neu_pot = tf.Variable(tf.zeros(output_size, dtype=tf.float32)) 
        
        # Set spike threshold at 1
        self.spike_threshold = 1

        # Initialize connection probability in range [-1, 1]
        self.in_hid_prob = tf.random.uniform([hidden_size, input_size], -1, 1.01, dtype=tf.float32)
        self.hid_hid_prob = tf.random.uniform([hidden_size, hidden_size], -1, 1.01, dtype=tf.float32)
        self.hid_out_prob = tf.random.uniform([output_size, hidden_size], -1, 1.01, dtype=tf.float32)
        
        self.in_hid_weights = tf.random.uniform([hidden_size, input_size], 0.01, 1, dtype=tf.float32)
        self.hid_hid_weights = tf.random.uniform([hidden_size, hidden_size], 0.01, 1, dtype=tf.float32)
        self.hid_out_weights = tf.random.uniform([output_size, hidden_size], 0.01, 1, dtype=tf.float32)        

        # Initialize hidden and output neurons as inactive
        self.hid_active = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False)
        self.output_record = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False)
    
    def update_hidden_pot(self, delta_pot):
        self.hid_neu_pot = tf.tensor_scatter_nd_update(self.hid_neu_pot, delta_pot.indices, delta_pot.values)
        spiked_indices = tf.where(self.hid_neu_pot >= self.spike_threshold)
        mask = tf.cast(self.hid_neu_pot < self.spike_threshold, tf.float32)
        self.hid_neu_pot = self.hid_neu_pot * mask

        return spiked_indices, self.hid_neu_pot  
    
    def update_output_pot(self, delta_pot):
        self.out_neu_pot = tf.tensor_scatter_nd_update(self.out_neu_pot, delta_pot.indices, delta_pot.values)
        spiked_indices = tf.where(self.out_neu_pot >= self.spike_threshold)
        mask = tf.cast(self.out_neu_pot < self.spike_threshold, tf.float32)
        self.out_neu_pot = self.out_neu_pot * mask

        return spiked_indices, self.out_neu_pot

    @tf.function
    def forward_step(self, t_step, input_times, in_hid_delta_pot,
                     hid_neu_pot, out_neu_pot, 
                     hid_active, output_record, 
                     ran_hid_hid, ran_hid_out):      
        
        # Get the current from input-to-hidden connections
        hid_delta_pot = tf.cond(tf.reduce_any(tf.equal(input_times, t_step)),
                                lambda: in_hid_delta_pot[tf.squeeze(tf.where(tf.equal(input_times, t_step))[0])], # Take the first index because there is only one input at a time
                                lambda: tf.zeros_like(hid_neu_pot))

        # Add current from hidden-to-hidden connections
        hid_delta_pot += tf.cond(
            tf.equal(hid_active.size(), 0),  # Check if hid_active is empty
            lambda: tf.zeros_like(hid_neu_pot),  # If empty, no additional current
            lambda: calculate_current(choose_active_weights(  # If not empty, calculate current from active neurons
            *get_active_by_mask(hid_active.stack(), 
                      self.hid_hid_prob, 
                      self.hid_hid_weights), 
            ran_hid_hid[t_step])))
        
        # Add current from hidden-to-output connections
        out_delta_pot = calculate_current(choose_active_weights(
            *get_active_by_mask(hid_active.stack(), 
                      self.hid_out_prob, 
                      self.hid_out_weights), 
            ran_hid_out[t_step]))

        # Check for spikes
        hid_spikes, hid_neu_pot = self.update_hidden_pot(hid_delta_pot)
        out_spikes, out_neu_pot = self.update_output_pot(out_delta_pot)

        # Record spikes using TensorArray write operations
        hid_active = hid_active.write(t_step, tf.squeeze(hid_spikes))
        output_record = output_record.write(t_step, tf.squeeze(out_spikes))
        
        return t_step, (input_times, in_hid_delta_pot,
                    hid_neu_pot, out_neu_pot,
                    hid_active, output_record,
                    ran_hid_hid, ran_hid_out)

    @tf.function
    def forward(self, input_indices, input_times, output_length):
        t_step = tf.constant(0)
        ran_in_hid = generate_random_pool(len(input_times), self.in_hid_prob.shape[0], self.in_hid_prob.shape[1])
        ran_hid_hid = generate_random_pool(output_length, self.hid_hid_prob.shape[0], self.hid_hid_prob.shape[1])
        ran_hid_out = generate_random_pool(output_length, self.hid_out_prob.shape[0], self.hid_out_prob.shape[1])

        # Pre_compute the in_hid current to avoid redundant computation in the loop
        in_hid_active_prob, in_hid_active_weights = tf.map_fn(lambda x: get_active_by_mask(x, self.in_hid_prob, self.in_hid_weights), input_indices)
        in_hid_chosen_weights = tf.map_fn(lambda x: choose_active_weights(in_hid_active_prob[x], in_hid_active_weights[x], ran_in_hid[x]), tf.range(len(input_times)))
        in_hid_delta_pot = tf.map_fn(lambda x: calculate_current(x), in_hid_chosen_weights)

        args = (t_step, input_times, in_hid_delta_pot,
                self.hid_neu_pot, self.out_neu_pot,
                self.hid_active, self.output_record,
                ran_hid_hid, ran_hid_out)
        condition = lambda t_step, args: tf.less(t_step, output_length)

        def body(t_step, input_times, in_hid_delta_pot, hid_neu_pot, out_neu_pot, hid_active, output_record, ran_hid_hid, ran_hid_out):
            new_t, new_state = self.forward_step(t_step, input_times, in_hid_delta_pot,
                                                hid_neu_pot, out_neu_pot,
                                                hid_active, output_record,
                                                ran_hid_hid, ran_hid_out)
            return (new_t + 1,
                    input_times, in_hid_delta_pot,
                    new_state[2], new_state[3],
                    new_state[4], new_state[5],
                    ran_hid_hid, ran_hid_out)

        t_step, args = tf.while_loop(condition, body, args)

        return args[5].stack(), args[6].stack()