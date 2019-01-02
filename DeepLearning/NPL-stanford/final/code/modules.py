# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out

class BiLSTMEncoder(object):


    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("BiLSTMEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out

        

class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, key_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output

        
class BiAttn(object):
    """Module for bi attention."""

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys, keys_mask):
        """
        Keys attend to values.
        For each key/value, return C2Q/Q2C attention distribution, 
        and a C2Q/Q2C attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, key_vec_size)
          keys_mask: Tensor shape (batch_size, num_keys).
            1s where there's real input, 0s where there's padding

        Outputs:
          C2Q_attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          
          Q2C_attn_dist: Tensor shape (batch_size, 1, num_keys).
            The distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
                   
          C2Q_output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the C2Q attention distribution as weights).
            
          Q2C_output: Tensor shape (batch_size, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the Q2C attention distribution as weights).
        """
        with vs.variable_scope("BiAttn"):

            # Calculate the similarity matrix
            batch_size = tf.shape(keys)[0]; 
            num_keys = tf.shape(keys)[1]
            num_values = tf.shape(values)[1]
            
            w_sim_1 = tf.get_variable('w_sim_1', shape=(self.key_vec_size),initializer=tf.contrib.layers.xavier_initializer()) # 2 * H
            w_sim_2 = tf.get_variable('w_sim_2', shape=(self.value_vec_size),initializer=tf.contrib.layers.xavier_initializer()) # 2 * H
            w_sim_3 = tf.get_variable('w_sim_3', shape =(self.key_vec_size),initializer=tf.contrib.layers.xavier_initializer()) # 2 * H

            # Compute matrix  of size BS x N x M x 2H which contains all c_i o q_j
            CW = tf.reshape(tf.reshape(keys, (-1, self.key_vec_size)) * tf.expand_dims(w_sim_3, 0), (-1, num_keys, self.key_vec_size)) # BS x N x 2H
            #Compute all dot products
            term1 = tf.reshape(tf.matmul(tf.reshape(keys, (batch_size * num_keys, self.key_vec_size)), tf.expand_dims(w_sim_1, -1)), (-1, num_keys)) # BS x N
            term2 = tf.reshape(tf.matmul(tf.reshape(values, (batch_size * num_values, self.value_vec_size)), tf.expand_dims(w_sim_2, -1)), (-1, num_values)) # BS x M
            term3 = tf.matmul(CW, tf.transpose(values, [0, 2, 1])) # BS x N x M
            S = tf.reshape(term1,(-1, num_keys, 1)) + term3 + tf.reshape(term2, (-1, 1, num_values))
         
            
            # Calculate C2Q attention distribution
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, C2Q_attn_dist = masked_softmax(S, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use C2Q attention distribution to take weighted sum of values
            C2Q_output = tf.matmul(C2Q_attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            C2Q_output = tf.nn.dropout(C2Q_output, self.keep_prob)

            
            # Calculate Q2C attention distribution
            m_sim = tf.reduce_max(S, axis=2) # shape (batch_size, num_keys)
            _, Q2C_attn_dist = masked_softmax(m_sim, keys_mask, 1) # shape (batch_size, num_keys). take softmax over values

            # Use Q2C attention distribution to take weighted sum of values
            Q2C_attn_dist_expand = tf.expand_dims(Q2C_attn_dist,axis=1) # shape (batch_size,1,num_keys)
            Q2C_output = tf.matmul(Q2C_attn_dist_expand, keys) # shape (batch_size,1,key_vec_size)

            # Apply dropout
            Q2C_output = tf.nn.dropout(Q2C_output, self.keep_prob)
                       
            return C2Q_attn_dist, C2Q_output, Q2C_attn_dist_expand, Q2C_output
 
        
class CoAttn(object):
    """Module for coattention."""

    def __init__(self, keep_prob, key_vec_size, value_vec_size, batch_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size
        self.batch_size = batch_size

    def build_graph(self, values, values_mask, keys, keys_mask):
        """
        Keys attend to values.
        For each key/value, return C2Q/Q2C attention distribution, a C2Q/Q2C attention output vector,
        and a Co attention ouput vector

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, key_vec_size)
          keys_mask: Tensor shape (batch_size, num_keys).
            1s where there's real input, 0s where there's padding

        Outputs:
          C2Q_attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          
          Q2C_attn_dist: Tensor shape (batch_size, num_values, 1).
            The distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
                   
          C2Q_output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the C2Q attention distribution as weights).
            
          Q2C_output: Tensor shape (batch_size, 1, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the Q2C attention distribution as weights).
            
          Co_output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the Q2C attention distribution as weights).
        """
        with vs.variable_scope("CoAttn"):
            
            batch_size = tf.shape(keys)[0]; 
            num_keys = tf.shape(keys)[1]
            num_values = tf.shape(values)[1]
            
            
            # Project values (i.e. question hidden states)
            w_Co = tf.get_variable('w_Co', shape=(self.value_vec_size, self.value_vec_size),initializer=tf.contrib.layers.xavier_initializer()) # 2*H x 2 * H
            b_Co = tf.get_variable('b_Co', shape=(1,self.value_vec_size),initializer=tf.contrib.layers.xavier_initializer()) # 2 * H
            values_pro = tf.nn.tanh(tf.reshape(tf.matmul(tf.reshape(values, (batch_size * num_values, self.value_vec_size)), w_Co)+b_Co, (-1, num_values,self.value_vec_size))) # BS x num_values x 2*H

            # Add sentinel vectors
            key_sentinel_vec = tf.get_variable('key_sentinel_vec', shape=(self.batch_size,1,self.key_vec_size),initializer=tf.contrib.layers.xavier_initializer()) # BS x 1 x 2 * H
            value_sentinel_vec = tf.get_variable('value_sentinel_vec', shape=(self.batch_size,1,self.value_vec_size),initializer=tf.contrib.layers.xavier_initializer()) # BS x 1 x 2 * H
            keys_sentinel = tf.concat([keys,key_sentinel_vec],axis=1) # BS x (num_keys+1) x 2*H
            values_sentinel = tf.concat([values_pro,value_sentinel_vec],axis=1) # BS x (num_values+1) x 2*H
            
            # Calculate the affinity matrix
            L = tf.matmul(keys_sentinel,tf.transpose(values_sentinel,perm=[0, 2, 1])) # BS x num_keys+1 x num_values+1       
            values_mask = tf.concat([values_mask, tf.constant(0,shape=(self.batch_size,1),dtype=tf.int32)], axis=1)
            values_mask = tf.expand_dims(values_mask, axis=1) # BS x 1 x M+1
            keys_mask = tf.concat([keys_mask, tf.constant(0,shape=(self.batch_size,1),dtype=tf.int32)], axis=1)
            keys_mask = tf.expand_dims(keys_mask, -1) # BS x N+1 x 1
            L_mask = tf.matmul(keys_mask,values_mask) # BS x N+1 x M+1
            
            
            # Calculate C2Q attention distribution
            _, C2Q_attn_dist = masked_softmax(L, L_mask, 2) # shape (batch_size, num_keys+1, num_values+1). take softmax over values

            # Use C2Q attention distribution to take weighted sum of values
            C2Q_output = tf.matmul(C2Q_attn_dist, values_sentinel) # shape (batch_size, num_keys+1, value_vec_size)
            
            # Calculate Q2C attention distribution
            _, Q2C_attn_dist = masked_softmax(L, L_mask, 1) # shape (batch_size, num_keys+1, num_values+1). take softmax over keys

            # Use Q2C attention distribution to take weighted sum of values
            Q2C_output = tf.matmul(tf.transpose(Q2C_attn_dist,perm=[0,2,1]), keys_sentinel)  # shape (batch_size, num_values+1, key_vec_size)        
                        
            # Use C2Q attention distribution to take weighted sum of Q2C attention
            Co_output = tf.matmul(C2Q_attn_dist,Q2C_output)  # shape (batch_size, num_keys+1,, key_vec_size)   
            
            # Apply dropout
            Q2C_output = tf.nn.dropout(Q2C_output, self.keep_prob)
            C2Q_output = tf.nn.dropout(C2Q_output, self.keep_prob)
            Co_output = tf.nn.dropout(Co_output, self.keep_prob)
            
            # Reduce dim
            C2Q_output = C2Q_output[:,:-1,:]
            Q2C_output = Q2C_output[:,:-1,:]
            Co_output = Co_output[:,:-1,:]
            
            return C2Q_attn_dist, C2Q_output, Q2C_attn_dist, Q2C_output, Co_output
 
        
        

def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist
