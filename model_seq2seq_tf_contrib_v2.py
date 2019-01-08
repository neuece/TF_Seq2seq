# using contrib.seq2seq to build a seq2seq model with attention

# embeddings can be trained or loaded. Please change accordingly.

# Reference: TF documentation
#            https://github.com/tensorflow/nmt

# MODEL

import tensorflow as tf

class Seq2seq(object):
	
    # place holder for input and output
    # inputs are batch of index tf.int32
	def build_inputs(self, config):
		self.input_seq = tf.placeholder(shape=(config.batch_size, None), dtype=tf.int32, name='input_seq')
		self.input_seq_length = tf.placeholder(shape=(config.batch_size,), dtype=tf.int32, name='input_seq_length')
		self.target_seq = tf.placeholder(shape=(config.batch_size, None), dtype=tf.int32, name='target_seq')
		self.target_seq_length = tf.placeholder(shape=(config.batch_size,), dtype=tf.int32, name='target_seq_length')
	
	# model definition
    # default settings: using Teachers, Attention and Beam search
	def __init__(self, config, w2i_target, Flag_TeacherForcing=True, Flag_Attention=True, Option_BeamSearch=1):
		# Option_BeamSearch is the beam width
        
        
        # initialization
		self.build_inputs(config)
               
		# build Encoder
		with tf.variable_scope("encoder"):
		    # initialize embedding randomly
			encoder_embedding = tf.Variable(tf.random_uniform([config.source_vocab_size, config.embedding_dim]),
                                   dtype=tf.float32, name='encoder_embedding')
          # encoder_embedding = # load pre-trained embeddings
			encoder_inputs_embedded = tf.nn.embedding_lookup(encoder_embedding, self.input_seq)
            
			with tf.variable_scope("gru_cell"):
				encoder_cell = tf.nn.rnn_cell.GRUCell(config.hidden_dim)
                
#          with tf.variable_scope("lstm_cell"):
#				encoder_cell = tf.nn.rnn_cell.LSTMCell(config.hidden_dim)
			
           # bidirection RNN is used. Output the final state and ouptus 
			((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell, cell_bw=encoder_cell, 
                                                                                                                                        inputs=encoder_inputs_embedded, 
                                                                                                                                        sequence_length=self.input_seq_length, 
                                                                                                                                        dtype=tf.float32)
          # combine the final state of bidirectional rnn                                                                                                                             
			encoder_state = tf.add(encoder_fw_final_state, encoder_bw_final_state) # encoder_final_states
			encoder_outputs = tf.add(encoder_fw_outputs, encoder_bw_outputs) 
		
		with tf.variable_scope("decoder"):
			
           # initialize embedding randomly 
			decoder_embedding = tf.Variable(tf.random_uniform([config.target_vocab_size, config.embedding_dim]),
                                   dtype=tf.float32, name='decoder_embedding')
            
			# add start string to decoder
          # w2i_target["_SS"] make sure "_SS"  
			start_string = tf.ones([config.batch_size], dtype=tf.int32, name='start_string') * w2i_target["_SS"]
            
			# Teacher Forcing helps to avoid error accumulation and speed up the training (it uses the seq in the actual decoding/target)
          #!!!!!!!!!!! only use TeacherForcing in training !!!!!!!!!!!!!!!
			if Flag_TeacherForcing:
				decoder_inputs = tf.concat([tf.reshape(start_string,[-1,1]), self.target_seq[:,:-1]], 1)
				helper = tf.contrib.seq2seq.TrainingHelper(tf.nn.embedding_lookup(decoder_embedding, decoder_inputs), self.target_seq_length)
			else:
				helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embedding, start_string, w2i_target["_EOS"])
				
			with tf.variable_scope("gru_cell"):
				decoder_cell = tf.nn.rnn_cell.GRUCell(config.hidden_dim)
                
#          with tf.variable_scope("lstm_cell"):
#				decoder_cell = tf.nn.rnn_cell.LSTMCell(config.hidden_dim)
				
               
                
				if Flag_Attention:
					if Option_BeamSearch > 1:
                        # this section follows the reference at TF https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BeamSearchDecoder
						tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=Option_BeamSearch)
						tiled_sequence_length = tf.contrib.seq2seq.tile_batch(self.input_seq_length, multiplier=Option_BeamSearch)
						attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=config.hidden_dim, memory=tiled_encoder_outputs, memory_sequence_length=tiled_sequence_length)
						decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism)
						tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=Option_BeamSearch)
						tiled_decoder_initial_state = decoder_cell.zero_state(batch_size=config.batch_size*Option_BeamSearch, dtype=tf.float32)
						tiled_decoder_initial_state = tiled_decoder_initial_state.clone(cell_state=tiled_encoder_final_state)
						decoder_initial_state = tiled_decoder_initial_state
                        
                        
#                        tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=beam_width)
#                        tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(encoder_final_state, multiplier=beam_width)
#                        tiled_sequence_length = tf.contrib.seq2seq.tile_batch(sequence_length, multiplier=beam_width)
#                        attention_mechanism = MyFavoriteAttentionMechanism(num_units=attention_depth,memory=tiled_inputs,
#                                                                           memory_sequence_length=tiled_sequence_length)
#                        attention_cell = AttentionWrapper(cell, attention_mechanism, ...)
#                        decoder_initial_state = attention_cell.zero_state(dtype, batch_size=true_batch_size * beam_width)
#                        decoder_initial_state = decoder_initial_state.clone(cell_state=tiled_encoder_final_state)
#                        
                        
					else:
						attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=config.hidden_dim, memory=encoder_outputs, memory_sequence_length=self.input_seq_length)
						# attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=config.hidden_dim, memory=encoder_outputs, memory_sequence_length=self.input_seq_length)
						decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism)
						decoder_initial_state = decoder_cell.zero_state(batch_size=config.batch_size, dtype=tf.float32)
						decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
				else:
					if Option_BeamSearch > 1:
						decoder_initial_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=Option_BeamSearch)
					else:
						decoder_initial_state = encoder_state
			
			if Option_BeamSearch > 1:
				decoder = tf.contrib.seq2seq.BeamSearchDecoder(decoder_cell, decoder_embedding, start_string, w2i_target["_EOS"],  decoder_initial_state, beam_width=Option_BeamSearch, output_layer=tf.layers.Dense(config.target_vocab_size))
			else:
				decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state, output_layer=tf.layers.Dense(config.target_vocab_size))
			
			decoder_outputs, decoder_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=tf.reduce_max(self.target_seq_length))
			
		if Option_BeamSearch > 1:
			self.out = decoder_outputs.predicted_ids[:,:,0]
		else:	
			decoder_logits = decoder_outputs.rnn_output
			self.out = tf.argmax(decoder_logits, 2)
			
			sequence_mask = tf.sequence_mask(self.target_seq_length, dtype=tf.float32)
			self.loss = tf.contrib.seq2seq.sequence_loss(logits=decoder_logits, targets=self.target_seq, weights=sequence_mask)
			
			self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)
			