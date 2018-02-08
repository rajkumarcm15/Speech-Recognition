# Train Recurrent Neural Network with Connectionist Temporal Classification Network
# for End-End Speech Recognition problem
# Name: Rajkumar Conjeevaram Mohan
# Email: Rajkumar.Conjeevaram-Mohan14@imperial.ac.uk
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "15"
import time
import numpy as np
import tensorflow as tf
import Data as d
import sys
import multiprocessing
#sys.stdout = open('log.txt','w')

def seq_bn(tensor,seq_lens):
    """
    Performs a sequence-wise batch normalization
    following a simple recipe from Deep Speech 2
    :param tensor: 3-D Tensor: [max_time,batch_size,feature_size]
    :param seq_lens: list consisting of sequence length of each data in mini-batch
    :return: 3-D Tensor: [max_time,batch_size,feature_size]
    """
    n = tf.cast(tf.reduce_sum(seq_lens),tf.float32)
    mew = tf.div(tf.reduce_sum(tensor,reduction_indices=[0,1], keep_dims=True),n)
    var_ = tf.div(tf.reduce_sum(tf.square(tf.sub(tensor,mew)),reduction_indices=[0,1],keep_dims=True),n)
    epsilon = 1e-5
    return tf.div(tf.sub(tensor,mew),tf.sqrt(tf.add(var_,epsilon)))


system = 'local'
test_path = '../../Data/OpenSLR/data_voip_en/test'
train_path = '../../Data/OpenSLR/data_voip_en/train'
val_path = '../../Data/OpenSLR/data_voip_en/dev'

#-----------RNN Configuration-----------
batch_size = 50
use_peephole = True
n_layers = 1
hidden_size = 250
num_units = hidden_size
momentum=0.9
max_grad_norm = 10

val_batch_size = 50
epochs = 250

#-----------Data Config-----------------

ms_to_sample = 20
overlap_ms = 10
#---------------------------------------
# layer_nLSTMunits_results.npy
results_fn = str('%d_%d_results.npy'%(n_layers,hidden_size))
NUM_CORES = int(multiprocessing.cpu_count()/2)
config = tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES,
                        intra_op_parallelism_threads=NUM_CORES)
if system == 'batch':
    test_path = '/vol/sml/data_voip_en/test'
    train_path = '/vol/sml/data_voip_en/train'
    val_path = '/vol/sml/data_voip_en/dev'
    gpu_options = tf.GPUOptions(allow_growth=True,allocator_type='BFC')
    config = tf.ConfigProto(gpu_options=gpu_options,
                            inter_op_parallelism_threads=NUM_CORES,
                            intra_op_parallelism_threads=NUM_CORES)

elif system == 'sycorax':
    test_path = '/data/mpd37/data_voip_en/test'
    train_path = '/data/mpd37/data_voip_en/train'
    val_path = '/data/mpd37/data_voip_en/dev'
    gpu_options = tf.GPUOptions(allow_growth=True,allocator_type='BFC')
    config = tf.ConfigProto(gpu_options=gpu_options,
                            inter_op_parallelism_threads=NUM_CORES,
                            intra_op_parallelism_threads=NUM_CORES)


#--------------------------------------
data_train = d.Data(batch_size,train_path,
                    'train_list.npy',
                    mode=3,
                    frame_overlap_flag=True,
                    overlap_ms=overlap_ms,
                    ms_to_sample=ms_to_sample)
data_val =   d.Data(val_batch_size, val_path,
                    'val_path.npy',
                    mode=3,
                    frame_overlap_flag=True,
                    overlap_ms=overlap_ms,
                    ms_to_sample=ms_to_sample)
n_chars = len(data_train.charmap) + 1
# frame_length = data_train.retain_fft
frame_length = (data_train.mfcc_coeff/2)*3


def encoder(seq_inputs,seq_lens,add_dropout,keep_prob,batch_normalise):
    initializer = tf.truncated_normal_initializer(mean=0, stddev=0.1)
    forward = tf.nn.rnn_cell.LSTMCell(num_units=num_units,
                                      # num_proj = hidden_size,
                                      use_peepholes=use_peephole,
                                      initializer=initializer,
                                      state_is_tuple=True)
    if add_dropout:
        forward = tf.nn.rnn_cell.DropoutWrapper(cell=forward, output_keep_prob=keep_prob)

    forward = tf.nn.rnn_cell.MultiRNNCell([forward] * n_layers, state_is_tuple=True)

    backward = tf.nn.rnn_cell.LSTMCell(num_units=num_units,
                                       # num_proj= hidden_size,
                                       use_peepholes=use_peephole,
                                       initializer=initializer,
                                       state_is_tuple=True)
    if add_dropout:
        backward = tf.nn.rnn_cell.DropoutWrapper(cell=backward, output_keep_prob=keep_prob)

    backward = tf.nn.rnn_cell.MultiRNNCell([backward] * n_layers, state_is_tuple=True)

    [fw_out, bw_out], _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=forward, cell_bw=backward, inputs=seq_inputs,
                                                          time_major=True, dtype=tf.float32,
                                                          sequence_length=tf.cast(seq_lens, tf.int64))

    if batch_normalise:
        # Batch normalize forward output
        fw_out = tf.transpose(fw_out, [1, 0, 2])
        mew, var_ = tf.nn.moments(fw_out, axes=[0])
        fw_out = tf.nn.batch_normalization(fw_out, mew, var_, 0.1, 1, 1e-6)

        # Batch normalize backward output
        bw_out = tf.transpose(bw_out, [1, 0, 2])
        mew, var_ = tf.nn.moments(bw_out, axes=[0])
        bw_out = tf.nn.batch_normalization(bw_out, mew, var_, 0.1, 1, 1e-6)

        fw_out = tf.reshape(fw_out, [-1, hidden_size])
        bw_out = tf.reshape(bw_out, [-1, hidden_size])

    # Linear Layer parameters
    W_fw = tf.Variable(tf.truncated_normal(shape=[hidden_size, n_chars], stddev=np.sqrt(2.0 / (2 * hidden_size))))
    W_bw = tf.Variable(tf.truncated_normal(shape=[hidden_size, n_chars], stddev=np.sqrt(2.0 / (2 * hidden_size))))
    b_out = tf.constant(0.1, shape=[n_chars])

    # Perform an affine transformation
    logits = tf.add(tf.add(tf.matmul(fw_out, W_fw), tf.matmul(bw_out, W_bw)), b_out)
    logits = tf.reshape(logits, [-1, batch_size, n_chars])

    return logits

def decoder(logits,seq_lens):
    # Use CTC Beam Search Decoder to decode pred string from the prob map
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_lens)
    return decoded, log_prob

def compute_loss(logits,decoded,targets,seq_lens):
    # Compute Loss
    loss = tf.reduce_mean(tf.nn.ctc_loss(logits, targets, seq_lens))

    # Compute error rate based on edit distance
    """
    Debugging Phase---------------------
    """
    predicted = tf.to_int32(decoded[0])
    "-----------------------------------"

    error_rate = tf.reduce_sum(tf.edit_distance(predicted, targets, normalize=False)) / \
                 tf.to_float(tf.size(targets.values))

    return loss, error_rate

def train(loss,max_grad_norm,lr):
    tvars = tf.trainable_variables()
    grad, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)

    optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=momentum)
    train_step = optimizer.apply_gradients(zip(grad, tvars))
    return train_step

graph = tf.Graph()
with graph.as_default():
    with tf.Session(config=config) as sess:

        # Graph creation
        graph_start = time.time()
        keep_prob = tf.placeholder(dtype=tf.float32, shape=[])
        seq_inputs = tf.placeholder(tf.float32, shape=[None, batch_size, frame_length], name="sequence_inputs")
        seq_lens = tf.placeholder(shape=[batch_size], dtype=tf.int32)
        seq_inputs = seq_bn(seq_inputs, seq_lens)
        # Target params
        indices = tf.placeholder(dtype=tf.int64, shape=[None, 2])
        values = tf.placeholder(dtype=tf.int32, shape=[None])
        shape = tf.placeholder(dtype=tf.int64, shape=[2])
        # Make targets
        targets = tf.SparseTensor(indices, values, shape)
        lr = tf.placeholder(dtype=tf.float32, shape=[])

        logits = encoder(seq_inputs,seq_lens,True,keep_prob,True)
        decoded,log_prob = decoder(logits,seq_lens)
        loss,error_rate = compute_loss(logits,decoded,targets,seq_lens)
        graph_end = time.time()
        print("Time elapsed for creating graph: %.3f" % (round(graph_end - graph_start, 3)))
        train_step = train(loss,max_grad_norm,lr)

        # steps per epoch
        # start_time = 0
        steps = int(np.ceil(len(data_train.files) / batch_size))

        loss_tr = []
        log_tr = []
        loss_vl = []
        log_vl = []
        err_tr = []
        err_vl = []

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        feed = None
        epoch = None
        step = None
        transcript = []
        try:
            for epoch in range(1,epochs+1):

                loss_val = 0
                l_pr = 0
                start_time = time.time()
                for step in range(steps):
                    if epoch < 50:
                        _lr_ = 1e-2
                    elif epoch >= 50 and epoch < 150:
                        _lr_ = 1e-3
                    elif epoch >= 150 and epoch < 200:
                        _lr_ = 1e-4
                    elif epoch >= 200 and epoch < 300:
                        _lr_ = 1e-5
                    else:
                        _lr_ = 1e-6


                    train_data, transcript, \
                    targ_indices, targ_values, \
                    targ_shape, n_frames = data_train.next_batch()
                    n_frames = np.reshape(n_frames,[-1])
                    feed = {seq_inputs: train_data, indices:targ_indices,
                            values:targ_values, shape:targ_shape,
                            seq_lens:n_frames,lr:_lr_,keep_prob:0.7}

                    # Evaluate loss value, decoded transcript, and log probability
                    _,loss_val,deco,l_pr,err_rt_tr = sess.run([train_step,loss,decoded,log_prob,error_rate],
                                                            feed_dict=feed)
                    loss_tr.append(loss_val)
                    log_tr.append(l_pr)
                    err_tr.append(err_rt_tr) 


                    # On validation set
                    val_data, val_transcript, \
                    targ_indices, targ_values, \
                    targ_shape, n_frames = data_val.next_batch()
                    n_frames = np.reshape(n_frames, [-1])
                    feed = {seq_inputs: val_data, indices: targ_indices,
                            values: targ_values, shape: targ_shape,
                            seq_lens: n_frames,lr:_lr_,keep_prob:1}

                    vl_loss, l_val_pr, err_rt_vl = sess.run([loss, log_prob, error_rate], feed_dict=feed)

                    loss_vl.append(vl_loss)
                    log_vl.append(l_val_pr)
                    err_vl.append(err_rt_vl)
                    print("epoch %d, step: %d, tr_loss: %.2f, vl_loss: %.2f, tr_err: %.2f, vl_err: %.2f"
                          % (epoch, step, np.mean(loss_tr), np.mean(loss_vl), err_rt_tr, err_rt_vl))

                end_time = time.time()
                elapsed = round(end_time - start_time, 3)

                if epoch % 10 == 0:
                    # On training set
                    # Select a random index within batch_size

                    sample_index = np.random.randint(1, batch_size)     # RELEASE AFTER DEBUGGING

                    # Fetch the decoded path from probability map
                    pred_sparse = tf.SparseTensor(deco[0].indices, deco[0].values, deco[0].shape)
                    pred_dense = tf.sparse_tensor_to_dense(pred_sparse)
                    ans = pred_dense.eval()

                    # Fetch the target transcript
                    actual_str = [data_train.reverse_map[i] for i in transcript[sample_index]]

                    pred = []
                    for i in ans[sample_index,:]:
                        if i == n_chars-1:
                            pred.append(data_train.reverse_map[0])
                        else:
                            pred.append(data_train.reverse_map[i])
                    print("time_elapsed for 200 steps: %.3f, " % (elapsed))

                    pred_str = "Sample mini-batch results: \n" \
                          "predicted string: ", np.array(pred)
                    act_str = "actual string1: ", np.array(actual_str)
                    print("%s \n %s \n"%(pred_str,act_str))

                # print("On training set, the loss: %.2f, log_pr: %.3f, error rate %.3f:" % (
                # loss_val, np.mean(l_pr), err_rt_tr))
                # print("On validation set, the loss: %.2f, log_pr: %.3f, error rate: %.3f" % (
                # vl_loss, np.mean(l_val_pr), err_rt_vl))

                # Save the trainable parameters after the end of an epoch
                #path = saver.save(sess, 'model_%d' % epoch)
                #print("Session saved at: %s" % path)

                #np.save(results_fn, np.array([loss_tr, log_tr, loss_vl, log_vl, err_tr, err_vl], dtype=np.object))

            # path = saver.save(sess, 'Working')
            # print("Session saved at: %s" % path)
            # np.save(results_fn, np.array([loss_tr, log_tr, loss_vl, log_vl, err_tr, err_vl], dtype=np.object))

        except (KeyboardInterrupt, SystemExit, Exception), e:
            print("Error/Interruption: %s" % str(e))
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print("Line no: %d" % exc_tb.tb_lineno)
            # print("Saving model: %s" % saver.save(sess, 'Last.cpkt'))
            print("Current batch: %d" % data_train.b_id)
            print("Current epoch: %d" % epoch)
            print("Current step: %d"%step)
            # np.save(results_fn, np.array([loss_tr, log_tr, loss_vl, log_vl, err_tr, err_vl], dtype=np.object))
            print("Clossing TF Session...")
            sess.close()
            print("Terminating Program...")
            sys.exit(0)
    print("Finished")
