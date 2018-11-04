#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function  ::  DONE
    #   Use tf.saved_model.loader.load to load the model and weights

    ## Provide ENCODER
    print("Loading VGG & extracting layers | Encoder")
    vgg_tag = 'vgg16'                                               # VGG 16 encoder
    vgg_input_tensor_name = 'image_input:0'                         # VGG first layer
    vgg_keep_prob_tensor_name = 'keep_prob:0'                       # VGG second layer
    vgg_layer3_out_tensor_name = 'layer3_out:0'                     # VGG third layer
    vgg_layer4_out_tensor_name = 'layer4_out:0'                     # VGG forth layer 
    vgg_layer7_out_tensor_name = 'layer7_out:0'                     # VGG seventh layer

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)             # Load model
    graph = tf.get_default_graph();                                 # Get default graph
    enc01 = graph.get_tensor_by_name(vgg_input_tensor_name)         # Get input tensor
    enc02 = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)     # Get keep_prob tenseor
    enc03 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)    # Get layer 3
    enc04 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)    # Get layer 4
    enc05 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)    # Get layer 7
    print("DONE")
    return enc01, enc02, enc03, enc04, enc05                        # return layers

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    
    ## Provide DECODER
    print("Creating layers | Decoder") 

    kernel_init = tf.random_normal_initializer(stddev=.0001);                  # Initialize Kernel with randomn_normal_initializer --- TUNE FOR OPTIMAL RESULT

    # Add Layer to layer7 output --> 1x1 convolution
    l7out_conv1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding = 'same', kernel_initializer = kernel_init)

    # Add Layer to layer4 ouput --> 1x1 convolution
    l4out_conv1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding = 'same', kernel_initializer = kernel_init)

    # Add Layer to layer3 output --> 1x1 convolution
    l3out_conv1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding = 'same', kernel_initializer = kernel_init)

    # Upsample L7out_conv1x1 and add
    l7out_up = tf.layers.conv2d_transpose(l7out_conv1x1, num_classes, 4, 2, padding = 'same', kernel_initializer = kernel_init) 
    
    #Add l7 upsampling output and the l4 output after 1x1 convolution
    l4in_up = tf.add(l7out_up, l4out_conv1x1)
    
    #Upsamle new layer
    l4out_up = tf.layers.conv2d_transpose(l4in_up, num_classes, 4, 2, padding = 'same', kernel_initializer = kernel_init)

    #Add l4 upsamling output and l3 output after 1x1 convolution
    l3in_up = tf.add(l4out_up, l3out_conv1x1)

    #Upsamle new layer
    l3out_up = tf.layers.conv2d_transpose(l3in_up, num_classes, 16, (8,8), padding = 'same', kernel_initializer = kernel_init)
    print("DONE")
    #RTeturn last upsampled layer
    return l3out_up

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    beta = 0.002 # Use this factor for L2 regulazition loss
    print("Creating optimizer operation for training")
    logits = tf.reshape(nn_last_layer, (-1, num_classes))                                               #Resize nn output to 2d
    labels = tf.reshape(correct_label, (-1, num_classes))                                               #Resize labels as well
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))  #Softmax_cross_entropy_with_logits 


    opt = tf.train.AdamOptimizer(learning_rate)         #use adam optimizer with placeholder learning rate
    train_opt = opt.minimize(cross_entropy_loss)        #tell Optimizer what to minimize


    # Output cross_entropy_loss for display in training operation (train_nn)
    # Outout optimizer fro training operation
    print("done")
    return logits, train_opt, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate,KEEPPROB,LEARNINGRATE):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
   
    print("Training...")
    for ep in range(epochs): # Train for each epoch 
        batch_counter = 0
        for im, lbl in get_batches_fn(batch_size):
            batch_counter += 1
            result = sess.run([train_op, cross_entropy_loss],                 # Run training operation
                                   feed_dict = {input_image: im,                    # Fill input_image tf_placeholder 
                                                correct_label: lbl,                 # Fill correct_label tf_placeholder
                                                keep_prob: KEEPPROB,                # Fill keep_prob tf_placeholder with dropout probability
                                                learning_rate: LEARNINGRATE})       # Fill learning_rate tf_placeholder with dropout probability

            print("Epoch: ", ep+1, " of ", epochs, " - batch number", batch_counter, ": Loss:", result[1])    # Output training information

    print("DONE")
    pass

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'

    #Initialize placeholders like in project_tests.py
    correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes])
    learning_rate = tf.placeholder(tf.float32)


    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model if not available
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        
        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, l3, l4, l7 = load_vgg(sess, vgg_path)
        final_layer = layers(l3, l4, l7, num_classes)
        # TODO: Train NN using the train_nn function
        logits, train_opt, cross_entropy_loss = optimize(final_layer, correct_label, learning_rate, num_classes)

        #Init vars
        sess.run(tf.global_variables_initializer())

        #Define epochs, batch size, dropout and learning rate --- TUNE FOR OPTIMAL RESULT
        EPOCHS = 5
        BATCHSIZE = 2
        KEEPPROB = 0.5
        LEARNINGRATE = 0.0001


        train_nn(sess, EPOCHS, BATCHSIZE, get_batches_fn,train_opt,cross_entropy_loss,input_image,correct_label,keep_prob,learning_rate,KEEPPROB,LEARNINGRATE)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()

# Training History
# Nr    |   DATE & TIME |   EPOCHS  |   BATCH SIZE  |   LEARNING RATE   |   KEEP PROB   |   L2_KERNEL REG   |  RESULT   |   Comment
# 1.    |   07-09,20:00 |   5       |   10          |   0.0001          |   0.66        |   1e-3            |   Bad     |   first try
# 2.    |   07-10,08:00 |   7       |   5           |   0.00001         |   0.4         |   1e-5            |   Bad     |   added l2_loss to cost
# 3.    |   07-10,13:00 |   7       |   2           |   0.0001          |   0.5         |   1e-3            |   Bad     |   Maximum batch size due to gpu memory constraint: 2
# 4.    |   07-10,13:35 |   5       |   2           |   0.0001          |   0.5         |   1e-3            |   Semi    |   added scaling to layer 3,4
# 5.    |   07-10,15:35 |   10      |   2           |   0.0001          |   0.5         |   1e-3            |   Good    |   used kernel initialyzer = 1e-3 --> loss did not decay
# 6.    |   07-10,15:35 |   10      |   2           |   0.0001          |   0.5         |   -               |   Bad     |   used kernel initialyzer = 2.0
# 7.    |   07-10,15:35 |   5       |   2           |   0.0001          |   0.5         |   -               |   Bad     |   used kernel initialyzer = .1
# 8.    |   07-10,20:00 |   5       |   2           |   0.0001          |   0.5         |   -               |   Bad     |   used kernel initialyzer = .01
# 9.    |   07-10,20:00 |   5       |   2           |   0.0001          |   0.5         |   -               |   Good    |   used kernel initialyzer = .001
# 10.   |   07-10,20:00 |   5       |   2           |   0.0001          |   0.5         |   -               |   Good    |   used kernel initialyzer = .0001
# 10.   |   07-10,20:00 |   5       |   2           |   0.0001          |   0.5         |   -               |   Good    |   used kernel initialyzer = .00001