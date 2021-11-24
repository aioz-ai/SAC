# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
import numpy as np
import random
from utils.utils import sentences_to_indices, read_glove_vecs, cosine_annealing, num_params
import _pickle as pickle
slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'train_dir', None,
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers', 8,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 8,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 100,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 100,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 1e-5, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'momentum',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.9, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', 'car', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', './Car/Data',
    'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                                'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'train_image_size', 448, 'Train image size')

tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', './pre_trained/inception_v3.ckpt',
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', "InceptionV3/bilinear_pooling,InceptionV3/roi_pooling",
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

tf.app.flags.DEFINE_string(
    'gpus', "0",
    'gpu list'
)

tf.app.flags.DEFINE_string(
    'feature_maps', None,
    'the layer name of feature maps')

tf.app.flags.DEFINE_string(
    'attention_maps', None,
    'the layer name of attention maps')

tf.app.flags.DEFINE_integer(
    'num_parts', None,
    'number of parts'
)

FLAGS = tf.app.flags.FLAGS


# cosine annealing learning rate schedule


def _configure_learning_rate(num_samples_per_epoch, global_step):
    """Configures the learning rate.

    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.

    Returns:
      A `Tensor` representing the learning rate.

    Raises:
      ValueError: if
    """
    decay_steps = int(num_samples_per_epoch / (FLAGS.batch_size * FLAGS.num_clones) *
                      FLAGS.num_epochs_per_decay)
    if FLAGS.sync_replicas:
        decay_steps /= FLAGS.replicas_to_aggregate

    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.learning_rate,
                                          global_step,
                                          decay_steps,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(FLAGS.learning_rate,
                                         global_step,
                                         decay_steps,
                                         FLAGS.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
    """Configures the optimizer used for training.

    Args:
      learning_rate: A scalar or `Tensor` learning rate.

    Returns:
      An instance of an optimizer.

    Raises:
      ValueError: if FLAGS.optimizer is not recognized.
    """
    if FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=FLAGS.adadelta_rho,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=FLAGS.adam_beta1,
            beta2=FLAGS.adam_beta2,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=FLAGS.ftrl_learning_rate_power,
            initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
            l1_regularization_strength=FLAGS.ftrl_l1,
            l2_regularization_strength=FLAGS.ftrl_l2)
    elif FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=FLAGS.momentum,
            name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=FLAGS.rmsprop_decay,
            momentum=FLAGS.rmsprop_momentum,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
    return optimizer


def _get_init_fn():
    """Returns a function run by the chief worker to warm-start the training.

    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
      An init function run by the supervisor.
    """
    if FLAGS.checkpoint_path is None:
        return None

    # Warn the user if a checkpoint exists in the train_dir. Then we'll be
    # ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(FLAGS.train_dir):
        tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists in %s'
            % FLAGS.train_dir)
        return None

    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Fine-tuning from %s' % checkpoint_path)

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=FLAGS.ignore_missing_vars)


def _get_variables_to_train():
    """Returns a list of variables to train.

    Returns:
      A list of variables to train by the optimizer.
    """
    if FLAGS.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        #######################
        # Config model_deploy #
        #######################
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=FLAGS.clone_on_cpu,
            replica_id=FLAGS.task,
            num_replicas=FLAGS.worker_replicas,
            num_ps_tasks=FLAGS.num_ps_tasks)

        # Create global_step
        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()

        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        ######################
        # Select the network #
        ######################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            FLAGS.dataset_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            weight_decay=FLAGS.weight_decay,
            is_training=True)

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=True)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        with tf.device(deploy_config.inputs_device()):
            classes = pickle.load(open('data/%s_classes.pkl' % FLAGS.dataset_name, 'rb'))
            classes = np.array(classes)
            word2index, index2word, _ = read_glove_vecs('data/%s_glove6b_init_300d.npy' % FLAGS.dataset_name,
                                               'data/%s_dictionary.pkl' % FLAGS.dataset_name)
            word2index[index2word[0]] = len(word2index)
            indices = sentences_to_indices(classes, word2index, 3)
            indices = tf.Variable(tf.convert_to_tensor(indices))
            indices = tf.broadcast_to(indices, [FLAGS.batch_size, len(classes), 3])

            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=FLAGS.num_readers,
                common_queue_capacity=20 * FLAGS.batch_size,
                common_queue_min=10 * FLAGS.batch_size)
            [image, label] = provider.get(['image', 'label'])
            label -= FLAGS.labels_offset

            train_image_size = FLAGS.train_image_size or network_fn.default_image_size

            image = image_preprocessing_fn(image, train_image_size, train_image_size)

            images, labels = tf.train.batch(
                [image, label],
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_preprocessing_threads,
                capacity=5 * FLAGS.batch_size)
            labels = slim.one_hot_encoding(
                labels, dataset.num_classes - FLAGS.labels_offset)
            batch_queue = slim.prefetch_queue.prefetch_queue(
                [images, labels, indices], capacity=2 * deploy_config.num_clones)



        ####################
        # Define the model #
        ####################

        def calculate_pooling_center_loss(features, label, alfa, nrof_classes, weights, name):
            features = tf.reshape(features, [features.shape[0], -1])
            label = tf.argmax(label, 1)

            nrof_features = features.get_shape()[1]
            centers = tf.get_variable(name, [nrof_classes, nrof_features], dtype=tf.float32,
                                      initializer=tf.constant_initializer(0), trainable=False)
            label = tf.reshape(label, [-1])
            centers_batch = tf.gather(centers, label)
            centers_batch = tf.nn.l2_normalize(centers_batch, axis=-1)

            diff = (1 - alfa) * (centers_batch - features)
            centers = tf.scatter_sub(centers, label, diff)

            with tf.control_dependencies([centers]):
                distance = tf.square(features - centers_batch)
                distance = tf.reduce_sum(distance, axis=-1)
                center_loss = tf.reduce_mean(distance)

            center_loss = tf.identity(center_loss * weights, name=name + '_loss')
            return center_loss

        def attention_crop(attention_maps):
            batch_size, height, width, num_parts = attention_maps.shape
            bboxes = []
            for i in range(batch_size):
                attention_map = attention_maps[i]
                part_weights = attention_map.mean(axis=0).mean(axis=0)
                part_weights = np.sqrt(part_weights)
                part_weights = part_weights / np.sum(part_weights)
                selected_index = np.random.choice(np.arange(0, num_parts), 1, p=part_weights)[0]

                mask = attention_map[:, :, selected_index]

                threshold = random.uniform(0.4, 0.6)
                itemindex = np.where(mask >= mask.max() * threshold)

                ymin = itemindex[0].min() / height - 0.1
                ymax = itemindex[0].max() / height + 0.1
                xmin = itemindex[1].min() / width - 0.1
                xmax = itemindex[1].max() / width + 0.1

                bbox = np.asarray([ymin, xmin, ymax, xmax], dtype=np.float32)
                bboxes.append(bbox)
            bboxes = np.asarray(bboxes, np.float32)
            return bboxes

        def attention_drop(attention_maps):
            batch_size, height, width, num_parts = attention_maps.shape
            masks = []
            for i in range(batch_size):
                attention_map = attention_maps[i]
                part_weights = attention_map.mean(axis=0).mean(axis=0)
                part_weights = np.sqrt(part_weights)
                part_weights = part_weights / np.sum(part_weights)
                selected_index = np.random.choice(np.arange(0, num_parts), 1, p=part_weights)[0]
                mask = attention_map[:, :, selected_index:selected_index + 1]

                # soft mask
                threshold = random.uniform(0.2, 0.5)
                mask = (mask < threshold * mask.max()).astype(np.float32)
                masks.append(mask)
            masks = np.asarray(masks, dtype=np.float32)
            return masks

        def clone_fn(batch_queue):
            """Allows data parallelism by creating multiple clones of network_fn."""
            images, labels, indices = batch_queue.dequeue()
            if FLAGS.model_name == 'inception_v3_topk':
                logits_1, end_points = network_fn(images, indices)

                attention_maps = end_points['attention_maps']
                attention_maps = tf.image.resize_bilinear(attention_maps, [train_image_size, train_image_size])

                # attention crop
                bboxes = tf.py_func(attention_crop, [attention_maps], [tf.float32])
                bboxes = tf.reshape(bboxes, [FLAGS.batch_size, 4])
                box_ind = tf.range(FLAGS.batch_size, dtype=tf.int32)
                images_crop = tf.image.crop_and_resize(images, bboxes, box_ind,
                                                       crop_size=[train_image_size, train_image_size])

                # attention drop
                masks = tf.py_func(attention_drop, [attention_maps], [tf.float32])
                masks = tf.reshape(masks, [FLAGS.batch_size, train_image_size, train_image_size, 1])
                images_drop = images * masks

                logits_2, end_points_2 = network_fn(images_crop, indices, reuse=True)
                logits_3, end_points_3 = network_fn(images_drop, indices, reuse=True)

                #Topk_logits
                topk_logits = end_points['topk_logits']

                slim.losses.softmax_cross_entropy(logits_1, labels, weights=1.0 / 3, scope='cross_entropy_1')
                slim.losses.softmax_cross_entropy(logits_2, labels, weights=1.0 / 3, scope='cross_entropy_2')
                slim.losses.softmax_cross_entropy(logits_3, labels, weights=1.0 / 3, scope='cross_entropy_3')
                slim.losses.softmax_cross_entropy(topk_logits, labels, weights=0.01, scope='cross_entropy_4')

                embeddings = end_points['embeddings']
                center_loss = calculate_pooling_center_loss(features=embeddings, label=labels,
                                                            alfa=0.95, nrof_classes=dataset.num_classes,
                                                            weights=1.0, name='center_loss')
                slim.losses.add_loss(center_loss)
                return end_points

            elif 'resnet' in FLAGS.model_name :
                logits, end_points = network_fn(images, None)
                slim.losses.softmax_cross_entropy(logits, labels, weights=1, scope='cross_entropy')
                return end_points

            else:
                logits_1, end_points = network_fn(images, None)

                attention_maps = end_points['attention_maps']
                attention_maps = tf.image.resize_bilinear(attention_maps, [train_image_size, train_image_size])

                # attention crop
                bboxes = tf.py_func(attention_crop, [attention_maps], [tf.float32])
                bboxes = tf.reshape(bboxes, [FLAGS.batch_size, 4])
                box_ind = tf.range(FLAGS.batch_size, dtype=tf.int32)
                images_crop = tf.image.crop_and_resize(images, bboxes, box_ind,
                                                       crop_size=[train_image_size, train_image_size])

                # attention drop
                masks = tf.py_func(attention_drop, [attention_maps], [tf.float32])
                masks = tf.reshape(masks, [FLAGS.batch_size, train_image_size, train_image_size, 1])
                images_drop = images * masks

                logits_2, end_points_2 = network_fn(images_crop, None, reuse=True)
                logits_3, end_points_3 = network_fn(images_drop, None, reuse=True)

                slim.losses.softmax_cross_entropy(logits_1, labels, weights=1 / 3.0, scope='cross_entropy_1')
                slim.losses.softmax_cross_entropy(logits_2, labels, weights=1 / 3.0, scope='cross_entropy_2')
                slim.losses.softmax_cross_entropy(logits_3, labels, weights=1 / 3.0, scope='cross_entropy_3')

                embeddings = end_points['embeddings']
                center_loss = calculate_pooling_center_loss(features=embeddings, label=labels,
                                                            alfa=0.95, nrof_classes=dataset.num_classes,
                                                            weights=1.0, name='center_loss')
                slim.losses.add_loss(center_loss)
                return end_points

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
        first_clone_scope = deploy_config.clone_scope(0)
        # Gather update_ops from the first clone. These contain, for example,
        # the updates for the batch_norm variables created by network_fn.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        # Add summaries for end_points.
        end_points = clones[0].outputs
        for end_point in end_points:
            x = end_points[end_point]
            summaries.add(tf.summary.histogram('activations/' + end_point, x))
            summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                            tf.nn.zero_fraction(x)))

        # Add summaries for losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
            summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

        # Add summaries for variables.
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        #################################
        # Configure the moving averages #
        #################################
        if FLAGS.moving_average_decay:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, global_step)
        else:
            moving_average_variables, variable_averages = None, None

        #########################################
        # Configure the optimization procedure. #
        #########################################
        with tf.device(deploy_config.optimizer_device()):
            num_iter_per_epoch = dataset.num_samples * 80 / FLAGS.batch_size
            n_cycles = 20
            learning_rate = tf.py_func(cosine_annealing, [global_step, num_iter_per_epoch, n_cycles, FLAGS.learning_rate], [tf.float32])
            # learning_rate = _configure_learning_rate(dataset.num_samples, global_step)
            learning_rate = tf.reshape(learning_rate, [])
            optimizer = _configure_optimizer(learning_rate)
            summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        if FLAGS.sync_replicas:
            # If sync_replicas is enabled, the averaging will be done in the chief
            # queue runner.
            optimizer = tf.train.SyncReplicasOptimizer(
                opt=optimizer,
                replicas_to_aggregate=FLAGS.replicas_to_aggregate,
                total_num_replicas=FLAGS.worker_replicas,
                variable_averages=variable_averages,
                variables_to_average=moving_average_variables)
        elif FLAGS.moving_average_decay:
            # Update ops executed locally by trainer.
            update_ops.append(variable_averages.apply(moving_average_variables))

        # Variables to train.
        variables_to_train = _get_variables_to_train()

        #  and returns a train_tensor and summary_op
        total_loss, clones_gradients = model_deploy.optimize_clones(
            clones,
            optimizer,
            var_list=variables_to_train)
        # Add total_loss to summary.
        summaries.add(tf.summary.scalar('total_loss', total_loss))

        # Create gradient updates.
        grad_updates = optimizer.apply_gradients(clones_gradients,
                                                 global_step=global_step)
        update_ops.append(grad_updates)

        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
            train_tensor = tf.identity(total_loss, name='train_op')

        # Add the summaries from the first clone. These contain the summaries
        # created by model_fn and either optimize_clones() or _gather_clone_loss().
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                           first_clone_scope))

        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.visible_device_list = FLAGS.gpus

        #create file log


        ###########################
        # Kicks off the training. #
        ###########################
        train_vars = tf.trainable_variables()
        print("Num params: %d" % num_params(train_vars))
        slim.learning.train(
            train_tensor,
            logdir=FLAGS.train_dir,
            master=FLAGS.master,
            is_chief=(FLAGS.task == 0),
            init_fn=_get_init_fn(),
            summary_op=summary_op,
            number_of_steps=FLAGS.max_number_of_steps,
            log_every_n_steps=FLAGS.log_every_n_steps,
            save_summaries_secs=FLAGS.save_summaries_secs,
            save_interval_secs=FLAGS.save_interval_secs,
            sync_optimizer=optimizer if FLAGS.sync_replicas else None,
            session_config=config
        )


if __name__ == '__main__':
    tf.app.run()
