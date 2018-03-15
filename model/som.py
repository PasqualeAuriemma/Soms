import tensorflow as tf
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os


# https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/
# https://wonikjang.github.io/deeplearning_unsupervised_som/2017/06/30/som.html
# http://www.ai-junkie.com/ann/som/som1.html

class Som(object):
    # To check if the SOM has been trained
    _trained = False

    def __init__(self, m, n, dim, directory, n_iterations=100, alpha=None, sigma=None):

        self._training_op = None
        self._locations = None
        self._centroid_grid = None
        self._weightages = None
        self.save_dir = directory
        self.m = m
        self.n = n
        self.dim = dim
        if alpha is None:
            self.alpha = 0.3
        else:
            self.alpha = float(alpha)
        if sigma is None:
            self.sigma = max(m, n) / 2.0
        else:
            self.sigma = float(sigma)
        self._n_iterations = abs(n_iterations)

    def model(self):
        #  RESET GRAPH
        tf.reset_default_graph()
        # Create weight vectors Randomly initialized for all neurons of size [m*n,dim]
        #with tf.name_scope('Weights'):
        self.weight_vects = tf.Variable(tf.random_normal([self.m*self.n, self.dim]), name='w1')

        # Matrix of size [m*n, 2] for SOM grid locations of neurons
        #with tf.name_scope('Locations'):
        self.location_vects = tf.constant(np.array(list(self.neuron_locations(self.m, self.n))), name='w2')

        # PLACEHOLDERS FOR TRAINING INPUTS
        with tf.name_scope('Train_input'):
            # The training vector
            self.vect_input = tf.placeholder("float", [self.dim])
            # Iteration number
            self.iter_input = tf.placeholder("float")

        # CONSTRUCT TRAINING OP PIECE BY PIECE

        with tf.name_scope('BMU_index'):
            # To compute the Best Matching Unit given a vector
            # Basically calculates the Euclidean distance between every
            # neuron's weightage vector and the input, and returns the
            # index of the neuron which gives the least value
            pack = tf.stack([self.vect_input for _ in range(self.m*self.n)])
            sub = tf.subtract(self.weight_vects, pack)
            po = tf.pow(sub, 2)
            summary = tf.reduce_sum(po, 1)
            bmu_index = tf.argmin(tf.sqrt(summary), 0)

        # This will extract the location of the BMU based on the BMU's index
        with tf.name_scope('BMU_location'):
            resh = tf.reshape(bmu_index, [1])
            slice_input = tf.pad(resh, np.array([[0, 1]]))
            sli = tf.slice(self.location_vects, slice_input, tf.constant(np.array([1, 2]), dtype=tf.int64))
            bmu_loc = tf.reshape(sli, [2])

        # To compute the alpha and sigma values based on iteration number
        # sigma(t) = sigma(0) exp(-(iteration/tot_iteration))
        with tf.name_scope('Learning_rate_iteration'):
            learning_rate_op = tf.subtract(1.0, tf.div(self.iter_input, self._n_iterations))
        with tf.name_scope('Alpha'):
            alpha_op = tf.multiply(self.alpha, learning_rate_op)
            tf.summary.scalar(name='alpha', tensor=alpha_op)
        with tf.name_scope('Sigma'):
            sigma_op = tf.multiply(self.sigma, learning_rate_op)
            tf.summary.scalar(name='sigma', tensor=sigma_op)

        # Construct the op learning rates for all neurons, based on iteration number and location w.r.t. BMU
        # this is a vector of size [n*m]
        with tf.name_scope('Learning_rate_neighbour'):
            with tf.name_scope('Distance_centroid_from_BMU'):
                pack1 = tf.stack([bmu_loc for _ in range(self.m*self.n)])
                sub1 = tf.subtract(self.location_vects, pack1)
                pow1 = tf.pow(sub1, 2)
                bmu_distance_squares = tf.reduce_sum(pow1, 1)

            # Theta(t) = exp(-(dist^2)/sigma^2(t))
            with tf.name_scope('Tet_a'):
                sig_2 = tf.pow(sigma_op, 2)
                neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(bmu_distance_squares, "float32"), sig_2)))

            learning_rate_op = tf.multiply(alpha_op, neighbourhood_func)
            # Create update weightage vectors of all neurons based on a particular input
            learning_rate_multiplier = tf.stack([tf.tile(tf.slice(learning_rate_op, np.array([i]), np.array([1])),
                                                         [self.dim]) for i in range(self.m*self.n)])

        # Finally update weight
        with tf.name_scope('Update_weight'):
            # W_delta = L(t) * ( V(t)-W(t) )
            sub2 = tf.subtract(tf.stack([self.vect_input for _ in range(self.m*self.n)]), self.weight_vects)
            weightage_delta = tf.multiply(learning_rate_multiplier, sub2)
            # W(t+1) = W(t) + W_delta
            new_weightages_op = tf.add(self.weight_vects, weightage_delta, name='op_training')
            # Update weightge_vects by assigning new_weightages_op to it.
            self._training_op = tf.assign(self.weight_vects, new_weightages_op, name='op_to_restore')

        return self._training_op

    def neuron_locations(self, xm, xn):
            """
            Yields one by one the 2-D locations of the individual neurons
            in the SOM.
            """
            # Nested iterations over both dimensions
            # to generate all 2-D locations in the map
            for i in range(xm):
                for j in range(xn):
                    yield np.array([i, j])

    def train(self, input_vects):
        """
        Trains the SOM.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Current weightage vectors for all neurons(initially random) are
        taken as starting conditions for training.
        """

        self.model()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Create a saver object which will save all the variables
            saver = tf.train.Saver()

            log_path_train = 'logdir' + '/train_{}'.format(datetime.utcnow().strftime("%Y%m%d%H%M%S"))
            train_writer = tf.summary.FileWriter(log_path_train, sess.graph)
            summaries_train = tf.summary.merge_all()

            pp = None
            for epoch in tqdm(range(self._n_iterations)):  # for epoch in range(self._n_iterations):
                # Train with each vector one by one
                for input_v1 in input_vects:
                    up_weight, summaries = sess.run([self._training_op, summaries_train],
                                                    feed_dict={self.vect_input: input_v1,
                                                               self.iter_input: epoch})
                    pp = summaries
                if epoch % 10 == 0:
                    train_writer.add_summary(pp, global_step=epoch)
                    saver.save(sess, os.path.join(self.save_dir, 'train.ckpt'),
                               global_step=epoch)
            # Store a centroid grid for easy retrieval later on
            # centroid_grid = [[] for _ in range(self.m)]
            self._weightages = list(sess.run(self.weight_vects))
            self._locations = list(sess.run(self.location_vects))
            self.calculate_centroid()
            train_writer.close()
            self._trained = True
            # Now, save the graph
            saver.save(sess, os.path.join(self.save_dir, 'my_test_model'), global_step=1000)

    def calculate_centroid(self):
        centroid_grid = [[] for _ in range(self.m)]
        for i, loc in enumerate(self._locations):
                        centroid_grid[loc[0]].append(self._weightages[i])
        self._centroid_grid = centroid_grid

    def get_centroids(self):
        """
        Returns a list of 'm' lists, with each inner list containing
        the 'n' corresponding centroid locations as 1-D NumPy arrays.
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid

    def test_model(self, directory, x):

        with tf.Session() as sess:
            # First let's load meta graph and restore weights
            saver = tf.train.import_meta_graph(os.path.join(directory, 'my_test_model-1000.meta'))
            saver.restore(sess, tf.train.latest_checkpoint(directory))

            # Now, let's access and create placeholders variables and
            # create feed-dict to feed new data
            graph = tf.get_default_graph()
            w1 = graph.get_tensor_by_name("w1:0")
            w2 = graph.get_tensor_by_name("w2:0")

            self._weightages = list(sess.run(w1))
            self._locations = list(sess.run(w2))
            self.calculate_centroid()
            self._trained = True
            bmu_map = self.map_vects([x])
            return bmu_map

    def map_vects(self, input_vects):
        """
        Maps each input vector to the relevant neuron in the SOM
        grid.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Returns a list of 1-D NumPy arrays containing (row, column)
        info for each input vector(in the same order), corresponding
        to mapped neuron.
        """
        if not self._trained:
            raise ValueError("SOM not trained yet 1")

        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self._weightages))],
                            key=lambda x: np.linalg.norm(np.float32(vect)-self._weightages[x]))
            to_return.append(self._locations[min_index])
        return to_return
