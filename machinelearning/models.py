import numpy as np

import backend
import nn


class Model(object):
    """Base model class for the different applications"""

    def __init__(self):
        self.get_data_and_monitor = None
        self.learning_rate = 0.0

    def run(self, x, y=None):
        raise NotImplementedError("Model.run must be overriden by subclasses")

    def train(self):
        """
        Train the model.

        `get_data_and_monitor` will yield data points one at a time. In between
        yielding data points, it will also monitor performance, draw graphics,
        and assist with automated grading. The model (self) is passed as an
        argument to `get_data_and_monitor`, which allows the monitoring code to
        evaluate the model on examples from the validation set.
        """
        for x, y in self.get_data_and_monitor(self):
            graph = self.run(x, y)
            graph.backprop()
            graph.step(self.learning_rate)


class RegressionModel(Model):
    """
    TODO: Question 4 - [Application] Regression

    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.1
        self.hidden_size = 300

        self.w1 = nn.Variable(1, self.hidden_size)
        self.w2 = nn.Variable(self.hidden_size, self.hidden_size)
        self.w3 = nn.Variable(self.hidden_size, 1)
        self.b1 = nn.Variable(self.hidden_size)
        self.b2 = nn.Variable(self.hidden_size)
        self.b3 = nn.Variable(1)

    def run(self, x, y=None):
        """
        TODO: Question 4 - [Application] Regression

        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"
        self.graph = nn.Graph([self.w1, self.w2, self.w3, self.b1, self.b2, self.b3])

        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            input_x = nn.Input(self.graph, x)
            input_y = nn.Input(self.graph, y)
            xw1 = nn.MatrixMultiply(self.graph, input_x, self.w1)
            xw1_plus_b1 = nn.MatrixVectorAdd(self.graph, xw1, self.b1)
            l1 = nn.ReLU(self.graph, xw1_plus_b1)
            l1w2 = nn.MatrixMultiply(self.graph, l1, self.w2)
            l2w2_plus_b2 = nn.MatrixVectorAdd(self.graph, l1w2, self.b2)
            l2 = nn.ReLU(self.graph, l2w2_plus_b2)
            l2w3 = nn.MatrixMultiply(self.graph, l2, self.w3)
            l2w3_plus_b3= nn.MatrixVectorAdd(self.graph, l2w3, self.b3)
            loss = nn.SquareLoss(self.graph, l2w3_plus_b3, input_y)

            return self.graph
            
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            input_x = nn.Input(self.graph, x)
           
            xw1 = nn.MatrixMultiply(self.graph, input_x, self.w1)
            xw1_plus_b1 = nn.MatrixVectorAdd(self.graph, xw1, self.b1)
            l1 = nn.ReLU(self.graph, xw1_plus_b1)
            l1w2 = nn.MatrixMultiply(self.graph, l1, self.w2)
            l2w2_plus_b2 = nn.MatrixVectorAdd(self.graph, l1w2, self.b2)
            l2 = nn.ReLU(self.graph, l2w2_plus_b2)
            l2w3 = nn.MatrixMultiply(self.graph, l2, self.w3)
            l2w3_plus_b3= nn.MatrixVectorAdd(self.graph, l2w3, self.b3)

            return self.graph.get_output(l2w3_plus_b3)



class OddRegressionModel(Model):
    """
    TODO: Question 5 - [Application] OddRegression

    A neural network model for approximating a function that maps from real
    numbers to real numbers.

    Unlike RegressionModel, the OddRegressionModel must be structurally
    constrained to represent an odd function, i.e. it must always satisfy the
    property f(x) = -f(-x) at all points during training.
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.1
        self.hidden_size = 400
        self.neg_1 = np.array([[-1.]])

        self.w1 = nn.Variable(1, self.hidden_size)
        self.w2 = nn.Variable(self.hidden_size, self.hidden_size)
        self.w3 = nn.Variable(self.hidden_size, 1)
        self.b1 = nn.Variable(self.hidden_size)
        self.b2 = nn.Variable(self.hidden_size)
        self.b3 = nn.Variable(1)


    def run(self, x, y=None):
        """
        TODO: Question 5 - [Application] OddRegression

        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"
        self.graph = nn.Graph([self.w1, self.w2, self.w3, self.b1, self.b2, self.b3])

        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            input_x = nn.Input(self.graph, x)
            input_neg_x = nn.Input(self.graph, -x)
            input_neg_one = nn.Input(self.graph, self.neg_1)
            input_y = nn.Input(self.graph, y)

            xw1 = nn.MatrixMultiply(self.graph, input_x, self.w1)
            xw1_plus_b1 = nn.MatrixVectorAdd(self.graph, xw1, self.b1)
            l1 = nn.ReLU(self.graph, xw1_plus_b1)
            l1w2 = nn.MatrixMultiply(self.graph, l1, self.w2)
            l2_pos = nn.MatrixVectorAdd(self.graph, l1w2, self.b2)
            l2 = nn.ReLU(self.graph, l2_pos)
            l2w3 = nn.MatrixMultiply(self.graph, l2, self.w3)
            l3_pos= nn.MatrixVectorAdd(self.graph, l2w3, self.b3)

            xw1 = nn.MatrixMultiply(self.graph, input_neg_x, self.w1)
            xw1_plus_b1 = nn.MatrixVectorAdd(self.graph, xw1, self.b1)
            l1 = nn.ReLU(self.graph, xw1_plus_b1)
            l1w2 = nn.MatrixMultiply(self.graph, l1, self.w2)
            l2_neg = nn.MatrixVectorAdd(self.graph, l1w2, self.b2)
            l2 = nn.ReLU(self.graph, l2_neg)
            l2w3 = nn.MatrixMultiply(self.graph, l2, self.w3)
            l3_neg = nn.MatrixVectorAdd(self.graph, l2w3, self.b3)


            neg_l3_neg = nn.MatrixMultiply(self.graph, l3_neg, input_neg_one)
            odd_f = nn.Add(self.graph, l3_pos, neg_l3_neg)

            loss = nn.SquareLoss(self.graph, odd_f, input_y)

            return self.graph

        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            input_x = nn.Input(self.graph, x)
            input_neg_x = nn.Input(self.graph, -x)
            input_neg_one = nn.Input(self.graph, self.neg_1)

            xw1 = nn.MatrixMultiply(self.graph, input_x, self.w1)
            xw1_plus_b1 = nn.MatrixVectorAdd(self.graph, xw1, self.b1)
            l1 = nn.ReLU(self.graph, xw1_plus_b1)
            l1w2 = nn.MatrixMultiply(self.graph, l1, self.w2)
            l2_pos = nn.MatrixVectorAdd(self.graph, l1w2, self.b2)
            l2 = nn.ReLU(self.graph, l2_pos)
            l2w3 = nn.MatrixMultiply(self.graph, l2, self.w3)
            l3_pos= nn.MatrixVectorAdd(self.graph, l2w3, self.b3)

            xw1 = nn.MatrixMultiply(self.graph, input_neg_x, self.w1)
            xw1_plus_b1 = nn.MatrixVectorAdd(self.graph, xw1, self.b1)
            l1 = nn.ReLU(self.graph, xw1_plus_b1)
            l1w2 = nn.MatrixMultiply(self.graph, l1, self.w2)
            l2_neg = nn.MatrixVectorAdd(self.graph, l1w2, self.b2)
            l2 = nn.ReLU(self.graph, l2_neg)
            l2w3 = nn.MatrixMultiply(self.graph, l2, self.w3)
            l3_neg = nn.MatrixVectorAdd(self.graph, l2w3, self.b3)


            neg_l3_neg = nn.MatrixMultiply(self.graph, l3_neg, input_neg_one)
            odd_f = nn.Add(self.graph, l3_pos, neg_l3_neg)

            return self.graph.get_output(odd_f)


class DigitClassificationModel(Model):
    """
    TODO: Question 6 - [Application] Digit Classification

    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_digit_classification

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.1
        self.hidden_size = 400

        self.w1 = nn.Variable(784, self.hidden_size)
        self.w2 = nn.Variable(self.hidden_size, self.hidden_size)
        self.w3 = nn.Variable(self.hidden_size, 10)
        self.b1 = nn.Variable(1,self.hidden_size)
        self.b2 = nn.Variable(1,self.hidden_size)
        self.b3 = nn.Variable(1,10)

    def run(self, x, y=None):
        """
        TODO: Question 6 - [Application] Digit Classification

        Runs the model for a batch of examples.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 10) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should predict a (batch_size x 10) numpy array of scores,
        where higher scores correspond to greater probability of the image
        belonging to a particular class. You should use `nn.SoftmaxLoss` as your
        training loss.

        Inputs:
            x: a (batch_size x 784) numpy array
            y: a (batch_size x 10) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 10) numpy array of scores (aka logits)
        """
        "*** YOUR CODE HERE ***"
        self.graph = nn.Graph([self.w1, self.w2, self.w3, self.b1, self.b2, self.b3])

        if y is not None:
            print self.w1.data.shape,self.w2.data.shape,self.w3.data.shape,self.b1.data.shape,self.b2.data.shape,self.b3.data.shape
            "*** YOUR CODE HERE ***"
            input_x = nn.Input(self.graph, x)
            input_y = nn.Input(self.graph, y)

            xw1 = nn.MatrixMultiply(self.graph, input_x, self.w1)
            xw1_plus_b1 = nn.MatrixVectorAdd(self.graph, xw1, self.b1)
            l1 = nn.ReLU(self.graph, xw1_plus_b1)
            l1w2 = nn.MatrixMultiply(self.graph, l1, self.w2)
            l2w2_plus_b2 = nn.MatrixVectorAdd(self.graph, l1w2, self.b2)
            l2 = nn.ReLU(self.graph, l2w2_plus_b2)
            l2w3 = nn.MatrixMultiply(self.graph, l2, self.w3)
            l2w3_plus_b3 = nn.MatrixVectorAdd(self.graph, l2w3, self.b3)

            print l2w3_plus_b3,input_y.data.shape,type(input_y)

            loss = nn.SoftmaxLoss(self.graph, l2w3_plus_b3, input_y)
            return self.graph

        else:
            "*** YOUR CODE HERE ***"

            input_x = nn.Input(self.graph, x)
           
            xw1 = nn.MatrixMultiply(self.graph, input_x, self.w1)
            xw1_plus_b1 = nn.MatrixVectorAdd(self.graph, xw1, self.b1)
            l1 = nn.ReLU(self.graph, xw1_plus_b1)

            l1w2 = nn.MatrixMultiply(self.graph, l1, self.w2)
            l2w2_plus_b2 = nn.MatrixVectorAdd(self.graph, l1w2, self.b2)
            l2 = nn.ReLU(self.graph, l2w2_plus_b2)

            l2w3 = nn.MatrixMultiply(self.graph, l2, self.w3)
            l2w3_plus_b3 = nn.MatrixVectorAdd(self.graph, l2w3, self.b3)

            return self.graph.get_output(l2w3_plus_b3)


class DeepQModel(Model):
    """
    TODO: Question 7 - [Application] Reinforcement Learning

    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.

    (We recommend that you implement the RegressionModel before working on this
    part of the project.)
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_rl

        self.num_actions = 2
        self.state_size = 4

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        self.learning_rate = 0.05
        self.hidden_size = 50

        self.w1 = nn.Variable(4,self.hidden_size)
        self.w2 = nn.Variable(self.hidden_size,self.hidden_size)
        self.w3 = nn.Variable(self.hidden_size,2)
        self.b1 = nn.Variable(1,self.hidden_size)
        self.b2 = nn.Variable(1,self.hidden_size)
        self.b3 = nn.Variable(1,2)

    def run(self, states, Q_target=None):
        """
        TODO: Question 7 - [Application] Reinforcement Learning

        Runs the DQN for a batch of states.

        The DQN takes the state and computes Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]

        When Q_target == None, return the matrix of Q-values currently computed
        by the network for the input states.

        When Q_target is passed, it will contain the Q-values which the network
        should be producing for the current states. You must return a nn.Graph
        which computes the training loss between your current Q-value
        predictions and these target values, using nn.SquareLoss.

        Inputs:
            states: a (batch_size x 4) numpy array
            Q_target: a (batch_size x 2) numpy array, or None
        Output:
            (if Q_target is not None) A nn.Graph instance, where the last added
                node is the loss
            (if Q_target is None) A (batch_size x 2) numpy array of Q-value
                scores, for the two actions
        """
        "*** YOUR CODE HERE ***"
        self.graph = nn.Graph([self.w1, self.w2, self.w3, self.b1, self.b2, self.b3])
        if Q_target is not None:
            input_x = nn.Input(self.graph, states)
            input_y = nn.Input(self.graph, Q_target)
            l1 = nn.ReLU(self.graph,nn.MatrixVectorAdd(self.graph,nn.MatrixMultiply(self.graph,input_x,self.w1),self.b1))
            l2 = nn.ReLU(self.graph,nn.MatrixVectorAdd(self.graph,nn.MatrixMultiply(self.graph,l1,self.w2),self.b2))
            l3 = nn.MatrixVectorAdd(self.graph,nn.MatrixMultiply(self.graph,l2,self.w3),self.b3)
            loss = nn.SquareLoss(self.graph, l3, input_y)
            print "loss!!!",loss.shape
            return self.graph
        else:
            input_x = nn.Input(self.graph, states)
            l1 = nn.ReLU(self.graph,nn.MatrixVectorAdd(self.graph,nn.MatrixMultiply(self.graph,input_x,self.w1),self.b1))
            l2 = nn.ReLU(self.graph,nn.MatrixVectorAdd(self.graph,nn.MatrixMultiply(self.graph,l1,self.w2),self.b2))           
            l3 = nn.MatrixVectorAdd(self.graph,nn.MatrixMultiply(self.graph,l2,self.w3),self.b3)
            print "output", self.graph.get_output(l3).shape
            return self.graph.get_output(l3)

    def get_action(self, state, eps):
        """
        Select an action for a single state using epsilon-greedy.

        Inputs:
            state: a (1 x 4) numpy array
            eps: a float, epsilon to use in epsilon greedy
        Output:
            the index of the action to take (either 0 or 1, for 2 actions)
        """
        if np.random.rand() < eps:
            return np.random.choice(self.num_actions)
        else:
            scores = self.run(state)
            return int(np.argmax(scores))


class LanguageIDModel(Model):
    """
    TODO: Question 8 - [Application] Language Identification

    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_lang_id

        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.05
        self.hidden_size = 275

        self.w1 = nn.Variable(self.num_chars,self.hidden_size)
        self.w2 = nn.Variable(self.hidden_size,self.num_chars)
        self.b1 = nn.Variable(self.num_chars)

    def run(self, xs, y=None):
        """
        TODO: Question 8 - [Application] Language Identification

        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        (batch_size x self.num_chars) numpy array, where every row in the array
        is a one-hot vector encoding of a character. For example, if we have a
        batch of 8 three-letter words where the last word is "cat", we will have
        xs[1][7,0] == 1. Here the index 0 reflects the fact that the letter "a"
        is the inital (0th) letter of our combined alphabet for this task.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 5) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node that represents a (batch_size x hidden_size)
        array, for your choice of hidden_size. It should then calculate a
        (batch_size x 5) numpy array of scores, where higher scores correspond
        to greater probability of the word originating from a particular
        language. You should use `nn.SoftmaxLoss` as your training loss.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a (batch_size x self.num_chars) numpy array
            y: a (batch_size x 5) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 5) numpy array of scores (aka logits)

        Hint: you may use the batch_size variable in your code
        """
        "*** YOUR CODE HERE ***"
        batch_size = xs[0].shape[0]

        self.graph = nn.Graph([self.w1,self.w2,self.b1])

        if y is not None:
            self.input_y = nn.Input(self.graph,y)
            self.h_vec = nn.Input(self.graph,np.zeros((batch_size,self.hidden_size)))  

            #iterate through all characters list
            for chars in xs:
                charz = nn.Input(self.graph,chars)
                mm = nn.MatrixMultiply(self.graph,charz,self.w1)
                #keep updating h_vec
                self.h_vec = nn.ReLU(self.graph,nn.Add(self.graph,mm,self.h_vec))

            #final matrix computation
            another_mm = nn.MatrixMultiply(self.graph,self.h_vec,self.w2)
            self.h_final = nn.MatrixVectorAdd(self.graph,another_mm,self.b1)

            #loss
            loss = nn.SoftmaxLoss(self.graph,self.h_final,self.input_y)

            return self.graph
        else:
            self.h_vec = nn.Input(self.graph,np.zeros((batch_size,self.hidden_size)))  

            #iterate through all characters list
            for chars in xs:
                charz = nn.Input(self.graph,chars)
                mm = nn.MatrixMultiply(self.graph,charz,self.w1)
                #keep updating h_vec
                self.h_vec = nn.ReLU(self.graph,nn.Add(self.graph,mm,self.h_vec))

            #final matrix computation
            another_mm = nn.MatrixMultiply(self.graph,self.h_vec,self.w2)
            self.h_final = nn.MatrixVectorAdd(self.graph,another_mm,self.b1)

            #predict
            return self.graph.get_output(self.h_final)
