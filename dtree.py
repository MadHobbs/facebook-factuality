"""
Name: Shota Yasunaga
Assignment: PS2 Project
"""

"""
Author      : Yi-Chieh Wu
Class       : HMC CS 158
Date        : 2018 Aug 11
Description : Decision Tree Classifier
"""

# Use only the provided packages!
import collections
from util import *

# numpy libraries
import numpy as np

# scikit-learn libraries
from sklearn import tree

######################################################################
# classes
######################################################################

class Tree(object) :
    """
    Array-based representation of a binary decision tree.
    
    See tree._tree.Tree (a Python wrapper around a C class).
    The binary tree is represented as a number of parallel arrays. The i-th
    element of each array holds information about the node 'i'. Node 0 is the
    tree's root. NOTE: Some of the arrays only apply to either leaves or split
    nodes, resp. In this case the values of nodes of the other type are
    arbitrary!
    
    Attributes
    --------------------
      node_count : int
          The number of nodes (internal nodes + leaves) in the tree.
      
      children_left : array of int, shape [node_count]
          children_left[i] holds the node id of the left child of node i.
          For leaves, children_left[i] == TREE_LEAF. Otherwise,
          children_left[i] > i. This child handles the case where
          X[:, feature[i]] <= threshold[i].
      
      children_right : array of int, shape [node_count]
          children_right[i] holds the node id of the right child of node i.
          For leaves, children_right[i] == TREE_LEAF. Otherwise,
          children_right[i] > i. This child handles the case where
          X[:, feature[i]] > threshold[i].
      
      feature : array of int, shape [node_count]
          feature[i] holds the feature to split on, for the internal node i.
      
      threshold : array of double, shape [node_count]
          threshold[i] holds the threshold for the internal node i.
      
      value : array of double, shape [node_count, 1, max_n_classes]
          value[i][0] holds the counts of each class reaching node i
      
      impurity : array of double, shape [node_count]
          impurity[i] holds the impurity at node i.
      
      n_node_samples : array of int, shape [node_count]
          n_node_samples[i] holds the number of training samples reaching node i.          
    """
    TREE_LEAF = tree._tree.TREE_LEAF
    TREE_UNDEFINED = tree._tree.TREE_UNDEFINED
    
    def __init__(self, n_features, n_classes, n_outputs=1) :
        if n_outputs != 1 :
            raise NotImplementedError("each sample must have a single label")
        
        self.n_features     = n_features
        self.n_classes      = n_classes
        self.n_outputs      = n_outputs
        
        capacity = 2047 # arbitrary, allows max_depth = 10
        self.node_count     = capacity
        self.children_left  = np.empty(self.node_count, dtype=int)
        self.children_right = np.empty(self.node_count, dtype=int)
        self.feature        = np.empty(self.node_count, dtype=int)
        self.threshold      = np.empty(self.node_count)
        self.value          = np.empty((self.node_count, n_outputs, n_classes))
        self.impurity       = np.empty(self.node_count)
        self.n_node_samples = np.empty(self.node_count, dtype=int)
        
        # private
        self._next_node     = 1 # start at root
        self._classes       = None
    
    #========================================
    # helper functions
    
    def _get_value(self, y) :
        """
        Get count of each class.
        
        Parameters
        --------------------
            y     -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            value -- numpy array of shape (n_classes,), class counts
                     value[i] holds count of each class
        """
        if len(y) == 0 :
            raise Exception("cannot separate empty set")
        
        counter = collections.defaultdict(lambda: 0)
        for label in y :
            counter[label] += 1
        
        value = np.empty((self.n_classes,))
        for i, label in enumerate(self._classes) :
            value[i] = counter[label]
        
        return value
    
    def _entropy(self, y) :
        """
        Compute entropy.
        
        Parameters
        --------------------
            y -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            H -- entropy
        """
        
        # compute counts
        _, counts = np.unique(y, return_counts=True)
        
        ### ========== TODO : START ========== ###
        # part a: compute entropy
        # hint: use np.log2 to take log
        H=0
        for i in counts:
            P_i = float(i)/sum(counts)
            H -= float(i)/sum(counts) * np.log2(float(i)/sum(counts))
        

        ### ========== TODO : END ========== ###
        
        return H
    
    def _information_gain(self, Xj, y) :
        """
        Compute information gain.
        
        Parameters
        --------------------
            Xj             -- numpy array of shape (n,), samples (one feature only)
            y              -- numpy array of shape (n,), target classes
                
        Returns
        --------------------
            info_gain      -- float, information gain using best threshold
            best_threshold -- float, threshold with best information gain
        """
        n = len(Xj)
        if n != len(y) :
            raise Exception("feature vector and class vector must have same length")
        
        # compute entropy
        H = self._entropy(y)
        
        # reshape feature vector to shape (n,1)
        Xj = Xj.reshape((n,1))
        values = np.unique(Xj) # unique values in Xj, sorted
        n_values = len(values)
        
        # compute optimal conditional entropy by trying all thresholds
        thresholds = np.empty(n_values - 1) # possible thresholds
        H_conds = np.empty(n_values - 1)    # associated conditional entropies
        for i in xrange(n_values - 1) :
            threshold = (values[i] + values[i+1]) / 2.
            thresholds[i] = threshold
            
            X1, y1, X2, y2 = self._split_data(Xj, y, 0, threshold)
            ### ========== TODO : START ========== ###
            # part c: compute conditional entropy

            length = len(y1) + len(y2)
            _,counts1 = np.unique(y1,return_counts =True)
            _,counts2 = np.unique(y2, return_counts = True)
            H_cond = 0
            for count in counts1:
                P = float(count)/float(sum(counts1))
                H_cond -= float(len(y1))/float(length)*P*np.log2(P)
            for count in counts2:
                P = float(count)/float(sum(counts2))
                H_cond -= float(len(y2))/float(length)*P*np.log2(P)

            ### ========== TODO : END ========== ###
            H_conds[i] = H_cond
        
        # find minimium conditional entropy (maximum information gain)
        # and associated threshold
        best_H_cond = H_conds.min()
        indices = np.where(H_conds == best_H_cond)[0]
        best_index = np.random.choice(indices)
        best_threshold = thresholds[best_index]
        
        # compute information gain
        info_gain = H - best_H_cond
        
        return info_gain, best_threshold
        
    def _split_data(self, X, y, feature, threshold) :
        """
        Split dataset (X,y) into two datasets (X1,y1) and (X2,y2)
        based on feature and threshold.
        
        (X1,y1) contains the subset of (X,y) such that X[i,feature] <= threshold.
        (X2,y2) contains the subset of (X,y) such that X[i,feature] > threshold.
        
        Parameters
        --------------------
            X         -- numpy array of shape (n,d), samples
            y         -- numpy array of shape (n,), target classes
            feature   -- int, feature index to split on
            threshold -- float, feature threshold
        
        Returns
        --------------------
            X1        -- numpy array of shape (n1,d), samples
            y1        -- numpy array of shape (n1,), target classes
            X2        -- numpy array of shape (n2,d), samples
            y2        -- numpy array of shape (n2,), target classes 
        """
        n, d = X.shape
        if n != len(y) :
            raise Exception("feature vector and label vector must have same length")
        
        X1, X2 = [], []
        y1, y2 = [], []
        ### ========== TODO : START ========== ###
        # part b: split data set
        # This is inefficient, but easy to code...
        for i in range(n):
            if X[i][feature] > threshold:
                if X2 == []:
                    X2 = np.array([X[i]])
                    y2 = np.array([y[i]])
                else:
                    X2 = np.vstack((X2,X[i]))
                    y2 = np.append(y2,y[i])
            else:
                if X1 == []:
                    X1 = np.array([X[i]])
                    y1 = np.array([y[i]])
                else:
                    X1 = np.vstack((X1,X[i]))
                    y1 = np.append(y1,y[i])
        
        ### ========== TODO : END ========== ###
        X1, X2 = np.array(X1), np.array(X2)
        y1, y2 = np.array(y1), np.array(y2)
        return X1, y1, X2, y2
    
    def _choose_feature(self, X, y) :
        """
        Choose a feature with max information gain from (X,y).
        
        Parameters
        --------------------
            X             -- numpy array of shape (n,d), samples
            y             -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            best_feature   -- int, feature to split on
            best_threshold -- float, feature threshold
        """
        n, d = X.shape
        if n != len(y) :
            raise Exception("feature vector and label vector must have same length")
        
        # compute optimal information gain by trying all features
        thresholds = np.empty(d) # best threshold for each feature
        scores     = np.empty(d) # best information gain for each feature
        for j in xrange(d) :
            if (X[:,j] == X[0,j]).all() :
                # skip if all feature values equal
                score, threshold = -1, None # use an invalid (but numeric) score
            else :
                score, threshold = self._information_gain(X[:,j], y)
            thresholds[j] = threshold
            scores[j] = score
        
        # find maximum information gain
        # and associated feature and threshold
        best_score = scores.max()
        indices = np.where(scores == best_score)[0]
        best_feature = np.random.choice(indices)
        best_threshold = thresholds[best_feature]
        
        return best_feature, best_threshold
    
    def _create_new_node(self, node, feature, threshold, value, impurity) :
        """
        Create a new internal node.
        
        Parameters
        --------------------
            node      -- int, current node index
            feature   -- int, feature index to split on
            threshold -- float, feature threshold
            value     -- numpy array of shape (n_classes,), class counts of current node
            impurity  -- float, impurity of current node
        """
        self.children_left[node]  = self._next_node
        self._next_node += 1
        self.children_right[node] = self._next_node
        self._next_node += 1
        
        self.feature[node]        = feature
        self.threshold[node]      = threshold
        self.value[node]          = value
        self.impurity[node]       = impurity
        self.n_node_samples[node] = sum(value)
    
    def _create_new_leaf(self, node, value, impurity) :
        """
        Create a new leaf node.
        
        Parameters
        --------------------
            node     -- int, current node index
            value    -- numpy array of shape (n_classes,), class counts of current node
            impurity -- float, impurity of current node
        """
        self.children_left[node]  = Tree.TREE_LEAF
        self.children_right[node] = Tree.TREE_LEAF
        
        self.feature[node]        = Tree.TREE_UNDEFINED
        self.threshold[node]      = Tree.TREE_UNDEFINED
        self.value[node]          = value
        self.impurity[node]       = impurity
        self.n_node_samples[node] = sum(value)
                
    def _build_helper(self, X, y, node=0) :
        """
        Build a decision tree from (X,y) in depth-first fashion.
        
        Parameters
        --------------------
            X        -- numpy array of shape (n,d), samples
            y        -- numpy array of shape (n,), target classes
            node     -- int, current node index (index of root for current subtree)
        """
        
        n, d = X.shape
        value = self._get_value(y)
        impurity = self._entropy(y)
        
        ### ========== TODO : START ========== ###
        # part d: decision tree induction algorithm
        # you can modify any code within this TODO block
        
        # base case
        # 1) all samples have same labels
        # 2) all feature values are equal
        uniquey = np.unique(y)
        uniqueX = np.unique(X, axis = 0)
        if len(uniquey) < 2 or len(uniqueX) < 2: # you should modify this condition
            # this line is so that the code can run
            # you can comment it out (or not) once you add your own code
            
            # create a single leaf
            self._create_new_leaf(node, value, impurity)
            
        else:
            # this line is so that the code can run
            # you can comment it out (or not) once you add your own code
            
            # choose best feature (and find associated threshold)
            best_feature,best_threshold = self._choose_feature(X,y)
            # make new decision tree node
            self._create_new_node(node,best_feature, best_threshold, value,impurity)
            next_left = self._next_node-2
            next_right = self._next_node-1
            # split data on best feature
            X1,y1,X2,y2 = self._split_data(X,y,best_feature,best_threshold)
            # build left subtree using recursion
            

            self._build_helper(X1,y1,node = next_left)
            # build right subtree using recursion
            self._build_helper(X2,y2,node = next_right)
        ### ========== TODO : END ========== ###
    
    #========================================
    # main functions
    
    def fit(self, X, y) :
        """
        Build a decision tree from (X,y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        
        # y must contain only integers
        if not np.equal(np.mod(y, 1), 0).all() :
            raise NotImplementedError("y must contain only integers")
        
        # store classes
        self._classes = np.unique(y)
        
        # build tree
        self._build_helper(X, y)
        
        # resize arrays
        self.node_count     = self._next_node
        self.children_left  = self.children_left[:self.node_count]
        self.children_right = self.children_right[:self.node_count]
        self.feature        = self.feature[:self.node_count]
        self.threshold      = self.threshold[:self.node_count]
        self.value          = self.value[:self.node_count]
        self.impurity       = self.impurity[:self.node_count]
        self.n_node_samples = self.n_node_samples[:self.node_count]
        
        return self
    
    def predict(self, X) :
        """
        Predict target for X.
        
        Parameters
        --------------------
            X -- numpy array of shape (n,d), samples
        
        Returns
        --------------------
            y -- numpy array of shape (n,n_classes), values
        """
        
        n, d = X.shape
        y = np.empty((n, self.n_classes))
        
        ### ========== TODO : START ========== ###
        # part e: make predictions
        
        # for each sample
        #   start at root of tree
        #   follow edges to leaf node
        #   find value at leaf node
        
        for i in range(n):
            pointer = 0 
            while self.children_right[pointer] != Tree.TREE_LEAF:
                if X[i][self.feature[pointer]] > self.threshold[pointer]:
                    pointer = self.children_right[pointer]
                else:
                    pointer = self.children_left[pointer]
            y[i] = self.value[pointer]

        
        ### ========== TODO : END ========== ###
        
        return y


class Classifier(object) :
    """
    Classifier interface.
    """
    
    def fit(self, X, y):
        raise NotImplementedError()
        
    def predict(self, X):
        raise NotImplementedError()


class DecisionTreeClassifier(Classifier) :
    
    def __init__(self, criterion="entropy", random_state=None) :
        """
        A decision tree classifier.
        
        Attributes
        --------------------
            classes_    -- numpy array of shape (n_classes, ), the classes labels
            n_classes_  -- int, the number of classes
            n_features_ -- int, the number of features
            n_outputs_  -- int, the number of outputs
            tree_       -- the underlying Tree object
        """
        if criterion != "entropy":
            raise NotImplementedError()
        
        self.n_features_ = None
        self.classes_    = None
        self.n_classes_  = None
        self.n_outputs_  = None
        self.tree_       = None
        self.random_state = random_state
    
    def fit(self, X, y) :
        """
        Build a decision tree classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        
        n_samples, self.n_features_ = X.shape
        
        # determine number of outputs
        if y.ndim != 1 :
            raise NotImplementedError("each sample must have a single label")
        self.n_outputs_ = 1
        
        # determine classes
        classes = np.unique(y)
        self.classes_ = classes
        self.n_classes_ = classes.shape[0]
        
        # set random state
        np.random.seed(self.random_state)
        
        # main
        self.tree_ = Tree(self.n_features_, self.n_classes_, self.n_outputs_)
        self.tree_.fit(X, y)
        return self
    
    def predict(self, X) :
        """
        Predict class value for X.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        
        if self.tree_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        
        # defer to self.tree_
        X = X.astype(tree._tree.DTYPE)
        proba = self.tree_.predict(X)
        predictions = self.classes_.take(np.argmax(proba, axis=1), axis=0)
        return predictions



def print_tree(decision_tree, feature_names=None, class_names=None, root=0, depth=1):
    """
    Print decision tree.
    
    Only works with decision_tree.n_outputs = 1.
    https://healthyalgorithms.com/2015/02/19/ml-in-python-getting-the-decision-tree-out-of-sklearn/
        
    Parameters
    --------------------
        decision_tree -- tree (sklearn.tree._tree.Tree or Tree)
        feature_names -- list, feature names
        class_names   -- list, class names
    """
    
    t = decision_tree
    if t.n_outputs != 1:
        raise NotImplementedError()
    
    if depth == 1:
        print 'def predict(x):'
    
    indent = '    ' * depth
    
    # determine node numbers of children
    left_child = t.children_left[root]
    right_child = t.children_right[root]
    
    # determine predicted class for this node
    values = t.value[root][0]
    class_ndx = np.argmax(values)
    if class_names is not None:
        class_str = class_names[class_ndx]
    else:
        class_str = str(class_ndx)
        
    # determine node string     
    node_str = "(node %d: impurity = %.2f, samples = %d, value = %s, class = %s)" % \
        (root, t.impurity[root], t.n_node_samples[root], values, class_str)
    
    # main code
    if left_child == tree._tree.TREE_LEAF:
        print indent + 'return %s # %s' % (class_str, node_str)
    else:
        # determine feature name
        if feature_names is not None:
            name = feature_names[t.feature[root]]
        else:
            name = "x_%d" % t.feature[root]
        
        print indent + 'if %s <= %.2f: # %s' % (name, t.threshold[root], node_str)
        print_tree(t, feature_names, class_names, root=left_child, depth=depth+1)
        
        print indent + 'else:'
        print_tree(t, feature_names, class_names, root=right_child, depth=depth+1)

