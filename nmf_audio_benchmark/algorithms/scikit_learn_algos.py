"""
Class for the NMF algorithm implemented in scikit-learn. Never tested.

See the actual scikit-learn API for more details. https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
"""

from sklearn.decomposition import NMF as sklearn_NMF

class SKNMF():
    """
    NMF object, following the scikit-learn and PyTorch API.
    
    The method to compute NMF is called 'run', may be discussed.
    """
    def __init__(self, rank, loss, beta, init, update_rule = "mu", n_iter = 100, tol=0.0001, random_state=None, alpha_W=0.0, alpha_H='same', l1_ratio=0.0, verbose=0, shuffle=False):
        """
        Instanciate the NMF object with the parameters.

        Follows the scikit-learn API.
        """
        self.rank = rank
        self.loss = loss
        self.init = init
        self.beta = beta
        self.solver = update_rule
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.alpha_W = alpha_W
        self.alpha_H = alpha_H
        self.l1_ratio = l1_ratio
        self.verbose = verbose
        self.shuffle = shuffle
    
    def run(self, data):
        """
        Actually compute the NMF. Returns the W and H matrices.

        Parameters
        ----------  
        data : np.ndarray
            Data matrix to decompose.

        Returns
        -------
        W : np.ndarray
            W matrix of the NMF decomposition.  
        H : np.ndarray
            H matrix of the NMF decomposition.
        """
        nmf = sklearn_NMF(n_components=self.rank, init=self.init, beta_loss=self.beta, solver=self.solver, max_iter=self.n_iter, tol=self.tol, random_state=self.random_state, alpha=self.alpha_W, l1_ratio=self.l1_ratio, verbose=self.verbose, shuffle=self.shuffle)
        W = nmf.fit_transform(data)
        H = nmf.components_
        return W, H
