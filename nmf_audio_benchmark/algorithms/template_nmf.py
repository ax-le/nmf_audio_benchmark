"""
Template to add a new NMF method.

The code must implement a class, following the nowadays standard scikit-learn and PyTorch API. The class must have the following methods:
- __init__: Initialize the NMF object with the parameters.
- run: Run the NMF algorithm on the data provided. The W_0 and H_0 matrices can be provided as well, for custom initialization. The harmonic initialization is also available, and requires a feature object to be provided. 
"""

class TemplateNMF():
    """
    NMF object, following the scikit-learn and PyTorch API.
    
    The method to compute NMF is called 'run', may be discussed.
    """
    def __init__(self, rank, beta, init, nmf_type):
        """
        Instanciate the NMF object with the parameters.

        Parameters
        ----------
        rank : int
            Rank of the NMF decomposition.
        beta : float
            Beta parameter of the NMF decomposition.
        init : str
            Initialization method of the NMF decomposition. Can be "random", "nndsvd", "custom" or "harmonic".
            See the actual NMF object for details.
            Harmonic initialization is a custom initialization, with W following the pitch of a harmonic notes. They follow the MIDI pitches, from 1 to 88.
        nmf_type : str, optional
            Type of NMF decomposition. Can be "unconstrained", "min_vol" or "sparse". The default is "unconstrained".
            This is a placeholder for experiments, in order to implement and compare different NMF types.
        """
        self.rank = rank
        self.beta = beta
        self.init = init
        self.nmf_type = nmf_type
        # You can add your own parameters here.

    def run(self, data, W_0 = None, H_0 = None):
        """
        Actually compute the NMF. Returns the W and H matrices.

        Parameters
        ----------  
        data : np.ndarray
            Data matrix to decompose.
        W_0 : np.ndarray, optional
            Initial W matrix. The default is None.
            Only used if init is set to "custom" (or "harmonic").
        H_0 : np.ndarray, optional
            Initial H matrix. The default is None.
            Only used if init is set to "custom" (or "harmonic").
        feature_object : object, optional
            Feature object, used ony for the "harmonic" initialization. The default is None.

        Returns
        -------
        W : np.ndarray
            W matrix of the NMF decomposition.  
        H : np.ndarray
            H matrix of the NMF decomposition.
        """
        raise NotImplementedError("TODO. Implement the NMF algorithm here.")
