"""
NMF code from the nn_fac library [1]. The code is adapted to the nmf_audio_benchmark library.

The code implements a class, following the nowadays standard scikit-learn and PyTorch API. The class is called nn_fac_NMF, and has the following methods:
- __init__: Initialize the NMF object with the parameters.
- run: Run the NMF algorithm on the data provided. The W_0 and H_0 matrices can be provided as well, for custom initialization. The harmonic initialization is also available, and requires a feature object to be provided. 

For now, only implements the unconstrained NMF type.

TODO: Sparse NMF could be implemented, and the min_vol NMF type could be implemented as well.

References
----------
[1] A. Marmoret and J. Cohen, "NN-FAC: Nonnegative factorization techniques toolbox," 2020.
"""

import nmf_audio_benchmark.algorithms.utils.init_helper as init_helper # For the "harmonic" initialization, i.e. init of the W matrix according to harmonic spectrum.

import nn_fac.nmf as NMF

import numpy as np
import warnings

class unconstrained_NMF():
    """
    NMF object, following the scikit-learn and PyTorch API.
    
    The method to compute NMF is called 'run', may be discussed.
    """
    # TODO: Implement the min_vol and sparse NMF types

    def __init__(self, rank, beta, init, nmf_type="unconstrained", update_rule = "mu", n_iter = 200, tol=1e-6, sparsity_coefficients=[None, None], fixed_modes = [], normalize=[True, False], verbose=False):
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
        update_rule : str, optional
            Update rule of the NMF decomposition. Can be "mu" or "hals". The default is "mu".
            "hals" only works for beta = 2.
        n_iter : int, optional
            Maximum number of iterations. The default is 100.
        tol : float, optional
            Tolerance of the stopping criterion. The default is 0.0001. 
        sparsity_coefficients : list, optional
            Sparsity coefficients of the W and H matrices. The default is [None, None].
        fixed_modes : list, optional
            Fixed modes of the W and H matrices. The default is [].
        normalize : list, optional
            Normalize the W and H matrices. The default is [False, False].
        verbose : bool, optional
            Verbose mode. The default is False.
        nmf_type : str, optional
            Type of NMF decomposition. Can be "unconstrained", "min_vol" or "sparse". The default is "unconstrained".
            This is a placeholder for experiments, in order to implement and compare different NMF types.
        """
        if init == "harmonic": # The harmonic init means that W is set as harmonic spectra, following the pitch of a harmonic notes. They follow the MIDI pitches, from 1 to 88.
            self.init = "custom"
            self._harmonic_init = True
            self.rank = 88
            if rank != 88:
                warnings.warn("Harmonic initialization is set. Rank is set to 88.")
        else:
            self.init = init
            self._harmonic_init = False
            self.rank = rank

        self.beta = beta
        self.update_rule = update_rule
        self.n_iter_max = n_iter
        self.tol = tol
        self.sparsity_coefficients = sparsity_coefficients
        self.fixed_modes = fixed_modes
        self.normalize = normalize
        self.verbose = verbose
        self.nmf_type = nmf_type

    def run(self, data, W_0 = None, H_0 = None, feature_object = None):
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
        if self._harmonic_init:
            assert feature_object is not None, "Feature object must be provided to use the harmonic initialization"
            if W_0 is not None or H_0 is not None:
                warnings.warn("Custom initialization is overriden by the harmonic initialization")
            W_0, H_0 = self.harmonic_init(data, feature_object)

        if self.init == "custom" and (W_0 is None or H_0 is None):
            raise ValueError("Custom initialization requires W_0 and H_0 to be provided.")
            # TODO: for future developements, only W_0 could be provided, with H_0 initialized with a random matrix or via one pass of MU.
            # Useful for dictionary learning or any kind of prior information on W.
                
        # TODO: Different NMF types will be probably implemented in different classes. Will depend on future developments.
        # For now, only the unconstrained NMF type is implemented.
        match self.nmf_type:
            case "unconstrained":

                if self.sparsity_coefficients != [None, None]:
                    warnings.warn("Sparsity coefficients are not used in unconstrained NMF.")

                if self.beta == 2:
                    self.update_rule = "hals"
                else:
                    self.update_rule = "mu"

                W, H = NMF.nmf(data, rank=self.rank, init = self.init, U_0 = W_0, V_0 = H_0, n_iter_max=self.n_iter_max, tol=self.tol,
                       update_rule = self.update_rule, beta = self.beta, sparsity_coefficients = [None, None], fixed_modes = self.fixed_modes, normalize = self.normalize,
                       verbose=self.verbose, return_costs=False, deterministic=True)

            case "min_vol":
                raise NotImplementedError("Min_vol NMF not implemented yet.")
                # TODO
            case "sparse":
                raise NotImplementedError("Sparse NMF not implemented yet.")
                # TODO
            case _:
                raise NotImplementedError(f"NMF type not understood: {self.nmf_type}")
        
        return W, H

    def harmonic_init(self, data, feature_object):
        """
        Initialize W_0 with a harmonic template, following the pitch of a piano. 
        Credits to Meinard Müller and Tim Zunner. https://www.audiolabs-erlangen.de/resources/MIR/FMP/C8/C8S3_NMFSpecFac.html.
        
        Parameters
        ----------
        data : np.ndarray
            Data matrix to decompose.
        feature_object : object
            Feature object, used to get the sampling rate and the number of FFT points.
        
        Returns
        -------
        W_0 : np.ndarray
            W matrix of the NMF decomposition, initialized with a harmonic template.
        H_0 : np.ndarray
            H matrix of the NMF decomposition, initialized with a random matrix.
        """

        # Initialize W_0 with a harmonic template, following the pitch of a piano. Credits to Meinard Müller and Tim Zunner. https://www.audiolabs-erlangen.de/resources/MIR/FMP/C8/C8S3_NMFSpecFac.html.
        W_0 = init_helper.init_nmf_template_pitch(K=data.shape[0], pitch_set=range(1,89), freq_res=feature_object.sr/feature_object.n_fft, tol_pitch=0.05)

        # One pass of MU for H
        H_0 = NMF.mu.switch_alternate_mu(data, W_0, np.random.rand(W_0.shape[1], data.shape[1]), self.beta, "H")
        return W_0, H_0



