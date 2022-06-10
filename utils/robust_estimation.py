from typing import Tuple, Union

import numpy as np
from scipy.special import erfc
from sklearn.utils.extmath import randomized_svd
from sklearn.covariance import EmpiricalCovariance
from sklearn.utils import check_random_state


class BeingRobust(EmpiricalCovariance):
    """Being Robust (in High Dimensions) Can Be Practical: robust estimator of location (and potentially covariance).
    This estimator is to be applied on Gaussian-distributed data. For other distributions some changes might be
    required. Please check out the original paper and/or Matlab code.
    Parameters
    ----------
    eps : float, optional
        Fraction of perturbed data points, by default 0.1
    tau : float, optional
        Significance level, by default 0.1
    cher : float, optional
        Factor filter criterion, by default 2.5
    use_randomized_svd : bool, optional
        If True use `sklearn.utils.extmath.randomized_svd`, else use full SVD, by default True
    debug : bool, optional
        If True print debug information, by default False
    assume_centered : bool
        If True, the data is not centered beforehand, by default False
    random_state : Union[int, np.random.RandomState],
        Determines the pseudo random number generator for shuffling the data.
        Pass an int for reproducible results across multiple function calls. By default none
    keep_filtered : bool, optional
        If True teh filtered data point are kept (`BeingRobust#filtered_`) the, by default False
    Attributes
    ----------
    location_ : np.ndarray of shape (n_features,)
        Estimated robust location.
    filtered_ : np.ndarray of shape (?, n_features)
        Remaining data points for estimating the mean.
    Examples
    --------
    #>>> import numpy as np
    #>>> from being_robust import BeingRobust
    #>>> real_cov = np.array([[.8, .3], [.3, .4]])
    #>>> rng = np.random.RandomState(0)
    #>>> X = rng.multivariate_normal(mean=[0, 0], cov=real_cov, size=500)
    #>>> br = BeingRobust(random_state=0, keep_filtered=True).fit(X)
    #>>> br.location_
    #array([0.0622..., 0.0193...])
    #>>> br.filtered_
    #array([[-1.6167..., -0.6431...], ...
    """

    def __init__(self,
                 eps: float = 0.1,
                 tau: float = 0.1,
                 cher: float = 2.5,
                 use_randomized_svd: bool = True,
                 debug: bool = False,
                 assume_centered: bool = False,
                 random_state: Union[int, np.random.RandomState] = None,
                 keep_filtered: bool = False):
        super().__init__()
        self.eps = eps
        self.tau = tau
        self.cher = cher
        self.use_randomized_svd = use_randomized_svd
        self.debug = debug
        self.random_state = random_state
        self.assume_centered = assume_centered
        self.keep_filtered = keep_filtered

    def fit(self, X, y=None) -> 'BeingRobust':
        """Fits the data to obtain the robust estimate.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y: Ignored
            Not used, present for API consistence purpose.
        Returns
        -------
        self : BeingRobust
        """
        X = self._validate_data(X, ensure_min_samples=1, estimator='BeingRobust')
        random_state = check_random_state(self.random_state)

        self.location_, X = filter_gaussian_mean(X,
                                                 eps=self.eps,
                                                 tau=self.tau,
                                                 cher=self.cher,
                                                 use_randomized_svd=self.use_randomized_svd,
                                                 debug=self.debug,
                                                 assume_centered=self.assume_centered,
                                                 random_state=random_state)
        if self.keep_filtered:
            self.filtered_ = X

        return self


def filter_gaussian_mean(X: np.ndarray,
                         eps: float = 0.1,
                         tau: float = 0.1,
                         cher: float = 2.5,
                         use_randomized_svd: bool = True,
                         debug: bool = False,
                         assume_centered: bool = False,
                         random_state: int = None) -> Tuple[float, np.ndarray]:
    """Being Robust (in High Dimensions) Can Be Practical: robust estimator of location (and potentially covariance).
    This estimator is to be applied on Gaussian-distributed data. For other distributions some changes might be
    required. Please check out the original paper and/or Matlab code.
    Parameters
    ----------
    eps : float, optional
        Fraction of perturbed data points, by default 0.1
    tau : float, optional
        Significance level, by default 0.1
    cher : float, optional
        Factor filter criterion, by default 2.5
    use_randomized_svd : bool, optional
        If True use `sklearn.utils.extmath.randomized_svd`, else use full SVD, by default True
    debug : bool, optional
        If True print debug information, by default False
    assume_centered : bool
        If True, the data is not centered beforehand, by default False
    random_state : Union[int, np.random.RandomState],
        Determines the pseudo random number generator for shuffling the data.
        Pass an int for reproducible results across multiple function calls. By default none
    Returns
    -------
    Tuple[float, np.ndarray]
        The robust location estimate, the filtered version of `X`
    """
    n_samples, n_features = X.shape

    emp_mean = X.mean(axis=0)

    if assume_centered:
        centered_X = X
    else:
        centered_X = (X - emp_mean) / np.sqrt(n_samples)

    if use_randomized_svd:
        U, S, Vh = randomized_svd(centered_X.T, n_components=1, random_state=random_state)
    else:
        U, S, Vh = np.linalg.svd(centered_X.T, full_matrices=False)

    lambda_ = S[0]**2
    v = U[:, 0]

    if debug:
        print(f'\nRecursing on X of shape {X.shape}')
        print(f'lambda_ < 1 + 3 * eps * np.log(1 / eps) -> {lambda_} < {1 + 3 * eps * np.log(1 / eps)}')
    if lambda_ < 1 + 3 * eps * np.log(1 / eps):
        return emp_mean, X

    delta = 2 * eps
    if debug:
        print(f'delta={delta}')

    projected_X = X @ v
    med = np.median(projected_X)
    projected_X = np.abs(projected_X - med)
    sorted_projected_X_idx = np.argsort(projected_X)
    sorted_projected_X = projected_X[sorted_projected_X_idx]

    for i in range(n_samples):
        T = sorted_projected_X[i] - delta
        filter_crit_lhs = n_samples - i
        filter_crit_rhs = cher * n_samples * \
            erfc(T / np.sqrt(2)) / 2 + eps / (n_samples * np.log(n_samples * eps / tau))
        if filter_crit_lhs > filter_crit_rhs:
            break

    if debug:
        print(f'filter data at index {i}')

    if i == 0 or i == n_samples - 1:
        return emp_mean, X

    return filter_gaussian_mean(
        X[sorted_projected_X_idx[:i + 1]],
        eps=eps,
        tau=tau,
        cher=cher,
        use_randomized_svd=use_randomized_svd,
        debug=debug,
        assume_centered=assume_centered,
        random_state=random_state
    )