import logging
import numpy as np
import sys
sys.path.insert(0, '..')

import chronostar.transform as tf

def convertPolarToCartesian(pos):
    """Take a point in polar space to cartesian space"""
    r, theta = pos.T
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.array([x, y]).T

def convertManyPolarToCartesian(points):
    """
    Take many points in polar space and take to cartesian space

    Parameters
    ----------
    points : [n, 2] array
        Array of polar points (r,theta)

    Returns
    -------
    result : [n,2] array
        Array of cartesian points (x,y)
    """
    rs, thetas = points.T
    xs = rs * np.cos(thetas)
    ys = rs * np.sin(thetas)
    return np.vstack((xs,ys)).T

def test_polar():
    """Generates a collection of points in polar space from a covariance matrix
    Transforms covariance matrix to cartesian, and compares to a numpy
    covariance matrix fitted to the points in cartesian space"""

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    logging.info("Beginning polar demo")

    # setting up polar points
    r_mean = np.sqrt(10 ** 2 + 10 ** 2)
    theta_mean = 14 * np.pi / 6.
    r_std = 1.
    theta_std = np.pi / 24.
    C_rt = 0.0

    pol_mean = np.array([r_mean, theta_mean])
    pol_cov = np.array([
        [r_std ** 2, C_rt * r_std * theta_std],
        [C_rt * r_std * theta_std, theta_std ** 2],
    ])

    nsamples = 100000
    pol_samples = np.random.multivariate_normal(pol_mean, pol_cov, nsamples)

    # converting to cartesian manually for comparison
    res = convertManyPolarToCartesian(pol_samples) #[:, 0], pol_samples[:, 1])

    cart_mean = convertPolarToCartesian(pol_mean)
    logging.debug("Covariance matrix:\n{}".format(pol_cov))
    logging.debug("Polar mean: {}".format(pol_mean))
    cart_cov = tf.transform_covmatrix(pol_cov, convertPolarToCartesian,
                                      pol_mean, dim=2)

    cart_samples = np.random.multivariate_normal(cart_mean, cart_cov, nsamples)

    estimated_mean = np.mean(res, axis=0)
    estimated_cov = np.cov(res.T)

    assert np.allclose(estimated_mean, cart_mean, rtol=1e-1)
    assert np.allclose(estimated_cov, cart_cov, rtol=1e-1)

if __name__ == '__main__':
    test_polar()