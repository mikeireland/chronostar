import numpy as np
import matplotlib.pyplot as plt

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def draw_from_ellipsoid(covmat, cent, npts):
    # random uniform points within ellipsoid as per: https://ma.ttpitk.in/blog/?p=368
    ndims = covmat.shape[0]

    # calculate eigenvalues (e) and eigenvectors (v)
    eigenValues, eigenVectors = np.linalg.eig(covmat)
    idx = (-eigenValues).argsort()[::-1][:ndims]
    e = eigenValues[idx]
    v = eigenVectors[:, idx]
    e = np.diag(e)

    # generate radii of hyperspheres
    rs = np.random.uniform(0, 1, npts)

    # generate points
    pt = np.random.normal(0, 1, [npts, ndims]);


    # get scalings for each point onto the surface of a unit hypersphere
    fac = np.sum(pt ** 2, axis=1)

    # calculate scaling for each point to be within the unit hypersphere
    # with radii rs
    fac = (rs ** (1.0 / ndims)) / np.sqrt(fac)

    pnts = np.zeros((npts, ndims));

    # scale points to the ellipsoid using the eigenvalues and rotate with
    # the eigenvectors and add centroid
    d = np.sqrt(np.diag(e))
    d.shape = (ndims, 1)

    for i in range(0, npts):
        # scale points to a uniform distribution within unit hypersphere
        pnts[i, :] = fac[i] * pt[i, :]
        pnts[i, :] = np.dot(np.multiply(pnts[i, :], np.transpose(d)), np.transpose(v)) + cent

    return pnts


# Small and dense
npts = 5000
cent = np.array([1, 2, 3])
covmat = np.array([[0.2, 0.1, 0.1], [0.1, 0.1, 0], [0.1, 0, 0.15]])
pnts = draw_from_ellipsoid(covmat, cent, npts)

# Big and sparse
npts = 1000
cent = np.array([1, 2, 3])
covmat = np.array([[0.5, 0, 0], [0, 0.2, 0], [0, 0, 0.3]])
pnts2 = draw_from_ellipsoid(covmat, cent, npts)

fig = plt.figure()
# ~ ax=fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)
# ~ ax.scatter(pnts[:,0], pnts[:,1], pnts[:,2], s=2)
ax.scatter(pnts[:, 0], pnts[:, 1], s=2)
print(pnts[:,0])

# ~ ax.scatter(pnts2[:,0], pnts2[:,1], pnts2[:,2], s=2)
plt.show()
