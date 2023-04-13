import numpy as np
from scipy import optimize

# Vectorized cross and dot product functions
def cross(x1,y1,z1,x2,y2,z2):
    xc = y1 * z2 - z1 * y2
    yc = z1 * x2 - x1 * z2
    zc = x1 * y2 - y1 * x2
    return xc, yc, zc

def dot(x1,y1,z1,x2,y2,z2):
    return x1*x2+y1*y2+z1*z2

def nr(M, ecc):
    def kep(E, M, ecc):
        return E - (ecc*np.sin(E)) - M

    if isinstance(M, list):
        return optimize.newton(kep, np.ones(len(M)), args=(M, ecc))
    else:
        return optimize.newton(kep, 1.0, args=(M, ecc))

def PQW(Omega, omega, inc):
    """
    Rotation Matrix Components (Orbit Frame => Inertial Frame)
    http://biomathman.com/pair/KeplerElements.pdf (End)
    http://astro.geo.tu-dresden.de/~klioner/celmech.pdf (Eqn. 2.30)
    NB: Shapiro Notation Tricky; Multiplies x = R.T * X, Dresden x = R * X
    """

    Px = np.cos(omega) * np.cos(Omega) - \
         np.sin(omega) * np.cos(inc) * np.sin(Omega)
    Py = np.cos(omega) * np.sin(Omega) + \
         np.sin(omega) * np.cos(inc) * np.cos(Omega)
    Pz = np.sin(omega) * np.sin(inc)

    Qx = - np.sin(omega) * np.cos(Omega) - \
           np.cos(omega) * np.cos(inc) * np.sin(Omega)
    Qy = - np.sin(omega) * np.sin(Omega) + \
           np.cos(omega) * np.cos(inc) * np.cos(Omega)
    Qz =   np.sin(inc) * np.cos(omega)

    return Px, Py, Pz, Qx, Qy, Qz