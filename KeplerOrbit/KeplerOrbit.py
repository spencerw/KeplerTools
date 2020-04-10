import numpy as np
from scipy import integrate
from .MathHelpers import cross, dot, nr, PQW

G = 6.6759e-8 # cgs units

def cart2kep(X, Y, Z, vx, vy, vz, m1, m2):
    """
    Convert a single set of cartesian positions and velocities
    (in heliocentric coordinates) to kepler orbital elements. This
    function expects CGS units.
    
    Parameters
    ----------
    X: float
        X position of body
    Y: float
        Y position of body
    Z: float
        Z position of body
    vx: float
        x velocity of body
    vy: float
        y velocity of body
    vz: float
        z velocity of body
    m1: float
        mass of central body
    m2: float
        mass of orbiting body
    
    Returns
    -------
    float
        a, e, inc, asc_node, omega, M - Semimajor axis, eccentricity
        inclination, longitude of ascending node, longitude of 
        perihelion and mean anomaly of orbiting body
    """
    
    r = np.array([X, Y, Z])
    v = np.array([vx, vy, vz])
    mu = G*(m1+m2)
    magr = np.sqrt(X**2. + Y**2. + Z**2.)
    magv = np.sqrt(vx**2. + vy**2. + vz**2.)
    
    h = np.cross(r, v)
    magh = np.sqrt(np.sum(h**2.))
    evec = 1./mu * np.cross(v, h) - r/magr
    e = np.sqrt(np.sum(evec**2.))
    a = np.dot(h, h) / (mu * (1. - e**2.))
    
    ivec = [1., 0., 0.]
    jvec = [0., 1., 0.]
    kvec = [0., 0., 1.]
    
    inc = np.arccos(np.dot(kvec, h) / magh)
    
    n = np.cross(kvec, h)
    if inc == 0.:
        asc_node = 0.
    else:
        nmag = np.sqrt(np.sum(n**2.))
        asc_node = np.arccos(np.dot(ivec, n) / nmag)
        if np.dot(n, jvec) < 0:
            asc_node = 2.*np.pi - asc_node

    if inc == 0.:
        omega = np.arctan2(evec[1]/e, evec[0]/e)
    else:
        omega = np.arccos(np.dot(n, evec) / (nmag*e))
        if np.dot(evec, kvec) < 0.:
            omega = 2.*np.pi - omega
    
    theta = np.arccos(np.dot(evec, r) / (e * magr))
    if np.dot(r, v) < 0.:
        theta = 2.*np.pi - theta
    E = np.arccos((e + np.cos(theta)) / (1 + e * np.cos(theta)))
    if theta > np.pi and theta < 2.*np.pi:
        E = 2.*np.pi - E
    M = E - e*np.sin(E)
    
    return a, e, inc, asc_node, omega, M

def cart2kepX(X, Y, Z, vx, vy, vz, m1, m2):
    """
    Convert an array of cartesian positions and velocities
    (in heliocentric coordinates) to an array of kepler orbital
    elements. This function expects CGS units. This function is
    fully vectorized.

    Parameters
    ----------
    X: numpy array of floats
        X position of body
    Y: numpy array of floats
        Y position of body
    Z: numpy array of floats
        Z position of body
    vx: numpy array of floats
        x velocity of body
    vy: numpy array of floats
        y velocity of body
    vz: numpy array of floats
        z velocity of body
    m1: numpy array of floats
        mass of central body
    m2: numpy array of floats
        mass of orbiting body

    Returns
    -------
    numpy array of floats
        a, e, inc, asc_node, omega, M - Semimajor axis, eccentricity
        inclination, longitude of ascending node, longitude of 
        perihelion and mean anomaly of orbiting body
    """
    
    mu = G*(m1 + m2)
    magr = np.sqrt(X**2. + Y**2. + Z**2.)
    magv = np.sqrt(vx**2. + vy**2. + vz**2.)
    
    hx, hy, hz = cross(X, Y, Z, vx, vy, vz)
    magh = np.sqrt(hx**2. + hy**2. + hz**2.)
    tmpx, tmpy, tmpz = cross(vx, vy, vz, hx, hy, hz)
    evecx = tmpx/mu - X/magr
    evecy = tmpy/mu - Y/magr
    evecz = tmpz/mu - Z/magr

    e = np.sqrt(evecx**2. + evecy**2. + evecz**2.)
    
    a = dot(hx, hy, hz, hx, hy, hz) / (mu * (1. - e**2.))
    
    ivec = [1., 0., 0.]
    jvec = [0., 1., 0.]
    kvec = [0., 0., 1.]
    
    inc = np.arccos(dot(kvec[0], kvec[1], kvec[2], hx, hy, hz) / magh)
    
    nx, ny, nz = cross(kvec[0], kvec[1], kvec[2], hx, hy, hz)
    nmag = np.sqrt(nx**2. + ny**2. + nz**2.)
    asc_node = np.where(inc == 0., 0., np.arccos(dot(ivec[0], ivec[1], ivec[2], nx, ny, nz) / nmag))
    asc_node[dot(nx, ny, nz, jvec[0], jvec[1], jvec[2]) < 0.] = 2.*np.pi - asc_node[dot(nx, ny, nz, jvec[0], jvec[1], jvec[2]) < 0.]

    omega = np.where(inc == 0., np.arctan2(evecy/e, evecx/e), np.arccos(dot(nx, ny, nz, evecx, evecy, evecz) / (nmag*e)))
    omega[dot(evecx, evecy, evecz, kvec[0], kvec[1], kvec[2]) < 0.] = 2.*np.pi - omega[dot(evecx, evecy, evecz, kvec[0], kvec[1], kvec[2]) < 0.]
    
    theta = np.arccos(dot(evecx, evecy, evecz, X, Y, Z) / (e * magr))
    theta = np.where(dot(X, Y, Z, vx, vy, vz) < 0., 2*np.pi - theta, theta)
                     
    E = np.arccos((e + np.cos(theta)) / (1 + e * np.cos(theta)))
    E = np.where(np.logical_and(theta > np.pi, theta < 2*np.pi), 2*np.pi - E, theta)
    M = E - e*np.sin(E)
    
    return a, e, inc, asc_node % (2*np.pi), omega, M

def kep2cart(a, ecc, inc, Omega, omega, M, mass, m_central):
    """
    Convert a single set of kepler orbital elements into cartesian
    positions and velocities. This function expects CGS units. Note
    that because of the Newton-Raphson function, this routine cannot
    be vectorized.

    Parameters
    ----------
    a: float
        semi-major axis body
    ecc: float
        eccentricity of body
    inc: float
        inclination of body
    Omega: float
        longitude of ascending node of body
    omega: float
        longitude of perihelion of body
    M: float
        Mean anomaly of body
    mass: float
        mass of body
    m_central: float
        mass of central body

    Returns
    -------
    float
        pos, vel - Tuples containing the x, y and z positions
        and velocities of the body
    """

    a = a
    mass = mass
    m_central = m_central

    if inc == 0.:
        Omega = 0.
    if ecc == 0.:
        omega = 0.
    E = nr(M, ecc)
    
    X = a * (np.cos(E) - ecc)
    Y = a * np.sqrt(1. - ecc**2.) * np.sin(E)
    mu = m_central + mass
    n = np.sqrt(mu / a**3.)
    Edot = n / (1. - ecc * np.cos(E))
    Vx = - a * np.sin(E) * Edot
    Vy = a * np.sqrt(1. - ecc**2.) * Edot * np.cos(E)
    
    Px, Py, Pz, Qx, Qy, Qz = PQW(Omega, omega, inc)

    # Rotate Positions
    x = X * Px + Y * Qx
    y = X * Py + Y * Qy
    z = X * Pz + Y * Qz

    # Rotate Velocities
    vx = Vx * Px + Vy * Qx
    vy = Vx * Py + Vy * Qy
    vz = Vx * Pz + Vy * Qz
    
    pos = x, y, z
    vel = vx, vy, vz
    
    return pos, vel

def orb_params(snap):
    """
    Takes a Pynbody snapshot of particles and calculates gives them fields
    corresponding to Kepler orbital elements. Assumes that the central star
    is particle #0 and that the positions and velocities are in barycentric
    coordinates.

    Parameters
    ----------
    snap: SimArray
        A snapshot of the particles

    Returns
    -------
    SimArray
        A copy of 'snap' with additional fields 'a', 'ecc', 'inc', 'asc_node',
        'omega' and 'M' added.
    """

    x = snap.d['pos']
    x_h = x[1:] - x[0]
    v = snap.d['vel']
    v_h = v[1:] - v[0]
    m1 = snap['mass'][0]
    pl = snap[1:]
    m2 = pl['mass']

    pl['a'], pl['e'], pl['inc'], pl['asc_node'], pl['omega'], pl['M'] \
        = cart2kepX(x_h[:,0], x_h[:,1], x_h[:,2], v_h[:,0], v_h[:,1], v_h[:,2], m1, m2)
    
    return pl

def kep2poinc(a, e, i, omega, Omega, M , m1, m2):
    """
    Convert kepler orbital elements into Poincaire variables.
    Can accept either single values or lists of coordinates

    Parameters
    ----------
    a: float
        semi-major axis body
    ecc: float
        eccentricity of body
    inc: float
        inclination of body
    Omega: float
        longitude of ascending node of body
    omega: float
        longitude of perihelion of body
    M: float
        Mean anomaly of body
    mass: float
        mass of body
    m_central: float
        mass of central body

    Returns
    -------
    float
        lam, gam, z, Lam, Gam, Z - Poincaire coordinates
    """

    mu = m1 + m2
    mustar = m1*m2/(m1 + m2)

    lam = M + omega + Omega
    gam = -omega - Omega
    z = -Omega
    Lam = mustar*np.sqrt(mu*a)
    Gam = mustar*np.sqrt(mu*a)*(1 - np.sqrt(1 - e**2))
    Z = mustar*np.sqrt(mu*a*np.sqrt(1 - e**2))*(1 - np.cos(i))

    return lam, gam, z, Lam, Gam, Z

def kep2del(a, e, i, omega, Omega, M , m1, m2):
    """
    Convert kepler orbital elements into Delunay variables.
    Can accept either single values or lists of coordinates

    Parameters
    ----------
    a: float
        semi-major axis body
    ecc: float
        eccentricity of body
    inc: float
        inclination of body
    Omega: float
        longitude of ascending node of body
    omega: float
        longitude of perihelion of body
    M: float
        Mean anomaly of body
    mass: float
        mass of body
    m_central: float
        mass of central body

    Returns
    -------
    float
        l, g, h, L, G, H - Delunay coordinates
    """

    L = np.sqrt((m1 + m2)*a)
    G = L*np.sqrt(1 - e**2)
    H = G*np.cos(i)
    l = M
    g = omega
    h = Omega

    return l, g, h, L, G, H

def kep2mdel(a, e, i, omega, Omega, M, m1, m2):
    """
    Convert kepler orbital elements into modified Delunay variables.
    Can accept either single values or lists of coordinates

    Parameters
    ----------
    a: float
        semi-major axis body
    ecc: float
        eccentricity of body
    inc: float
        inclination of body
    Omega: float
        longitude of ascending node of body
    omega: float
        longitude of perihelion of body
    M: float
        Mean anomaly of body
    mass: float
        mass of body
    m_central: float
        mass of central body

    Returns
    -------
    float
        Lam, P, Q, lam, p, q - Delunay coordinates
    """
    
    L = np.sqrt((m1 + m2)*a)
    G = L*np.sqrt(1 - e**2)
    H = G*np.cos(i)
    l = M
    g = omega
    h = Omega

    Lam = L
    P = L - G
    Q = G - H
    lam = l + g + h
    p = -g - h
    q = -h
    return Lam, P, Q, lam, p, q

def lap(j, s, alpha):
    """ Laplace coefficient """
    def int_func(x):
        return np.cos(j*x)/(1. - (2.*alpha*np.cos(x)) + alpha**2.)**s
    integral = integrate.quad(int_func, 0., 2.*np.pi)[0]
    return 1./np.pi*integral

def d_lap(j, s, alpha):
    """ First alpha derivative of Laplace coefficient """
    return s*(lap(j-1, s+1, alpha) - 2.*alpha*lap(j, s+1, alpha) + lap(j+1, s+1, alpha))

def f_d(j, alpha):
    """ Direct terms of distubring function (M+D table 8.1) """
    l = lap(j, 0.5, alpha)
    dl = d_lap(j, 0.5, alpha)
    return 1/2*(-2*j*l - alpha*dl)

def f_d2(j, alpha):
    """ Direct terms of distubring function for 2nd order interior MMR (M+D table 8.1) """
    l = lap(j, 0.5, alpha)
    dl = d_lap(j, 0.5, alpha)
    return 1/8*(-5*j*l + 4*j**2*l - 2*alpha*dl + 4*j*alpha*dl + alpha**2*dl**2)

def f_s1(alpha):
    """ Secular terms in disturbing function (from table 8.4 in M+D) """
    c1 = 1/4*alpha*d_lap(0, 1/2, alpha)
    c2 = 1/16*alpha**2*d_lap(1, 3/2, alpha)
    c3 = -1/8*alpha**3*d_lap(0, 3/2, alpha)
    c4 = 1/16*alpha**2*d_lap(1, 3/2, alpha)
    c5 = -1/8*alpha**2*lap(0, 3/2, alpha)
    return c1 + c2 + c3 + c4 + c5

def C_r(m_pert, m_c, a, j1, j2):
    """ Constant from resonant part of disturbing function (M+D equation 8.32) """
    alpha = (j2/(j2-1))**(2/3)
    P = 2*np.pi*np.sqrt(a**3/m_c)
    n = 2*np.pi/P
    return (m_pert/m_c)*n*alpha*f_d(j1, alpha)

def curlypidot_d(m, m_c, a, e, j1, j2, j4, phi):
    """ Direct resonance term from M+D equation 8.30 """
    C_r_val = C_r(m, m_c, a, j1, j2)
    return j4*C_r_val*e**(j4 - 2)*np.cos(phi)

def curlypidot_sec(m_pert, m_central, a, alpha):
    """ Secular term from M+D equation 8.30 """
    P = 2*np.pi*np.sqrt(a**3/m_central)
    n = 2*np.pi/P
    C_s = m_pert/m_central*n*alpha*f_s1(alpha)
    return 2*C_s

def res_width(m, m_c, ecc, j1, j2):
    """ Libration width of interior first or second order resonance (Murray + Dermott 8.76) """
    alpha = (-j2/j1)**(2./3.)
    if j2 == 1-j1:
        alpha_f_d = alpha*f_d(j1, alpha)
    elif j2 == 2-j1:
        alpha_f_d = alpha*f_d2(j1, alpha)
    Cr_n = np.fabs(m/m_c*alpha_f_d)
    if j2 == 1-j1:
        da_a = np.sqrt(16./3.*Cr_n*ecc)*np.sqrt(1.+1./(27.*j2**2.*ecc**3)*Cr_n)-(2./(9.*j2*ecc)*Cr_n)
    elif j2 == 2-j1:
        da_a = np.sqrt(16./3.*Cr_n*ecc)
    return da_a

def res_lib_freq(m, m_c, a, ecc, j1, j2, j4):
    """ Libration frequency of first order resonance (Murray + Dermott 8.47) """
    alpha = (j2/(j2-1))**(2./3.)
    alpha_f_d = alpha*f_d(j1, alpha)
    P = 2*np.pi*np.sqrt(a**3/m_c)
    n = 2*np.pi/P
    Crn = n**2*np.fabs(m/m_c*alpha_f_d)
    return np.sqrt(3*j2**2*Crn*ecc**np.fabs(j4))
