import MathHelpers
import numpy as np

G = 2.959e-4*(1.15741e-5)**2. #AU^3 / s^2 / Msun

# Convert heliocentric coordinates and velocities to kepler orbital elements
def cart2kep(pos, vel, m1, m2):
    # pos in AU, vel in AU / s
    X, Y, Z = pos
    vx, vy, vz = vel
    
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

# Vectorized version
def cart2kepX(X, Y, Z, vx, vy, vz, m1, m2):
    mu = G*(m1+m2)
    magr = np.sqrt(X**2. + Y**2. + Z**2.)
    magv = np.sqrt(vx**2. + vy**2. + vz**2.)
    
    hx, hy, hz = MathHelpers.cross(X, Y, Z, vx, vy, vz)
    magh = np.sqrt(hx**2. + hy**2. + hz**2.)
    tmpx, tmpy, tmpz = MathHelpers.cross(vx, vy, vz, hx, hy, hz)
    evecx = tmpx/mu - X/magr
    evecy = tmpy/mu - Y/magr
    evecz = tmpz/mu - Z/magr
    e = np.sqrt(evecx**2. + evecy**2. + evecz**2.)
    
    a = MathHelpers.dot(hx, hy, hz, hx, hy, hz) / (mu * (1. - e**2.))
    
    ivec = [1., 0., 0.]
    jvec = [0., 1., 0.]
    kvec = [0., 0., 1.]
    
    inc = np.arccos(MathHelpers.dot(kvec[0], kvec[1], kvec[2], hx, hy, hz) / magh)
    
    nx, ny, nz = MathHelpers.cross(kvec[0], kvec[1], kvec[2], hx, hy, hz)
    nmag = np.sqrt(nx**2. + ny**2. + nz**2.)
    asc_node = np.where(inc == 0., 0., np.arccos(MathHelpers.dot(ivec[0], ivec[1], ivec[2], nx, ny, nz) / nmag))
    asc_node[MathHelpers.dot(nx, ny, nz, jvec[0], jvec[1], jvec[2]) < 0.] = 2.*np.pi - asc_node[MathHelpers.dot(nx, ny, nz, jvec[0], jvec[1], jvec[2]) < 0.]

    omega = np.where(inc == 0., np.arctan2(evecy/e, evecx/e), np.arccos(MathHelpers.dot(nx, ny, nz, evecx, evecy, evecz) / (nmag*e)))
    omega[MathHelpers.dot(evecx, evecy, evecz, kvec[0], kvec[1], kvec[2]) < 0.] = 2.*np.pi - omega[MathHelpers.dot(evecx, evecy, evecz, kvec[0], kvec[1], kvec[2]) < 0.]
    
    theta = np.arccos(MathHelpers.dot(evecx, evecy, evecz, X, Y, Z) / (e * magr))
    theta = np.where(MathHelpers.dot(X, Y, Z, vx, vy, vz) < 0., 2.*np.pi - theta, theta)
                     
    E = np.arccos((e + np.cos(theta)) / (1 + e * np.cos(theta)))
    E = np.where(np.logical_and(theta > np.pi, theta < 2.*np.pi), 2.*np.pi - E, theta)
    M = E - e*np.sin(E)
    
    return a, e, inc, asc_node % (2.*np.pi), omega, M

# Convert kepler orbital elements to heliocentric cartesian coordinates
def kep2cart(a, ecc, inc, Omega, omega, M, mass, m_central):
    if inc == 0.:
        Omega = 0.
    if ecc == 0.:
        omega = 0.
    E = MathHelpers.nr(M, ecc)
    
    X = a * (np.cos(E) - ecc)
    Y = a * np.sqrt(1. - ecc**2.) * np.sin(E)
    G = 1.
    mu = G * (m_central + mass)
    n = np.sqrt(mu / a**3.)
    Edot = n / (1. - ecc * np.cos(E))
    Vx = - a * np.sin(E) * Edot
    Vy = a * np.sqrt(1. - ecc**2.) * Edot * np.cos(E)
    
    Px, Py, Pz, Qx, Qy, Qz = MathHelpers.PQW(Omega, omega, inc)

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

# Get orbital parameters for particles in a pynbody snapshot
# This assumes that the first dark matter particle is the central star
def orb_params(snap):
    x2 = snap.d['pos']
    com_x = np.sum(x2[:,0]*snap.d['mass'])/np.sum(snap.d['mass'])
    com_y = np.sum(x2[:,1]*snap.d['mass'])/np.sum(snap.d['mass'])
    com_z = np.sum(x2[:,2]*snap.d['mass'])/np.sum(snap.d['mass'])
    xpos = x2[1:][:,0] - com_x
    ypos = x2[1:][:,1] - com_y
    zpos = x2[1:][:,2] - com_z
    v2 = snap.d['vel'].in_units('au s**-1')
    v_com_x = np.sum(v2[:,0]*snap.d['mass'])/np.sum(snap.d['mass'])
    v_com_y = np.sum(v2[:,1]*snap.d['mass'])/np.sum(snap.d['mass'])
    v_com_z = np.sum(v2[:,2]*snap.d['mass'])/np.sum(snap.d['mass'])
    xvel = v2[1:][:,0] - v_com_x
    yvel = v2[1:][:,1] - v_com_y
    zvel = v2[1:][:,2] - v_com_z
    m1 = np.max(snap['mass'])
    planetesimals = snap.d[1:]
    m2 = planetesimals['mass']

    a, e, inc, asc_node, omega, M = cart2kepX(xpos, ypos, zpos, xvel, yvel, zvel, m1, m2)
    planetesimals['a'] = a
    planetesimals['e'] = e
    planetesimals['inc'] = inc
    planetesimals['asc_node'] = asc_node
    planetesimals['omega'] = omega
    planetesimals['M'] = M
    
    return planetesimals

# Functions for computing resonance width (see section 8.7 of Murray and Dermott)

from scipy import integrate
# Laplace coefficient
def lap(j, s, alpha):
    def int_func(x):
        return np.cos(j*x)/(1. - (2.*alpha*np.cos(x)) + alpha**2.)**s
    integral = integrate.quad(int_func, 0., 2.*np.pi)[0]
    return 1./np.pi*integral

# First alpha derivative of disturbing function
def d_lap(j, s, alpha):
    return s*(lap(j-1, s+1, alpha) - 2.*alpha*lap(j, s+1, alpha) + lap(j+1, s+1, alpha))

# Direct terms of distubring function (Winter + Murray 1998)
def f_d(j, alpha):
    return (j*lap(j, 0.5, alpha)) + (alpha/2.*d_lap(j, 0.5, alpha))

# Libration width of first order resonance
def res_width_fo(m, m_c, a, ecc, j2):
    alpha = (j2/(j2+1))**(-2./3.)
    alpha_f_d = alpha*f_d(j2, alpha)
    Cr_n = np.fabs(m/m_c*alpha_f_d)
    da_a = np.sqrt(16./3.*Cr_n*ecc)*np.sqrt(1.+1./(27.*j2**2.*ecc**3)*Cr_n)-(2./(9.*j2*ecc)*Cr_n)
    return a*da_a
