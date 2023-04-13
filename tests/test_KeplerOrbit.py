import numpy as np
import KeplerOrbit
def test_cart2kep2cart():
	tol = 1e-10

	# Earth orbiting Sun, slightly inclined so angles are defined
	m1, m2 = 1, 1e-20

	a = 1
	e = 0.05
	inc = 0.1
	asc_node = np.pi
	omega = np.pi
	M = np.pi

	X, Y, Z, vx, vy, vz = KeplerOrbit.kep2cart(a, e, inc, asc_node, omega, M, m1, m2)

	assert np.fabs(X - -1.05) < tol, "cart2kep2kart position axis does not match"
	assert np.fabs(Y - 3.782338790704024e-16) < tol, "cart2kep2kart position axis does not match"
	assert np.fabs(Z - -2.5048146051777413e-17) < tol, "cart2kep2kart position axis does not match"
	assert np.fabs(vx - -3.490253699036788e-16) < tol, "cart2kep2kart velocity axis does not match"
	assert np.fabs(vy - -0.9464377445249709) < tol, "cart2kep2kart velocity axis does not match"
	assert np.fabs(vz - 0.09496052074620637) < tol, "cart2kep2kart velocity axis does not match"

	a, e, inc, asc_node, omega, M = KeplerOrbit.cart2kep(X, Y, Z, vx, vy, vz, m1, m2)

	assert np.fabs(a - 1) < tol, "cart2kep semimajor axis does not match"
	assert np.fabs(e - 0.05) < tol, "cart2kep eccentricity does not match"
	assert np.fabs(inc - 0.1) < tol, "cart2kep inclination does not match"
	assert np.fabs(asc_node - np.pi) < tol, "cart2kep Omega does not match"
	assert np.fabs(omega - np.pi) < tol, "cart2kep omega does not match"
	assert np.fabs(M - np.pi) < tol, "cart2kep mean anomaly does not match"

	# Now try converting back to cartesian
	X1, Y1, Z1, vx1, vy1, vz1 = KeplerOrbit.kep2cart(a, e, inc, asc_node, omega, M, m1, m2)

	assert np.fabs(X1 - X) < tol, "cart2kep2kart position axis does not match"
	assert np.fabs(Y1 - Y) < tol, "cart2kep2kart position axis does not match"
	assert np.fabs(Z1 - Z) < tol, "cart2kep2kart position axis does not match"
	assert np.fabs(vx1 - vx) < tol, "cart2kep2kart velocity axis does not match"
	assert np.fabs(vy1 - vy) < tol, "cart2kep2kart velocity axis does not match"
	assert np.fabs(vz1 - vz) < tol, "cart2kep2kart velocity axis does not match"

	return None

def test_snap():
	import os
	import numpy as np
	import pynbody as pb
	import KeplerOrbit

	tol = 1e-8

	path = os.path.dirname(os.path.abspath(__file__))
	snap = pb.load(path + '/test.ic')
	pl = KeplerOrbit.orb_params(snap)

	assert np.mean(pl['a']) - 2.96233536 < tol, "orb_params semimajor axis does not match"
	assert np.mean(pl['e']) - 0.06809352 < tol, "orb_params eccentricity axis does not match"
	assert np.mean(pl['inc']) - 0.005003 < tol, "orb_params inclination axis does not match"
	assert np.mean(pl['asc_node']) - 3.15458748 < tol, "orb_params Omega axis does not match"
	assert np.mean(pl['omega']) - 3.13253456, "orb_params omega axis does not match"
	assert np.mean(pl['M']) - 3.13972053 < tol, "orb_params mean anomaly axis does not match"

	return None
