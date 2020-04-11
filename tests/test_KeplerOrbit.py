def test_cart2kep2cart():
	import math
	import KeplerOrbit

	tol = 1e-10

	# Earth orbiting Sun, slightly inclined so angles are defined
	m1, m2 = 1, 1e-20

	a = 1
	e = 0.05
	inc = 0.1
	asc_node = math.pi
	omega = math.pi
	M = math.pi

	X, Y, Z, vx, vy, vz = KeplerOrbit.kep2cart(a, e, inc, asc_node, omega, M, m1, m2)

	assert math.fabs(X - -1.05) < tol, "cart2kep2kart position axis does not match"
	assert math.fabs(Y - 3.782338790704024e-16) < tol, "cart2kep2kart position axis does not match"
	assert math.fabs(Z - -2.5048146051777413e-17) < tol, "cart2kep2kart position axis does not match"
	assert math.fabs(vx - -3.490253699036788e-16) < tol, "cart2kep2kart velocity axis does not match"
	assert math.fabs(vy - -0.9464377445249709) < tol, "cart2kep2kart velocity axis does not match"
	assert math.fabs(vz - 0.09496052074620637) < tol, "cart2kep2kart velocity axis does not match"

	a, e, inc, asc_node, omega, M = KeplerOrbit.cart2kep(X, Y, Z, vx, vy, vz, m1, m2)

	assert math.fabs(a - 1) < tol, "cart2kep semimajor axis does not match"
	assert math.fabs(e - 0.05) < tol, "cart2kep eccentricity does not match"
	assert math.fabs(inc - 0.1) < tol, "cart2kep inclination does not match"
	assert math.fabs(asc_node - math.pi) < tol, "cart2kep Omega does not match"
	assert math.fabs(omega - math.pi) < tol, "cart2kep omega does not match"
	assert math.fabs(M - math.pi) < tol, "cart2kep mean anomaly does not match"

	# Now try converting back to cartesian
	X1, Y1, Z1, vx1, vy1, vz1 = KeplerOrbit.kep2cart(a, e, inc, asc_node, omega, M, m1, m2)

	assert math.fabs(X1 - X) < tol, "cart2kep2kart position axis does not match"
	assert math.fabs(Y1 - Y) < tol, "cart2kep2kart position axis does not match"
	assert math.fabs(Z1 - Z) < tol, "cart2kep2kart position axis does not match"
	assert math.fabs(vx1 - vx) < tol, "cart2kep2kart velocity axis does not match"
	assert math.fabs(vy1 - vy) < tol, "cart2kep2kart velocity axis does not match"
	assert math.fabs(vz1 - vz) < tol, "cart2kep2kart velocity axis does not match"

	return None

# Other tests to write:
# - test vectorized version of this function