def test_cart2kep():
	import KeplerOrbit

    # Earth orbiting Sun
	X, Y, Z = 1.496e13, 0, 0
	vx, vy, vz = 0, 2979246.96861, 0
	m1, m2 = 1.989e33, 6e27
	a, e, inc, asc_node, omega, M = KeplerOrbit.cart2kep(X, Y, Z, vx, vy, vz, m1, m2)

	assert a > 0, "cart2kep semimajor axis does not match"
	assert e > 0, "cart2kep eccentricity does not match"

	return None