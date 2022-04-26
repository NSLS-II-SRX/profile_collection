# Deployment acceptance tests
# Running the tests from IPython
# %run -i ~/.ipython/profile_collection/acceptance_tests/tests.py


def test_change_energy():
    energy.move(7.2)


def test_set_roi():
    setroi(1, 'Fe')


def test_clear_roi():
    clearroi()


def test_get_binding_energies():
    getbindingE('Fe')


def test_get_emmision_energies():
    getemissionE('Fe')


def test_xanes_scan():
    """
    Xanes scan test.
    If db.table() and export scan complete without errors than it was successful.
    """
    print("Starting xanes scan test")
    Fe_K = getbindingE('Fe')
    uid, = RE(xanes_plan([Fe_K - 100, Fe_K - 20, Fe_K + 50, Fe_K + 150],
                         [2.0, 1.0, 2.0], samplename='Fe Foil',
                         filename='FeFoilStd', acqtime=1.0))
    print("Fly scan complete")
    print("Reading scan from databroker ...")
    db[uid].table(fill=True)
    print("Exporting scan ...")
    export_scan(db[uid].start['scan_id'])
    print("Test is complete")


def test_fly_scan1():
    """
    Fly scan test 2.
    If db.table() and export scan complete without errors than it was successful.
    """
    print("Starting fly scan test 2")
    uid, = RE(nano_scan_and_fly(-5, 5, 11, 0, 2, 3, 0.1, shutter=False)
    print("Fly scan complete")
    print("Reading scan from databroker")
    db[uid].table(fill=True)
    print("Exporting scan")
    export_scan(db[uid].start['scan_id'])
    print("Test is complete")


def test_fly_scan2():
    """
    Fly scan test 2.
    If db.table() and export scan complete without errors than it was successful.
    """
    print("Starting fly scan test 2")
    uid, = RE(nano_y_scan_and_fly(-5, 5, 11, 0, 2, 3, 0.1, shutter=False))
    print("Fly scan complete")
    print("Reading scan from databroker")
    db[uid].table(fill=True)
    print("Exporting scan")
    export_scan(db[uid].start['scan_id'])
    print("Test is complete")


def test_step_scan():
    """
    Fly scan test 2.
    If db.table() and export scan complete without errors than it was successful.
    """
    print("Starting fly scan test 2")
    uid, = RE(nano_xrf(-5, 5, 1, 0, 2, 1, 0.1, shutter=False))
    print("Fly scan complete")
    print("Reading scan from databroker")
    db[uid].table(fill=True)
    print("Exporting scan")
    export_scan(db[uid].start['scan_id'])
    print("Test is complete")


def test_pyxrf_tools():
    run-pyxrftools


def test_make_hdf():
    make_hdf(12345)


test_change_energy()
