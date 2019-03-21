from pkg_resources import Requirement, resource_filename
from os.path import join
from lever.reader import load_mat

def test_loadmat():
    gerald_log = join(resource_filename(Requirement.parse("lever"), "lever/test/data"), "lever.mat")
    log = load_mat(gerald_log)
    assert(log.trial_time == 5)
    assert(log.timestamps.shape[0] == 117)
    assert(log.sample_rate == 256)
    assert(log.stimulus['config']['reward'] == 6)
    assert(abs(log.stimulus['config']['lev_baseline'] - 3.94221418) < 1E-6)
