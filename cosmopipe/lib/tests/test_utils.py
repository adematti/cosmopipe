from cosmopipe.lib import setup_logging
from cosmopipe.lib.utils import round_measurement


def test_round():
    assert round_measurement(0.01,-1.0,1,sigfigs=2) == ('0.0', '-1.0', '1.0')
    assert round_measurement(0.01,-1.0,0.8,sigfigs=2) == ('0.01', '-1.00', '0.80')
    assert round_measurement(0.0001,-1.0,1,sigfigs=2)  == ('0.0', '-1.0', '1.0')
    assert round_measurement(1e4,-1.0,1,sigfigs=2) == ('1.00000e4', '-1.0', '1.0')
    assert round_measurement(-0.0001,-1.0,1,sigfigs=2) == ('0.0', '-1.0', '1.0')


if __name__ == '__main__':

    setup_logging()
    test_round()
