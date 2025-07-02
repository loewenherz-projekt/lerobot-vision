import numpy as np
from lerobot_vision.kinematics import forward_kinematics, inverse_kinematics


def test_roundtrip():
    joints = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
    pos, rot = forward_kinematics(joints)
    out = inverse_kinematics(pos, rot)
    assert np.allclose(out, joints)
