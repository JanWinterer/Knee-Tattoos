import numpy as np
import pytest

from marker_tracking.frame import apply, check_transform, compose, identity, invert, load, make_transform, save

# 90 deg about z: camera x -> world y, camera y -> world -x
RZ90 = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
T = make_transform(RZ90, [1.0, 2.0, 3.0])


def test_identity_changes_nothing():
    p = np.array([[0.1, -0.2, 0.7], [0.0, 0.05, 0.65]])
    assert np.array_equal(apply(identity(), p), p)


def test_known_rotation_and_translation():
    assert np.allclose(apply(T, [1.0, 0.0, 0.0]), [1.0, 3.0, 3.0])
    assert np.allclose(apply(T, [0.0, 1.0, 0.0]), [0.0, 2.0, 3.0])


@pytest.mark.parametrize('shape', [(3,), (5, 3), (4, 6, 3)])
def test_keeps_shape(shape):
    p = np.random.default_rng(0).normal(size=shape)
    assert apply(T, p).shape == shape


def test_matches_matrix_multiply():
    p = np.random.default_rng(1).normal(size=(10, 3))
    homog = np.c_[p, np.ones(10)] @ T.T
    assert np.allclose(apply(T, p), homog[:, :3])


def test_invert_round_trip():
    p = np.random.default_rng(2).normal(size=(10, 3))
    assert np.allclose(apply(invert(T), apply(T, p)), p)
    assert np.allclose(compose(T, invert(T)), np.eye(4))


def test_compose_order():
    T_shift = make_transform(t=[0.0, 0.0, 1.0])
    p = [1.0, 0.0, 0.0]
    assert np.allclose(apply(compose(T, T_shift), p), apply(T, apply(T_shift, p)))


@pytest.mark.parametrize('bad', [
    np.eye(3),                                   # wrong shape
    np.diag([2.0, 1.0, 1.0, 1.0]),               # scaling
    np.diag([-1.0, 1.0, 1.0, 1.0]),              # mirror
    np.vstack([np.eye(4)[:3], [0, 0, 1, 1]]),    # bad bottom row
    np.full((4, 4), np.nan),                     # NaN
])
def test_rejects_non_rigid(bad):
    with pytest.raises(ValueError):
        check_transform(bad)


def test_rejects_points_without_xyz():
    with pytest.raises(ValueError):
        apply(T, [1.0, 2.0])


def test_load_none_is_identity():
    T_loaded, source = load(None)
    assert np.array_equal(T_loaded, np.eye(4))
    assert source == 'identity'


def test_save_load_round_trip(tmp_path):
    path = tmp_path / 'world.json'
    save(path, T, 'plate')
    T_loaded, source = load(path)
    assert np.allclose(T_loaded, T)
    assert source == 'plate'


def test_load_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path / 'nope.json')


def test_load_reads_key_from_bigger_file(tmp_path):
    path = tmp_path / 'camera.json'
    path.write_text('{"fx": 645.5, "T_world_cam": [[1,0,0,0],[0,1,0,0],[0,0,1,0.5],[0,0,0,1]]}')
    T_loaded, source = load(path)
    assert np.allclose(apply(T_loaded, [0, 0, 0]), [0, 0, 0.5])
    assert source == 'unknown'
