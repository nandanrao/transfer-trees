from data import stack_data, Data, sample_split_data
import numpy as np

def test_stack_data():
    X1, X2 = np.array([[1],[2],[3]]), np.array([[4],[5]])
    y1, y2 = np.array([1,1,1]), np.array([2,2])
    W1, W2 = np.array([[1,1], [0,0], [1,1]]), np.array([[2,2], [3,3]])
    z1, z2 = np.ones(1), np.zeros(1)
    d1 = Data(z1, W1, X1, y1)
    d2 = Data(z2, W2, X2, y2)

    stacked = stack_data([d1, d2], z1)

    assert np.all(stacked.W == np.array([[1,1], [0, 0], [1,1], [2,2], [3,3]]))
    assert np.all(stacked.y == np.array([1,1,1,2,2]))
    assert np.all(stacked.z == z1)


def test_sample_split_data_non_stratified():
    X = np.array([[1, 3], [2, 6], [3, 9]])
    y = np.array([1, 2, 1])
    W = np.array([[1, 1], [0, 0], [1, 1]])
    z = np.ones(1)

    data = Data(z, W, X, y)
    d1, d2 = sample_split_data(data, 0.66, seed=1)

    assert np.all(d1.X == np.array([[1, 3], [3, 9]]))
    assert np.all(d1.W == np.array([[1, 1], [1, 1]]))
    assert np.all(d2.X == np.array([[2, 6]]))
    assert np.all(d2.W == np.array([[0, 0]]))
    assert np.all(d1.y == np.array([1, 1]))
    assert np.all(d2.y == np.array([2]))



def test_sample_split_data_stratified():

    X = np.array([[1, 3], [2, 6], [3, 9], [4, 12]])
    y = np.array([1, 21, 2, 22])
    W = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    z = np.ones(1)

    data = Data(z, W, X, y)

    context_idxs = np.array([0, 1, 0, 1])

    d1, d2 = sample_split_data(data, 0.5, context_idxs=context_idxs, seed=1)

    assert np.all(d1.X == np.array([[3, 9], [4, 12]]))
    assert np.all(d2.X == np.array([[1, 3], [2, 6]]))

    assert np.all(d1.W == np.array([[2, 2], [3, 3]]))
    assert np.all(d2.W == np.array([[0, 0], [1, 1]]))

    assert np.all(d1.y == np.array([2, 22]))
    assert np.all(d2.y == np.array([1, 21]))
