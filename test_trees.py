import numpy as np
import trees as t

def test_mse():
    assert t.mse(None, np.array([10.,10.,10.,10.,30.,30.,30.,30.]))[1] == 100.
    assert t.mse(None, np.array([10.,10.,10.,10.]))[1] == 0.
    assert t.mse(None, np.array([5,5,5,5,10,10,10,10]))[1] == 6.25

def test_find_threshold_mse():
    # x,y
    # x is one dimension
    # should go through x and get score at every split

    X = np.array([1,2,3,4,5,6,7,8]).reshape(-1, 1)
    y = np.array([5,5,5,5,10,10,10,10])
    idx,score = t.find_threshold(t.mse, X, y, 1, X.shape[0])
    assert idx == 4
    assert score == 2.5**2

def test_find_threshold_mae():
    # x,y
    # x is one dimension
    # should go through x and get score at every split

    X = np.array([1,2,3,4,5,6]).reshape(-1, 1)
    y = np.array([5,10,15,25,30,35])
    idx,score = t.find_threshold(t.mae, X, y, 1, X.shape[0])
    assert idx == 3
    assert score == 10 - np.mean([5., 0., 5.])

def test_sort_and_extract():
    X = np.array([[1,10],
                  [2,9],
                  [3,8],
                  [4,7],
                  [5,6],
                  [6,5],
                  [7,4],
                  [8,3]])
    y = np.array([10,20,30,40,50,60,70,80])

    # sorts by given x and returns 2-d array
    xo,yo = t.sort_for_dim(X, y, 0)
    assert np.all(xo == X)
    assert np.all(yo == y)

    xo,yo = t.sort_for_dim(X, y, 1)

    assert np.all(xo == np.array([[8,3],
                                  [7,4],
                                  [6,5],
                                  [5,6],
                                  [4,7],
                                  [3,8],
                                  [2,9],
                                  [1,10]]))
    assert np.all(yo == np.flip(y))


def test_find_next_split_1():
    X = np.array([[1,1],
                  [2,90],
                  [3,63],
                  [4,7],
                  [5,60],
                  [6,5],
                  [7,4],
                  [8,3]])
    y = np.array([10,20,30,40,50,60,70,80])
    nxt = t.find_next_split(t.mse, X, y, np.array([0,1]), 2)
    assert nxt == t.Split(0, 4, 5, 400.)

def test_find_next_split_2():
    X = np.array([[1,10],
                  [20,9],
                  [30,8],
                  [4,7],
                  [5,6],
                  [60,5],
                  [7,4],
                  [8,3]])
    y = np.array([10,20,30,40,50,60,70,80])
    nxt = t.find_next_split(t.mse, X, y, np.array([0,1]), 2)
    assert nxt == t.Split(1, 4, 7, 400.)

def test_split_data():
    X = np.array([[1,10],
                  [20,9],
                  [30,8],
                  [4,7],
                  [5,6],
                  [60,5],
                  [7,4],
                  [8,3]])
    y = np.array([10,20,30,40,50,60,70,80])
    split = t.Split(1,3,6,100.)
    Xl, yl, Xr, yr = t.split_data(X, y, split)
    assert np.all(Xl == np.array([[8,3],
                                  [7,4],
                                  [60,5]]))

    assert np.all(yl == np.array([80,70,60]))

    assert np.all(Xr == np.array([[5,6],
                                  [4,7],
                                  [30,8],
                                  [20,9],
                                  [1,10]]))

    assert np.all(yr == np.array([50,40,30,20,10]))


def test_build_tree_obeys_min_samples():
    X = np.array([[1,10],
                  [20,9],
                  [30,8],
                  [4,7],
                  [5,6],
                  [60,5],
                  [7,4],
                  [8,3]])
    y = np.array([10,20,30,40,50,60,70,80])

    tree = t.build_tree(t.mse, X, y,
                        dims = np.array([0,1]),
                        k = 30,
                        min_gain = 0.1,
                        min_samples = 2)

    assert tree.thresh == 7.0
    assert tree.left.thresh == 5.0
    assert tree.right.thresh == 9.0
    assert type(tree.left.left) == t.Leaf
    assert type(tree.right.right) == t.Leaf

    tree = t.build_tree(t.mse, X, y,
                        dims = np.array([0,1]),
                        k = 30,
                        min_gain = 0.1,
                        min_samples = 3)

    assert tree.thresh == 7.0
    assert type(tree.left) == t.Leaf
    assert type(tree.right) == t.Leaf
