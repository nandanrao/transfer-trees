import numpy as np
import trees as t

def test_mse():
    assert t.mse(None, np.array([10.,10.,10.,10.,30.,30.,30.,30.]))[1] == 100.
    assert t.mse(None, np.array([10.,10.,10.,10.]))[1] == 0.
    assert t.mse(None, np.array([5.,5.,5.,5.,10.,10.,10.,10.]))[1] == 6.25

def test_get_indices():
    ar = np.array([0,0,0,1,2,3,3,3,5]).reshape(-1, 1)
    idxs = t.get_indices(ar, 0, 2, 7)
    assert np.all(idxs == np.array([3,4,5]))
    idxs = t.get_indices(ar, 0, 1, 8)
    assert np.all(idxs == np.array([3,4,5,8]))
    idxs = t.get_indices(ar, 0, 0, 9)
    assert np.all(idxs == np.array([0,3,4,5,8]))

def test_get_indices_works_with_one_el():
    ar = np.array([0,0,0,1,2,3,3,3,5], dtype=np.float64).reshape(-1, 1)
    idxs = t.get_indices(ar, 0, 3, 3)
    assert np.all(idxs == np.array([3]))

def test_find_threshold_mse():
    # x,y
    # x is one dimension
    # should go through x and get score at every split

    X = np.array([1,2,3,4,5,6,7,8]).reshape(-1, 1)
    y = np.array([5,5,5,5,10,10,10,10], dtype=np.float64)
    idx,score,thresh = t.find_threshold(t.mse, X, y, 0, 1, X.shape[0] - 1)
    assert idx == 4
    assert score == 2.5**2
    assert thresh == 4.5

def test_find_threshold_mae():
    # x,y
    # x is one dimension
    # should go through x and get score at every split

    X = np.array([1,2,3,4,5,6]).reshape(-1, 1)
    y = np.array([5,10,15,25,30,35], dtype=np.float64)
    idx,score,thresh = t.find_threshold(t.mae, X, y, 0, 1, X.shape[0] - 1)
    assert idx == 3
    assert score == 10 - np.mean([5., 0., 5.])
    assert thresh == 3.5

def test_sort_for_dim():
    X = np.array([[1,10],
                  [2,9],
                  [3,8],
                  [4,7],
                  [5,6],
                  [6,5],
                  [7,4],
                  [8,3]])
    y = np.array([10,20,30,40,50,60,70,80], dtype=np.float64)

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
    y = np.array([10,20,30,40,50,60,70,80], dtype=np.float64)
    nxt = t.find_next_split(t.mse, X, y, np.array([0,1]), 2)
    assert nxt == t.Split(0, 4, 4.5, 400.)

def test_find_next_split_2():
    X = np.array([[1,10],
                  [20,9],
                  [30,8],
                  [4,7],
                  [5,6],
                  [60,5],
                  [7,4],
                  [8,3]])
    y = np.array([10,20,30,40,50,60,70,80], dtype=np.float64)
    nxt = t.find_next_split(t.mse, X, y, np.array([0,1]), 2)
    assert nxt == t.Split(1, 4, 6.5, 400.)

def test_split_data():
    X = np.array([[1,10],
                  [20,9],
                  [30,8],
                  [4,7],
                  [5,6],
                  [60,5],
                  [7,4],
                  [8,3]])
    y = np.array([10,20,30,40,50,60,70,80], dtype=np.float64)
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
    y = np.array([10,20,30,40,50,60,70,80], dtype=np.float64)

    tree = t.build_tree(t.mse, X, y,
                        dims = np.array([0,1]),
                        k = 30,
                        min_gain = 0.1,
                        min_samples = 2)

    assert tree.thresh == 6.5
    assert tree.left.thresh == 4.5
    assert tree.right.thresh == 8.5
    assert type(tree.left.left) == t.Leaf
    assert type(tree.right.right) == t.Leaf

    tree = t.build_tree(t.mse, X, y,
                        dims = np.array([0,1]),
                        k = 30,
                        min_gain = 0.1,
                        min_samples = 3)

    assert tree.thresh == 6.5
    assert type(tree.left) == t.Leaf
    assert type(tree.right) == t.Leaf

    tree = t.build_tree(t.mse, X, y,
                        dims = np.array([0,1]),
                        k = 30,
                        min_gain = 0.1,
                        min_samples = 1)

    assert tree.thresh == 6.5
    assert tree.left.thresh == 4.5

    # TODO: the flip of dims isn't actually desired behavior
    # it's just a side effect of the order of argmax returning
    # first on a tie
    assert tree.left.left.dim == 0
    assert tree.left.left.thresh == 7.5
    assert tree.right.thresh == 8.5
    assert tree.right.right.dim == 0
    assert tree.right.right.thresh == 10.5


# def test_build_tree_works_with_repetitions():
#     X = np.array([[0,10],
#                   [0,9],
#                   [0,8],
#                   [0,7],
#                   [0,6],
#                   [2,5],
#                   [3,4],
#                   [10,3]])
#     y = np.array([10,20,30,40,50,60,70,80])

#     tree = t.build_tree(t.mse, X, y,
#                         dims = np.array([0,1]),
#                         k = 30,
#                         min_gain = 0.1,
#                         min_samples = 2)

#     print(tree)
#     assert False
