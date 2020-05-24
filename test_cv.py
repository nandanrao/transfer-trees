import numpy as np
from cv import pick_alpha, get_score_for_alpha

def test_get_score_for_alpha():

    tree_path = [(.1, 2.5),
                 (.2, 1.0),
                 (.5, 5.5)]

    score = get_score_for_alpha(0.4, tree_path)
    assert score == 1.0

    score = get_score_for_alpha(0.5, tree_path)
    assert score == 5.5

    score = get_score_for_alpha(0.0, tree_path)
    assert score == 2.5


def test_pick_alpha():
    scored_paths = [[(-np.inf, 7.5),
                     (.1, 2.5),
                     (.5, 5.5)],
                    [(-np.inf, 8.5),
                     (.2, 1.0),
                     (.4, 2.75)]]
    alpha, score = pick_alpha(scored_paths)
    assert alpha == .2

    scored_paths = [[(-np.inf, 7.5),
                     (.1, 2.5),
                     (.5, 5.5)],
                    [(-np.inf, 0.5),
                     (.2, 1.0),
                     (.4, 2.75)]]
    alpha, score = pick_alpha(scored_paths)
    assert alpha == .1

    scored_paths = [[(-np.inf, 0.0),
                     (.1, 2.5),
                     (.5, 5.5)],
                    [(-np.inf, 0.5),
                     (.2, 1.0),
                     (.4, 2.75)]]
    alpha, score = pick_alpha(scored_paths)
    assert alpha == -np.inf
