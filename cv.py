import numpy as np

def collect_score(node):
    try:
        left = collect_score(node.left)
        right = collect_score(node.right)
        return left + right
    except AttributeError:
        return node.scores[0]

def get_score_for_alpha(alpha, paths):
    filtered = [(a, s) for a, s in paths if a <= alpha]

    if len(filtered) == 0:
        return paths[0][1]
        
    alpha, score = filtered[-1]
    return score

def pick_alpha(paths):
    alphas = {a for p in paths for a,s in p}
    alphas = sorted(list(alphas))

    means = [(a, np.mean(np.array([get_score_for_alpha(a, p) for p in paths])))
             for a in alphas]

    means = sorted(means, key=lambda x: x[1])
    alpha, score = means[0]
    return alpha, score
    

# scored = [[(alpha, collect_score(tree)) for alpha, tree in tree_path] for tree_path in tree_paths]
