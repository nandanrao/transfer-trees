# the score of me
# compared to the weighted sum of the scores of all the terminal nodes in my branch
# divided by the |T| - 1, where |T| is the number of terminal nodes in the tree

# this is just node.tot_gain / (node.leaves() - 1)

# that gives effective_alpha

# then trim node with lowest effect alpha (turn from node into leaf)
# store score of


from collections import namedtuple
from copy import deepcopy

PruneNode = namedtuple('PruneNode', ['eff_alpha', 'tot_gain', 'Node'])

def create_prune_tree(node):
    tot_gain = node.gain
    denom = node.leaves() - 1
    for child in [left, right]:
        if isinstance(child, Node):

            tot_gain += create_prun_tree(child).tot_gain

    return PruneNode(tot_gain / denom, tot_gain, node)



# write tot_gains
# give tree eff_alphas
# find node with lowest
# trim
