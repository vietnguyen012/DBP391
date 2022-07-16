from bisect import bisect

def count_inversions(a):
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):
        j = bisect(sorted_so_far, u)
        inversions += i - j
        sorted_so_far.insert(j, u)
    return inversions


def kendall_tau(ground_truth, predictions,return_low_example=False):
    total_inversions = 0
    total_2max = 0
    low_score = 1000
    low_score_item_dict = {}
    # twice the maximum possible inversions across all instances
    for i,(gt, pred) in enumerate(zip(ground_truth, predictions)):
        ranks = [gt.index(x) for x in pred]  # rank predicted order in terms of ground truth
        total_inversions_item = count_inversions(ranks)
        total_inversions += total_inversions_item
        n = len(gt)
        total_2max_item = n * (n - 1)
        total_2max += total_2max_item
        score_item = 1 - 4 * total_inversions_item / total_2max_item
        if score_item < low_score:
            low_score = score_item
            low_score_item_dict["gt"] = gt
            low_score_item_dict["pred"] = pred
            low_score_item_dict["id"] = ground_truth.axes[0][i]
    return (1 - 4 * total_inversions / total_2max),low_score_item_dict