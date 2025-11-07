def precision(tp, fp):
    """ Computes precision metric.
    
    Arguments:
    tp {int} -- True positives.
    fp {int} -- False positives.
    
    Returns:
    {float} -- Precision value.
    """
    if (tp + fp) == 0: return 0.0
    return tp / (tp + fp)

def recall(tp, fn):
    """ Computes recall metric.
    
    Arguments:
    tp {int} -- True positives.
    fn {int} -- False negatives.
    
    Returns:
    {float} -- Recall value.
    """
    if (tp + fn) == 0: return 0.0
    return tp / (tp + fn)

def f1_score(prec, rec):
    """ Computes F1 score.
    
    Arguments:
    prec {float} -- Precision value.
    rec {float} -- Recall value.
    
    Returns:
    {float} -- F1 score.
    """
    if (prec + rec) == 0: return 0.0
    return 2 * ((prec * rec) / (prec + rec))