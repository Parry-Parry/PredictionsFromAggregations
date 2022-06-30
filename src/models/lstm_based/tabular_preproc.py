import pandas as pd

def tabular_aggregate(dataset : pd.DataFrame, aggr_col : list, buckets=None):
    """
    Perform an aggregation over a tabular dataset

    ::param dataset: tabular dataset as pandas Frame for simplicity
    ::param list aggr_col: Set of columns over which to aggregate
    ::param int buckets: optional parameter to group numerical data from aggregation
    """
    pass