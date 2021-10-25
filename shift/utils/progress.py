import sys


def progress_bar(index, length, explanation=None, indexing=False, num_refresh=50,
                 marker_done="#", marker_undone="_", bar_edges="|",
                 bar_edge_left=None, bar_edge_right=None, bar_length=50):
    """Progress bar for a for-loop.

    Parameter
    ---------
    index : int
        Current index of for-loop.
    length : int
        Total length of the for-loop.
    explanation : str/optional
        The message output.
    num_refresh : int
        The number of refreshes for entire progress bar, i.e. rate of progress bar printing.
        Default means 50 refreshes per 100%, i.e. refreshed every 2%.
    marker_done : str
        The marker used for the progress bar.
    marker_undone : str
        The marker used for the progress bar which has not yet been completed.
    bar_edges : str
        The edges used for the progress bar.
    bar_edge_left : str
        The left bar edge.
    bar_edge_right : str
        The right bar edge
    """
    _percentage = (float(index + 1) / float(length)) * 100.
    _bar_counts = int((_percentage / 100.) * float(bar_length) )
    if bar_edge_left is None:
        bar_edge_left = bar_edges
    if bar_edge_right is None:
        bar_edge_right = bar_edges
    _progress_bar = bar_edge_left
    if _bar_counts != 0 and _bar_counts != bar_length:
        _progress_bar = _progress_bar + marker_done * _bar_counts + marker_undone * (bar_length - _bar_counts)
    elif _bar_counts == 0:
        _progress_bar = _progress_bar + marker_undone * bar_length
    else:
        _progress_bar = _progress_bar + marker_done * bar_length
    _progress_bar = _progress_bar + bar_edge_right
    if indexing is True:
        _index_string = "[%d/%d]"%(index+1,length)
    else:
        _index_string = ""
    if num_refresh >= length:
        num_refresh = length
    _refresh = float(length) / float(num_refresh)
    if _percentage < 10:
        _percentage_string = "   %d%% " % _percentage
    elif _percentage != 100:
        _percentage_string = "  %d%% " % _percentage
    else:
        _percentage_string = " %d%% " % _percentage
    if (index + 1) % int(_refresh) == 0 or index + 1 == length:
        if explanation is not None:
            sys.stdout.write("\r" + explanation + " : " + _progress_bar + _percentage_string + _index_string)
        else:
            sys.stdout.write("\r" + _progress_bar + _percentage_string + _index_string)
        sys.stdout.flush()
        if index + 1 == length:
            if explanation is not None:
                sys.stdout.write("\r" + explanation + " : " + _progress_bar + _percentage_string + _index_string + '\n')
            else:
                sys.stdout.write("\r" + _progress_bar + _percentage_string + _index_string + '\n')
            sys.stdout.flush()
    else:
        pass
