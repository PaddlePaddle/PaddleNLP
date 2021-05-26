"""Contains various utility functions."""


def subsequence(first_sequence, second_sequence):
    """
    Returns whether the first sequence is a subsequence of the second sequence.

    Inputs:
        first_sequence (list): A sequence.
        second_sequence (list): Another sequence.

    Returns:
        Boolean indicating whether first_sequence is a subsequence of second_sequence.
    """
    for startidx in range(len(second_sequence) - len(first_sequence) + 1):
        if second_sequence[startidx:startidx + len(
                first_sequence)] == first_sequence:
            return True
    return False
