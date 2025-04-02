import numpy as np

from typing import Optional


def diagonal_average(
    matrix: np.ndarray,
    reverse: Optional[bool] = True,
    samesize: Optional[bool] = False,
    averaging: Optional[bool] = True,
) -> np.ndarray:
    """
    Perform Hankel averaging (or diagonal averaging) on the input matrix

    The main function of this function is to average or sum the diagonals of the input matrix.
    According to the parameter settings, you can choose to extract the diagonals in the forward or reverse direction,
    and you can choose whether to calculate the average value of the diagonal elements.
    The function returns a one-dimensional array representing the processed result.

    :param 2D_ndarray matrix: The input matrix to be averaged.
    :param bool reverse: If True, diagonals are taken in reverse order (from the bottom-right to the top-left).
    :param bool samesize: If True, only diagonals from the main to the leftmost are taken.
    :param bool averaging: If True, the mean value of each diagonal is taken; otherwise, the sum is taken.
    :return vector: 1D ndarray,
        The resulting vector after diagonal averaging.

    :Note:

    * If samesize = False:
        - If reverse = False, diagonals from the bottom-left to the top-right are taken.
        - If reverse = True, diagonals from the bottom-right to the top-left are taken.
    * If samesize = True:
        - If reverse = False, diagonals from the bottom-left to the main diagonal are taken.
        - If reverse = True, diagonals from the bottom-right to the main diagonal are taken.
    """

    # Get the number of rows and columns in the matrix
    (rows, columns) = matrix.shape

    # Calculate the total number of diagonals in the matrix
    n_diags = rows + columns - 1

    # Adjust the number of diagonals if samesize is True
    if samesize:
        n_diags = n_diags // 2 + 1

    # Initialize the output vector with zeros
    out = np.zeros(n_diags, dtype=matrix.dtype)

    # Loop through each diagonal
    for idx_from_bottom in np.arange(n_diags):
        # Calculate the actual diagonal index
        idx = idx_from_bottom - rows + 1

        # Extract the diagonal elements
        diag = get_diagonal(matrix, idx, reverse=reverse)

        # Perform averaging or summing based on the averaging parameter
        if not reverse:
            if averaging:
                out[idx_from_bottom] = np.mean(
                    diag
                )  # Take the mean of the diagonal elements
            else:
                out[idx_from_bottom] = np.sum(
                    diag
                )  # Take the sum of the diagonal elements
        else:
            if averaging:
                out[n_diags - idx_from_bottom - 1] = np.mean(
                    diag
                )  # Take the mean of the diagonal elements
            else:
                out[idx_from_bottom] = np.sum(
                    diag
                )  # Take the sum of the diagonal elements

    # Return the resulting vector
    return out


def get_diagonal(
    matrix: np.ndarray, idx: int, reverse: Optional[bool] = False
) -> np.ndarray:
    """
    Extract the specified diagonal from a matrix

    The main function of this function is to extract the diagonal elements of the specified index from the input matrix.
    The index of the diagonal is calculated from the main diagonal (index is 0),
    the positive index represents the diagonal to the right of the main diagonal,
    and the negative index represents the diagonal to the left of the main diagonal.
    The function also supports reverse extraction of diagonals,
    that is, starting from the lower right corner of the matrix.

    :param 2D_ndarray matrix: The input matrix from which the diagonal is extracted.
    :param int idx: The index of the diagonal relative to the main diagonal (zero diagonal).
                    Positive indices are to the right of the main diagonal, and negative indices are to the left.
    :param bool reverse: If True, extract the diagonal in reverse order (from the bottom-right to the top-left).
    :return diag: 1D ndarray,
        The extracted diagonal elements.

    :Notes:

    * If reverse = False:
        - idx = 0: main diagonal
        - idx > 0: diagonals to the left of the main diagonal
        - idx < 0: diagonals to the right of the main diagonal
    * If reverse = True:
        - idx = 0: main backward diagonal
        - idx > 0: diagonals to the right of the main backward diagonal
        - idx < 0: diagonals to the left of the main backward diagonal

    :Example:

    >>> a = [1, 2, 3, 4, 5]
    >>> b = signals.matrix.toeplitz(a)[:3, :]
    >>> print(b)
    >>> print(get_diagonal(b,0))  # zero diagonal
    >>> print(get_diagonal(b,-2)) # 2 diagonals to the left
    >>> print(get_diagonal(b,3))  # 3 diagonals to the right
    >>> print(get_diagonal(b,0,reverse=True))  # zero backward diagonal
    >>> print(get_diagonal(b,-1,reverse=True)) # 1 right backward diagonal
    >>> print(get_diagonal(b,1,reverse=True))  # 1 left backward diagonal
    """
    # Get the number of rows and columns in the matrix
    (rows, columns) = matrix.shape

    # Calculate the total number of diagonals in the matrix
    n_diags = rows + columns - 1

    # Calculate the index of the diagonal from the bottom
    idx_from_bottom = idx + rows - 1

    # Check if the specified index is within the valid range of diagonals
    if idx_from_bottom >= n_diags or idx_from_bottom < 0:
        raise ValueError("idx value out of matrix shape ")

    # Calculate the length of the specified diagonal
    len_of_diag = _length_of_diag_(matrix, idx_from_bottom)

    # Initialize the output array to store the diagonal elements
    out = np.zeros(len_of_diag, dtype=matrix.dtype)

    # Extract the diagonal elements based on the reverse parameter
    if not reverse:
        # Extract the diagonal in the forward direction
        if idx >= 0:
            # Extract the diagonal to the right of the main diagonal
            for i in np.arange(len_of_diag):
                out[i] = matrix[i, i + idx]
        if idx < 0:
            # Extract the diagonal to the left of the main diagonal
            idx = np.abs(idx)
            for i in np.arange(len_of_diag):
                out[i] = matrix[i + idx, i]
    else:
        # Extract the diagonal in the reverse direction
        if idx >= 0:
            # Extract the diagonal to the right of the main backward diagonal
            for i in np.arange(len_of_diag):
                indexes = columns - 1 - i - idx
                out[i] = matrix[i, indexes]
        if idx < 0:
            # Extract the diagonal to the left of the main backward diagonal
            idx = np.abs(idx)
            for i in np.arange(len_of_diag):
                indexes = columns - 1 - i
                out[i] = matrix[i + idx, indexes]

    # Return the extracted diagonal elements
    return out


# --------------------------------------------------
def _length_of_diag_(matrix: np.ndarray, idx):
    """
    Get the length of the specified diagonal in a matrix

    The main function of this function is to calculate the length of the specified diagonal in the matrix.
    The index of the diagonal starts from the main diagonal (index is 0),
    the positive index represents the diagonal to the right of the main diagonal,
    and the negative index represents the diagonal to the left of the main diagonal.
    The function returns the length of the specified diagonal.

    :param matrix: 2D ndarray, The input matrix.
    :param idx: int, The index of the diagonal relative to the main diagonal (zero diagonal).
    :return len: int, The length of the specified diagonal.

    Notes:
    * The index is calculated from the element (0,0).
      For instance, the main diagonal (0) has length 1, the next one has length 2, etc.
    * The function handles both positive and negative indices.
    """
    # Convert the input matrix to a NumPy array
    matrix = np.asarray(matrix)

    # Get the number of rows and columns in the matrix
    (rows, columns) = matrix.shape

    # Calculate the total number of diagonals in the matrix
    n_diags = rows + columns - 1

    # Check if the specified index is within the valid range of diagonals
    if idx >= n_diags or idx < -n_diags:
        raise ValueError(
            "Index is out of diagonal number range. Total diagonals: ", n_diags
        )

    # Initialize the length of the diagonal
    length = 0

    # Calculate the rank of the matrix (minimum of rows and columns)
    rank = min(rows, columns)

    # Determine the length of the diagonal based on its index
    if idx < n_diags // 2:
        # For diagonals before the middle diagonal
        length = min(idx + 1, rank)
    else:
        # For diagonals after the middle diagonal
        length = min(n_diags - idx, rank)

    # Ensure the length is non-negative
    return max(length, 0)
