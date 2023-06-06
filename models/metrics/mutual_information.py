import torch


def mutual_info_score(Z1: torch.Tensor, Z2: torch.Tensor) -> float:
    """
   Compute the Mutual Information score between two assignment matrices.

   Parameters
   ----------
   Z1: torch.Tensor
       One assignment matrix.

   Z2: torch.Tensor
       Another assignment matrix.

   Returns
   -------
   score: float
       Normalized Mutual Information between Z1 and Z2.
   """
    P = Z1 @ Z2.T
    PXY = P / P.sum()
    PXPY = PXY.sum(dim=1).reshape(-1, 1) @ PXY.sum(dim=0).reshape(1, -1)
    ind = PXY > 0
    mi = (PXY[ind] * torch.log(PXY[ind] / PXPY[ind])).sum()
    return float(mi)


def normalized_mutual_info_score(Z1: torch.Tensor, Z2: torch.Tensor) -> float:
    """
    Compute the Normalized Mutual Information score between two assignment matrices.

    Parameters
    ----------
    Z1: torch.Tensor
        One assignment matrix.

    Z2: torch.Tensor
        Another assignment matrix.

    Returns
    -------
    score: float
        Normalized Mutual Information between Z1 and Z2.
    """
    nmi = 2 * mutual_info_score(Z1, Z2) / (mutual_info_score(Z1, Z1) + mutual_info_score(Z2, Z2))
    return nmi
