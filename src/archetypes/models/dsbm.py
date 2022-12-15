import torch
import torch.nn as nn
from .utils import *
from torch.nn.functional import softplus
from torch import sigmoid

class DSBM(nn.Module):
    def __init__(self, k, data, likelihood="normal", assignment="soft"):
        super(DSBM, self).__init__()
        self.X = data
        self.m, self.n = self.X.shape
        self.k = k

        implemented_likelihood = ["normal", "bernoulli", "poisson"]
        if likelihood not in implemented_likelihood:
            raise NotImplementedError(f"{likelihood} are not implemented. Select one of {implemented_likelihood}.")
        self._likelihood = likelihood
        self._likelihood_f = getattr(self, f"{likelihood}_loss")

        implemented_assignment = ["soft", "hard"]
        if assignment not in implemented_assignment:
            raise NotImplementedError(f"Assignment {assignment} are not implemented. Select one of"
                                      f" {implemented_assignment}.")
        self._assignment = assignment


        self._a = torch.nn.Parameter(torch.randn(self.m), requires_grad=True)
        self._d = torch.nn.Parameter(torch.randn(self.n), requires_grad=True)

        if self._assignment == "hard":
            self._A = hardmax(torch.randn(self.m, self.k[0]), dim=1)
            self._D = hardmax(torch.randn(self.k[1], self.n), dim=0)
        else:
            self._A = nn.Parameter(torch.randn(self.m, self.k[0]), requires_grad=True)
            self._D = nn.Parameter(torch.randn(self.k[1], self.n), requires_grad=True)

        self._Z = torch.nn.Parameter(torch.randn(self.k[0], self.k[1]), requires_grad=True)

        self.losses = []


    def normal_loss(self):
        """
        The negative log-likelihood of a normal distribution
        """
        AtXDt = self.A.T @ self.X @ self.D.T
        DDt = self.D @ self.D.T
        AtA = self.A.T @ self.A
        Z = self.Z

        loss = - 2 * (Z * AtXDt).sum() + ((Z @ DDt) * (AtA @ Z)).sum()  # Optimized
        return loss

    def bernoulli_loss(self):
        """
        The negative log-likelihood of a Bernoulli distribution
        """
        P = self.a[:, None] * self.A @ self.Z @ self.D * self.d[None, :]

        e = 1e-8
        P[P == 0] = e
        P[P == 1] = 1 - e

        loss = - (self.X * torch.log(P) + (1 - self.X) * torch.log(1 - P)).sum()  # Not optimized
        return loss

    def poisson_loss(self):
        """
        The negative log-likelihood of a Poisson distribution
        """
        P = self.a[:, None] * self.A @ self.Z @ self.D * self.d[None, :]

        loss = - (self.X * torch.log(P) - P).sum()  # Not optimized

        return loss

    def fit(self, iterations, learning_rate=0.01, print_loss=False):
        if self._assignment == "soft":
            optimizer_A = torch.optim.Adam(params=[self._a, self._A], lr=learning_rate)
            optimizer_D = torch.optim.Adam(params=[self._D, self._d], lr=learning_rate)
        else:
            optimizer_A = torch.optim.Adam(params=[self._a], lr=learning_rate)
            optimizer_D = torch.optim.Adam(params=[self._d], lr=learning_rate)


        optimizer_Z = torch.optim.Adam(params=[self._Z], lr=learning_rate)

        for _ in range(iterations):
            loss = self._likelihood_f()
            optimizer_A.zero_grad()
            loss.backward()
            optimizer_A.step()
            if self._assignment == "hard":
                self._A = update_clusters_A(self.X,
                                            self.A,
                                            self.Z @ self.D * self.d[None, :],
                                            self._likelihood)

            loss = self._likelihood_f()
            optimizer_Z.zero_grad()
            loss.backward()
            optimizer_Z.step()

            loss = self._likelihood_f()
            optimizer_D.zero_grad()
            loss.backward()
            optimizer_D.step()

            if self._assignment == "hard":
                self._D = update_clusters_D(self.X,
                                            self.D,
                                            self.a[:, None] * self.A @ self.Z,
                                            self._likelihood)


            loss = self._likelihood_f()
            optimizer_Z.zero_grad()
            loss.backward()
            optimizer_Z.step()

            loss = self._likelihood_f()
            self.losses.append(loss.item())
            if print_loss and _ % 500 == 0:
                print(f'Loss at the {_} iteration: {loss.item()}')

    @property
    def a(self):
        if self._likelihood == "bernoulli":
            return sigmoid(self._a)
        elif self._likelihood == "poisson":
            return softplus(self._a)
        else:
            return self._a

    @property
    def d(self):
        if self._likelihood == "bernoulli":
            return sigmoid(self._d)
        elif self._likelihood == "poisson":
            return softplus(self._d)
        else:
            return self._d

    @property
    def Z(self):
        if self._likelihood == "bernoulli":
            return sigmoid(self._Z)
        elif self._likelihood == "poisson":
            return softplus(self._Z)
        else:
            return self._Z

    @property
    def A(self):
        if self._assignment == "hard":
            return self._A
        else:
            return softmax(self._A, dim=1)

    @property
    def D(self):
        if self._assignment == "hard":
            return self._D
        else:
            return softmax(self._D, dim=0)
