import torch

from .utils import *

class DSBM(torch.nn.Module):
    def __init__(self, k, data, likelihood="normal", assignment="soft", device="cpu", biaa_model=None):
        super(DSBM, self).__init__()
        self.X = data.to(device)
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

        if biaa_model:
            # check assignment is the same
            if assignment != biaa_model._assignment:
                raise ValueError(f"Assignment {assignment} is not the same as the biaa_model assignment {biaa_model._assignment}.")

            if assignment == "soft":
                self._A = torch.nn.Parameter(biaa_model._A.detach().to(device), requires_grad=True)
                self._D = torch.nn.Parameter(biaa_model._D.detach().to(device), requires_grad=True)
            else:
                self._A = biaa_model._A.detach().to(device)
                self._D = biaa_model._D.detach().to(device)

            self._a = torch.nn.Parameter(biaa_model._a.detach().to(device), requires_grad=True)
            self._d = torch.nn.Parameter(biaa_model._d.detach().to(device), requires_grad=True)
            self._Z = torch.nn.Parameter(torch.logit(biaa_model.Z.detach()).to(device), requires_grad=True)
        else:
            self._a = torch.nn.Parameter(torch.ones(self.m, device=device), requires_grad=True)
            self._d = torch.nn.Parameter(torch.ones(self.n, device=device), requires_grad=True)

            if self._assignment == "hard":
                self._A = hardmax(torch.rand(self.m, self.k[0], device=device), dim=1)
                self._D = hardmax(torch.rand(self.k[1], self.n, device=device), dim=0)
            else:
                self._A = torch.nn.Parameter(torch.randn(self.m, self.k[0], device=device), requires_grad=True)
                self._D = torch.nn.Parameter(torch.randn(self.k[1], self.n, device=device), requires_grad=True)

            self._Z = torch.nn.Parameter(torch.randn(self.k[0], self.k[1], device=device), requires_grad=True)

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

    def fit(self, iterations, learning_rate=0.01, threshold=1e-5, print_loss=False):
        if self._assignment == "soft":
            optimizer = torch.optim.Adam(params=[self._A, self._Z, self._D, self._a, self._d], lr=learning_rate)
        else:
            optimizer = torch.optim.Adam(params=[self._Z, self._a, self._d], lr=learning_rate)

        for _ in range(iterations):
            loss = self._likelihood_f()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self._assignment != "soft":
                self._A = update_clusters_A(self.X / self.a[:, None],
                                            self.A,
                                            self.d[None, :] * (self.Z @ self.D),
                                            self._likelihood)

                self._D = update_clusters_D(self.X / self.d[None, :],
                                            self.D,
                                            self.a[:, None] * (self.A @ self.Z),
                                            self._likelihood)

            self.losses.append(loss.item())
            if print_loss and _ % 500 == 0:
                print(f'Loss at the {_} iteration: {loss.item()}')

            if _ > 1 and abs(self.losses[-1] - self.losses[-2]) < threshold:
                break

    @property
    def a(self):
        return torch.sigmoid(self._a)

    @property
    def d(self):
        return torch.sigmoid(self._d)


    @property
    def Z(self):
        if self._likelihood == "bernoulli":
            return torch.sigmoid(self._Z)
        elif self._likelihood == "poisson":
            return torch.relu(self._Z)
        else:
            return self._Z

    @property
    def A(self):
        if self._assignment != "soft":
            return self._A
        else:
            return torch.softmax(self._A, dim=1)

    @property
    def D(self):
        if self._assignment != "soft":
            return self._D
        else:
            return torch.softmax(self._D, dim=0)
