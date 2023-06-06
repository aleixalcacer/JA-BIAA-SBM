from .utils import *


class DBiAA(torch.nn.Module):
    def __init__(self, k, data, likelihood="normal", assignment="soft", device="cpu"):
        super(DBiAA, self).__init__()
        self.X = data.to(device)
        self.m, self.n = self.X.shape
        self.k = k

        self._a = torch.nn.Parameter(torch.ones(self.m, device=device), requires_grad=True)
        self._d = torch.nn.Parameter(torch.ones(self.n, device=device), requires_grad=True)

        if assignment == "soft":
            self._A = torch.nn.Parameter(torch.randn(self.m, self.k[0], device=device), requires_grad=True)
            self._D = torch.nn.Parameter(torch.randn(self.k[1], self.n, device=device), requires_grad=True)
        else:
            self._A = hardmax(torch.rand(self.m, self.k[0], device=device), dim=1)
            self._D = hardmax(torch.rand(self.k[1], self.n, device=device), dim=0)

        self._B = torch.nn.Parameter(torch.randn(self.k[0], self.m, device=device), requires_grad=True)
        self._C = torch.nn.Parameter(torch.randn(self.n, self.k[1], device=device), requires_grad=True)

        self.losses = []
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


    def normal_loss(self):
        """
        The negative log-likelihood of a normal distribution
        """
        BXC = self.B @ self.X @ self.C
        AtXDt = self.A.T @ self.X @ self.D.T
        DDt = self.D @ self.D.T
        AtA = self.A.T @ self.A

        loss = - 2 * (BXC * AtXDt).sum() + ((BXC @ DDt) * (AtA @ BXC)).sum()  # Optimized
        return loss

    def bernoulli_loss(self):
        """
        The negative log-likelihood of a Bernoulli distribution
        """
        P = self.a[:, None] * self.A @ self.B @ self.X @ self.C @ self.D * self.d[None, :]

        e = 1e-8
        P[P == 0] = e
        P[P == 1] = 1 - e
        loss = - (self.X * torch.log(P) + (1 - self.X) * torch.log(1 - P)).sum()  # Not optimized
        return loss

    def poisson_loss(self):
        """
        The negative log-likelihood of a Poisson distribution
        """
        P = self.a[:, None] * self.A @ self.B @ self.X @ self.C @ self.D * self.d[None, :]
        loss = - (self.X * torch.log(P) - P).sum()

        return loss


    def fit(self, iterations, learning_rate=0.01, threshold=1e-5, print_loss=False):
        if self._assignment == "soft":
            optimizer = torch.optim.Adam(params=[self._a, self._A, self._D, self._d, self._B, self._C],
                                         lr=learning_rate)
        else:
            optimizer = torch.optim.Adam(params=[self._a, self._B, self._C, self._d], lr=learning_rate)

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
            if print_loss and _ % 10 == 0:
                print(f'Loss at the {_} iteration: {loss.item()}')

            if _ > 1 and abs(self.losses[-1] - self.losses[-2]) < threshold:
                break

    @property
    def a(self):
        if self._likelihood == "bernoulli":
            return torch.sigmoid(self._a)
        elif self._likelihood == "poisson":
            return torch.nn.functional.softplus(self._a)
        else:
            return self._a

    @property
    def d(self):
        if self._likelihood == "bernoulli":
            return torch.sigmoid(self._d)
        elif self._likelihood == "poisson":
            return torch.nn.functional.softplus(self._d)
        else:
            return self._d

    @property
    def A(self):
        if self._assignment != "soft":
            return self._A
        else:
            return torch.softmax(self._A, dim=1)

    @property
    def B(self):
        return torch.softmax(self._B, dim=1)

    @property
    def C(self):
        return torch.softmax(self._C, dim=0)

    @property
    def D(self):
        if self._assignment != "soft":
            return self._D
        else:
            return torch.softmax(self._D, dim=0)

    @property
    def Z(self):
        return self.B @ self.X @ self.C
