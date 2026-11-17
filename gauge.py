import numpy as np

from gauge_group import GaugeGroup

class GaugeLattice():
    def __init__(self, grp: GaugeGroup, M, beta, start = 'cold', dim = 4):
        self.grp = grp
        self.M = M
        self.dim = dim
        self.beta = beta

        if start == 'cold': 
            self.L = self.grp.zero(size = self.dim * self.M ** self.dim)
        elif start == 'hot': 
            self.L = self.grp.random(size = self.dim * self.M ** self.dim)
        else:
            raise ValueError("Start must be 'cold' or 'hot'.")
        
        self.build_nbr()

    def build_nbr(self):
        self.nbr = [[0] * self.dim for _ in range(self.M ** self.dim)]
        self.inbr = [[0] * self.dim for _ in range(self.M ** self.dim)]

        for idx in range(self.M ** self.dim):
            pos = self.idx_to_pos(idx)
            for mu in range(self.dim):
                pos_mu = pos.copy()
                pos_mu[mu] = (pos_mu[mu] + 1) % self.M
                self.nbr[idx][mu] = self.pos_to_idx(pos_mu)

                pos_mui = pos.copy()
                pos_mui[mu] = (pos_mui[mu] - 1) % self.M
                self.inbr[idx][mu] = self.pos_to_idx(pos_mui)

    def pos_to_idx(self, pos):
        if len(pos) != self.dim: raise ValueError("Position must have the same dimension as the gauge.")
        return sum([pos[i] * self.M ** i for i in range(self.dim)])

    def idx_to_pos(self, idx):
        if idx < 0 or idx >= self.M ** self.dim: raise ValueError("Index out of bounds.")   # line 34

        pos = [0] * self.dim
        for i in range(self.dim):
            pos[i] = idx % self.M
            idx = idx // self.M
        return pos

    def link(self, site, mu):
        return self.L[self.dim * site + mu]

    def plaquette(self, idx, mu, nu):
        if mu == nu: return self.grp.zero()
        if idx < 0 or idx >= self.M ** self.dim: raise ValueError("Index out of bounds.")

        return self.grp.plaquette(self.link(idx, mu), 
                                  self.link(self.nbr[idx][mu], nu), 
                                  self.link(self.nbr[idx][nu], mu), 
                                  self.link(idx, nu))

    def plaquette_new(self, idx, mu, nu, new_value):
        if mu == nu: return self.grp.zero()
        if idx < 0 or idx >= self.M ** self.dim: raise ValueError("Index out of bounds.")

        return self.grp.plaquette(new_value, 
                                  self.link(self.nbr[idx][mu], nu), 
                                  self.link(self.nbr[idx][nu], mu), 
                                  self.link(idx, nu))

    def plaquette_new_inv(self, idx, mu, nu, new_value):
        if mu == nu: return self.grp.zero()
        if idx < 0 or idx >= self.M ** self.dim: raise ValueError("Index out of bounds.")

        return self.grp.plaquette(self.link(idx, mu), 
                                  self.link(self.nbr[idx][mu], nu), 
                                  new_value, 
                                  self.link(idx, nu))

    def delta_action(self, idx, new_value, mu):
        delta_S = 0
        old_value = self.link(idx, mu)
        for nu in range(self.dim):
            if nu == mu: continue
            # P \mu \nu (x)
            old_plaquette_1 = self.plaquette(idx, mu, nu)
            delta_S += self.grp.re_tr(self.plaquette_new(idx, mu, nu, new_value)) - \
                       self.grp.re_tr(old_plaquette_1)

            # P \mu \nu (x - \nu)
            idx_minu = self.inbr[idx][nu]
            old_plaquette_2 = self.plaquette(idx_minu, mu, nu)
            delta_S += self.grp.re_tr(self.plaquette_new_inv(idx_minu, mu, nu, new_value)) - \
                       self.grp.re_tr(old_plaquette_2)

        return -self.beta * delta_S

    def re_tr_pl(self):
        S = 0
        for idx in range(self.M ** self.dim):
            for mu in range(self.dim):
                for nu in range(mu + 1, self.dim):
                    S += self.grp.re_tr(self.plaquette(idx, mu, nu))
        return S

    def action(self):
        return -self.beta * self.re_tr_pl()

    def avg_plaquette(self):
        return 1 - self.re_tr_pl() / (self.M ** self.dim * self.dim * (self.dim - 1) / 2)

    def sim(self, iters, record = 0):
        history = []
        for i in range(iters):
            idx = np.random.randint(0, self.M ** self.dim)
            mu = np.random.randint(0, self.dim)
            new_value = self.grp.random()

            delta_S = self.delta_action(idx, new_value, mu)
            if delta_S < 0 or np.random.rand() < np.exp(-delta_S):
                self.L[self.dim * idx + mu] = self.grp.unitarize(new_value)

            if record > 0 and i % record == 0:
                history.append(self.avg_plaquette())
        return history

if __name__ == "__main__":
    from gauge_group import ZN
    g = GaugeLattice(ZN(4), 3, beta=1.0, dim=2)
    print(g.re_tr_pl())
    print(g.pos_to_idx([1, 2]))
    print(g.idx_to_pos(7))