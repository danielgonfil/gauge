import numpy as np

from gauge_group import GaugeGroup

class GaugeLattice():
    def __init__(self, grp: GaugeGroup, M, beta, start = 'cold', dim = 4, seed = 42, eps = 0.1):
        self.grp = grp
        self.M = M if M % 2 == 0 else M + 1 # ensure even lattice size for checkerboard update
        self.dim = dim
        self.n_sites = self.M ** self.dim
        self.n_links = self.dim * self.n_sites
        self.beta = beta
        self.eps = eps

        if start == 'cold': 
            self.L = self.grp.zero(size = self.n_links)
        elif start == 'hot': 
            self.L = self.grp.random(size = self.n_links)
        else:
            raise ValueError("Start must be 'cold' or 'hot'.")
        
        self._build_nbr()
        self._build_A()
        self._build_colors()

        self.rng = np.random.default_rng(seed) # use new numpy random generator for better performance and reproducibility

    def _build_colors(self):
        self.colors = []
        for mu in range(self.dim):
            parity = np.array([sum(self.idx_to_pos(s)[i] for i in range(self.dim) if i != mu) % 2
                            for s in range(self.n_sites)])
            self.colors.append([np.where(parity == c)[0] for c in (0, 1)])

        self.colors = np.array(self.colors, dtype=int)  # shape (dim, 2, n_sites/2)

    def _build_A(self):
        self.A = [[] for _ in range(self.n_links)]
        self.inv = []
        
        for link_idx in range(self.n_links):
            idx = link_idx // self.dim
            mu = link_idx % self.dim
            for nu in range(self.dim):
                if nu == mu: continue
                self.A[link_idx].append([self.link_idx(self.nbr[idx, mu], nu), 
                                        self.link_idx(self.nbr[idx, nu], mu), 
                                        self.link_idx(idx, nu)])

                self.inv.append([False, True, True])  # first link is forward, second and third are backward

                idx_minu = self.inbr[idx][nu]
                self.A[link_idx].append([self.link_idx(self.nbr[idx_minu, mu], nu),
                                        self.link_idx(idx_minu, mu),
                                        self.link_idx(idx_minu, nu)])                               

                self.inv.append([True, True, False])  # first link is forward, second and third are backward

        self.A = np.array(self.A, dtype=int)
        self.inv = np.array(self.inv, dtype=bool)

    def _build_nbr(self):
        self.nbr = [[0] * self.dim for _ in range(self.n_sites)]
        self.inbr = [[0] * self.dim for _ in range(self.n_sites)]

        for idx in range(self.n_sites):
            pos = self.idx_to_pos(idx)
            for mu in range(self.dim):
                pos_mu = pos.copy()
                pos_mu[mu] = (pos_mu[mu] + 1) % self.M
                self.nbr[idx][mu] = self.pos_to_idx(pos_mu)

                pos_mui = pos.copy()
                pos_mui[mu] = (pos_mui[mu] - 1) % self.M
                self.inbr[idx][mu] = self.pos_to_idx(pos_mui)

        self.nbr, self.inbr = np.array(self.nbr, dtype=int), np.array(self.inbr, dtype=int)

    def pos_to_idx(self, pos):
        if len(pos) != self.dim: raise ValueError("Position must have the same dimension as the gauge.")
        return sum([pos[i] * self.M ** i for i in range(self.dim)])

    def idx_to_pos(self, idx):
        if idx < 0 or idx >= self.n_sites: raise ValueError("Index out of bounds.")   # line 34

        pos = [0] * self.dim
        for i in range(self.dim):
            pos[i] = idx % self.M
            idx = idx // self.M
        return pos

    def link_idx(self, site, mu):
        return self.dim * site + mu
    
    def link(self, site, mu):
        return self.L[self.link_idx(site, mu)]

    def plaquette(self, idx, mu, nu):
        if mu == nu: return self.grp.zero()
        if idx < 0 or idx >= self.n_sites: raise ValueError("Index out of bounds.")

        return self.grp.plaquette(self.link(idx, mu), 
                                  self.link(self.nbr[idx, mu], nu), 
                                  self.link(self.nbr[idx, nu], mu), 
                                  self.link(idx, nu))

    def delta_retr(self, link_idxs, new_values):
        g = self.grp
        old_values = self.L[link_idxs]
        
        # Initialize result
        delta_retr = np.zeros(len(link_idxs), dtype=float)
        
        # Loop over all 2(dim-1) staples
        for m in range(2 * (self.dim - 1)):
            # Get the three link indices for this staple
            idx0 = self.A[link_idxs, m, 0]
            idx1 = self.A[link_idxs, m, 1]
            idx2 = self.A[link_idxs, m, 2]

            # Get the three links
            x = [self.L[idx0], self.L[idx1], self.L[idx2]]

            # Apply inversion flags
            x = [g.inverse(v) if self.inv[m, j] else v for j, v in enumerate(x)]
            
            # Build staple: S = x[0] · x[1] · x[2]
            S = g.mult(g.mult(x[0], x[1]), x[2])
            
            # Add contribution: re_tr(new·S) - re_tr(old·S)
            delta_retr += g.re_tr(g.mult(new_values, S)) - g.re_tr(g.mult(old_values, S))
        
        return delta_retr

    def re_tr_pl(self):
        S = 0
        idxs = np.arange(self.n_sites, dtype=int)
        for mu in range(self.dim):
            for nu in range(mu + 1, self.dim):
                P = self.grp.plaquette(self.L[self.dim * idxs + mu],
                                       self.L[self.dim * self.nbr[idxs, mu] + nu],
                                       self.L[self.dim * self.nbr[idxs, nu] + mu],
                                       self.L[self.dim * idxs + nu])
                S += self.grp.re_tr(P).sum()
        
        return S

    def action(self):
        return -self.beta * self.re_tr_pl()

    def avg_plaquette(self):
        return 1 - self.re_tr_pl() / (self.n_sites * self.dim * (self.dim - 1) / 2)

    def sim(self, n_sweeps = 1, local = True, fhistory = 1):
        history = {"avg_plaquette": [], "action": [], "polyakov_corr": [], "polyakov_mean": [], "acc_ratio": 0}
        acc_ratio = 0
        
        for _ in range(n_sweeps):
            for mu in range(self.dim):
                for color in [0, 1]:
                    lk = self.dim * self.colors[mu][color] + mu  # batch of ~128 link indices
                    # local: U -> X(eps) U with X near the identity; else: uniform over the group
                    new = (self.grp.random_id(self.L[lk], eps=self.eps) if local
                           else self.grp.random(self.L[lk].shape[0]))  # one numpy call generates all
                    
                    dS = -self.beta * self.delta_retr(lk, new)  # vectorised staple sum
                    a = (dS <= 0) | (self.rng.random(len(lk)) < np.exp(-dS))  # vectorised accept/reject
                    acc_ratio += np.sum(a)
                    self.L[lk[a]] = new[a]  # batch update

            if fhistory > 0 and n_sweeps % fhistory == 0:
                history["avg_plaquette"].append(self.avg_plaquette())
                history["action"].append(self.action()) 
                corr = self.polyakov_corr()
                history["polyakov_corr"].append(corr[0])
                history["polyakov_mean"].append(corr[1])

        if fhistory > 0:
            history["acc_ratio"] = acc_ratio / (n_sweeps * self.n_sites * self.dim)
        
        return history

    def tune_eps(self, target=0.5, n_sweeps=1, n_iter=20, lo=1e-3, hi=1.0, tol=5e-3):
        lo, hi = np.log(lo), np.log(hi)
        for i in range(n_iter):
            mid = 0.5 * (lo + hi)
            self.eps = float(np.exp(mid))
            hist = self.sim(n_sweeps=n_sweeps, local=True)
            acc = hist["acc_ratio"]
            if abs(acc - target) < tol: break
            if acc > target: lo = mid   # too timid -> grow eps
            else:            hi = mid   # too bold  -> shrink eps
        
        return self.eps, acc

    def polyakov_loops(self, mu=3):
        """Normalized complex Polyakov loop starting at every site."""
        cur = np.arange(self.n_sites)
        P = self.grp.zero(size=self.n_sites)          # group identity, not 1.0
        
        for _ in range(self.M):
            P = self.grp.mult(P, self.link(cur, mu))   # mult(x1, x2) = x1·x2, ordered
            cur = self.nbr[cur, mu]

        return self.grp.tr(P)

    def polyakov_corr(self, mu=3):
        """Returns C[d] = (1/V_s) Σ_m P(m+d) P(m)*  over the spatial sublattice, plus ⟨P⟩.
        V_s = M^(dim-1) is the spatial volume, and d is a spatial displacement vector."""
        P = self.polyakov_loops(mu)
        sl = P[np.arange(self.n_sites) % self.M == 0]      # t = 0 slice, mu = 0
        Ps = sl.reshape((self.M,) * (self.dim - 1))
        Pk = np.fft.fftn(Ps)
        C = np.fft.ifftn(np.abs(Pk) ** 2).real / Ps.size
        print(C)
        return C, Ps.mean()

if __name__ == "__main__":
    from gauge_group import ZN
    g = GaugeLattice(ZN(4), 3, beta=1.0, dim=2)
    print(g.re_tr_pl())
    print(g.pos_to_idx([1, 2]))
    print(g.idx_to_pos(7))