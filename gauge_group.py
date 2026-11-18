import numpy as np

class GaugeGroup:
    def __init__(self, seed = 42):
        self.rng = np.random.default_rng(seed)
        return

    def random(self, size = None):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def random_id(self, x, eps = 0.1):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def zero(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def plaquette(self, u1, u2, u3, u4):
        m1 = self.mult(u1, u2)
        m2 = self.imult(m1, u3)
        m3 = self.imult(m2, u4)
        return self.unitarize(m3)

    def tr(self, u):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def re_tr(self, u):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def unitarize(self, u):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def mult(self, u1, u2):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def imult(self, u1, u2):
        return self.mult(u1, self.inverse(u2))

    def inverse(self, u):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def staple(self, u1, u2, u3):
        m1 = self.imult(u1, u2)
        m2 = self.imult(m1, u3)
        return self.unitarize(m2)

    def istaple(self, u1, u2, u3):
        m1 = self.mult(self.inverse(u3), u1)
        m2 = self.mult(m1, u2)
        return self.unitarize(m2)

class ZN(GaugeGroup):
    def __init__(self, N):
        super().__init__()
        self.N = N
        self.cos_tb = np.cos(2 * np.pi * np.arange(self.N) / self.N)

    def random(self, size = None):
        return self.rng.integers(0, self.N, size=size, dtype=int)

    def random_id(self, x, eps = 0.1):
        return self.random(x.shape)

    def zero(self, size = None):
        return np.zeros(size, dtype=int)

    def tr(self, u):
        return np.exp(2j * np.pi * u / self.N)

    def re_tr(self, u):
        return self.cos_tb[u % self.N]

    def unitarize(self, u):
        return u % self.N

    def mult(self, u1, u2):
        return self.unitarize(u1 + u2)

    def inverse(self, u):
        return self.unitarize(-u)

class U1(GaugeGroup):
    def __init__(self):
        super().__init__()

    def random(self, size = None):
        return self.rng.uniform(0, 2 * np.pi, size=size)

    def random_id(self, x, eps = 0.1):
        return self.rng.uniform(-eps, eps, size=x.shape) + x

    def zero(self, size = None):
        return np.zeros(size)

    def tr(self, u):
        return np.exp(1j * u)
    
    def re_tr(self, u):
        return np.cos(u)

    def unitarize(self, u):
        return u % (2 * np.pi)

    def mult(self, u1, u2):
        return self.unitarize(u1 + u2)

    def inverse(self, u):
        return self.unitarize(-u)

class SU2(GaugeGroup):
    def __init__(self):
        super().__init__()

    def random(self, size = None):
        shape = (2,) if size is None else (size, 2)
        return self.unitarize(self.rng.normal(size=shape)
                    + 1j * self.rng.normal(size=shape))

    def random_id_X(self, shape, eps = 0.1):
        r = np.stack([self.rng.uniform(-0.5, 0.5, size=shape) for _ in range(4)], axis=-1)
        # print(f"r shape: {r.shape}, inp shape: {inp.shape}")  # Debugging line
        r_norm = np.linalg.norm(r[..., 1:], axis=-1)
        # print(f"r_norm shape: {r_norm.shape}, r shape: {r.shape}, inp shape: {inp.shape}")  # Debugging line
        r_norm = np.maximum(r_norm, 1e-12)  # avoid division by zero
        # print(f"{r[..., 0].shape}, {(r[..., 1] / r_norm).shape}, r shape: {(np.sign(r[..., 0])*np.sqrt(1-eps**2)).shape}, inp shape: {inp.shape}")  # Debugging line
        
        x = np.stack([  np.ones_like(r[..., 0]) * np.sqrt(1-eps**2), # np.sign(r[..., 0])*np.sqrt(1-eps**2)
                        eps * r[..., 1] / r_norm, 
                        eps * r[..., 2] / r_norm, 
                        eps * r[..., 3] / r_norm], axis=-1)

        # pauli matrices representation, X is distributed symmetrically around the identity
        # so that X and X-1 are equally likely, which is important for the Metropolis algorithm
        # and detailled balance
        X = np.stack([x[..., 0] + 1j * x[..., 3], x[..., 1] + 1j * x[..., 2]], axis=-1)

        return X

    def random_id(self, inp, eps = 0.1):
        return self.mult(self.random_id_X(inp.shape[0], eps), inp)

    def zero(self, size = None):
        if size is None:
            return np.array([1, 0], dtype=complex)
        else:
            a = np.ones((size,), dtype=complex)
            b = np.zeros((size,), dtype=complex)
            return np.concatenate((a[:, None], b[:, None]), axis=1)

    def unitarize(self, u):
        norm = np.linalg.norm(u, axis=-1, keepdims=True)
        return u / norm

    def tr(self, u):
        return np.real(u[..., 0])

    def re_tr(self, u):
        return np.real(u[..., 0])

    def mult(self, u1, u2):
        o = np.stack([u1[..., 0] * u2[..., 0] - np.conj(u2[..., 1]) * u1[..., 1],
                         u1[..., 0] * u2[..., 1] + np.conj(u2[..., 0]) * u1[..., 1]], axis=-1)
        return self.unitarize(o)
    
    def inverse(self, u):
        o = np.stack([np.conj(u[..., 0]), -u[..., 1]], axis=-1)
        return self.unitarize(o)

class SU3(GaugeGroup):
    def __init__(self):
        super().__init__()
        self.su2_random = SU2().random_id_X

    def beta_mod(self):
        return 1.0 / 3.0

    def random(self, size = None):
        shape = (2, 3) if size is None else (size, 2, 3)
        z = (self.rng.normal(size=shape) + 1j * self.rng.normal(size=shape))  # scale is irrelevant
        u = z[..., 0, :]
        v = z[..., 1, :]
        return self.unitarize(np.stack([u, v], axis=-1))

    def random_id_X(self, shape, eps = 0.1):
        R_2, S_2, T_2 = self.su2_random(shape, eps), self.su2_random(shape, eps), self.su2_random(shape, eps)

        R1 = np.stack([R_2[..., 0], R_2[..., 1], np.zeros_like(R_2[..., 0])], axis=-1)
        R2 = np.stack([-np.conj(R_2[..., 1]), np.conj(R_2[..., 0]), np.zeros_like(R_2[..., 0])], axis=-1)
       
        S1 = np.stack([S_2[..., 0], np.zeros_like(S_2[..., 0]), S_2[..., 1]], axis=-1)
        S2 = np.stack([np.zeros_like(S_2[..., 0]), np.ones_like(S_2[..., 0]), np.zeros_like(S_2[..., 0])], axis=-1)

        T1 = np.stack([np.ones_like(T_2[..., 0]), np.zeros_like(T_2[..., 0]), np.zeros_like(T_2[..., 0])], axis=-1)
        T2 = np.stack([np.zeros_like(T_2[..., 0]), T_2[..., 0], T_2[..., 1]], axis=-1)

        R = np.stack([R1, R2], axis=-1)
        S = np.stack([S1, S2], axis=-1)
        T = np.stack([T1, T2], axis=-1)

        X = self.mult(self.mult(R, S), T)

        flip = self.rng.random(X.shape[0]) < 0.5
        X[flip] = self.inverse(X[flip])

        return X

    def random_id(self, inp, eps = 0.1):
        return self.mult(self.random_id_X(inp.shape[0], eps), inp)

    def zero(self, size = None):
        if size is None:
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=complex).T
        else:
            u, v = np.zeros((size, 3), dtype=complex), np.zeros((size, 3), dtype=complex)
            u[:, 0], v[:, 1] = 1, 1
            return np.stack([u, v], axis=-1)

    def unitarize(self, x):
        u, v = x[..., 0], x[..., 1]
        u_new = u / np.linalg.norm(u, axis=-1, keepdims=True)
        v_p = v - u_new * np.sum(np.conj(u_new) * v, axis=-1, keepdims=True)
        v_new = v_p / np.linalg.norm(v_p, axis=-1, keepdims=True)
        return np.stack([u_new, v_new], axis=-1)

    def tr(self, x):
        u, v = x[..., 0], x[..., 1]
        w3 = np.conj(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0])
        return (u[..., 0] + v[..., 1] + w3) / 3

    def re_tr(self, x):
        u, v = x[..., 0], x[..., 1]
        w3 = np.conj(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0])
        return np.real(u[..., 0] + v[..., 1] + w3) / 3

    def mult(self, x1, x2):
        u1, v1 = x1[..., 0], x1[..., 1]
        u2, v2 = x2[..., 0], x2[..., 1]
        # w1 = np.conj(np.cross(u1, v1, axis=-1))
        w2 = np.conj(np.cross(u2, v2, axis=-1))

        o1 = np.stack([u1[..., 0] * u2[..., 0] + u1[..., 1] * v2[..., 0] + u1[..., 2] * w2[..., 0],
                       u1[..., 0] * u2[..., 1] + u1[..., 1] * v2[..., 1] + u1[..., 2] * w2[..., 1],
                       u1[..., 0] * u2[..., 2] + u1[..., 1] * v2[..., 2] + u1[..., 2] * w2[..., 2]], axis=-1)
        o2 = np.stack([v1[..., 0] * u2[..., 0] + v1[..., 1] * v2[..., 0] + v1[..., 2] * w2[..., 0],
                       v1[..., 0] * u2[..., 1] + v1[..., 1] * v2[..., 1] + v1[..., 2] * w2[..., 1],
                       v1[..., 0] * u2[..., 2] + v1[..., 1] * v2[..., 2] + v1[..., 2] * w2[..., 2]], axis=-1)

        return self.unitarize(np.stack([o1, o2], axis=-1))

    def inverse(self, x):
        u, v = x[..., 0], x[..., 1]
        w1 = u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1]
        w2 = u[..., 2] * v[..., 0] - u[..., 0] * v[..., 2]
        o1 = np.stack([np.conj(u[..., 0]), np.conj(v[..., 0]), w1], axis=-1)
        o2 = np.stack([np.conj(u[..., 1]), np.conj(v[..., 1]), w2], axis=-1)

        return self.unitarize(np.stack([o1, o2], axis=-1))

if __name__ == "__main__":
    su2 = SU2()
    random_elements = su2.random(size=5)
    print(random_elements)

    print(su2.random_id(random_elements, eps=0.1))
    print(su2.inverse(random_elements))
