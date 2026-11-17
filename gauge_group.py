import numpy as np

class GaugeGroup:
    def __init__(self):
        return

    def random(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def random_id(self, x, eps):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def zero(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def plaquette(self, u1, u2, u3, u4):
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


class ZN(GaugeGroup):
    def __init__(self, N):
        super().__init__()
        self.N = N
        self.cos_tb = [np.cos(2 * np.pi * k / self.N) for k in range(self.N)]

    def random(self, size = None):
        return np.random.randint(0, self.N, size=size, dtype=int)

    def random_id(self, x, eps = 0):
        return self.random()

    def zero(self, size = None):
        return np.zeros(size, dtype=int)

    def plaquette(self, u1, u2, u3, u4):
        return self.unitarize(u1 + u2 - u3 - u4)

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
        return np.random.uniform(0, 2 * np.pi, size=size)

    def random_id(self, x, eps):
        return np.random.uniform(-eps, eps) * x

    def zero(self, size = None):
        return np.zeros(size)

    def plaquette(self, u1, u2, u3, u4):
        return self.unitarize(u1 + u2 - u3 - u4)

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
        return self.unitarize(np.random.normal(size=shape)
                    + 1j * np.random.normal(size=shape))

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
    
    def random_id(self, x, eps):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def plaquette(self, u1, u2, u3, u4):
        m1 = self.mult(u1, u2)
        m2 = self.imult(m1, u3)
        m3 = self.imult(m2, u4)
        return self.unitarize(m3)

    def re_tr(self, u):
        return np.real(u[..., 0])

    def mult(self, u1, u2):
        return np.stack([u1[..., 0] * u2[..., 0] - np.conj(u2[..., 1]) * u1[..., 1],
                         u1[..., 0] * u2[..., 1] + np.conj(u2[..., 0]) * u1[..., 1]], axis=-1)

    def inverse(self, u):
        return np.stack([np.conj(u[..., 0]), -u[..., 1]], axis=-1)
    

if __name__ == "__main__":
    size = 10
    # print([(np.complex128(1), np.complex128(0))] * size)
    # print(np.zeros(None)
    size = None
    a = np.ones((size,), dtype=complex)
    b = np.zeros((size,), dtype=complex)
    su2_elements = np.concatenate((a[:, None], b[:, None]), axis=1)
    print(su2_elements)


    