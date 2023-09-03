import numpy as np

from minjax import lax
from minjax.interpreters import ad


def main():
    x, y = np.random.normal(size=(2,))
    primal, tangent = ad.jvp_v1(lambda x, y: x < y, (x, y), (1.0, 1.0))
    print(x, y)
    print(primal)
    print(tangent)


if __name__ == "__main__":
    main()
