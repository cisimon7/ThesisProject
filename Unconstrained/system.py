import numpy as np
from Unconstrained.LTIWithConstantTerm import LTIWithConstantTerm


def random_lti_system_with_const_term(n_size=4, u_size=3):
    n = n_size  # size of state vector
    u = u_size  # size of controller vector
    A = np.round(np.random.randint(1, 10) * np.random.rand(n, n), 4)
    B = np.round(np.random.randint(1, 10) * np.random.rand(n, u), 4)
    c = np.round(np.random.randint(1, 10) * np.random.rand(n), 4)
    init = np.round(np.random.randint(1, 10) * np.random.rand(n), 4)

    print(f"A:\n{A}\n")
    print(f"B:\n{B}\n")
    print(f"c:\n{c}\n")
    print(f"initial state:\n{init}\n")

    return LTIWithConstantTerm(A=A, B=B, c=c)
