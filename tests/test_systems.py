import numpy as np
from LinearConstraintStateSpaceModel import LinearConstraintStateSpaceModel
from OrthogonalDecomposition import matrix_rank, subspaces_from_svd

constraint_system5 = LinearConstraintStateSpaceModel(
    A=np.array([
        [0.29388389, 0.60619791, 0.61711047, 0.98134008, 0.8411186],
        [0.17693043, 0.03707734, 0.4704481, 0.37946392, 0.26073155],
        [0.17234994, 0.00099042, 0.09406578, 0.13462686, 0.59497191],
        [0.02387212, 0.71286746, 0.70349125, 0.5670503, 0.37909315],
        [0.49230875, 0.14454622, 0.78801647, 0.03618107, 0.91140465]
    ]),
    B=np.array([
        [0.57687472, 0.98192325, 0.49108052],
        [0.89164231, 0.82203857, 0.62980234],
        [0.29258367, 0.86144339, 0.61083259],
        [0.26803695, 0.28221136, 0.21249502],
        [0.82645489, 0.0847191, 0.27868093]
    ]),
    G=np.array([
        [5.33, 2.692, 4.241, 4.564, 1.835],
        [2.619, 2.781, 4.491, 6.801, 3.085],
        [31.98, 16.152, 25.446, 27.384, 11.01],
        [37.034, 27.454, 43.91, 59.062, 25.85]
    ]),
    F=np.array([
        [0.90561087, 0.46770737, 0.49420815, 0.1559242, 0.09016879, 0.58504117],
        [0.75716534, 0.873799, 0.11249054, 0.46628263, 0.13524094, 0.89432745],
        [0.18143612, 0.16911702, 0.88149943, 0.42200387, 0.88007125, 0.24401606],
        [0.88693047, 0.01659639, 0.46908306, 0.19753509, 0.02025264, 0.07489282],
        [0.88566959, 0.07450498, 0.81403114, 0.34240506, 0.70330564, 0.05149097]
    ]),
    init_state=np.array([0.5473899, 0.94628836, 0.45888643, 0.86570615, 0.27199946])
)
constraint_system4 = LinearConstraintStateSpaceModel(
    A=np.array([
        [0.79898356, 0.93832862, 0.30943309, 0.9329607, 0.52567468],
        [0.43480883, 0.77582657, 0.71322907, 0.56725567, 0.02921253],
        [0.07447839, 0.74912897, 0.12599992, 0.27272414, 0.33534004],
        [0.30690427, 0.0917924, 0.29935492, 0.93765648, 0.56331568],
        [0.74560995, 0.43528731, 0.32654366, 0.00334108, 0.83872725]
    ]),
    B=np.array([
        [0.53678534, 0.93797791, 0.37248788],
        [0.50745662, 0.23616261, 0.58877284],
        [0.97512747, 0.32282102, 0.7786647],
        [0.54521594, 0.63116388, 0.27805539],
        [0.99799447, 0.72076249, 0.06001138]
    ]),
    G=np.array([
        [0.472, 0.19, 0.293, 0.351, 0.396],
        [0.627, 0.586, 0.134, 0.558, 0.203],
        [0.298, 0.282, 0.66, 0.604, 0.067],
        [10.878, 8.182, 8.036, 11.5, 5.261],
        [4.601, 3.32, 5.926, 6.19, 2.256]
    ]),
    F=np.array([
        [0.42111223, 0.08285465, 0.31751467, 0.50741614, 0.78448954, 0.39717398],
        [0.1612954, 0.89717344, 0.7428686, 0.14443119, 0.6704288, 0.03088425],
        [0.03849139, 0.73873068, 0.79369482, 0.13195193, 0.15142974, 0.61221381],
        [0.07526989, 0.12390503, 0.62527005, 0.36274025, 0.99327459, 0.25669884],
        [0.82678452, 0.27598196, 0.43322599, 0.59732486, 0.80116294, 0.01101946]
    ]),
    init_state=np.array([0.56375954, 0.18230589, 0.46795963, 0.79780289, 0.0667068])
)

constraint_system2 = LinearConstraintStateSpaceModel(
    A=np.array([
        [-1.01416031, -0.85202817, -0.23718763],
        [-0.24703406, -1.20508211, 0.47151586],
        [0.37052781, 1.13780232, -0.05080365]
    ]),
    B=np.array([
        [1.62100904, 1.11382971, 1.68311231, 0.90885391],
        [-0.51764834, -1.15764292, -0.53954765, -0.13238155],
        [-0.60691388, -0.50319479, 0.79373825, 1.07906922]
    ]),
    G=np.array([
        [1, 2, 3],
        [1, 1, 2],
        [1, 2, 3]
    ]),
    F=np.eye(3, 6),
    init_state=np.array([3, 5, 10])
)

constraint_system1 = LinearConstraintStateSpaceModel(
    A=np.eye(4),
    B=np.array([
        [0, 1 / 67, 0, 1 / 67],
        [0, -(4 / 67), 0, -(4 / 67)],
        [0, 7 / 67, 0, 7 / 67],
        [0, 1 / 67, 0, 1 / 67]
    ]),
    G=np.array([[1, 4, 2, 1], [3, 3, 1, 2], [0, 1, 0, 4]]),
    F=np.eye(4, 6),
    init_state=10 * np.array([1, 1, 1, 1])
)


def random_system(n_size=4, u_size=3, k_size=4):
    n = n_size  # size of state vector
    u = u_size  # size of controller vector
    k = k_size  # number of constraints
    A = np.random.rand(n, n)
    B = np.random.rand(n, u)
    F = np.random.rand(n, 6)
    state = np.random.rand(n)

    # Generate random row space basis
    row_space = 10 * np.round(np.random.random_sample() * np.random.rand(np.random.choice(np.arange(1, k)), n), 4)

    # Make sure rows are independent
    assert matrix_rank(row_space) == len(row_space), f"row_space vector:\n{row_space}\nnot independent, try again"

    part = np.asarray([
        np.sum(np.asarray([np.random.randint(0, 10) * vector for vector in row_space]), axis=0)
        for _ in range(k - len(row_space))
    ])  # Generate random linear combination of row space of length n - len(row_space)

    G = np.block([[row_space], [part]])  # Form Constraint matrix from row space
    R, _, _, N = subspaces_from_svd(G)

    print(f"A:\n{A}\n")
    print(f"B:\n{B}\n")
    print(f"G:\n{G}\n")
    print(f"F:\n{F}\n")
    print(f"initial state:\n{state}\n")

    return LinearConstraintStateSpaceModel(A=A, B=B, G=G, F=F, init_state=state)
