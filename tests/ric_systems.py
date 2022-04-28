import numpy as np

from Constrained.ConstraintRiccati import ConstraintRiccatiSystem
from OrthogonalDecomposition import matrix_rank, subspaces_from_svd

# SOME INTERESTING SYSTEMS GENERATED FROM RANDOM SAMPLE

ric_constraint_system7 = ConstraintRiccatiSystem(
    A=np.array([
        [3.5175e+00, 3.7784e+00, 3.9815e+00, 3.0900e-02, 6.0360e-01, 9.9000e-02, 3.3397e+00, 2.1230e-01],
        [3.3083e+00, 1.4009e+00, 7.3860e-01, 3.7526e+00, 3.3430e-01, 3.5030e-01, 3.2635e+00, 2.5944e+00],
        [3.0273e+00, 1.2420e-01, 1.7706e+00, 3.8579e+00, 3.7401e+00, 3.4919e+00, 1.2784e+00, 1.6159e+00],
        [1.2380e-01, 1.0662e+00, 8.5670e-01, 1.1100e+00, 2.5140e+00, 3.2959e+00, 8.9710e-01, 1.6441e+00],
        [1.1872e+00, 4.4590e-01, 3.1112e+00, 2.4438e+00, 3.6598e+00, 2.3000e-03, 1.5928e+00, 2.6180e+00],
        [1.2948e+00, 2.2468e+00, 7.3800e-01, 1.1927e+00, 3.4111e+00, 2.3376e+00, 2.9133e+00, 2.5197e+00],
        [1.3872e+00, 2.2388e+00, 3.8173e+00, 3.9225e+00, 1.4574e+00, 1.8966e+00, 3.3304e+00, 5.2190e-01],
        [3.7770e+00, 3.6191e+00, 3.9289e+00, 2.3191e+00, 1.7955e+00, 8.0030e-01, 1.2420e+00, 2.1280e+00]
    ]),
    B=np.array([
        [1.8523, 2.0467, 1.9634, 1.1682, 3.0047],
        [2.0785, 1.4625, 1.4116, 1.2709, 2.826],
        [0.3739, 1.6422, 3.57, 2.8438, 2.5507],
        [0.3736, 2.2157, 3.8489, 1.6366, 0.9476],
        [3.8931, 2.6746, 2.4799, 2.8149, 1.355],
        [4.0609, 0.5715, 0.9327, 1.3313, 0.3812],
        [2.9899, 1.4384, 1.2483, 4.8903, 2.8246],
        [2.9585, 0.1546, 2.4605, 0.9792, 1.6699]
    ]),
    G=np.array([
        [4.558, 7.323, 1.159, 6.005, 7.222, 4.059, 3.622, 4.248],
        [8.166, 5.353, 5.505, 9.113, 2.712, 1.896, 7.023, 4.117],
        [1.145, 8.234, 9.281, 6.723, 5.439, 4.355, 6.027, 6.774],
        [96.306, 110.405, 91.305, 132.929, 75.052, 50.72, 101.803, 81.141],
    ]),
    F=np.array([
        [3.1029, 1.6583, 3.3316, 3.7932, 0.4456, 3.2001],
        [0.1223, 0.0811, 3.0397, 0.8824, 3.1312, 3.9104],
        [3.0384, 0.7866, 1.3156, 2.5453, 3.0569, 2.6369],
        [3.3162, 3.0024, 3.8197, 1.4177, 0.1939, 0.245],
        [3.3852, 1.7952, 1.5984, 0.4032, 3.707, 3.8668],
        [2.3943, 1.9457, 0.144, 0.8227, 0.6771, 1.4288],
        [0.8957, 0.8843, 0.6604, 2.3004, 2.8711, 0.6526],
        [1.2993, 3.2436, 2.2567, 3.3612, 3.9969, 2.4852]
    ]),
    init_state=np.array([4.3093, 2.0962, 2.8854, 2.2051, 4.1785, 4.9742, 3.3541, 0.6962])
)

ric_constraint_system6 = ConstraintRiccatiSystem(
    A=np.array([
        [1.3062, 3.5495, 3.4974, 1.277, 1.3043, 6.8957, 4.2683],
        [4.1653, 3.3022, 3.7618, 7.695, 7.699, 1.6591, 2.3765],
        [4.0102, 7.312, 2.3005, 5.8425, 0.6391, 6.686, 1.2101],
        [2.6521, 4.5329, 2.6833, 1.4578, 0.6255, 0.5169, 6.3852],
        [7.6552, 0.1763, 8.2501, 0.2699, 0.3245, 3.6695, 4.3044],
        [5.1276, 6.7918, 3.9073, 8.7601, 3.0387, 4.6854, 4.425],
        [4.6032, 5.5747, 1.9444, 2.3762, 6.3462, 3.4319, 6.1985]
    ]),
    B=np.array([
        [0.2989, 0.373, 0.022, 0.4413],
        [0.9781, 0.1205, 0.7935, 0.2742],
        [0.134, 0.2106, 0.5448, 0.8398],
        [0.2379, 0.2601, 0.863, 0.8344],
        [0.1246, 0.6611, 0.163, 0.5208],
        [0.9529, 0.2726, 0.8249, 0.9981],
        [0.6672, 0.7439, 0.594, 0.4416]
    ]),
    G=np.array([
        [3.627, 6.868, 8.678, 4.913, 6.627, 8.365, 6.523],
        [0.987, 3.33, 2.65, 4.234, 7.262, 2.215, 8.091],
        [5.759, 0.81, 1.132, 4.435, 8.065, 5.591, 0.963],
        [66.344, 66.65, 72.514, 89.482, 147.686, 98.682, 104.084]
    ]),
    F=np.array([
        [1.2436, 3.0835, 1.7777, 5.6664, 3.1442, 3.044],
        [3.8605, 3.5469, 2.742, 5.9887, 4.2676, 5.3153],
        [3.4615, 4.8691, 4.9149, 4.249, 0.9487, 4.4745],
        [2.7704, 0.2275, 3.837, 4.8859, 4.7911, 4.5483],
        [1.7921, 4.4615, 2.5691, 1.5664, 2.1011, 2.2057],
        [3.0171, 4.7265, 3.7799, 4.4727, 4.716, 3.9503],
        [1.7583, 5.3771, 4.0654, 3.6596, 4.4334, 0.5287]
    ]),
    init_state=np.array([2.66094476, 4.05521636, 3.33001269, 2.76215208, 2.69931146, 3.6949612, 2.45076177])
)

ric_constraint_system5 = ConstraintRiccatiSystem(
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
ric_constraint_system4 = ConstraintRiccatiSystem(
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

ric_constraint_system3 = ConstraintRiccatiSystem(
    A=np.array([
        [1.44, 1.9865, 1.6216, 0.339, 1.2941, 1.187],
        [1.3345, 1.1552, 1.5322, 1.9704, 1.965, 1.4749],
        [0.4209, 0.2875, 1.063, 1.424, 0.2876, 0.8386],
        [0.1133, 1.5101, 0.2184, 1.3016, 1.2341, 1.1086],
        [0.1726, 1.11, 1.7345, 1.3155, 1.6133, 1.6221],
        [1.4266, 1.5264, 1.1581, 1.2312, 0.5986, 1.7198]
    ]),
    B=np.array([
        [0.538, 0.8329, 0.7768, 0.4362, 0.5359],
        [0.5954, 0.0986, 0.5126, 0.359, 0.9718],
        [0.3817, 0.0728, 0.0158, 0.3586, 0.9953],
        [0.6043, 0.6163, 0.9614, 0.2555, 0.5027],
        [0.9913, 0.894, 0.9436, 0.1045, 0.5303],
        [0.2437, 0.5011, 0.165, 0.0693, 0.4701]
    ]),
    G=np.array([
        [1.236, 2.941, 0.545, 3.714, 2.315, 2.066],
        [6.18, 14.705, 2.725, 18.57, 11.575, 10.33],
        [4.944, 11.764, 2.18, 14.856, 9.26, 8.264],
        [2.472, 5.882, 1.09, 7.428, 4.63, 4.132]
    ]),
    F=np.array([
        [1.5562, 1.7207, 1.261, 1.9366, 0.6044, 0.8595],
        [1.2586, 1.3344, 0.5662, 1.2553, 0.772, 1.1163],
        [1.6968, 0.446, 0.6225, 0.9418, 1.0405, 1.0066],
        [0.2889, 0.4398, 1.4273, 1.4194, 1.9851, 0.8691],
        [0.401, 0.4281, 0.3188, 1.0976, 0.6569, 0.1071],
        [0.5386, 1.0654, 1.0566, 0.5824, 1.6186, 1.8364]
    ]),
    init_state=np.array([0.086, 0.9024, 0.2862, 0.0788, 0.7835, 0.0407])
)

ric_constraint_system2 = ConstraintRiccatiSystem(
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

ric_constraint_system1 = ConstraintRiccatiSystem(
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


def ric_random_system(n_size=4, u_size=3, k_size=4):
    n = n_size  # size of state vector
    u = u_size  # size of controller vector
    k = k_size  # number of constraints
    A = np.round(np.random.randint(1, 10) * np.random.rand(n, n), 4)
    B = np.round(np.random.randint(1, 10) * np.random.rand(n, u), 4)
    F = np.round(np.random.randint(1, 10) * np.random.rand(n, 6), 4)
    state = np.round(np.random.randint(1, 10) * np.random.rand(n), 4)

    # TODO(Assert matrix A and B is controllable) Equation 10
    # TODO(Assert matrix A and C is observable)

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

    return ConstraintRiccatiSystem(A=A, B=B, G=G, F=F, init_state=state)
