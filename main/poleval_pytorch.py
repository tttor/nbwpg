import torch

def get_rpi_s(Rsa, PIsa):
    # Get the reward of state under a policy
    nS, _ = PIsa.shape
    rpi_s = []
    for s in range(nS):
        rpi_s.append(torch.dot(Rsa[s], PIsa[s]))
    rpi_s = torch.vstack(rpi_s).squeeze()
    return rpi_s

def get_ppisteady_s(Ppi_ss, PI_sa):
    nS, _ = PI_sa.shape
    A = torch.transpose(Ppi_ss, 0, 1) - torch.eye(nS); A[-1, :] = torch.ones(nS)
    b = torch.zeros(nS).double(); b[-1] = 1; B = b.unsqueeze(dim=1)
    ppi_star, _ = torch.solve(B, A)
    ppi_star = ppi_star.squeeze()
    return ppi_star

def get_Ppi_ss(Psas, PIsa):
    # Get the probability of next state given current state under a policy.
    # That is, p(a, s'|s) = p(s'|s, a) x p(a|s)
    nS, nA = PIsa.shape
    Psas = Psas.reshape(nS, -1)
    Ppi_ss = []
    for s in range(nS):
        pi_s = torch.vstack([torch.diag(PIsa[s, a].repeat(nS)) for a in range(nA)])
        Ppi_ss.append(torch.matmul(Psas[s, :], pi_s))
    Ppi_ss = torch.vstack(Ppi_ss)
    assert torch.allclose(Ppi_ss.sum(), torch.tensor(nS).double()), Ppi_ss.sum().item()
    return Ppi_ss

def get_Qsa(gain, bias, Psas, Rsa, nA_list):
    # Return state-action values of (n=0)-discount optimality, aka bias state-action values
    nS, nA = Rsa.shape
    Qsa = torch.zeros(nS, nA)
    for s in range(nS):
        for a in range(nA_list[s]):
            Qsa[s, a] = Rsa[s, a] - gain
            for snext in range(nS):
                Qsa[s, a] += Psas[s, a, snext]*bias[snext]
    return Qsa

def get_Qsa_1(bias, v1, Psas, nA_list):
    # Return state-action values of (n=1)-discount optimality
    nS = len(nA_list); nA = max(nA_list)
    Qsa = torch.zeros(nS, nA)
    for s in range(nS):
        for a in range(nA_list[s]):
            Qsa[s, a] = -bias[s]
            for snext in range(nS):
                Qsa[s, a] += Psas[s, a, snext]*v1[snext]
    return Qsa
