import torch
from util_bwopt import feature_dimth, action_dimth, sample_dimth

### Policy network #############################################################
class OneDimensionStateAndTwoActionPolicyNetwork(torch.nn.Module):
    def __init__(self, nA_list):
        super(OneDimensionStateAndTwoActionPolicyNetwork, self).__init__()
        input_dim = output_dim = 1; check_allstate_have_same_nA(nA_list)
        self.linear_net = torch.nn.Linear(input_dim, output_dim, bias=True)
        self.distrib_net = TwoActionSigmoidCategoricalDistributionNetwork()
        self.state_feature_extractor_id = 'one_dim_state_index'
        self.weight_x_name = 'linear_net.weight'
        self.weight_y_name = 'linear_net.bias'

    def forward(self, observ):
        action_preference = self.linear_net(observ)
        return self.distrib_net(action_preference)

class TwoDimensionStateAndTwoActionPolicyNetwork(torch.nn.Module):
    def __init__(self, nA_list):
        super(TwoDimensionStateAndTwoActionPolicyNetwork, self).__init__()
        check_allstate_have_same_nA(nA_list)
        self.linear_net_1 = torch.nn.Linear(1, 1, bias=False)
        self.linear_net_2 = torch.nn.Linear(1, 1, bias=False)
        self.distrib_net = TwoActionSoftmaxCategoricalDistributionNetwork()
        self.state_feature_extractor_id = 'two_dim_xy_coord'
        self.weight_x_name = 'linear_net_1.weight'
        self.weight_y_name = 'linear_net_2.weight'

    def forward(self, observ):
        lin_1_out = self.linear_net_1(observ[:, 0].reshape(-1, 1))
        lin_2_out = self.linear_net_2(observ[:, 1].reshape(-1, 1))
        return self.distrib_net(torch.hstack([lin_1_out, lin_2_out]))

### Action distrib network #####################################################
class TwoActionSigmoidCategoricalDistributionNetwork():
    def __init__(self):
        self.sigmoid_fn = torch.nn.Sigmoid()

    def __call__(self, preference_forward):
        p_forward = self.sigmoid_fn(preference_forward)
        p_backward = 1.0 - p_forward
        p = torch.cat([p_forward, p_backward], dim=action_dimth)
        return torch.distributions.Categorical(probs=p)

class TwoActionSoftmaxCategoricalDistributionNetwork():
    def __init__(self):
        self.softmax_fn = torch.nn.Softmax(dim=action_dimth)

    def __call__(self, linear_output):
        probs = self.softmax_fn(linear_output)
        return torch.distributions.Categorical(probs=probs)

### util #######################################################################
policynetclass_dict = {}
policynetclass_dict['OneDimensionStateAndTwoActionPolicyNetwork'] = OneDimensionStateAndTwoActionPolicyNetwork
policynetclass_dict['TwoDimensionStateAndTwoActionPolicyNetwork'] = TwoDimensionStateAndTwoActionPolicyNetwork

def policy_net2tabular(allstatefeature, pi_net, requires_grad):
    if requires_grad:
        pi = pi_net(allstatefeature)
    else:
        with torch.no_grad():
            pi = pi_net(allstatefeature)
    return pi.probs

def check_allstate_have_same_nA(nA_list):
    if not(nA_list.count(nA_list[0])==len(nA_list)):
        # In policy approx setting (where the number of parameters is much less
        # than the number of state-action pairs, hence generalization),
        # having different numbers of actions at some states may slow down
        # the computation because we have to check nA individually per state.
        # In contrast, it is easy to modify the MDP from having one single action
        # to having multiple action with the same reward and next states.
        raise RuntimeError('Only handle the same nA for all states')
    return True
