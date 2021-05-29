import os
import gym, yaml, numpy as np

class SymbolicRepresentation(gym.Env):
    def __init__(self, cfg_fname):
        checks = []; self.dpath = dpath = os.path.dirname(os.path.realpath(__file__))
        cfg_fpath = os.path.join(dpath, 'config', cfg_fname)
        with open(cfg_fpath, 'r') as f:
            self.cfg = cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.envid_short = cfg_fname.replace('.yaml', '') # without prefix `gym_symbol:`
        self.state_list = cfg['state_list']
        self.statename_list = [i['state_name'] for i in self.state_list]
        self.nS = len(self.state_list)
        self.nA_list = [len(i['action_list']) for i in self.state_list]
        self.nA = max(self.nA_list)

        self.isd = [i['init_prob'] for i in self.state_list] # isd: initial state distrib
        self.state = self.reset() # the current state index
        checks.append(sum(self.isd)==1)
        checks.append(self.isd.index(max(self.isd))==0) # for now, for simplicity on isd

        self.tmix_cfg = cfg['tmix']
        self.tmix_cfg['rtol'] = float(self.tmix_cfg['rtol'])
        self.tmix_cfg['atol'] = float(self.tmix_cfg['atol'])
        self.tmix_upperbound = self.tmix_cfg['upperbound'] # only an approximate
        self.tmax_xep = None if (self.tmix_upperbound is None) else self.tmix_upperbound
        self.tmax_xep += 5 # plus some TOL to compensate the approx of tmix_upperbound

        # self.transient_stateidx_list = \
        #     [sidx for sidx, state in enumerate(self.state_list) \
        #     if state['class_under_all_policies']=='transient']
        # self.recurrent_stateidx_list = \
        #     [sidx for sidx, state in enumerate(self.state_list) \
        #     if state['class_under_all_policies']=='recurrent']
        # n_tr, n_rc = len(self.transient_stateidx_list), len(self.recurrent_stateidx_list)
        # if (n_tr > 0) or (n_rc > 0):
        #     checks.append((n_tr + n_rc)==self.nS)

        if not(all(checks)):
            raise RuntimeError('not(all(checks))')

    def step(self, action):
        s = self.state # the current state index
        a = action # the current (intented) action index
        snext_probs, snext_names, rnext_vals = zip(*[(i['prob'], i['name'], i['reward'])
            for i in self.state_list[s]['action_list'][a]['nextstate_list']])
        next_idx = np.random.multinomial(1, snext_probs).tolist().index(1)
        snext = self.statename_list.index(snext_names[next_idx])
        rnext_val = rnext_vals[next_idx]
        self.state = snext
        return (snext, rnext_val)

    def reset(self):
        self.state = np.random.multinomial(1, self.isd).tolist().index(1)
        return self.state

    def get_Psas(self):
        # Psas: element (s, a, s') denotes the prob p(s'|s, a)
        Psas = np.zeros((self.nS, self.nA, self.nS))
        for s, state in enumerate(self.state_list):
            for a, action in enumerate(state['action_list']):
                for statenext in action['nextstate_list']:
                    snext = self.statename_list.index(statenext['name'])
                    Psas[s, a, snext] = statenext['prob']
        assert np.sum(Psas)==sum(self.nA_list)
        return Psas

    def get_Rsa(self):
        Rsa = np.zeros((self.nS, self.nA))
        for s, state in enumerate(self.state_list):
            for a, action in enumerate(state['action_list']):
                rew = 0 # accumulator for getting E[R|s, a, s']
                for statenext in action['nextstate_list']:
                    rew += statenext['prob']*statenext['reward']
                Rsa[s, a] = rew
        return Rsa

    def get_allstatefeature(self, feature_id):
        return self.get_statefeature(range(self.nS), feature_id)

    def get_statefeature(self, state_idx_list, feature_id):
        fea = []
        for i, state_idx in enumerate(state_idx_list):
            if feature_id=='one_dim_state_index':
                fea_i = np.array([state_idx]).astype(np.double)
            elif feature_id=='two_dim_xy_coord':
                if not('gridnav' in self.envid_short):
                    raise NotImplementedError
                envid_parts = self.envid_short.split('_'); assert len(envid_parts)==3
                n_grid = int(envid_parts[1])
                order = 'F' # column-major (Fortran-style) from top-left
                x, y = np.unravel_index([state_idx], (n_grid, n_grid), order=order)
                fea_i = np.hstack([x, y]).astype(np.double)
            elif feature_id=='two_dim_for_example_10_1_2_v1':
                if state_idx==0: # (where 2 _different_ actions must be chosen)
                    fea_i =  np.array([1, 1]).astype(np.double)
                elif state_idx==1:
                    fea_i =  np.array([0, 0]).astype(np.double)
                else:
                    raise NotImplementedError(state_idx)
            else:
                raise NotImplementedError(feature_id)
            fea.append(fea_i)
        fea = np.array(fea); assert fea.ndim==2, fea.ndim
        return fea

    # def get_Rsas(self):
    #     Rsas = np.zeros((self.nS, self.nA, self.nS))
    #     for s, state in enumerate(self.state_list):
    #         for a, action in enumerate(state['action_list']):
    #             for statenext in action['nextstate_list']:
    #                 snext = self.statename_list.index(statenext['name'])
    #                 Rsas[s, a, snext] = statenext['reward']
    #     return Rsas
