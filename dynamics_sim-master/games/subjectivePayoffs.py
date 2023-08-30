import math
from games.game import SymmetricNPlayerGame
import numpy as np
from games.ddmGame import thresh_count as tc
from games.ddmGame import def_params
from copy import copy

thresh_count = tc
total_thresh_count = thresh_count + 1
WOL_thresholds = [round(tau / (2 * thresh_count), 2) for tau in range(total_thresh_count)]
thresholds = [round((tau + 1) / (2 * thresh_count), 2) for tau in range(total_thresh_count - 1)]
rr = np.arange(-10, 10, 2)
integral_boxes = 100
updated_def_params = copy(def_params)
updated_def_params['reward_range'] = rr

# TODO Add subjective utility extension through redefining the strategy space as the values of subjective utilities then taking the optimal action given the SU


class subjectiveDDM(SymmetricNPlayerGame):
    """ A class used to represent the CWOL game with continuous information.
    Evolutionary simulations are run for a single agent type who optimizes over subjective value of defection
    """
    DEFAULT_PARAMS = updated_def_params
    PLAYER_LABELS = ['Agent']
    STRATEGY_LABELS = []
    for reward in rr:
        STRATEGY_LABELS.append('Added value of ' + str(round(reward, 2)) + ' to defect')
    EQUILIBRIA_LABELS = range(len(rr))

    def __init__(self, d_i, punish_defect, c_min_max, var, search_cost, no_look_reward, reward_range, equilibrium_tolerance=0.2):
        self.defect_util = d_i - punish_defect

        self.c_min_max = c_min_max
        self.expected_c = (self.c_min_max[0] + self.c_min_max[1]) / 2
        self.c_dist = [count * (self.c_min_max[1] - self.c_min_max[0]) / (integral_boxes - 1) + self.c_min_max[0] for count in
                       range(integral_boxes)]
        self.prior = 1 - (self.c_min_max[1] - self.defect_util) / (self.c_min_max[1] - self.c_min_max[0])

        self.no_look_reward = no_look_reward
        self.var = var
        self.search_cost = search_cost
        self.reward_range = rr

        self.defect_probs = []
        self.time_spent = []
        self.coop_probs = []

        self.payoff_list = []

        # for reward in self.reward_range:  # Finds payoffs for fixed thresholds
        #     threshold = 1/4
        #     _, payoff, defect_prob = self.adjusted_threshold_outcomes(-threshold, threshold, reward)
        #     self.payoff_list.append(payoff)
        #     self.defect_probs.append(defect_prob)

        for reward in self.reward_range:  # Finds optimal thresholds for each reward
            fake_tau_list = []
            real_tau_list = []
            for tau_lower in thresholds:
                tau_lower = -tau_lower
                for tau_upper in thresholds:
                    observed_payoff, payoff, defect_prob = self.adjusted_threshold_outcomes(tau_lower, tau_upper, reward)
                    fake_tau_list.append(observed_payoff)
                    real_tau_list.append(payoff)
            chosen_threshold_index = fake_tau_list.index(max(fake_tau_list))
            self.payoff_list.append(real_tau_list[chosen_threshold_index])

        payoff_matrix_1 = self.payoff_list
        payoff_matrix = [[payoff for _ in reward_range] for payoff in payoff_matrix_1]

        super(subjectiveDDM, self).__init__(payoff_matrix=payoff_matrix, n=1, equilibrium_tolerance=equilibrium_tolerance)

    def adjusted_threshold_outcomes(self, tau_lower, tau_upper, reward):
        adj_payoff, payoff = self.u_thresh(tau_lower, tau_upper, reward)
        self.coop_probs.append([])

        if tau_lower >= 0:
            defect_prob = 0
            
            self.coop_probs[-1] = [1 for _ in self.c_dist]
        elif tau_upper <= 0:
            defect_prob = 1
            self.coop_probs[-1] = [0 for _ in self.c_dist]
        else:
            defect_prob = 0
            for c in self.c_dist:
                alpha = 2 * (self.defect_util - (c - reward)) / (self.var ** 2)
                defect_prob += self.compute_defect_prob(alpha, tau_lower, tau_upper) / len(self.c_dist)
                self.coop_probs[-1].append(1 - self.compute_defect_prob(alpha, tau_lower, tau_upper))

        return adj_payoff, payoff, defect_prob

    def u_thresh(self, tau_lower, tau_upper, reward=0):
        if tau_lower == 0 and tau_upper == 0:  # We require either CWOL or DWOL
            return -100, -100
        elif tau_upper <= 0:  # DWOL
            return self.defect_util + reward, self.defect_util

        total_utility = 0
        adj_utility = 0
        for c in self.c_dist:
            if tau_lower >= 0:  # CWOL
                total_utility += (c + self.no_look_reward)
                adj_utility += (c + self.no_look_reward)
            else:
                adj_u, t_u = self.u_fixed_state(tau_lower, tau_upper, c, reward)
                adj_utility += adj_u
                total_utility += t_u

        return adj_utility / len(self.c_dist), total_utility / len(self.c_dist)

    def u_forced_wait(self, tau_lower, tau_upper, length):
        pass  # TODO Set up

    @staticmethod
    def compute_defect_prob(alpha, tau_lower, tau_upper):
        if alpha == 0:
            return - tau_lower / (-tau_lower + tau_upper)
        else:
            numerator = 1 - math.e ** (alpha * - tau_lower)
            denominator = math.e ** (-alpha * tau_upper) - math.e ** (alpha * - tau_lower)
            return numerator / denominator

    def compute_exp_stopping(self, alpha, tau_lower, tau_upper):
        if alpha == 0:
            return tau_lower * tau_upper
        else:
            stopping_numerator = tau_upper * (1 - math.e ** (alpha * -tau_lower)) - tau_lower * (1 - math.e ** (- alpha * tau_upper))
            stopping_denominator = (alpha * (self.var ** 2) / 2) * (math.e ** (- alpha * tau_upper) - math.e ** (alpha * -tau_lower))
            return stopping_numerator / stopping_denominator

    def u_fixed_state(self, tau_lower, tau_upper, c, reward=0):  # TODO Figure out how to fix prior
        alpha = 2 * (self.defect_util + reward - c) / (self.var ** 2)

        p_defect = self.compute_defect_prob(alpha, tau_lower, tau_upper)
        exp_stopping = self.compute_exp_stopping(alpha, tau_lower, tau_upper)
        utility = (1 - p_defect) * c + p_defect * self.defect_util - exp_stopping * self.search_cost
        adj_utility = (1 - p_defect) * c + p_defect * (self.defect_util + reward) - exp_stopping * self.search_cost

        return adj_utility, utility

    @classmethod
    def classify(cls, params, state, tolerance):
        outcome = state[0]
        np_state = np.array(outcome)
        modal_strat = np_state.argmax()
        return modal_strat.flatten()

