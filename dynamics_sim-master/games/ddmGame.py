import math
from games.game import SymmetricNPlayerGame
import numpy as np

thresh_count = 10
total_thresh_count = thresh_count + 1
thresholds = [round(tau / (2 * thresh_count), 2) for tau in range(total_thresh_count)]
integral_boxes = 100
def_params = dict(d_i=0, punish_defect=10, c_min_max=[-20, 0], no_look_reward=1, var=0.5, search_cost=1)


class DDM(SymmetricNPlayerGame):
    """ A class used to represent the CWOL game with continuous information.
    Evolutionary simulations are run for a single agent type
    """
    DEFAULT_PARAMS = def_params
    PLAYER_LABELS = ['Agent']
    STRATEGY_LABELS = []
    for tau_lower in thresholds:
        tau_lower = -tau_lower
        for tau_upper in thresholds:
            STRATEGY_LABELS.append('Thresholds: ' + str(round(tau_lower + 1/2, 2)) +
                                   ', ' + str(round(tau_upper + 1/2, 2)))
    EQUILIBRIA_LABELS = range(total_thresh_count ** 2)

    def __init__(self, d_i, punish_defect, c_min_max, var, search_cost, no_look_reward, equilibrium_tolerance=0.2):
        self.defect_util = d_i - punish_defect

        self.c_min_max = c_min_max
        self.expected_c = (self.c_min_max[0] + self.c_min_max[1]) / 2
        self.c_dist = [count * (self.c_min_max[1] - self.c_min_max[0]) / (integral_boxes - 1) + self.c_min_max[0] for count in
                       range(integral_boxes)]
        self.no_look_reward = no_look_reward
        self.prior = 1 - (self.c_min_max[1] - self.defect_util) / (self.c_min_max[1] - self.c_min_max[0])

        self.var = var
        self.search_cost = search_cost

        self.tau_list = []  # Thresholds are relative to 1/2, a threshold of -0.3 actually is a threshold of 0.2
        self.defect_probs = []
        self.time_spent = []
        self.coop_probs = []
        for tau_lower in thresholds:
            tau_lower = -tau_lower
            for tau_upper in thresholds:
                payoff, defect_prob = self.threshold_outcomes(tau_lower, tau_upper)
                self.tau_list.append(payoff)
                self.defect_probs.append(defect_prob)

        payoff_matrix_1 = self.tau_list
        payoff_matrix = [[payoff for _ in range(total_thresh_count ** 2)] for payoff in payoff_matrix_1]

        super(DDM, self).__init__(payoff_matrix=payoff_matrix, n=1, equilibrium_tolerance=equilibrium_tolerance)

    def threshold_outcomes(self, tau_lower, tau_upper):
        payoff = self.u_thresh(tau_lower, tau_upper)
        self.coop_probs.append([])

        if tau_lower >= 0:
            defect_prob = 0
            self.coop_probs[-1] = [1 for c in self.c_dist]
        elif tau_upper <= 0:
            defect_prob = 1
            self.coop_probs[-1] = [0 for c in self.c_dist]
        else:
            defect_prob = 0
            for c in self.c_dist:
                alpha = 2 * (self.defect_util - c) / (self.var ** 2)
                defect_prob += self.compute_defect_prob(alpha, tau_lower, tau_upper) / len(self.c_dist)
                self.coop_probs[-1].append(1 - self.compute_defect_prob(alpha, tau_lower, tau_upper))

        return payoff, defect_prob

    def u_thresh(self, tau_lower, tau_upper):
        if tau_lower == 0 and tau_upper == 0:  # Both thresholds at 0 is undefined
            return -100
        elif tau_upper <= 0:  # DWOL
            return self.defect_util

        total_utility = 0
        for c in self.c_dist:
            if tau_lower >= 0:  # CWOL
                total_utility += (c + self.no_look_reward)
            else:
                total_utility += self.u_fixed_state(tau_lower, tau_upper, c)

        return total_utility / len(self.c_dist)

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

    def u_fixed_state(self, tau_lower, tau_upper, c):
        alpha = 2 * (self.defect_util - c) / (self.var ** 2)

        p_defect = self.compute_defect_prob(alpha, tau_lower, tau_upper)
        exp_stopping = self.compute_exp_stopping(alpha, tau_lower, tau_upper)
        utility = (1 - p_defect) * c + p_defect * self.defect_util - exp_stopping * self.search_cost

        return utility

    @classmethod
    def classify(cls, params, state, tolerance):
        outcome = state[0]
        cwol_prop = dwol_prop = 0  # Mapping thresholds of 1/2 to the appropriate WOL evenly
        for idx, freq in enumerate(outcome):  # Finding proportions of CWOL and DWOL
            if idx % total_thresh_count == 0:  # DWOL
                dwol_prop += freq
                outcome[idx] = 0
            if idx < total_thresh_count == 0:  # CWOL
                cwol_prop += freq
                outcome[idx] = 0

        outcome[total_thresh_count] = dwol_prop
        outcome[1] = cwol_prop

        np_state = np.array(outcome)
        modal_strat = np_state.argmax()
        return modal_strat.flatten()

