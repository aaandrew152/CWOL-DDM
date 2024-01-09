# Testing main functions using the Hawk Dove and Hawk Dove Bourgeois games

from wrapper import GameDynamicsWrapper, VariedGame
from dynamics.wright_fisher import WrightFisher
from games.ddmGame import DDM, def_params, total_thresh_count
from games.subjectivePayoffs import subjectiveDDM
import plotting.plot_thresholds

import unittest
import numpy as np

dynam_params = dict(selection_strengthI=30, mu=0.01)

start_state = [[[0 for _ in range(total_thresh_count ** 2)]]]
if total_thresh_count % 2 == 1:
    half = (total_thresh_count - 1) / 2
else:
    half = total_thresh_count / 2
center = int(half * total_thresh_count + half)
start_state[0][0][center] += 100

d_params = (-15, 15, 3) # 20
num_iters = 1 # 3

# https://academic.oup.com/ej/advance-article-abstract/doi/10.1093/ej/uead055/7230363?redirectedFrom=fulltext&login=false
class TestCase(unittest.TestCase):
    def setUp(self):
        pass #import logging
        #logging.basicConfig(filename='debug.log', level=logging.DEBUG)

    def est_contour_plots(self):
        s = GameDynamicsWrapper(DDM, WrightFisher, game_kwargs=def_params,
                                dynamics_kwargs=dynam_params)
        simulation = s.simulate(num_gens=10, graph=False, return_labeled=False, start_state=start_state)
        frequencies = convert_output(simulation, total_thresh_count ** 2)

        # plotting.plot_thresholds.single_plot_reaction_pdf(frequencies, total_time=0.1, count=100)
        frequencies = plotting.plot_thresholds.plot_logit(10)
        
        plotting.plot_thresholds.prepare_contour_data(frequencies, True)


    def test_defection_over_d_i(self):  # Simulates while changing a single variable over time
        s = VariedGame(DDM, WrightFisher, game_kwargs=def_params, dynamics_kwargs=dynam_params)
        output = s.vary_param('d_i', d_params, num_gens=200, num_iterations=num_iters, graph=False)

        # plotting.plot_thresholds.parametric_modal_strat(output.data, d_params)
        #plotting.plot_thresholds.plot_defection_rates(output, 1000000, 25, d_params)
        plotting.plot_thresholds.plot_reaction_pdf(output, d_params, total_time=0.1, count=200) # 100
        # plotting.plot_thresholds.plot_coop_rates(output, d_params)

    def est_subj_plots(self):
        del def_params['no_look_reward']
        s = GameDynamicsWrapper(subjectiveDDM, WrightFisher, game_kwargs=def_params,
                                dynamics_kwargs=dynam_params)
        simulation = s.simulate(num_gens=5000, return_labeled=False) # , graph=dict(shading='redblue'))
        frequencies = convert_output(simulation, len(simulation[1][0]))
        plotting.plot_thresholds.single_plot_reaction_pdf(frequencies, total_time=0.1, count=100)


def convert_output(simulation, strat_length):
    frequencies = np.zeros(strat_length)
    for period in np.array(simulation[1][0]):  # Take average strategies
        frequencies += period / len(simulation[1][0])

    return frequencies


if __name__ == '__main__':
    unittest.main()
