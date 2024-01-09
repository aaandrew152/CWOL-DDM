import matplotlib.pyplot as plt
import numpy as np
from math import exp, pi, sin, e
from plotting.plotHelperFunct import *
from games.ddmGame import DDM, total_thresh_count, thresholds, def_params, integral_boxes
from operator import itemgetter
from plotting.plot import superExpandRBShades, redBlueShades
import random
from itertools import chain

superExpandRBShades.reverse()
font_size = 18
path = "C:/Users/abf204/OneDrive - University of Exeter/Desktop/Projects/CWOL/Figures/"

class GraphOptions:
    COLORS_KEY = 'colors'
    SUPER_EXPAND_RB_COLORS_KEY = 'Red to Blue with 100 colors'
    EXTRA_COLORS = 'extra colors if more than 7 to be shown'
    Y_LABEL_KEY = 'y_label'
    LEGEND_LOCATION_KEY = 'legend_location'
    SHOW_GRID_KEY = 'grid'
    LEGEND_LABELS_KEY = 'legend_labels'
    TITLE_KEY = 'title'
    MARKERS_KEY = 'markers'
    NO_MARKERS_KEY = 'hide_markers'
    PLAYER_TYPES = 'Group certain graphs'

    default = {COLORS_KEY: ['Cyan', 'Blue', 'Green', 'Yellow', 'Red', 'Magenta', 'Black', 'BlueViolet', 'Crimson', 'Indigo'] * 5,  # If extra colors needed it repeats
               SUPER_EXPAND_RB_COLORS_KEY: [(value[0] / 255, value[1] / 255, value[2] / 255) for value in superExpandRBShades],
               MARKERS_KEY: "o.v8sh+xD|_ ",
               NO_MARKERS_KEY: False,
               Y_LABEL_KEY: "Proportion of Population",
               LEGEND_LOCATION_KEY: 'upper right',
               SHOW_GRID_KEY: True,
               TITLE_KEY: lambda player_i: "Population Dynamics for Player %d" % player_i,
               LEGEND_LABELS_KEY: lambda graph_i, cat_i: "X_%d,%d" % (graph_i, cat_i),
               PLAYER_TYPES: False}


def _append_options(options):
    old_options = GraphOptions.default.copy()
    if options is not None:
        old_options.update(options)
    return old_options


def generate_game(d_i=None):
    if d_i is not None:
        punish_defect, c_min_max, no_look_reward, var, search_cost = \
            itemgetter('punish_defect', 'c_min_max', 'no_look_reward', 'var', 'search_cost')(def_params)
    else:
        d_i, punish_defect, c_min_max, no_look_reward, var, search_cost = \
            itemgetter('d_i', 'punish_defect', 'c_min_max', 'no_look_reward', 'var', 'search_cost')(def_params)

    copied_game = DDM(d_i, punish_defect, c_min_max, var, search_cost, no_look_reward)

    return copied_game


def single_plot_defection_rate(frequencies, indiv_count, play_count):
    outcomes = []
    weights = np.array([])
    bins = np.arange(0, play_count+1, 1)

    for idx, freq in enumerate(frequencies):
        outcomes.extend(np.random.binomial(play_count, generate_game().defect_probs[idx], indiv_count))
        weights = np.append(weights, [freq / indiv_count for _ in range(indiv_count)])

    weights = weights / sum(weights)
    plt.hist(outcomes, bins=bins, edgecolor='blue', weights=weights)

    x_label = "Defections out of " + str(play_count) + " plays"
    setup_plot(x_label, "Proportion", "Defection Count")


def plot_coop_rates(output, d_params):
    d_list = [x * (d_params[1] - d_params[0]) / d_params[2] + d_params[0] for x in range(d_params[2] + 1)]
    
    coop_outcomes = [0 for c in range(integral_boxes)]
    
    for idx, sim in enumerate(output.data):
        modal_strat = sim.argmax()
        d_i = d_list[idx]
        coop_probs = generate_game(d_i).coop_probs[modal_strat]

        for c_idx, c_prob in enumerate(coop_probs):
            coop_outcomes[c_idx] += c_prob / len(d_list)
            
    x_label = "Cooperation Value"
    c_values = generate_game(d_list[0]).c_dist
    
    plt.ylim([0, 1.05])
    plt.plot(c_values, coop_outcomes)
    setup_plot(x_label, "Cooperation Probability", "$d_i$ varies from " + str(d_params[0]) + " to " + str(d_params[1]))


def plot_defection_rates(output, indiv_count, play_count, d_params):
    d_list = [x * (d_params[1] - d_params[0]) / d_params[2] + d_params[0] for x in range(d_params[2] + 1)]

    def_outcomes = []
    bins = np.arange(0, play_count+1, 1)

    di_bins = [[] for _ in bins]

    for idx, sim in enumerate(output.data):
        modal_strat = sim.argmax()
        d_i = d_list[idx]
        def_prob = generate_game(d_i).defect_probs[modal_strat]

        outcome = np.random.binomial(play_count, def_prob, indiv_count)
        def_outcomes.extend(outcome)
        for num_defects in range(play_count+1):
            di_bins[num_defects].extend(d_i for _ in range(np.count_nonzero(outcome == num_defects)))

    weights = np.array([1 / len(def_outcomes) for _ in def_outcomes])
    plt.hist(def_outcomes, bins=bins, edgecolor='blue', weights=weights)

    x_label = "Defections out of " + str(play_count) + " plays"
    setup_plot(x_label, "Proportion", "$d_i$ varies from " + str(d_params[0]) + " to " + str(d_params[1]))

    avg_dis, x_values = [], []
    for defect_count, di_bin in enumerate(di_bins):
        if len(di_bin) > 0:
            avg_dis.append(sum(di_bin) / len(di_bin))
            x_values.append(defect_count)
        else:
            avg_dis.append(0)
            x_values.append(x_values[-1]+1)

    # avg_dis_dict = {idx: avg_dis[idx] for idx, defect_num in enumerate(avg_dis)}
    # plt.plot(x_values[1:], avg_dis[1:])
    # plt.scatter(x_values[0], avg_dis[0])
    plt.bar(np.arange(play_count+1), avg_dis)
    plt.axis([None, None, d_list[0], d_list[-1]])
    setup_plot("Number of Defections", "Expected $d_i$", 'Expected $d_i$ after observing defections')


def setup_plot(x_label, y_label, title):
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    # plt.title(title, fontsize=font_size)
    plt.savefig(path + title + ".svg")
    plt.show()


class ReactionAction:  # Thresholds compared to 0
    def __init__(self, upper, lower, drift, var, sum_accuracy=50):
        self.lower = lower - 1/2
        self.upper = upper - 1/2
        self.drift = drift
        self.sum_accuracy = sum_accuracy

        self.D = var ** 2 / 2
        self.distance = self.upper - self.lower

    def compute_defect_prob(self, alpha):
        if self.upper <= 0.01:  # DWOL:

            return 1
        elif self.lower >= -0.01:
            return 0

        if alpha == 0:
            return - self.lower / (-self.lower + self.upper)
        else:
            numerator = 1 - e ** (alpha * - self.lower)
            denominator = e ** (-alpha * self.upper) - e ** (alpha * - self.lower)
            return numerator / denominator

    def end_pdf(self, total_time=0.5, count=50):  # 1) Note 0 is skipped, unless WOL 2) Normalizing to 1
        if self.upper <= 0.01 or self.lower >= -0.01:
            temp_pdf = np.zeros(count)
            temp_pdf[0] = 1
            return temp_pdf

        pdf = []
        for t in range(count):
            time = (t + 1) * total_time / count
            pdf.append(self.compute_pdf_t(time))

        pdf = np.array(pdf) / sum(pdf)
        return pdf

    def defect_pdf(self, total_time=0.1, count=50):
        if self.upper <= 0.01:
            temp_pdf = np.zeros(count)
            temp_pdf[0] = 1
            return temp_pdf, 1
        elif self.lower >= -0.01:
            return np.zeros(count), 0

        def_pdf = []
        for t in range(count):
            time = (t + 1) * total_time / count

            def_pdf.append(self.compute_pdf_t(time, defect=True))

        def_pdf = np.array(def_pdf) / sum(def_pdf)

        defect_prob = self.compute_defect_prob(2 * self.drift / (2 * self.D))

        return def_pdf, defect_prob

    def compute_pdf_t(self, t, defect=False):
        v_upper_t = self.v_x(self.upper, t)
        if defect:
            return self.D * v_upper_t

        v_lower_t = self.v_x(self.lower, t)
        return self.D * (v_upper_t - v_lower_t)

    def v_x(self, x, t):  # Sum accuracy indicates the distance out to which the sum should be computed
        summation = 0
        for n in range(self.sum_accuracy):
            n += 1
            lambda_i = self.lambda_n(n)
            summation += n * pi / self.distance * sin(n * pi * x / self.distance) * exp(- lambda_i * t)

        v = 2 / self.distance * exp(self.drift * x / (2 * self.D)) * summation
        return v

    def lambda_n(self, n):
        frac_1 = n * pi / self.distance
        frac_2 = self.drift / (2 * self.D)
        return self.D * (frac_1 ** 2 + frac_2 ** 2)


def pdf_to_cdf(count, pdf):
    cdf = np.zeros(count)

    cdf[0] = pdf[0] / sum(pdf)
    for idx, value in enumerate(pdf[1:]):
        cdf[idx + 1] = cdf[idx] + value / sum(pdf)

    return cdf


def pdf_for_game(sim, copied_game, total_time, count, defect=False):
    pdf = np.zeros(count)

    for strat_idx, freq in enumerate(sim):
        upper, lower = map_strat_num_to_xy(strat_idx, thresholds)
        prop = freq / len(copied_game.c_dist)

        if defect:
            for c in copied_game.c_dist:
                ra = ReactionAction(upper, lower, copied_game.defect_util - c, copied_game.var)
                new_def_pdf, new_def_prob = ra.defect_pdf(total_time=total_time, count=count)
                pdf = pdf + new_def_pdf * new_def_prob * prop
        else:
            for c in copied_game.c_dist:
                ra = ReactionAction(upper, lower, copied_game.defect_util - c, copied_game.var)
                pdf = pdf + ra.end_pdf(total_time=total_time, count=count) * prop

    return pdf


def create_coop_cdf(count, pdf, def_pdf):
    coop_pdf = pdf - def_pdf
    return pdf_to_cdf(count, coop_pdf)


def single_plot_reaction_pdf(frequencies, total_time=1, count=10):
    copied_game = generate_game()

    pdf = pdf_for_game(frequencies, copied_game, total_time, count)
    def_pdf = pdf_for_game(frequencies, copied_game, total_time, count, defect=True)

    cdf = pdf_to_cdf(count, pdf)
    def_cdf = pdf_to_cdf(count, def_pdf)
    coop_cdf = create_coop_cdf(count, pdf, def_pdf)

    x_values = [t * total_time / count for t in range(count)]

    plt.plot(x_values, cdf, color='brown', label='All Stopping Times')
    plt.plot(x_values, def_cdf, color='red', label='Defect Stopping Times')
    plt.plot(x_values, coop_cdf, color='blue', label='Cooperate Stopping Times')
    plt.ylim([0, 1.05])
    plt.legend()
    setup_plot("Time", "CDF", "Cumulative Stopping Times")


def plot_reaction_pdf(output, d_params, total_time=.1, count=10):
    d_list = get_d_list(d_params)
    pdf_list, def_pdf_list, coop_pdf_list = \
        [np.zeros(count) for _ in d_list], [np.zeros(count) for _ in d_list], [np.zeros(count) for _ in d_list]

    for idx, sim in enumerate(output.data):
        copied_game = generate_game(d_i=d_list[idx])

        pdf_list[idx] = pdf_for_game(sim, copied_game, total_time, count)
        def_pdf_list[idx] = pdf_for_game(sim, copied_game, total_time, count, defect=True)

        coop_pdf_list[idx] = pdf_list[idx] - def_pdf_list[idx]

    overall_pdf, overall_def_pdf = np.zeros(count), np.zeros(count)
    for pdf in pdf_list:
        overall_pdf = overall_pdf + pdf

    for def_pdf in def_pdf_list:
        overall_def_pdf = overall_def_pdf + def_pdf

    cdf, def_cdf = pdf_to_cdf(count, overall_pdf), pdf_to_cdf(count, overall_def_pdf)
    coop_cdf = create_coop_cdf(count, overall_pdf, overall_def_pdf)

    d_avgs = []
    for t_idx, _ in enumerate(coop_pdf_list[0]):
        mass_at_t = overall_pdf[t_idx] - overall_def_pdf[t_idx]
        d_avg = 0
        for d_idx, d_val in enumerate(d_list):
            d_avg += d_val * coop_pdf_list[d_idx][t_idx] / mass_at_t
        d_avgs.append(d_avg)

    x_values = [t * total_time / count for t in range(count)]

    plt.plot(x_values, cdf, color='brown', label='All Stopping Times')
    plt.plot(x_values, def_cdf, color='red', label='Defect Stopping Times')
    plt.plot(x_values, coop_cdf, color='blue', label='Cooperate Stopping Times')
    plt.ylim([0, 1.05])
    plt.legend()
    setup_plot("Time", "CDF", "Normalized stopping times when defect reward varies")

    plt.scatter(x_values[0], d_avgs[0], marker='o', edgecolors='b') # facecolors='none',
    plt.scatter(x_values[1], d_avgs[1], marker='o', facecolors='none', edgecolors='b')
    plt.plot(x_values[2:], d_avgs[2:], 'b')
    setup_plot("Time of Cooperation", "Expected $d_i$", 'Time-specific expected $d_i$ upon cooperation')


def get_d_list(d_params):
    return [x * (d_params[1] - d_params[0]) / d_params[2] + d_params[0] for x in range(d_params[2] + 1)]


def parametric_modal_strat(output, d_params):
    d_list = get_d_list(d_params)

    modal_strats = []
    for simulation in output:
        modal_strats.append(simulation.argmax())

    x_values = []
    y_values = []
    for modal_strat in modal_strats:
        x, y = map_strat_num_to_xy(modal_strat, thresholds)
        x_values.append(x)
        y_values.append(y)

    ax = plt.figure().add_subplot()
    line_array = generate_wol_lines('black')
    graphColoredLines(line_array, ax)

    for idx, _ in enumerate(x_values):  # Add slight scatter to CWOL and DWOL outcomes
        if x_values[idx] == 0.75 and y_values[idx] == 0.5:
            x_values[idx] = 0.75 + random.uniform(-0.02, 0.02)
        elif x_values[idx] == 0.5 and y_values[idx] == 0.25:
            y_values[idx] = 0.25 + random.uniform(-0.02, 0.02)

    interp_x_values, interp_y_values, interp_t_values = interpolate_values(x_values, y_values, d_list)
    plt.scatter(interp_x_values, interp_y_values, c=interp_t_values, clim=(d_list[0], d_list[-1]), cmap='plasma', marker='_')
    plt.scatter(x_values, y_values, c=d_list, cmap='plasma')

    plt.colorbar(pad=0.08, label='Defect Payoff ($d_i$ + punish_defect)')
    # If we want double ticks: https://stackoverflow.com/questions/27151098/draw-colorbar-with-twin-scales
    generate_axes(plt, x_values, y_values, font_size)
    plt.title('Modal Outcomes as Defect Payoff Varies', fontsize=font_size)
    plt.xlabel("Defection Threshold", fontsize=font_size)
    plt.ylabel("Cooperation Threshold", fontsize=font_size)
    plt.show()


def prepare_contour_data(frequencies, weight=False):
    frequencies = frequencies / sum(frequencies)

    cwol_prop, dwol_prop = find_wol_props(frequencies, total_thresh_count)

    if not weight:
        for idx, freq in enumerate(frequencies):  # Distribution CWOL and DWOL evenly
            if idx % total_thresh_count == 0 and idx > 0:
                frequencies[idx] = dwol_prop #/ (total_thresh_count - 1)
            if floor(idx / total_thresh_count) == 0 and idx > 0:
                frequencies[idx] = cwol_prop #/ (total_thresh_count - 1)

    twoD_frequencies = [[] for _ in range(total_thresh_count)]
    for idx, freq in enumerate(frequencies):
        twoD_frequencies[floor(idx / total_thresh_count)].append(min(freq, 1))

    if weight: # Scales maximum to one
        factor = max(list(chain(*twoD_frequencies)))
        twoD_frequencies = twoD_frequencies/factor

    upper_thresholds = [thresh + 1 / 2 for thresh in thresholds]

    ykey = 'Top Threshold'
    xkey = 'Bottom Threshold'
    plot_contour_data_set(twoD_frequencies, xkey, thresholds, ykey, upper_thresholds)
    
    
def plot_logit(weight=1):
    payoffs = generate_game().tau_list

    realistic_payoff_sum = 0
    for payoff in payoffs:  # Undo outlier
        realistic_payoff_sum += exp(weight*payoff)

    soft_utils = []
    for util in payoffs:
        soft_utils.append(exp(weight*util) / realistic_payoff_sum)

    normalized_soft_utils = []
    for soft_util in soft_utils:
        normalized_soft_utils.append((soft_util - min(soft_utils[1:])) / (max(soft_utils) - min(soft_utils[1:])))

    normalized_soft_utils = (np.array(normalized_soft_utils)) # Preparation for summation
    return normalized_soft_utils


def plot_contour_data_set(data, y_label, y_values, x_label, x_values, graph_options=None):
    # Note it seems as though the x and y values are switched for contour plots
    graph_options = _append_options(graph_options)
    plt.close('all')

    colors = graph_options[GraphOptions.SUPER_EXPAND_RB_COLORS_KEY]

    generate_axes(plt, x_values, y_values, font_size)

    levels = [x/len(colors) for x in range(len(colors)+1)]

    cs = plt.contourf(x_values, y_values, data, levels, colors=colors)

    plt.colorbar(cs, orientation='horizontal')
    line_array = generate_wol_lines()
    graphLines(line_array, plt)

    plt.tight_layout()
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    # plt.title(title, fontsize=font_size)

    plt.show()