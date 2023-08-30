import numpy
import matplotlib.pyplot as plt
import random
import math

random.seed()

from pyddm import Model
import pyddm.plot
from pyddm.models import DriftConstant, NoiseConstant, BoundConstant, OverlayNonDecision, ICPointSourceCenter
from pyddm.functions import fit_adjust_model, display_model
from pyddm import Fittable, Fitted
from pyddm.models import LossRobustBIC
from pyddm.functions import fit_adjust_model


class Player:
    def __init__(self, d):
        self.d = d
        self.history = []


# Simulation parameters
player_count = 3
observation_count = 2

# Game parameters
p = 0  # Fixed social cost of defection
defect_min = 0 - p
defect_max = 5 - p
# D is uniform on [min_defect, max_defect]

coop_min = 0
coop_max = 1
# C is uniform on [min_coop, max_coop]

no_look = 1 / 2

noise_level = 1
confidence = 1
# Level of confidence necessary to make decision, might need to be computed

players = []
for player in range(player_count):
    defect_value = random.uniform(defect_min, defect_max)
    players.append(Player(defect_value))


def game(player):  # Remember that this draws a single value of cooperation
    coop_value = random.uniform(coop_min, coop_max)

    if player.d < 2 * no_look:  # TODO double check that this is the correct value!
        player.history.append(None)
        return None

    model = Model(name='Single Choice',
                  drift=DriftConstant(drift=player.d - coop_value),
                  noise=NoiseConstant(noise=noise_level),
                  bound=BoundConstant(B=confidence),
                  overlay=OverlayNonDecision(nondectime=.1),
                  dx=.001, dt=.01, T_dur=2)

    print("Probability of correct is " + str(model.solve().prob_correct()))
    # display_model(model)

    player.history.append(coop_value)

    return model.solve()


def simulate(player, id):
    print("Player " + str(id + 1) + "'s defect value is " + str(player.d))

    for obs in range(observation_count):
        single_game = game(player)

        if not single_game:
            print("Cooperate without Looking")
        else:
            print("Game " + str(obs + 1) + "'s coop value is " + str(player.history[obs]))

            samp = single_game.resample(1000)

            # display_model(model_fit)

            # model_fit.parameters()

            # model_fit.get_fit_result().value() - loss function value

            pyddm.plot.plot_fit_diagnostics(sample=samp)
            # plt.savefig("output.png")
            plt.show()


for id, player in enumerate(players):
    simulate(player, id)

# TODO: Run replicator dynamics for fixed payoffs from defection, i.e., 0, 1/2, 1, ...
# Add strategy for don't look and defect

# MATH SECTION:
# Probability of choosing defect when the value of defection is low
d_i = 1
c = 1/2
var = 1

alpha = 2 * (d_i - c)/(var ** 2))
threshold = 1/6
lThresh = 1/2 - threshold
hThresh = 1/2 + threshold

p_defect = (1 - math.e ** (alpha * lThresh)) / (math.e)

# TODO: add section with subjective utility - ONLY OVER COOPERATING OR DEFECTING, NOT NOT LOOKING

