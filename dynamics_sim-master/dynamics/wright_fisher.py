__author__ = 'eblubin@mit.edu, anande01@g.harvard.edu'
import numpy as np
from dynamics.dynamics import DynamicsSimulator
from games.ddmGame import total_thresh_count
from games.subjectivePayoffs import rr


class WrightFisher(DynamicsSimulator):
    def __init__(self, mu = None, *args, **kwargs):
        """
        @param mu: mutation rate
        @type mu: float or a list of n lists, where n is the number of players and len(list_n) = number of player_n strategies. Defaults to zero if not specified.
        """

        # TODO don't allow pop_size of 0, wright fisher only works with finite pop size
        super(WrightFisher, self).__init__(*args,stochastic=True,**kwargs)

        if mu == None:
            mu = 0.0
        self.mu = mu

    def total_adj(self, previous_state, strat):  # Returns total number of adjacencies
        num_adj = 0

        # if strat >= total_thresh_count:  # Not LHS
        #     num_adj += find_left(previous_state, strat)
        # if strat < total_thresh_count ** 2 - total_thresh_count:  # Not RHS
        #     num_adj += find_right(previous_state, strat)
        # if strat % total_thresh_count != 0:  # Not bot
        #     num_adj += find_down(previous_state, strat)
        # if strat % total_thresh_count != total_thresh_count - 1:  # Not top
        #     num_adj += find_up(previous_state, strat)

        if strat > 0:  # Not minimal
            num_adj += find_down(previous_state, strat)
        if strat < len(rr) - 1: # Not maximal
            num_adj += find_up(previous_state, strat)

        return num_adj

    def next_generation(self, previous_state, group_selection, rate, local=False):
        next_state = []
        number_groups=len(previous_state)
        payoff = []
        avg_payoffs = []
        fitness = []

        for i in range(number_groups):
            p, avg_p = self.calculate_payoffs(previous_state[i])
            payoff.append(p)
            avg_payoffs.append(avg_p)
            fitness.append(self.calculate_fitnesses(payoff[i], self.selection_strengthI))

        # Creating the mutation matrix
        if type(self.mu) == float:
            mu_matrix = []
            for i in range(len(payoff[0])):
                mu_matrix.append(self.mu*np.ones(len(payoff[0][i])))
        else:
            mu_matrix = self.mu

        # Wright-Fisher between groups
        if group_selection and np.random.uniform(0,1)<rate:

            avg_fitness = []

            # Calculate the fitness of each group based on their average payoffs
            for k in range(len(avg_payoffs)):
                avg_fitness.append(self.fitness_func(avg_payoffs[k], self.selection_strengthG))
            # Groups reproduce proportional to their fitness

            new_group_distribution = np.random.multinomial(number_groups,[x / sum(avg_fitness) for x in avg_fitness])


            # Update the new distribution of groups
            for idx, group_freq in enumerate(new_group_distribution):
                for i in range(group_freq):
                    next_state.append(previous_state[idx])

        # Wright-Fisher inside groups
        else:
            for i in range(number_groups):
                new_group_state =[]
                for player_idx, (strategy_distribution, fitnesses, num_players) in enumerate(zip(previous_state[i], fitness[i], self.num_players)):
                    num_strats = len(strategy_distribution)
                    total_mutations = 0
                    new_player_state = np.zeros(num_strats)

                    for strategy_idx, n in enumerate(strategy_distribution):
                        f = fitnesses[strategy_idx]
                        mu_individual = mu_matrix[player_idx][strategy_idx]

                        # sample from binomial distribution to get number of mutations for strategy
                        if n == 0:
                            mutations = 0
                        else:
                            mutations = np.random.binomial(n, mu_individual)
                        n -= mutations
                        total_mutations += mutations
                        new_player_state[strategy_idx] = n * f

                        # distribute player strategies proportional n * f
                        # don't use multinomial, because that adds randomness we don't want yet.

                    if new_player_state.sum() != 0:
                        new_player_state *= float(num_players - total_mutations) / new_player_state.sum()
                        new_player_state = np.array(self.round_individuals(new_player_state))

                    else: # Make sure that mutations get randomly distributed if they lead to a zero population size
                        new_player_state = np.zeros(num_strats)

                    if local:  # Weigh mutations by number of adjacent Strategies
                        adj_list = []
                        for strat, mu_rate in enumerate(mu_matrix[0]):
                            adj_list.append(self.total_adj(previous_state, strat))
                        new_mutation_matrix = np.array(adj_list) / sum(adj_list)
                        new_player_state += np.random.multinomial(total_mutations, new_mutation_matrix)
                    else:
                        new_player_state += np.random.multinomial(total_mutations, [1. / num_strats] * num_strats)
                    new_group_state.append(new_player_state)
                next_state.append(new_group_state)

        return next_state, fitness


def find_up(previous_state, strat):
    return previous_state[0][0][strat + 1]

def find_down(previous_state, strat):
    return previous_state[0][0][strat - 1]

def find_right(previous_state, strat):
    return previous_state[0][0][strat + total_thresh_count]

def find_left(previous_state, strat):
    return previous_state[0][0][strat - total_thresh_count]
