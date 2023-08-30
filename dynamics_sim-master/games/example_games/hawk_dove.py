from games.game import SymmetricNPlayerGame


class HawkDove(SymmetricNPlayerGame):
    """
    A class used to to represent the 2 player hawk-dove game. See U{http://www.life.umd.edu/classroom/zool360/L18-ESS/ess.html}
    """
    DEFAULT_PARAMS = dict(v=30, c=100)
    STRATEGY_LABELS = ('Hawk', 'Dove')
    EQUILIBRIA_LABELS = ('All_Hawk', 'All_Dove', 'Hawk_Dove', 'Dove_Hawk')

    def __init__(self, v, c, equilibrium_tolerance=0.1):
        payoff_matrix = (((v - c) / 2.0, v),
                         (0, v / 2.0))

        super(HawkDove, self).__init__(payoff_matrix=payoff_matrix, n=2, equilibrium_tolerance=equilibrium_tolerance)

    @classmethod
    def classify(cls, params, state, tolerance):
        p = params
        threshold = 1-tolerance

        return super(HawkDove, cls).classify(params, state, tolerance)
