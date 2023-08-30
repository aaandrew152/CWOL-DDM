from games.game import Game, SymmetricNPlayerGame
from games.example_games.payoff_matrices.ctsDisc import generatePayoffs

n = 2  # Number of distinct values
m = n + 1

stratOptions1 = ['Punish iff S > ' + str(value) + '/' + str(n+2) for value in range(0, m+2)]
stratOptions2 = ['Punish iff S > ' + str(value) for value in range(0, m+2)]

class CtsDisc(SymmetricNPlayerGame):
    """ A class that creates a game to study categorical norms. See U{https://www.dropbox.com/s/t358isz782yk4b6/MH_2018_Categorical_Norms.pdf?dl=0}
    """
    DEFAULT_PARAMS = dict(a=4, b=0, c=2, d=4, errorRange=1/3)
    PLAYER_LABELS = ['']
    #STRATEGY_LABELS = (["Always Punish"] + stratOptions + ["Never Punish"])
    STRATEGY_LABELS = (stratOptions2)
    EQUILIBRIA_LABELS = ('Always punish', 'Never Punish', 'Coordinate on punishment')

    def __init__(self, a, b, c, d, errorRange, equilibrium_tolerance=0.2):
        payoff_matrix_p1 = [[0 for _ in range(m+2)] for _ in range(m+2)]
        payoff_matrix_p2 = [[0 for _ in range(m+2)] for _ in range(m+2)]

        values = (a, b, c, d, errorRange, n)  # For easier entry

        for i in range(m):  # General strategies coordinating punishment
            for j in range(m):
                payoff_matrix_p1[i+1][j+1], payoff_matrix_p2[i+1][j+1] = generatePayoffs(i, j, values)

        for i in range(m):  # Payoffs when player 1 never punishes
            payoff_matrix_p1[m+1][i+1], payoff_matrix_p2[m+1][i+1] = generatePayoffs(100, i, values)

        for i in range(m):  # Payoffs when player 2 never punishes
            payoff_matrix_p1[i+1][m+1], payoff_matrix_p2[i+1][m+1] = generatePayoffs(i, 100, values)

        payoff_matrix_p1[m+1][m+1], payoff_matrix_p2[m+1][m+1] = d, d  # Both never punish

        for i in range(m):  # Payoffs when player 1 always punishes
            payoff_matrix_p1[0][i+1], payoff_matrix_p2[0][i+1] = generatePayoffs(-100, i, values)

        for i in range(m):  # Payoffs when player 2 always punishes
            payoff_matrix_p1[i+1][0], payoff_matrix_p2[i+1][0] = generatePayoffs(i, -100, values)

        payoff_matrix_p1[0][0], payoff_matrix_p2[0][0] = a, a  # Both always punish

        payoff_matrix_p1[m+1][0], payoff_matrix_p2[m+1][0] = c, b  # P1 never punishes, P2 always punishes

        payoff_matrix_p1[0][m+1], payoff_matrix_p2[0][m+1] = b, c  # P1 always punishes, P2 never punishes

        payoff_matrix = payoff_matrix_p1
        super(CtsDisc, self).__init__(payoff_matrix=payoff_matrix, n=2, equilibrium_tolerance=equilibrium_tolerance)

    @classmethod
    def classify(cls, params, state, tolerance):
        threshold = 1-tolerance

        if state[0][0] > threshold:
            return 0  # Always punish
        elif state[0][m+1] > threshold:
            return 1  # Never punish
        else:
            for value in range(1, m+1):
                if state[0][value] > threshold:
                    return 2  # Coordination on threshold
        return super(CtsDisc, cls).classify(params, state, tolerance)
