from games.game import game
from games.example_games.payoff_matrics.cwol_multiple import p1Payoffs, p2Payoffs

class CwolMultiple(game):
    """ Cooperate without looking game with 4 players.
    """
    DEFAULT_PARAMS = dict(a=2, b=2, cl=3, ch=5, dl=2.5, dh=5, p=.8, q=.7, r=.3, \
                          wl=.1, wh=.7, p1lProp=.7, p2lProp=.7)
    PLAYER_LABELS = ('Low Player 1', 'High Player 1', 'Low Player 2', 'High Player 2')
    STRATEGY_LABELS = (('All D', 'Look Only L', 'Look All C', 'CWOL'),
                       ('All D', 'Look Only L', 'Look All C', 'CWOL'),
                       ('Exit', 'Exit Look', 'Exit Defect low', 'Exit Look Defect', 'Never Exit'),
                       ('Exit', 'Exit Look', 'Exit Defect low', 'Exit Look Defect', 'Never Exit'))
    EQUILIBRIA_LABELS = ('Defect Exit', 'CWOL separating', 'Cooperate stay')
    #Listing low, then high

    def __init__(self, a, b, cl, ch, dl, dh, p, q, r, wl, wh, p1lProp, p2lProp):

        p1hProp = (1 - p1lProp) / 2
        p1lProp /= 2

        p2hProp = (1 - p2lProp) /2
        p2lProp /= 2

        payoff_1l = p1Payoffs()
