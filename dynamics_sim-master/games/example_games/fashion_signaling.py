from games.game import Game
from games.example_games.payoff_matrices.fashion_signaling_matrices import senderPayoffs, receiverPayoffs

class FashionSignaling(Game):
    """ A class used to create a game that is a variation of the buried signaling game. See U{https://www.dropbox.com/s/gd655zcvom7pxzn/Modesty_Combined.pdf?dl=0}
    """
    DEFAULT_PARAMS = dict(lNormalCost=1.1, lHiddenCost=1.1, hNormalCost=1.1, hHiddenCost=1.1, llSender=0.8, lhSender=5,
                          hlSender=1.2, hhSender=15, llReceiver=-10, hlReceiver=5, lhReceiver=-10, hhReceiver=10,
                          lReceiverCost=3, hReceiverCost=1, lSenderProp=8, hSenderProp=2, lReceiverProp=8,
                          hReceiverProp=2, equilibrium_tolerance=0.2)
    PLAYER_LABELS = ('Low Sender', 'High Sender', 'Low Receiver', 'High Receiver')
    STRATEGY_LABELS = (('Silent', 'Normal', 'Hidden'),
                       ('Silent', 'Normal', 'Hidden'),
                       ('Accept all', 'Accept Silent and Normal', 'Accept Silent and Hidden',
                        'Accept Normal and Hidden', 'Accept Silent', 'Accept Normal', 'Accept Hidden',
                        'Reject'),
                       ('Accept all', 'Accept Silent and Normal', 'Accept Silent and Hidden',
                        'Accept Normal and Hidden', 'Accept Silent', 'Accept Normal', 'Accept Hidden',
                        'Reject'))#If a player accepts a hidden signal it is presumed that they are investing
    EQUILIBRIA_LABELS = ('Sophisticated Signaling', 'Common Separating', 'Pooling with Rejection', 'Pooling with Acceptance')

    def __init__(self, lNormalCost, lHiddenCost, hNormalCost, hHiddenCost, llSender, lhSender, hlSender,
                 hhSender, llReceiver, hlReceiver, lhReceiver, hhReceiver, lReceiverCost, hReceiverCost, lSenderProp,
                 hSenderProp, lReceiverProp, hReceiverProp, equilibrium_tolerance):

        LSenderProp = lSenderProp / (lSenderProp + hSenderProp)
        HSenderProp = hSenderProp / (lSenderProp + hSenderProp)

        LReceiverProp = lReceiverProp / (lReceiverProp + hReceiverProp)
        HReceiverProp = hReceiverProp / (lReceiverProp + hReceiverProp)

        payoff_matrix_LS = senderPayoffs(lNormalCost, lHiddenCost, llSender, lhSender, LReceiverProp, HReceiverProp, 'low')

        payoff_matrix_HS = senderPayoffs(hNormalCost, hHiddenCost, hlSender, hhSender, LReceiverProp, HReceiverProp, 'high')

        payoff_matrix_LR = receiverPayoffs(llReceiver, hlReceiver, lReceiverCost, LSenderProp, HSenderProp, 'low')

        payoff_matrix_HR = receiverPayoffs(lhReceiver, hhReceiver, hReceiverCost, LSenderProp, HSenderProp, 'high')

        payoff_matrix = [payoff_matrix_LS, payoff_matrix_HS, payoff_matrix_LR, payoff_matrix_HR]
        player_dist = (LSenderProp/2, HSenderProp/2, LReceiverProp/2, HReceiverProp/2)
        super(FashionSignaling, self).__init__(payoff_matrices=payoff_matrix, player_frequencies=player_dist, equilibrium_tolerance=equilibrium_tolerance)

    @classmethod
    def classify(cls, params, state, tolerance):
        # for convenience, we will guarantee that the state is normalized already to proportions, not absolute number of player
        threshold = 1 - tolerance

        if all(x >= threshold for x in [state[0][0], state[1][2], state[3][6]]) and (state[2][5] >= threshold or state[2][7] >= threshold):
            return 0#Sophisticated Signaling
        elif all(x >= threshold for x in [state[0][0], state[1][1], state[2][5], state[3][5]]):
            return 1#Common Separating
        elif all(x >= threshold for x in [state[0][0], state[1][0]]) and (x <= tolerance for x in [state[2][1], state[3][1]]):
            return 2#Pooling with Rejection
        elif all(x >= threshold for x in [state[0][0], state[1][0], state[2][1], state[3][1]]):
            return 3#Pooling with Acceptance
        else:
            return super(FashionSignaling, cls).classify(params, state, tolerance)
