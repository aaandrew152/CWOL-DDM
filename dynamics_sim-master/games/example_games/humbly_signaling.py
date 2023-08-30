from games.game import Game
from games.example_games.payoff_matrices.humbly_signaling_matrices import senderPayoffs, receiverPayoffs


class HumblySignaling(Game):
    """ A class used to create a game that is a variation of the buried signaling game. See U{https://www.dropbox.com/s/gd655zcvom7pxzn/Modesty_Combined.pdf?dl=0}
    """
    DEFAULT_PARAMS = dict(lReveal=.1, mReveal=.2, hReveal=.8, lCost=5, mCost=1,
                          hCost=1, llSender=1, lhSender=2, mlSender=3, mhSender=4, hlSender=1,
                          hhSender=10, llReceiver=-10, mlReceiver=4, hlReceiver=5, lhReceiver=-20,
                          mhReceiver=-12.5, hhReceiver=10, lSenderProp=5, mSenderProp=4,
                          high_sender_proportion=2, lReceiverProp=3, hReceiverProp=2, equilibrium_tolerance=.2)
    PLAYER_LABELS = ('Low Sender', 'Medium Sender', 'High Sender', 'Low Receiver', 'High Receiver')
    STRATEGY_LABELS = (('No Signal', 'Normal Signal', 'Hidden Signal'),
                       ('No Signal', 'Normal Signal', 'Hidden Signal'),
                       ('No Signal', 'Normal Signal', 'Hidden Signal'),
                       ('Accept all', 'Accept Normal', 'Accept Hidden', 'Accept None'),
                       ('Accept all', 'Accept Normal', 'Accept Hidden', 'Accept None'))
    EQUILIBRIA_LABELS = ('Pooling with Rejection', 'Pooling with Acceptance', 'Accept normal', 'Accept hidden', 'Strategic Obfuscation', 'High Accept All')

    def __init__(self, lReveal, mReveal, hReveal, lCost, mCost, hCost, llSender, lhSender, mlSender,
                 mhSender, hlSender, hhSender, llReceiver, mlReceiver, hlReceiver, lhReceiver,
                 mhReceiver, hhReceiver, lSenderProp, mSenderProp, high_sender_proportion, lReceiverProp,
                 hReceiverProp, equilibrium_tolerance):

        hSenderProp = high_sender_proportion  #Renaming for cleanliness

        LSenderProp = lSenderProp / (lSenderProp + mSenderProp + hSenderProp)
        MSenderProp = mSenderProp / (lSenderProp + mSenderProp + hSenderProp)
        HSenderProp = hSenderProp / (lSenderProp + mSenderProp + hSenderProp)

        LReceiverProp = lReceiverProp / (lReceiverProp + hReceiverProp)
        HReceiverProp = hReceiverProp / (lReceiverProp + hReceiverProp)# Normalize the proportions

        payoff_matrix_LS = senderPayoffs(lCost, llSender, lhSender, LReceiverProp, HReceiverProp, lReveal, 'l')

        payoff_matrix_MS = senderPayoffs(mCost, mlSender, mhSender, LReceiverProp, HReceiverProp, mReveal, 'm')

        payoff_matrix_HS = senderPayoffs(hCost, hlSender, hhSender, LReceiverProp, HReceiverProp, hReveal, 'h')

        payoff_matrix_LR = receiverPayoffs(lReveal, mReveal, hReveal, llReceiver, mlReceiver, hlReceiver, lSenderProp, mSenderProp, hSenderProp, 'l')

        payoff_matrix_HR = receiverPayoffs(lReveal, mReveal, hReveal, lhReceiver, mhReceiver, hhReceiver, lSenderProp, mSenderProp, hSenderProp, 'h')

        payoff_matrix = [payoff_matrix_LS, payoff_matrix_MS, payoff_matrix_HS, payoff_matrix_LR, payoff_matrix_HR]
        player_dist = (LSenderProp/2, MSenderProp/2, HSenderProp/2, LReceiverProp/2, HReceiverProp/2)
        super(HumblySignaling, self).__init__(payoff_matrices=payoff_matrix, player_frequencies=player_dist, equilibrium_tolerance=equilibrium_tolerance)

    @classmethod
    def classify(cls, params, state, tolerance):
        # for convenience, we will guarantee that the state is normalized already to proportions, not absolute number of player
        threshold = 1 - tolerance

        if state[4][3] > threshold:
            return 0#High receiver accept no one
        elif state[3][0] > threshold and state[4][0] > threshold:
            return 1#Accept everyone
        elif state[3][1] > threshold and state[4][1] > threshold:
            return 2#Accept normal signals both receivers
        elif state[3][2] > threshold and state[4][2] > threshold:
            return 3#Accept hidden signals both receivers
        elif state[2][2] > threshold and state[3][1] > threshold and state[4][2] > threshold:
            return 4#Strategic Obfuscation
        elif state[4][0] > threshold:
            return 5#High accepts all
        else:
            return super(HumblySignaling, cls).classify(params, state, tolerance)
