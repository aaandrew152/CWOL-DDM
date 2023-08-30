from games.game import SymmetricNPlayerGame

class ReactivePD(SymmetricNPlayerGame):
    """ A class used to represent a symmetric game, where the players are playing 4 reactive strategies associated
    with the Prisoners' Dilemma.
    """
    DEFAULT_PARAMS = dict(R=3, T=5, S=0, P=1, bias_strength=0)
    STRATEGY_LABELS = ('ALLD', 'ALLC', 'TFT', 'GTFT')
    EQUILIBRIA_LABELS = ('ALLD', 'ALLC', 'TFT', 'GTFT')

    def __init__(self, R, T, S, P, bias_strength):

        self.R = R; self.T = T; self.S = S; self.P = P

        p1 = 0; q1 = 0
        p2 = 1; q2 = 1
        p3 = 1; q3 = 0
        p4 = 1; q4 = 1/3

        payoff_matrix = ((self.payoff(p1,q1,p1,q1),self.payoff(p1,q1,p2,q2),self.payoff(p1,q1,p3,q3),self.payoff(p1,q1,p4,q4)),
                         (self.payoff(p2,q2,p1,q1),self.payoff(p2,q2,p2,q2),self.payoff(p2,q2,p3,q3),self.payoff(p2,q2,p4,q4)),
                         (self.payoff(p3,q3,p1,q1),self.payoff(p3,q3,p2,q2),self.payoff(p3,q3,p3,q3),self.payoff(p3,q3,p4,q4)),
                         (self.payoff(p4,q4,p1,q1),self.payoff(p4,q4,p2,q2),self.payoff(p4,q4,p3,q3),self.payoff(p4,q4,p4,q4)))


        super(ReactivePD, self).__init__(payoff_matrix, 2,bias_strength)

    def payoff(self,p1,q1,p2,q2):
        r1 = p1 - q1
        r2 = p2 - q2

        if q1 == 0 and q2 == 0:

            s1 = 0
            s2 = 0

        else:

            s1 = (q2*r1 + q1)/(1-(r1*r2))
            s2 = (q1*r2 + q2)/(1-(r1*r2))

        E = self.R * s1 * s2 + self.S * s1 * (1-s2) + self.T * (1-s1)*s2 + self.P * (1-s1) * (1-s2)
        return E

    @classmethod
    def classify(cls, params, state, tolerance):
        threshold = 1 - tolerance

        if state[0][0] >= threshold:
            return 0#ALLD
        elif state[0][1] >= threshold:
            return 1#ALLC
        elif state[0][2] >= threshold:
            return 2#TFT
        elif state[0][3] >= threshold:
            return 3#GTFT
        else:
            return super(ReactivePD, cls).classify(params, state, tolerance)
