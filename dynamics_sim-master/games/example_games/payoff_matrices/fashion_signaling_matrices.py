accept = [[0, 1, 2, 4], [0, 1, 3, 5], [0, 2, 3, 6]]
#Three possible moves for sender, inside each array the receivers move determines if accepted

def senderPayoffs(normalCost, hiddenCost, lSender, hSender, lReceiverProp, hReceiverProp, type):
     
    payoffs = [[[[0 for x in range(8)] for x in range(8)] for x in range(3)] for x in range(3)]           
    
    for LSindex, lowSMove in enumerate(payoffs):
        for HSindex, highSMove in enumerate(lowSMove):
            for LRindex, lowRMove in enumerate(highSMove):
                for HRindex, payoff in enumerate(lowRMove):
                    if type.lower() == 'low':
                        strat = LSindex
                    elif type.lower() == 'high':
                        strat = HSindex
                    else:
                        raise ValueError("Choose a type for the sender!")
                        
                    if HRindex in accept[strat]:#Accepted by high receivers
                        payoffs[LSindex][HSindex][LRindex][HRindex] += hSender * hReceiverProp
                    if LRindex in accept[strat]:#Accepted by low receivers
                        payoffs[LSindex][HSindex][LRindex][HRindex] += lSender * lReceiverProp 
                        
                    if strat == 1:#Pay cost of normal signal
                        payoffs[LSindex][HSindex][LRindex][HRindex] -= normalCost
                    elif strat == 2:#Pay cost of hidden signal
                        payoffs[LSindex][HSindex][LRindex][HRindex] -= hiddenCost
   
    return payoffs


def receiverPayoffs(lReceiver, hReceiver, receiverCost, lSenderProp, hSenderProp, type):

    payoffs = [[[[0 for x in range(8)] for x in range(8)] for x in range(3)] for x in range(3)]

    for LSindex, lowSMove in enumerate(payoffs):
        for HSindex, highSMove in enumerate(lowSMove):
            for LRindex, lowRMove in enumerate(highSMove):
                for HRindex, payoff in enumerate(lowRMove):
                    if type.lower() == 'low':
                        accepting = LRindex
                    elif type.lower() == 'high':
                        accepting = HRindex
                    else:
                        raise ValueError("Choose a type for the receiver!")
                    
                    if accepting in accept[LSindex]:#Accepting the low receiver
                        payoffs[LSindex][HSindex][LRindex][HRindex] += lReceiver * lSenderProp
                    if accepting in accept[HSindex]:#Accepting the high receiver
                        payoffs[LSindex][HSindex][LRindex][HRindex] += hReceiver * hSenderProp
                    if accepting in accept[2]:#Pay cost of investing to recognize the hidden signals
                        payoffs[LSindex][HSindex][LRindex][HRindex] -= receiverCost          
    print(payoffs[0][1][7][1])
    print(payoffs[0][1][5][1])
    return payoffs