def senderPayoffs(cost, lSender, hSender, lReceiverProp, hReceiverProp, reveal, type):
    payoffs = [[[[[0 for x in range(4)] for x in range(4)] for x in range(3)] for x in range(3)] for x in range(3)]
    
    for lsIndex, lsMove in enumerate(payoffs):
        for msIndex, msMove in enumerate(lsMove):
            for hsIndex, hsMove in enumerate(msMove):
                for lrIndex, lrMove in enumerate(hsMove):
                    for hrIndex, payoff in enumerate(lrMove):
                        if type.lower() == 'low' or type.lower() == 'l':
                            move = lsIndex
                        elif type.lower() == 'medium' or type.lower() == 'm':
                            move = msIndex
                        elif type.lower() == 'high' or type.lower() == 'h':
                            move = hsIndex
                        else:
                            raise ValueError("Please choose a type")
                        
                        if move == 0 or move == 1:#Reveal chance irrelevant
                            if move >= lrIndex:#sender is accepted by low receivers
                                payoffs[lsIndex][msIndex][hsIndex][lrIndex][hrIndex] += lSender * lReceiverProp
                            if move >= hrIndex: 
                                payoffs[lsIndex][msIndex][hsIndex][lrIndex][hrIndex] += hSender * hReceiverProp
                        else:#Trying to hide move
                            if lrIndex == 0:#Accepting all
                                payoffs[lsIndex][msIndex][hsIndex][lrIndex][hrIndex] += lSender * lReceiverProp
                            elif lrIndex == 1 or lrIndex == 2: #Accept if a signal appears
                                payoffs[lsIndex][msIndex][hsIndex][lrIndex][hrIndex] += lSender * lReceiverProp * reveal
                                
                            if hrIndex == 0:
                                payoffs[lsIndex][msIndex][hsIndex][lrIndex][hrIndex] += hSender * hReceiverProp
                            elif hrIndex == 1 or hrIndex == 2:
                                payoffs[lsIndex][msIndex][hsIndex][lrIndex][hrIndex] += hSender * hReceiverProp * reveal                            
                             
                        if move >= 1:#Pay cost of signal
                            payoffs[lsIndex][msIndex][hsIndex][lrIndex][hrIndex] -= cost
                            
    return payoffs


def receiverPayoffs(lReveal, mReveal, hReveal, lReceiver, mReceiver, hReceiver, lSenderProp, mSenderProp, hSenderProp, type):
    payoffs = [[[[[0 for x in range(4)] for x in range(4)] for x in range(3)] for x in range(3)] for x in range(3)]
    al = lReceiver * lSenderProp
    am = mReceiver * mSenderProp
    ah = hReceiver * hSenderProp
    
    for lsIndex, lsMove in enumerate(payoffs):
        for msIndex, msMove in enumerate(lsMove):
            for hsIndex, hsMove in enumerate(msMove):
                for lrIndex, lrMove in enumerate(hsMove):
                    for hrIndex, payoff in enumerate(lrMove):
                        if type.lower() == 'low' or type.lower() == 'l':
                            move = lrIndex
                        elif type.lower() == 'high' or type.lower() == 'h':
                            move = hrIndex
                        else:
                            raise ValueError("Please choose a valid type: l or h")
                            
                        if move == 0:#Accept everyone
                            payoffs[lsIndex][msIndex][hsIndex][lrIndex][hrIndex] += al + am + ah
                        elif move == 1:#Accept signals
                            if lsIndex == 1:#Normal
                                payoffs[lsIndex][msIndex][hsIndex][lrIndex][hrIndex] += al
                            if msIndex == 1:
                                payoffs[lsIndex][msIndex][hsIndex][lrIndex][hrIndex] += am
                            if hsIndex == 1:
                                payoffs[lsIndex][msIndex][hsIndex][lrIndex][hrIndex] += ah
                                
                            if lsIndex == 2:#Hidden
                                payoffs[lsIndex][msIndex][hsIndex][lrIndex][hrIndex] += al * lReveal
                            if msIndex == 2:
                                payoffs[lsIndex][msIndex][hsIndex][lrIndex][hrIndex] += am * mReveal
                            if hsIndex == 2:
                                payoffs[lsIndex][msIndex][hsIndex][lrIndex][hrIndex] += ah * hReveal
                        elif move == 2:#Only accept hidden
                            if lsIndex == 2:#Hidden
                                payoffs[lsIndex][msIndex][hsIndex][lrIndex][hrIndex] += al * lReveal
                            if msIndex == 2:
                                payoffs[lsIndex][msIndex][hsIndex][lrIndex][hrIndex] += am * mReveal
                            if hsIndex == 2:
                                payoffs[lsIndex][msIndex][hsIndex][lrIndex][hrIndex] += ah * hReveal
                        elif move == 3:#Reject all
                            payoffs[lsIndex][msIndex][hsIndex][lrIndex][hrIndex] = 0
                        else:
                            raise ValueError("HUH?")
                                
    return payoffs