
def p1Payoffs(a, c, w, p2lProp, p2hProp, type):
    payoffs = [[[[0 for x in range(5)] for x in range(5)] for x in range(4)] for x in range(4)]
    
    for p1lIndex, p1lMove in enumerate(payoffs):
        for p1hIndex, p1hMove in enumerate(p1lMove):
            if type.lower() == 'low':
                move = p1lMove
            elif type.lower() == 'high':
                move = p1hMove
            else:
                raise ValueError('Please enter a valid p1 type')
                
            for p2lIndex, p2lMove in enumerate(p1hMove):
                for p2hIndex, p2hMove in enumerate(p2lMove):
                    payoffFromLow = 0
                    payoffFromHigh = 0
                    
                    if p2lIndex == 0:
                        payoffFromLow = 
                    
