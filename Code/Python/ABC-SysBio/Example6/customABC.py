from numpy import *

def get_ES(ts):

    tO1 = [95,96,97,98,99]
    tO2 = [196,197,198,199,200]
    tSig = 100

    O1 = mean(ts[tO1])
    O2 = mean(ts[tO2])
    Omax = max( ts[tSig:] )
    Omin = min( ts[tSig:] )

    skip = False

    # check that we reached steady state before and after
    if sqrt(var(ts[tO1])) > 0.01*abs(O1) :
        #print "O1 not steady state"
        skip = True
    if sqrt(var(ts[tO2])) > 0.01*abs(O2) :
        #print "O2 not steady state"
        skip = True

    if abs(Omax-O1) > abs(Omin-O1):
        Op = Omax
        imax = argmax(ts[tSig:])
        O_next_min = min( ts[imax:] )
        if abs(O_next_min-O1) > 0.5 *abs(Op-O1):
            skip = True
            #print "skipped +ve", Omax, Omin
            
    else:
        Op = Omin
        imin = argmin(ts[tSig:])
        O_next_max = max( ts[imin:] )
        if abs(O_next_max-O1) > 0.5 *abs(Op-O1):
            skip = True
            #print "skipped -ve", Omin, Omax

    if skip == False:
        # (I2-I1)/I1 = 0.2
        E = abs(O2-O1)/(0.2*O1)
        S = abs(Op-O1)/(0.2*O1)
    else :
        E = None
        S = None

    return [E, S]

def distance(data1, data2, parameters, model):
    # data1 is simulated, and has shape npoints x beta
    # data2 is real

    #times = np.arange(0,200.1,0.1)
    d1 = None
    d2 = None

    E,S = get_ES(data1[:,0])

    d1 = E
    if S > 0 : d2 = 1/S

    #print [O1, O2, Op, d1, d2]
    return [d1, d2]


