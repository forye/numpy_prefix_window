'''
Created on Jul 29, 2014

@author: Idan Fonea
'''

if __name__ == '__main__':
    pass

import numpy
#for local varrience
from numpy import zeros,floor,isnan,nanmean,nanstd
from pandas import DataFrame,Series,ols#, Series
import statsmodels.api as sm

from src.tools.STATICS import FEATURE_WINDOW

import matplotlib.pyplot as plt  


def get_mean_window(z,W=FEATURE_WINDOW,Wh=0,interval=60):
    
    '''
    was 'super_advanced_mean'
    as fast as convoulute...
    but can determain the location of the window as well (Wh is the future samples window)
    '''
    W= W*60/interval
    Wh = int(Wh*60/interval)
    Wl=int(W-Wh)
    mask_tag=numpy.isnan(z)#o(n)
    mask=~mask_tag# only true values #o(n)
    WlZs=numpy.array([0 for _ in range(Wl)])#o(Wl)
    WhZs=numpy.array([0 for _ in range(Wh )])#o(Wh)
    x=numpy.where(mask_tag,0.,z)    
    x=numpy.append(WhZs, x)#o(n+Wl)
    x=numpy.append(x,WlZs)#o(n+Wh_Wl)
    y = numpy.cumsum(numpy.roll(x,-Wh)-numpy.roll(x,Wl))#[Wl-1:len(x)-Wh]
    mask1=numpy.append(WhZs, mask)#o(n+Wl)
    mask1=numpy.append(mask1,WlZs)#o(n+Wh_Wl)
    Ws=numpy.cumsum(numpy.roll(mask1,-Wh))-numpy.cumsum(numpy.roll(mask1,Wl))
    y=y/Ws
    if numpy.any(mask_tag):
        y[Wl-1:len(x)-Wh][mask_tag]=numpy.NaN
    return y[Wh:len(y)-Wl]

def get_local_variance(z,mu,W=FEATURE_WINDOW,Wh=0,interval=60):

    '''
    based 'super_advanced_mean'
    as fast as convolute...
    but can determain the location of the window as well (Wh is the future samples window)
    
    apply a mean on a mooving window
    w is the time window in [Minutes] 
    
    '''
    W= W*60/interval
    Wh = int(Wh*60/interval)
    Wl=int(W-Wh)
    mask_tag=numpy.isnan(z)#o(n)
    mask=~mask_tag# only true values #o(n)
    WlZs=numpy.array([0 for _ in range(Wl)])#o(Wl)
    WhZs=numpy.array([0 for _ in range(Wh )])    #o(Wh)
    x=numpy.where(mask_tag,0.,z)
    x=numpy.append(WhZs, x)#o(n+Wl)
    x=numpy.append(x,WlZs)#o(n+Wh_Wl)#[Wl-1:len(x)-Wh]
    mu=numpy.append(WhZs, mu)#o(n+Wl)
    mu=numpy.append(mu,WlZs)
    mask1=numpy.append(WhZs, mask)#o(n+Wl)
    mask1=numpy.append(mask1,WlZs)#o(n+Wh_Wl)    
    Ws=(numpy.cumsum(numpy.roll(mask1,-Wh))-numpy.cumsum(numpy.roll(mask1,Wl)))
    y1= (numpy.cumsum(numpy.roll(x,-Wh))-numpy.cumsum(numpy.roll(x,Wl)))
    x2=x**2
    y2 = (numpy.cumsum(numpy.roll(x2,-Wh))-numpy.cumsum(numpy.roll(x2,Wl)))
    y = y2 - 2* mu* y1
    y=y/Ws
    y=y+mu**2
    if numpy.any(mask_tag):
        y[Wl-1:len(x)-Wh][mask_tag]=numpy.NaN
    return numpy.sqrt(y[Wh:len(y)-Wl])


def get_local_variation(z,W=FEATURE_WINDOW,Wh=0,interval=60):

    '''
    based 'super_advanced_mean'
    as fast as convolute...
    but can determain the location of the window as well (Wh is the future samples window)
    
    apply a mean on a mooving window
    w is the time window in [Minutes] 
    
    '''
    W= W*60/interval
    Wh = int(Wh*60/interval)
    Wl=int(W-Wh)
    z_tag=numpy.roll(z,-1)
    z_tag[-1]=numpy.NaN
    y=z-z_tag
    mask_tag=numpy.isnan(y)#o(n)
    mask=~mask_tag# only true values #o(n)
    WlZs=numpy.array([0 for _ in range(Wl)])#o(Wl)
    WhZs=numpy.array([0 for _ in range(Wh )])    #o(Wh)
    x=numpy.where(mask_tag,0.,y)
    x=numpy.append(WhZs, x)#o(n+Wl)
    x=numpy.append(x,WlZs)#o(n+Wh_Wl)#[Wl-1:len(x)-Wh]
    mask1=numpy.append(WhZs, mask)#o(n+Wl)
    mask1=numpy.append(mask1,WlZs)#o(n+Wh_Wl)    
    Ws=(numpy.cumsum(numpy.roll(mask1,-Wh))-numpy.cumsum(numpy.roll(mask1,Wl)))
    x1 = numpy.absolute(x)
    x2 = numpy.square(x)
    y1= (numpy.cumsum(numpy.roll(x1,-Wh))-numpy.cumsum(numpy.roll(x1,Wl)))
    y2= (numpy.cumsum(numpy.roll(x2,-Wh))-numpy.cumsum(numpy.roll(x2,Wl)))
    y1=y1/Ws
    y2=y2/Ws
    if numpy.any(mask_tag):
        y1[Wl-1:len(x)-Wh][mask_tag]=numpy.NaN
        y2[Wl-1:len(x)-Wh][mask_tag]=numpy.NaN
    return numpy.sqrt(y2[Wh:len(y)-Wl]),y1[Wh:len(y)-Wl]




def get_local_trend( x , w , wh=0, timeInterval = 60):
    '''
    regress liniarly on a window of samples returning the found bias and slope
    '''    
     
    w= int(w*60/timeInterval)
    beta=numpy.random.rand(numpy.size(x))/16
    alpha=numpy.random.rand(numpy.size(x))/16
    x_0 = numpy.ones(w)
    x_1=numpy.array(range(w))
    if w>len(x):
        return None    
    wl =int(w-wh)
    wh=int(wh)            
    alpha[0]=numpy.NaN
    beta[0]=x[0]
    for n in range(1,wl):
        h=n            
        if isnan(x[n]) and h!=0:            
            alpha[n]=numpy.NaN
            beta[n]=numpy.NaN
        else:
            mask=~numpy.isnan(x[:n+1])
            inputLeft = numpy.vstack([numpy.array(range(n+1))[mask],numpy.ones(n+1)[mask]]).T
            alpha[n],beta[n]= numpy.linalg.lstsq(inputLeft,x[:n+1][mask])[0]
    for n in range(wl+1,len(x)-wh):
        h=wl+wh            
        if isnan(x[n]):            
            alpha[n]=numpy.NaN
            beta[n]=numpy.NaN
        else:
            mask = ~isnan(x[n-wl+1:n+wh+1])
            inputcenter = numpy.vstack([x_1[mask],x_0[mask]]).T
            alpha[n],beta[n]= numpy.linalg.lstsq(inputcenter,x[n-wl+1:n+wh+1][mask])[0]            
            h = sum(~mask)            
    for n in range((len(x)-wh+1),len(x)):
        h=len(x) - n
        if isnan(x[n]):            
            alpha[n]=numpy.NaN
            beta[n]=numpy.NaN
        else:
            mask=~numpy.isnan(x[n-wl+1:len(x)])
            inputRight = numpy.vstack([numpy.array(range(len(x)+wl-1-n))[mask],numpy.ones(len(x)+wl-1-n)[mask]]).T
            alpha[n],beta[n]= numpy.linalg.lstsq(inputRight,x[n-wl+1:len(x)][mask])[0]
    return alpha,beta

def get_local_trend_standard_error( x , w ,alpha,beta, wh=0, timeInterval = 60):
    '''
    caculate the standard error of a linear 1 variable regression model
    given alpha, bete result and window size (also slope and time interval) return the standart error vector    '''    
     
    w= int(w*60/timeInterval)
    se=numpy.zeros(numpy.size(x))
    x_0 = numpy.ones(w)
    x_1=numpy.array(range(w))
   
    wh=int(wh)
    wl =int(w-wh)            

    for n in range(1,wl):
        h=n            
        if isnan(x[n]) and h!=0:
            se[n]=numpy.NaN
        else:
            mask=~numpy.isnan(x[:n+1])
#             inputLeft = numpy.vstack([numpy.array(range(n+1))[mask],numpy.ones(n+1)[mask]]).T
#             se[n] = numpy.sum(numpy.dot( numpy.array([alpha[n],beta[n]]),inputLeft)-x[:n+1][mask])/numpy.sum(~mask)            
#             alpha[n],beta[n]= numpy.linalg.lstsq(inputLeft,x[:n+1][mask])[0]
#             se[n] = numpy.ones(n+1)[mask][mask]*beta[n]+numpy.array(range(n+1))[mask][mask]*alpha[n]-x[:n+1][mask]/numpy.sum(~mask)
            se[n] = numpy.linalg.norm(numpy.ones(n+1)[mask][mask]*beta[n]+numpy.array(range(n+1))[mask][mask]*alpha[n]-x[:n+1][mask])
    for n in range(wl+1,len(x)-wh):
        h=wl+wh            
        if isnan(x[n]):   
            se[n]=numpy.NaN
        else:
            mask = ~isnan(x[n-wl+1:n+wh+1])
#             inputcenter = numpy.vstack([x_1[mask],x_0[mask]]).T
#             se[n] = x_0[mask]*beta[n]+x_1[mask]*alpha[n]-x[n-wl+1:n+wh+1][mask]/numpy.sum(~mask)
            se[n] = numpy.linalg.norm(x_0[mask]*beta[n]+x_1[mask]*alpha[n]-x[n-wl+1:n+wh+1][mask])
#             se[n] = numpy.sum(numpy.dot( numpy.array([alpha[n][mask],beta[n][mask]]),inputcenter)-x[n-wl+1:n+wh+1][mask])/numpy.sum(~mask)
    for n in range((len(x)-wh+1),len(x)):
        h=len(x) - n
        if isnan(x[n]):            
            se[n]=numpy.NaN
        else:
            mask=~numpy.isnan(x[n-wl+1:len(x)])
#             inputRight = numpy.vstack([numpy.array(range(len(x)-n)),numpy.ones(len(x)-n)]).T
#             se[n] = numpy.sum(numpy.dot( numpy.array([alpha[n],beta[n]]),inputRight)-x[n-wl+1:len(x)][mask])/numpy.sum(~mask)
#             se[n] = numpy.ones(len(x)-n)[mask]*beta[n]+numpy.array(range(len(x)-n))[mask]*alpha[n]-x[n-wl+1:len(x)]/numpy.sum(~mask)
            se[n] = numpy.linalg.norm(numpy.ones(len(x)+wl-1-n)[mask]*beta[n]+numpy.array(range(len(x)+wl-1-n))[mask]*alpha[n]-x[n-wl+1:len(x)][mask])
            
#             inputRight = numpy.vstack([numpy.array(range(len(x)+wl-1-n))[mask],numpy.ones(len(x)+wl-1-n)[mask]]).T
#             alpha[n],beta[n]= numpy.linalg.lstsq(inputRight,x[n-wl+1:len(x)][mask])[0]
    return se




#calculate_standard_error_of_local_terend

# make qq plot
#count_errors et/se>1.65
# count et positives and negative (are close to w/2)

#self regress

#calculate_standard_error_of_self_regress

#calculate shock index

#event Identifier 1,2,3,4

# count events

#fft window
# get p fraction

# clean_untrend siganl
# get Disperrsion Dp

def make_qq_plot(x,mtitle='QQ Plot'):
    '''
    gets a vector and an optional graph title 
    and plots its qq plot
    '''    
    sm.qqplot(x)
    plt.title(mtitle)
    plt.show()
    return True



def get_p_value(x,p):
    '''
    calculates the p value from a vector x. returns x(p).
    returns x(p),p value reocurences,p 
    '''
    p_tag=int(p*1000)    
    hist,bin_edges =numpy.histogram(x, bins=1000,density =True)
    return [bin_edges[p_tag],hist[p_tag],p_tag]


def get_Dp(x,p):
    '''
    needs  get_p_value
    '''
    a= get_p_value(x, p)[0] - get_p_value(x, 1-p)[0]
    if numpy.isnan(a) or not x.any():
        Dp =numpy.NaN
    else:
        Dp=a
    return Dp

def get_p_value_window(x,W,p,Wh=0):
    '''
    gets the p value on window
    '''
    P=numpy.empty_like(x)
    Wl=int(W-Wh)
    Wh = int(Wh)
    P[0]=x[0]    
    acc=0
    for w in range(1,len(x)):
        if w<Wl:
            mask=~numpy.isnan(x[:w])
            P[w] = get_p_value(x[:w][mask],p)[0]
        elif w<len(x)-Wh:
            mask=~numpy.isnan(x[w-Wl:w+Wh-1])
            P[w] = get_p_value(x[w-Wl:w+Wh-1][mask],p)[0]
            acc+=1            
        else:
            mask=~numpy.isnan(x[w:])
            P[w] = get_p_value(x[w:][mask],p)[0]
    pass
    return P


def get_Dp_value_window(x,W,p,Wh=0):
    '''
    gets the p value on window
    '''
    Dp=numpy.empty_like(x)
    Wl=int(W-Wh)
    Wh = int(Wh)
    Dp[0]=x[0]
    
    acc=0
    for w in range(1,len(x)):
        if w<Wl:
            mask=~numpy.isnan(x[:w])
            Dp[w] = get_Dp(x[:w][mask],p)
        elif w<len(x)-Wh:
            mask=~numpy.isnan(x[w-Wl:w+Wh-1])
            Dp[w] = get_Dp(x[w-Wl:w+Wh-1][mask],p)
            acc+=1
            if acc==500:
                acc=0
            
        else:
            mask=~numpy.isnan(x[w:])
            Dp[w] = get_Dp(x[w:][mask],p)
    pass
    return Dp

def Vector2RooledDF(x,W,Wh=0):
    '''
    this gets a vector and a window and returns a dataframe that consists of this vector
     moved by one sample and padded by nans in the size of the window
     example:
     x=[1,2,3]
     W=3
     Wh=floor(W/2)

     DF=[[nan,nan,1,2,3],[nan,1,2,3,nan],[1,2,3,nan,nan]]
     
    w times memmory writing!!!
    '''
    Vs=numpy.empty(shape=(W,len(x)))
    Wl=W-Wh
    for w in range(W):
        #append from the right (first samples)
        temp=numpy.array(x) 
        if w<Wl:
            for _ in range(int(w)):

                temp=numpy.roll(temp, -1)
                temp[-1]=numpy.NaN
        elif w>Wl:
        #append from the right (last samples)
            for _ in range(int(w- Wl)):
                temp=numpy.roll(temp, 1)
                temp[0]=numpy.NaN
        Vs[w]=temp
    return Vs      

    
def pandas_self_regress_ex(x,W,Wh=0,minPeriods=2):
    '''
    3:1 pandas ols examle with min periods
    import pandas as pd
    MIN PERIODS!!=2
    xand y  are same length. w is the ampunt of x values as explainatory variables
    first variable is ones! (dc)
    y is first coloum and x is the folowing coloums of self regressed    
    checkout: pandas.stats.ols.MovingOLS
    '''
    mX=Vector2RooledDF(x,W,Wh)
    ones=numpy.ones(len(x))
    VS=numpy.vstack([ones,mX[1:,:]]).T
    X=DataFrame((VS))
    Y = Series(mX[0,:])

    a = ols(y = Y, x = X, min_periods = minPeriods)
    return a


def pandas_self_regress_window(x,W,D):
    '''
    perfroming self ordinary linear regression on a window (item by item
    D=number of past variables
    W= size of past window
    x=input vector
    return len(x)XD array of coefficients. where the first relates to the DC (ones vector) 
    
    '''
    regs=[]
    betas=[]
    nanis=[numpy.NaN]
    NaNBetaVal=Series(nanis*D)
    for n in range(W):
        betas.append(NaNBetaVal)
    for n in range(len(x)-W):
        try:
            regs.append(pandas_self_regress_ex(x[n:n+W],D,Wh=0,minPeriods=2))
            betas.append(regs[-1].beta)
        except:
            print n+W
            print x[n:n+W]
            betas.append(NaNBetaVal)
    return betas


