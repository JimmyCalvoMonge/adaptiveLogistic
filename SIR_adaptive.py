import numpy as np
import math
from scipy.integrate import odeint

"""
Adaptive human behavior in epidemics (Mean field version)

Baltazar Espinoza: Mathematica Version
Jimmy Calvo: Python Version

January 2024
Each function commented with the corresponding Mathematica code
"""

def B(pops, dispars, cv):

    """
    B[pops_,dispars_,cv_]:=Module[
    {
    s=pops[[1]],i=pops[[2]],r=pops[[3]],
    \[Beta]=dispars[[1]],
    cs=cv[[1]],ci=cv[[2]],cr=cv[[3]]
    },
    Return[\[Beta] cs s (ci i)/(cs s+ci i+cr r)]
    ]
    """

    s = pops[0]
    i = pops[1]
    r = pops[2]

    Beta = dispars[0]
    cs = cv[0]
    ci = cv[1]
    cr = cv[2]

    return Beta*(cs*s)*((ci*i)/(cs*s+ci*i+cr*r))


def state_odes_system(x, t, Beta, Lambda, Mu, Gamma, cs, ci, cr):

    s = x[0]
    i = x[1]
    r = x[2]

    dsdt = Lambda - B([s,i,r], [Beta, Gamma], [cs,ci,cr]) - Mu*s
    didt = B([s,i,r], [Beta, Gamma], [cs,ci,cr]) - (Gamma + Mu)*i
    drdt = Gamma*i - Mu*r

    return [dsdt, didt, drdt]


def SIRB(pops, poppars, dispars, cv, soltimes, **kwargs):

    """
    SIRB[pops_, poppars_, dispars_, cv_, soltimes_] :=
    
    (* 
    pops = vector with the S,I,R  initial conditions of the system,
    poppars = vector with the popualtion dynamics parameters \
    \[CapitalLambda] and \[Mu],
    dispars = vector with the disease parameters \[Beta] and \[Gamma],
    c = # of contacts made per disease class individuals,
    soltimes = times for the partial solution.
    *)
    
    Module[{
    s0 = pops[[1]], i0 = pops[[2]], r0 = pops[[3]], (*pops*)
    \[CapitalLambda] = poppars[[1]], \[Mu] = poppars[[2]], (*poppars*)
    \[Beta] = dispars[[1]], \[Gamma] = dispars[[2]], (*dispars*)
    cs = cv[[1]], ci = cv[[2]], cr = cv[[3]],(*c*)
    int = soltimes[[1]], fint = soltimes[[2]]
    },
    
    n = s0 + i0 + r0;
    
    (* The system *)
    ds = s'[t] == \[CapitalLambda] - B[{s[t], i[t], r[t]}, {\[Beta], \[Gamma]}, {cs, ci, cr}] - \[Mu] s[t];
    di = i'[t] == B[{s[t], i[t], r[t]}, {\[Beta], \[Gamma]}, {cs, ci, cr}] - (\[Gamma] + \[Mu]) i[t];
    dr = r'[t] == \[Gamma] i[t] - \[Mu] r[t];
    
    (* The solution *)
    solb = Flatten[
        NDSolve[{ds, di, dr, s[int] == s0, i[int] == i0, r[int] == r0},
        {s, i, r}, {t, int, fint}, MaxSteps -> 10000, 
        PrecisionGoal -> 20], 1];
    
    ]
    """

    s0 = pops[0]
    i0 = pops[1]
    r0 = pops[2]

    Lambda = poppars[0]
    Mu = poppars[1]

    Beta = dispars[0]
    Gamma = dispars[1]

    cs = cv[0]
    ci = cv[1]
    cr = cv[2]

    start = soltimes[0]
    end = soltimes[1]

    steps = kwargs.get('steps', 100)
    if steps:
        t = np.linspace(start, end, (end-start)*steps)
    else:
        t = np.arange(start, end)

    x = odeint(state_odes_system, [s0,i0,r0], t, args=(Beta, Lambda, Mu, Gamma, cs, ci, cr))

    s = x[:, 0]
    i = x[:, 1]
    r = x[:, 2]

    return s, i, r


def fs(x,y,z):
    """
    (*Single peak function*)
    fs[x_,y_,z_]:=(x z -z^2)^y
    """
    return (x*z - z**2)**y


def Maxer(pops, dispars, plan, disen):

    """
    (*The maxer code runs the optimization process*)
    Maxer[pops_,dispars_,plan_,disen_]:=Module[{
    s=pops[[1]],i=pops[[2]],r=pops[[3]],(* Populations*)
    \[Beta]=dispars[[1]],\[Gamma]=dispars[[2]], (* Disease parameters *)
    T=plan [[1]], (* Time horizon *)

    \[Nu]=plan[[2]],(* Risk perception *)
    cred=disen[[1]], (* Activity reduction of infected ones due to symptoms severity *)
    iutred=disen[[2]]  (* Reduced utility obtained while infected *)
    },

    b=48; (*maximum possible # of contacts, the optimal utility is at half of this*)
    maxcr=Floor[b/2];

    ci=maxcr (cred);
    cr=maxcr;

    \[Delta]=0.99986; (*discount factor*)
    """

    # Populations
    s = pops[0]
    i = pops[1]
    r = pops[2]

    # Disease parameters
    Beta = dispars[0]
    Gamma = dispars[1]

    T = plan[0] # Time horizon
    Nu = plan[1] # Risk perception
    Delta = plan[2] # Delta from Bellman
    maxcr = plan[3]

    cred = disen[0] # Activity reduction of infected ones due to symptoms severity
    iutred = disen[1] # Reduced utility obtained while infected

    b = maxcr*2
    ci = maxcr
    cr = maxcr

    """
    (* Vector storaging the optimal s utility *)
    EUS=Table[0,{t,1,T+1}];

    (* The individual's goal is to remain suceptible by the end of the planning horizon, so cs=maxcr=12 *)
    EUS[[-1]]=fs[b,\[Nu],maxcr];

    (* Vector storaging the optimal i utility *)
    EUI=Table[0,{t,1,T+1}];

    (* Vector storaging the optimal r utility *)
    EUR=Table[0,{t,1,T+1}];

    (* Vector storaging the optimal contact rate for susceptible individuals *)
    cs=Table[0,{i,1,T+1}];

    (* The last step is already known, the susceptible indivial wants to be susceptible at the end of the time horizon, then it is assumed that susceptible individuals have an optimized contact rate before the behavior change *)
    cs[[-1]]=maxcr;

    (* Pii=1-P^z, probability of continuing infectious, independent of the system, it's outside the loop *)
    Pii=Exp[-\[Gamma]];
    """

    # Vector storaging the optimal s utility
    EUS = [0]*(T+1)

    # The individual's goal is to remain suceptible by the end of the planning horizon, so cs=maxcr=12
    EUS[-1] = fs(b, Nu, maxcr)

    # Vector storaging the optimal i utility
    EUI = [0]*(T+1)

    # Vector storaging the optimal r utility
    EUR = [0]*(T+1)

    # Vector storaging the optimal contact rate for susceptible individuals
    cs = [0]*(T+1)

    # The last step is already known, the susceptible indivial wants to be susceptible at the end of the time horizon, 
    # then it is assumed that susceptible individuals have an optimized contact rate before the behavior change
    cs[-1] = maxcr

    # Pii=1-P^z, probability of continuing infectious, independent of the system, it's outside the loop
    Pii = math.exp(-1*Gamma)

    """
    (* Optimization procedure *)
    For[t=T,t>= 1,t--, (* Horizon times, this goes backwards *)

    (* individual's utility starts at zero: will increase*)
    maxu=0;

    (* Evaluating all the contact rates for susceptible individuals, all possible variable states *)
    For[csd=1,csd<=maxcr,csd++, 

    (* Probability of continuing susceptible, dependent on the prevalence, that's why it is inside the loop *)
    Pss=Exp[-\[Beta]((csd ci i)/(csd s+ci i+maxcr r))];

    (* Evaluating the expected utilities *)

    (*Utility gained by being currently susceptible*)
    EUS[[t]]= fs[b,\[Nu],csd] +\[Delta](Pss EUS[[t+1]]+(1-Pss) EUI[[t+1]]);

    (*Utility gained by being currently Infected*)
    EUI[[t]]=iutred fs[b,\[Nu],ci] +\[Delta](Pii EUI[[t+1]]+(1-Pii) EUR[[t+1]]);

    (*Utility gained by being currently Recovered*)
    EUR[[t]]=fs[b,\[Nu],cr] +\[Delta] EUR[[t+1]];

    (*Compare the new utility with the previous one, if it's bigger, then adapt the behavior *)
    If[EUS[[t]]>maxu,cs[[t]]=csd;maxu=EUS[[t]]]

    ]
    ];
    Clear[t];
    Return[cs]
    ]
    """

    # Optimization procedure

    # Horizon times, this goes backwards
    for t in range(T-1, -1, -1):
        
        # individual's utility starts at zero
        maxu = 0

        # Evaluating all the contact rates for susceptible individuals, all possible variable states
        for csd in range(1,maxcr+1):

            # Probability of continuing susceptible, dependent on the prevalence, that's why it is inside the loop
            Pss = math.exp(-1*Beta*(csd*ci*i)/(csd*s + ci*i + maxcr*r))

            if Pss<0.75:
                print('Here', Pss)

            # Evaluating the expected utilities

            # Utility gained by being currently susceptible
            EUS[t] = fs(b,Nu,csd) + Delta*(Pss*EUS[t+1] + (1-Pss)*EUI[t+1])

            # Utility gained by being currently Infected
            EUI[t] = iutred*fs(b,Nu,ci) + Delta*(Pii*EUI[t+1] + (1-Pii)*EUR[t+1])

            # Utility gained by being currently Recovered
            EUR[t] = fs(b,Nu,cr) + Delta*EUR[t+1]

            # Compare the new utility with the previous one, if it's bigger, then adapt the behavior
            if EUS[t] > maxu:
                cs[t] = csd
                maxu = EUS[t]

    return cs[0]


def RunAdaptiveSim(s0, i0, r0,
                   Lambda, Beta, Gamma, 
                   T, Nu, Delta,
                   maxcr, cred, iutred,
                   tft, DeltaT):

    """
    (* =-=-=-=-=-= The behavior epidemic model =-=-=-=-=-=-= *)

    (*Population parameters*)
    \[CapitalLambda]=0;\[Mu]=\[CapitalLambda]/n;

    (* Disease parameters *)
    \[Beta]=0.01; \[Gamma]=1/9.;

    (*Optimization parameters*)
    T=14;
    \[Nu]=0.1;
    maxcr=24;
    cred=1;
    iutred=0.;

    (* Initial conditions *)
    s0=9999;
    i0=1;
    r0=0;
    n=s0+i0+r0;

    vars={s[t],i[t],r[t]};

    (* Final time *)
    tft=200; 

    (*Timestep lengths*)
    \[CapitalDelta]t=1;

    (* Intervention times *)
    times=Table[i,{i,0,tft,\[CapitalDelta]t}];
    dim=Length[times];

    (* Infected and recovered contacts rates (fixed) *)
    ci=Floor[maxcr(cred)];cr=maxcr;

    (*Predefine lists*)
    ctime=Table[0,{k,0,tft}];

    ctime[[1]]=maxcr;
    """

    n = s0 + i0 + r0
    Mu = Lambda/n
    s, i, r = [s0], [i0], [r0]

    # Intervention times
    times = range(0, tft + 1, DeltaT)
    dim = len(times)

    # Infected and recovered contacts rates (fixed) *)
    ci = int(maxcr)
    cr = int(maxcr)

    # Predefine lists 
    ctime = [0]*(tft+1)
    ctime[0] = maxcr

    """
    (* Solutions loop *)

    popstime={};

    For[k=1,k<dim,k++,

    AppendTo[popstime,{s0,i0,r0}];

    (*=-=-=-=-=-=-=-=-=-=-=-= Disease dynamics =-=-=-=-=-=-=-=-=-=-=-=*)

    (* SIR solver under current system state *) 
    SIRB[{s0,i0,r0},{\[CapitalLambda],\[Mu]},{\[Beta],\[Gamma]},{ctime[[k]],ci,cr},{times[[k]],times[[k+1]]}];

    (* Updating population's distribution *)
    {s0,i0,r0}=Flatten[Table[#/.solb,{t,times[[k+1]],times[[k+1]]}]&/@vars];

    (* Updating susceptible individual's contacts *)
    cst=Maxer[{s0,i0,r0},{\[Beta],\[Gamma]},{T,\[Nu]},{cred,iutred}][[1]];

    ctime[[k+1]]=cst

    ]

    """

    # Solutions loop

    for k in range(dim-1):

        # =-=-=-=-=-=-=-=-=-=-=-= Disease dynamics =-=-=-=-=-=-=-=-=-=-=-=*)

        # SIR solver under current system state 
        s_new, i_new, r_new = SIRB([s[-1], i[-1], r[-1]], 
                                   [Lambda, Mu], 
                                   [Beta, Gamma], 
                                   [ctime[k], ci, cr], 
                                   [times[k], times[k+1]])

        # Updating population's distribution *)
        s.append(s_new[-1])
        i.append(i_new[-1])
        r.append(r_new[-1])

        # Updating susceptible individual's contacts *)
        cst = Maxer([s[-1], i[-1], r[-1]], [Beta, Gamma], [T, Nu, Delta, maxcr], [cred, iutred])

        ctime[k+1] = cst

    return s, i, r, ctime


if __name__ == '__main__':

    # Initial conditions
    s0 = 9999
    i0 = 1
    r0 = 0

    # Population parameters
    Lambda = 0 # Mortality rate is mu = Lambda/N

    # Disease parameters
    Beta = 0.01
    Gamma = 1/9

    # Optimization parameters
    T = 14 
    Nu = 0.1
    Delta = 0.99986
    maxcr = 24
    cred = 1
    iutred = 0

    # Final time 
    tft = 200 
    # Timestep lengths 
    DeltaT = 1

    s, i, r, ctime = RunAdaptiveSim(s0, i0, r0,
                   Lambda, Beta, Gamma, 
                   T, Nu, Delta,
                   maxcr, cred, iutred,
                   tft, DeltaT)

    print(ctime)