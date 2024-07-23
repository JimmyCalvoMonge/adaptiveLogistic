import numpy as np
import math

"""
Adaptive human behavior in epidemics (Mean field version)

Baltazar Espinoza: Mathematica Version
Jimmy Calvo: Python Version

January 2024
Each function commented with the corresponding Mathematica code
"""

def solve_logistic_system(p_0, t_0, t_1, k, r):
    """
    Logistic Growth equation is deterministic
    We give the solution vector betweeen x_start and x_end.
    That is we give [P(x_start), P(x_end)] solution for the system:

    dP/dt = rP(1-P/k)
    P(x_start) = p_0

    The solution is

    p(t) = kp_0e^{r(t-t_0)}/(k + p_0(e^{r(t-t_0)} - 1))
    We give the vector p(t_0), p(t_1) containing 100 values.
    """
    result = (k*p_0*math.exp(r*(t_1-t_0)))/(k + p_0*(math.exp(r*(t_1-t_0)) - 1)) 
    return result


def dynamic_programming_clark(T, M, c, xc, beta, alpha, lambda_, Y, x_0):
    """
    Example from Clark:

    # System Parameters

    T = 20 # Iterations
    M = 3 # Number of patches
    c = 10 # Maximum energy
    xc = 3 # Minimum energy for survival

    # Patch parameters

    beta = [0, 0.004, 0.02] # Patch probability of predation
    alpha = [1, 1, 1] # Patch foraging cost
    lambda_ = [0, 0.4, 0.6] # Patch probability of finding food
    Y = [0, 3, 5] # Increment in the state variable given that food is found 

    # Initial energy level

    x_0 = 4
    """

    f1 = [0]*xc + [1]*(c-xc) # Final survival proabilities
    # At t=T we only care if the forager is alive or dead
    # In this case all values x>xc are equally valuable

    fall = [] # Matrix to store all variable states
    dall = [] # Matrix to store all patch selections\

    def chop(x, a, b):
        # Energy store function
        if x < a:
            return a
        elif x>=a and x<=b:
            return x
        elif x > b:
            return b
        
    for _ in range(T - 1, 0, -1): # Times loop

        f0 = [0]*c # Survival probability state
        d = [0]*c #  Store the optimal patch index

        for x in range(xc, c): # Energy loop

            v = [0]*M # Store the patches survival probabilities on each iteration
            for j in range(M): # Patches loop

                # Evaluate survival probability for each patch
                v[j] = (1-beta[j])*(
                    lambda_[j] * f1[chop(x - alpha[j] + Y[j], xc - 1, c - 1)] + 
                    (1-lambda_[j]) * f1[chop(x-alpha[j], xc - 1, c - 1)]
                )

            #  Get the maximum survival probability 
            # value for each energy level x
            vm = np.max(v)

            # Store the maximum survival probability 
            # value for each energy level x
            f0[x] = round(vm, 3)

            # Get the patch that maximizes survival, 
            # the optimal decision at time t, or each energy level x
            d[x] = v.index(vm) + 1

        fall = fall + [f0] # Store the vector with the maximum 
        # survival probabilities at time t in the matrix fall
        dall = dall + [d] # Store the vector with the optimal
        # patch deicision at time t in the matrix dall

        # Update the survival probabilities
        f1 = f0

    fopt = np.transpose(np.array([ff[xc:] for ff in fall])).tolist()
    dopt = np.transpose(np.array([dd[xc:] for dd in dall])).tolist()

    x_vector = [x_0] # State variable

    fopttraj = [] # Variable state
    dopttraj = [] # Optimal patch index

    t = 0

    if xc <= x_0 and x_0 <= c:

        while t < T - 1:

            fopttraj = fopttraj + [fopt[x_vector[-1] - xc - 1][T - 2 - t]]
            dopttraj = dopttraj + [dopt[x_vector[-1] - xc - 1][T - 2 - t]]
            
            x_vector.append(
                chop(x_vector[-1] - alpha[dopttraj[-1] - 1] + Y[dopttraj[-1] - 1],
                     xc, c))

            t = t + 1

    return x_vector, fopttraj, dopttraj


def find_params_1(M, p_0, k, ft_cost, ft_prob_find_food, v_dig):
    
    p0 = p_0/k
    
    beta = [0]*M 
    # Patch (Foraging Time) probability of predation

    alpha = [int(ft_cost*i) + 1 for i in range(1, M+1)]
    # Patch (Foraging Time) foraging cost, depends on foraging time

    lambda_ = [round(ft_prob_find_food*i/p0, 3) for i in range(1, M+1)]
    # Patch (Foraging Time) probability of finding food. Decreasing in p, increasing in foraging time.

    Y = [int(v_dig*i) for i in range(0, 2*M, 2)]
    
    return beta, alpha, lambda_, Y


def find_energy_dynamic(p_0, k, energy_start, params):
    """
    Implements dynamic programming method to find the
    optimal energy and foraging time decision for each day.
    Depends on the current population density and parameters for each foraging time
    option.
    """

    # Call dynamic_programming_clark with appropiate parameters
    # Obtain the second energy level from this iteration

    # System Parameters

    T = params['T']
    M = params['M']
    c = params['c']
    xc = params['xc']
    ft_cost = params['ft_cost'] # cost of foraging
    ft_prob_find_food = params['ft_prob_find_food'] # gain food from foraging
    v_dig = params['v_dig'] # Velocity of digestion

    beta, alpha, lambda_, Y = find_params_1(M, p_0, k, ft_cost, # cost of foraging
                  ft_prob_find_food,  # gain food from foraging
                  v_dig # Velocity of digestion
                  )

    x_vector, _, dopttraj = dynamic_programming_clark(T, M, c, xc,
                                                      beta, alpha, lambda_, Y,
                                                      x_0=energy_start)
    if len(x_vector) > 1:
        final_energy_day = x_vector[1]
        final_foraging_day = dopttraj[1]
    else:
        final_energy_day = energy_start
        final_foraging_day = 0

    return final_energy_day, final_foraging_day


def adaptive_logistic(p_0, k, t_max, e_start, r_c, params, **kwargs):
    """
    Resolve logistic equation each day
    Update the r parameter daily by computing the
    decided energy to spend this day (this energy is obtained using a
    dynamic programming approach based on Clark and Mangel
    Dynamic Programming in Behavioral Ecology).
    """

    # Decide wether to use classical model or adaptive
    adaptive = kwargs.get('adaptive', True)
    p_vector = [p_0]

    energy_day = e_start # Initialize with max energy
    r_day = r_c*energy_day

    energy_vector = []
    r_vector = []
    foraging_vector = []

    for t in range(t_max):

        # Solve the logistic for one day.
        p_range_day = solve_logistic_system(p_vector[-1], t, t+1, k, r_day)

        if adaptive:

            # Obtain new energy consumption for this day using dynamic programming
            energy_day, foraging_day = find_energy_dynamic(
                p_vector[-1], k, energy_day, params)

            # Change the growth rate using this new energy.
            r_day = r_c*energy_day

            # Append energy, growth rate and foraging time to vector.
            energy_vector.append(energy_day)
            r_vector.append(r_day)
            foraging_vector.append(foraging_day)

        # Append solution for this day.
        p_vector.append(p_range_day)

    # Return all results
    return p_vector, energy_vector, r_vector, foraging_vector


if __name__ == '__main__':

    print("Running Adaptive Logistic experiments ...")