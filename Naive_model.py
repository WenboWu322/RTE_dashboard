import numpy as np
import time

# 开start_time
start_time = time.time()

# parameters
alpha = 0.008  # degradation rate
T = 45         # horizon
P = 5          # number of CSEM
K = 5.993

# initialization
leakage = 0.0                    
cost = 0.0                      

# same random pressure values as in the proposed model, need to be changed arrcoding to number of CSEM
# P_init = [3.19356, 3.47594, 3.36875, 3.30340, 3.08646, 3.08645, 3.03847, 3.43451, 3.30461]       
P_init = [3.19356, 3.47594, 3.36875, 3.30340, 3.08646]    

Pressure = np.zeros((P, T))       
v = np.zeros((P, T))             
x_action = np.zeros((P, T), dtype=int)  
y_action = np.zeros((P, T), dtype=int)  
cumulative_x_sum = np.zeros((P,), dtype=int)  


for i in range(P):
    Pressure[i, 0] = P_init[i]  # initial pressure

    if P_init[i] > 3.2:  # if initial pressure is greater than 3.2

        for t in range(1, T):
            if y_action[i, t - 1] == 1:  
                Pressure[i, t] = 3.5  # if y action is taken, pressure remains constant 3.5 bar for rest of the time steps
                continue

            Pressure[i, t] = (1 - alpha) * Pressure[i, t - 1]  # pressure degradation
            
            if 3.2 - Pressure[i, t] > 0.01:   # if pressure is less than 3.2 for the first time
                if cumulative_x_sum[i] < 2:  
                    x_action[i, t] = 1  
                    Pressure[i, t:] = [3.5] * (T - t) 
                    cumulative_x_sum[i] += 1  
                else:  
                    y_action[i, t] = 1  
                    Pressure[i, t:] = [3.5] * (T - t)  
                    break  

    elif P_init[i] < 3.2: 
        y_action[i, 0] = 1  
        for t in range(1, T):
            Pressure[i, t] = 3.5  

for i in range(P):
    for t in range(T):
        v[i, t] = max(Pressure[i, t] - Pressure[i, t - 1],0) 
        leakage += K * v[i, t]
        cost += (10 * x_action[i, t]) + (100 * y_action[i, t])

# end_time
end_time = time.time()

for i in range(P):
    print(f"Initial Pressure {i}: {P_init[i]:.4f}")
    print(f"Pressure over time: {Pressure[i]}")
    print(f"x actions: {x_action[i]}")
    print(f"y actions: {y_action[i]}")
    print("-" * 50)
print(f"Total leakage: {leakage:.4f}")
print(f"Total cost: {cost:.4f}")
print(f"Execution time:{end_time - start_time:.6f} seconds")