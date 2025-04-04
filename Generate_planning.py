import pandas as pd
import random

# define caissons nb of caissons
N = 40
T = 30
caissons = caissons = [i for i in range(N)]  # Index starts from 0

# define degradation rate
random.seed(42)
# alpha_values = {caisson: round(random.uniform(0.03, 0.08), 6) for caisson in caissons}
alpha_values = {caisson: round(random.uniform(0.01, 0.05), 6) for caisson in caissons}


# generate random initial pressure values
initial_p_values = {caisson: round(random.uniform(3.2, 3.5), 2) for caisson in caissons}

# generate a random number of 'x' before 'y' for each CAISSON (from 3 to 10)
random.seed(42)  # Set seed for reproducibility
x_counts = {caisson: random.randint(3, 4) for caisson in caissons}

def generate_tasks(initial_p_values, T, alpha_values, x_counts):
    # create a dataframe to store the maintenance plan
    columns = ['INDEX', 'CSEM', 'INITIALP', 'OPERATION', 'DATE']
    data = []

    # generate maintenance plan
    index = 0  
    for caisson, initial_p in initial_p_values.items():
        current_p = initial_p
        cr_count = 0
        total_days = 0
        alpha = alpha_values[caisson]
        max_x_count = x_counts[caisson]  # Get the specific random x count for this CAISSON
        
        while total_days < T:
            # calculate the time to reach 3.2 and take the integer part
            time_to_3_2 = int((current_p - 3.2) / alpha)

            # check if the total days exceed horizon T
            if total_days + time_to_3_2 + 1 > T:
                break

            # add the maintenance plan CR/RE
            if cr_count < max_x_count:
                operation = 'x'  # CR
                data.append([index, caisson, initial_p, operation, total_days + time_to_3_2])
                cr_count += 1
                total_days += time_to_3_2 + 1  # CR task takes 1 day
                current_p = 3.5  # reset the pressure to 3.5
            else:
                operation = 'y'  # RE
                data.append([index, caisson, initial_p, operation, total_days + time_to_3_2])
                total_days += time_to_3_2 + 1  # RE task takes 1 day
                index +=1
                break  # after RE, the maintenance plan is finished
            
            index += 1

    maintenance_plan = pd.DataFrame(data, columns=columns)
    return maintenance_plan

planning = generate_tasks(initial_p_values, T, alpha_values, x_counts)
print(planning)
planning.to_csv("maintenance_plan.csv", index=False)
print("CSV file has been saved successfully")
