import sys
sys.path.append('C:\\Program Files\\IBM\\ILOG\\CPLEX_Studio210\\cplex\\python\\20.1\\x64_win64')
sys.path.append("C:\\ProgramData\\anaconda3\\Lib\\site-packages")
import numpy as np
import pandas as pd
from docplex.mp.model import Model
import matplotlib.pyplot as plt
import numpy as np
import random

# set the parameters
P = 10   # NB OF CAISSONS
T = 60
gamma_cr = 3
gamma_re = 2
bcr = 1
bre = 3
r_cr = 3
r_re = 5
Rcr = 50
Rre = 50
Cmax = 3
a = 3


caissons = [i for i in range(P)]  # all the caissons
# define degradarion rate
random.seed(42)
alpha_values = {i: round(random.uniform(0.01, 0.05), 6) for i in range(P)}
# generate random initial pressure values
initial_p_values = {caisson: round(random.uniform(3.2, 3.5), 2) for caisson in caissons}


# define the window size
window_size = 14

# create the grouping model
def clustering_CR(df_x_y, window_size, window_start):

    N = df_x_y.shape[0]
    t = df_x_y["DATE"]
    
    mdlgp = Model('Maintance_grouping')
    M = 14

    # define the variables
    theta = mdlgp.binary_var_matrix(N, N, name = 'theta')
    center = mdlgp.binary_var_matrix(N, window_size, name="center")
    cluster = mdlgp.binary_var_list(N, name = 'cluster')
    o = mdlgp.integer_var_list(N, name = 'o')
    z = mdlgp.binary_var_matrix(N, T, name = 'z')

    mdlgp.minimize(mdlgp.sum(cluster[k] for k in range(N)))
    
    mdlgp.add_constraints(theta[n, k] <= cluster[k] for n in range(N) for k in range(N))
    mdlgp.add_constraints(sum(theta[n, k] for k in range(N)) == 1 for n in range(N))
    
    # Maximum number of tasks limit (group capacity constraint) 
    mdlgp.add_constraints(sum(theta[n, k] for n in range(N)) <= Cmax for k in range(N))
    
    # Each cluster can only have one center:
    mdlgp.add_constraints(sum(center[k, t] for t in range(window_size)) <= cluster[k] for k in range(N))

    # task grouping and center determination
    # definition of z
    mdlgp.add_constraints(z[k, t] <= M * center[k, t] for k in range(N) for t in range(T))
    mdlgp.add_constraints(z[k, t] >= o[k] - M * (1 - center[k, t]) for k in range(N) for t in range(T))
    mdlgp.add_constraints(z[k, t] <= o[k] for k in range(N) for t in range(T))

    mdlgp.add_constraints(z[k, t] >= df_x_y.loc[n, "DATE"] - a - M * (1 - theta[n, k]) - M * (1 - center[k, t]) for n in range(N) for k in range(N) for t in range(T))
    mdlgp.add_constraints(z[k, t] <= df_x_y.loc[n, "DATE"] + int((3.2 - 3.0) / alpha_values[n]) + M * (1 - theta[n, k]) for n in range(N) for k in range(N) for t in range(T))

    si2 = mdlgp.solve()
    val = mdlgp.objective_value

    mdlgp.set_log_output(True)  # 启用日志输出
    solution = mdlgp.solve()

    # iterate through all the tasks to find the unique center
    for n in range(len(df_x_y)):
        for k in range(N):  # iterate through all the clusters
            # if solution[center[k, df_x_y.loc[n, "DATE"]]]:  # check if the date is the center of the cluster
            #     task_center_time = solution.get_value(o[k]) if solution else None  # get the center time
            #     df_x_y.at[n, "CENTER"] = task_center_time  # store the center time in the dataframe
            #     break  # break the loop if the center is found
            for k in range(N):
                for t in range(window_size):
                    if solution.get_value(center[k,t]) > 0.9:  # 判断是否被选为center
                        task_center_time = t
                        break

    if solution is None:
        print("Model did not solve successfully.")
        print(f"Status: {mdlgp.solve_details.status}")
        
    else:
        print("Model solved successfully.")
        return df_x_y


def main(window_size):

    df_x_y = pd.read_csv('maintenance_plan.csv')

    # Initialize the sliding window parameters
    window_start = 0

    while window_start < T:
        # Define the window range
        window_end = min(window_start + window_size, T)
        # 关键修改：转换全局时间到窗口相对时间
        tasks_window = df_x_y[(df_x_y['DATE'] >= window_start) & (df_x_y['DATE'] < window_end)]
        tasks_window['window_time'] = tasks_window['DATE'] - window_start  # 转换为[0,13]
        
    
        # print(tasks_window)

        # reset the center column for the tasks that are grouped before but appear in the next window
        # df_x_y.loc[tasks_window.index, 'CENTER'] = None
        
        # Group by CAISSON and select the first task for each group
        first_tasks = tasks_window.groupby("CSEM").first().reset_index()

        if not tasks_window.empty:
            # Call clustering_CR function on the first tasks
            # df_combined_cr = clustering_CR(first_tasks.copy(), window_size)
            # 传递窗口起始时间给聚类函数
            df_combined_cr = clustering_CR(tasks_window.copy(), window_size, window_start) 
            if df_combined_cr is None:
                print("Error: clustering_CR returned None")
                exit(1)  # 或者 return 一个默认的 DataFrame
            # Update the DATE for the second task of each CAISSON within the window
            for caisson in df_combined_cr["CSEM"].unique():
                # Get all tasks for this CAISSON in the current window
                caisson_tasks = tasks_window[tasks_window["CSEM"] == caisson]
                
                if len(caisson_tasks) > 1:
                    # Get the index of the second task in the current window
                    second_task_index = caisson_tasks.index[1]
                    
                    # Update DATE for the second task only
                    # updated_date = int(df_combined_cr.loc[df_combined_cr["CSEM"] == caisson, "CENTER"].values[0] + 1 + (3.5-3.2)/alpha_values[caisson])
                    
                    center_value = df_combined_cr.loc[df_combined_cr["CSEM"] == caisson, "CENTER"].values
                    print(center_value)
                    if len(center_value) == 0 or pd.isna(center_value[0]):
                        raise ValueError(f"Error: No valid CENTER value found for CSEM = {caisson}")
                    
                    updated_date = int(center_value[0] + 1 + (3.5-3.2)/alpha_values[caisson])
                    df_x_y.loc[second_task_index, "DATE"] = updated_date
                    

        # Slide the window forward
        window_start += (window_size/2)

    return df_x_y


if __name__ == "__main__":
    # read the csv file
    # df_x_y = pd.read_csv('maintenance_plan.csv')
    result_df = main(window_size)
    print(result_df)
