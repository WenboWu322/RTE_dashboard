import sys
sys.path.append('C:\\Program Files\\IBM\\ILOG\\CPLEX_Studio210\\cplex\\python\\20.1\\x64_win64')
sys.path.append("C:\\ProgramData\\anaconda3\\Lib\\site-packages")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from docplex.mp.model import Model
import streamlit.components.v1 as components
import plotly.graph_objects as go


st.set_page_config(layout="wide")
hide_dataframe_row_index = """
            <style>
         div.c1{
                    background-color: #f5f5f5;
                    border: 2px solid;
                    padding: 20px 20px 20px 20px;
                    border-radius: 10px;
                    color: #ffc300;
                    box-shadow: 10px;
                    }
            </style>
            """


def data(x):
    df_p = x[x['value'] != 1].reset_index()
    df_x_y = x[x['value'].astype(int) == 1].reset_index()
    for i in range(df_p.shape[0]):
        r = df_p["name"].loc[i]
        e = r.split('_')[0]
        g = r.split('_')[1]
        h = r.split('_')[2]
        if (e == 'p'):
            df_p.loc[i, 'CAISSON'] = g
            df_p.loc[i, 'TEMPS'] = int(h)

    for i in range(df_x_y.shape[0]):
        r = df_x_y["name"].loc[i]
        e = r.split('_')[0]
        g = r.split('_')[1]
        h = r.split('_')[2]
        if(e == 'y') | (e == 'x'):
            df_x_y.loc[i, 'OPERATION'] = e
            df_x_y.loc[i, 'CAISSON'] = g
            df_x_y.loc[i, 'TEMPS'] = int(h)
        # else:
        #     df_x_y.loc[i, 'OPERATION'] = "NA"
        #     df_x_y.loc[i, 'CAISSON'] = g
        #     df_x_y.loc[i, 'TEMPS'] = 0
    df_x_y = df_x_y.dropna()[['value', 'OPERATION', 'CAISSON', 'TEMPS']].reset_index()
    

    
    return df_p, df_x_y


# Function to visualize the clustering
@st.cache_data
def vis_caisson(df_x_y, df_p, x, taux1):
    # Filter the data based on the selected caissons
    sns.color_palette("Paired")
    sns.set(rc={"figure.figsize":(12, 5)})
    sns.despine(left=True)
    x = [str(s) for s in x]
    df_p_caisson = df_p[df_p['CAISSON'].isin(x)]
    caisson = df_x_y[df_x_y['CAISSON'].isin(x)]


    chart = px.line(df_p_caisson, x='TEMPS', y='value', color='CAISSON', width=700, height=400)
    
    # Add the reference lines and vertical lines for the operations
    chart.add_hline(y=taux1, line_dash='dash', line_color='red', name='Taux 1')
    chart.add_hline(y=3.5, line_dash='dash', line_color='blue', name='3.5')
    
    for temp in caisson[caisson['OPERATION'] == 'x']['TEMPS'].values:
        chart.add_vline(x=temp, line_dash='dash', line_color='green', name='CR')
        chart.add_annotation(x=temp+0.5, y=2.95, text='CR', showarrow=False)
    
    for temp in caisson[caisson['OPERATION'] == 'y']['TEMPS'].values:
        chart.add_vline(x=temp, line_dash='dash', line_color='yellow', name='RE')
        chart.add_annotation(x=temp+0.5, y=2.95, text='RE', showarrow=False)
    
    
    # Set the x-axis tick angle and remove the legend title
    chart.update_layout(xaxis_tickangle=-45)
    
    # Set the y-axis range to start from 2.9
    chart.update_yaxes(range=[2.9, None])
    chart.update_xaxes(dtick='M1')

    # Remove legend
    chart.update_layout(showlegend=False)
   
   
    return chart


# newly editted function to show the legend index correctly
def vis_all_caisson(df_x_y, df_p, taux1):
    # Filter the data based on the selected caissons
    df_p['value'] = df_p['value'].astype(float)
    
    # Create a line chart of the pressure data using Plotly Express
    chart = px.line(df_p, x='TEMPS', y='value', color='CAISSON', width=700, height=400,
                    color_discrete_sequence=px.colors.qualitative.Plotly)
    
    # Add the reference lines
    chart.add_hline(y=taux1, line_dash='dash', line_color='green', name='Taux 1')
    chart.add_hline(y=3.5, line_dash='dash', line_color='blue', name='3.5')
    
    # Set the x-axis tick angle and legend title
    chart.update_layout(xaxis_tickangle=-45, legend_title='Caissons', plot_bgcolor='#f0fdf4')

    # Modify legend labels to start from 1
    for i, trace in enumerate(chart.data):
        chart.data[i].name = f'Caisson {i+1}'
    
    return chart


def create_model(P, T, Pinit, Rcr, Rre, r_cr, r_re, P_rempli, alpha,alpha_param,deg, taux1, taux2, gamma, M, couts1, couts2, b1, b2, k1, k2, k3, W, K):

    """ Model """
    mdl = Model('Maintenance_caisson')
    x = mdl.binary_var_matrix(P, T, name = 'x')  
    y = mdl.binary_var_matrix(P, T, name = 'y')
    
    f = mdl.binary_var_matrix(P, T, name = 'f')  
    g = mdl.binary_var_matrix(P, T, name = 'g')
    
    z = mdl.binary_var_matrix(P, T, name = 'z')
    w = mdl.binary_var_matrix(P, T, name = 'w')
    
    e = mdl.integer_var_matrix(P, T, name="e")

    p = mdl.continuous_var_matrix(P, T, name='p')

    d = mdl.binary_var_matrix(P, T, name = 'd')

    v = mdl.continuous_var_matrix(P, T, name = 'v')

    # 计算递增速率
    slope = mdl.continuous_var_matrix(P, T, name='slope')

    # leakageQuantity = mdl.continuous_var_list(P, name='leakageQuantity')
    leakageQuantity = 0.0


    # Add constraints and objective function to the model
    # CONTRAINTES D'INITIALISATION DE PRESSION(OU DE PRESSIONS INITIAUX)
    mdl.add_constraints((p[i, 0] == (Pinit[i].item()) for i in range(P)), names = 'SPinit')
    mdl.add_constraints((p[i, t] >= 3.001 for i in range(P) for t in range(T)), names = 'SPmin')
    mdl.add_constraints((p[i, t] <= 3.5 for i in range(P) for t in range(T)), names = 'SPmax')

    
    # # CONTRAINTE DE RESOURCES 
    mdl.add_constraints((sum(r_cr*x[k, tp] for k in range(P) for tp in range(T) if (tp <= t) and (tp + b1 - 1>= t)) <= Rcr for t in range(T)))
    mdl.add_constraints((sum(r_re*y[k, tp] for k in range(P) for tp in range(T) if (tp <= t) and (tp + b2 - 1>= t)) <= Rre for t in range(T)))
    #-----------------------------------------------------------------------------------------------------------------------------

    if deg == 'Linear degradation':
        mdl.add_constraints((p[i, t+1] >= (1- alpha)*p[i, t] - M*(f[i, t] + (g[i, t] + e[i,t])) for i in range(P) for t in range(T-1)))
        mdl.add_constraints((p[i, t+1] <= (1- alpha)*p[i, t] + M*(f[i, t] + (g[i, t] + e[i,t])) for i in range(P) for t in range(T-1)))    
    else:
        mdl.add_constraints((p[i, t+1] >= (1- alpha_param[i][0]**(t/(t + 34) - 1))*p[i, t] - M*(f[i, t] + g[i, t]) for i in range(P) for t in range(1, T-1)))
        mdl.add_constraints((p[i, t+1] <= (1- alpha_param[i][0]**(t/(t + 34) - 1))*p[i, t] + M*(f[i, t] + g[i, t]) for i in range(P) for t in range(1, T-1)))
        
    #-----------------------------------------------------------------------------------------------------------------------------
    mdl.add_constraints((p[i, t+b1] >= P_rempli - M*(1 - x[i, t]) for i in range(P) for t in range(T-b1)), names = 'AFTER CR')
    mdl.add_constraints((p[i, t+b1] <= P_rempli + M*(1 - x[i, t]) for i in range(P) for t in range(T-b1)), names = 'AFTER CR')
    # mdl.add_constraints(p[i, t + k] >= p[i, t] + k * slope[i, t] - M * (1 - f[i, t+k]) for i in range(P) for t in range(T-b1) for k in range(1, b1))
    # mdl.add_constraints(p[i, t + k] <= p[i, t] + k * slope[i, t] - M * (1 - f[i, t+k]) for i in range(P) for t in range(T-b1) for k in range(1, b1))

    # Add linear interpolation constraints
    for i in range(P):
        for t in range(T - b1):  # Ensure that time frames are not exceeded
            for k in range(1, b1):
                # Calculate the desired pressure value at time t+k
                expected_pressure = p[i, t] + (k / b1) * (P_rempli - p[i, t])
                mdl.add_constraint(
                    p[i, t + k] >= expected_pressure - M * (1 - x[i, t])
                )
                mdl.add_constraint(
                    p[i, t + k] <= expected_pressure + M * (1 - x[i, t])
                )
        for t in range(T - b1, T):
            for k in range(1, T - t):
                expected_pressure = p[i, t] + (k / (T-t)) * (P_rempli - p[i, t])
                mdl.add_constraint(
                    p[i, t + k] >= expected_pressure - M * (1 - x[i, t])
                )
                mdl.add_constraint(
                    p[i, t + k] <= expected_pressure + M * (1 - x[i, t])
                )

    # for i in range(P):
    #     for t in range(T - 1):
    #         mdl.add_constraints((p[i, tp] >= p[i, t] - M*(1 - y[i, t]) for tp in range(t+1, T)), names = 'AFTER RE')
    #         mdl.add_constraints((p[i, tp] <= p[i, t] + M*(1 - y[i, t]) for tp in range(t+1, T)), names = 'AFTER RE')
    for i in range(P):
        for t in range(T):
            mdl.add_constraints((p[i, tp] >= p[i, t] - M*(1 - y[i, t]) for tp in range(t+b2, T)), names = 'AFTER RE')
            mdl.add_constraints((p[i, tp] <= p[i, t] + M*(1 - y[i, t]) for tp in range(t+b2, T)), names = 'AFTER RE')
    
        
                 
    mdl.add_constraints((y[i, tp] <= 1 - y[i, t] for i in range(P) for t in range(T-b2) for tp in range(t+b2, T)), names = 'AFTER RE')

    # mdl.add_constraints(x[i, t] + y[i, t] <= 1 for i in range(P) for t in range(T))

    # Additional constraints to define d[i, t]
    # mdl.add_constraints((p[i, t] - 3.2 <= (1 - d[i, t]) for i in range(P) for t in range(T)), names='d_upper_bound')
    # mdl.add_constraints((3.2 - p[i, t] <= d[i, t] for i in range(P) for t in range(T)), names='d_lower_bound')

    #calculate the action slope, if x[i, t] == 1，then slope = (3.5 - p[i, t]) / b
    # mdl.add_constraints((slope[i, t] == (3.5 - p[i, t])/b1 for i in range(P) for t in range(T)), names='slope_calculation')
    
            
    #-----------------------------------------------------------------------------------------------------------------------------
    # definition for cumulating the sum of RE action
    for t in range(T):
        for i in range(P):
                mdl.add_constraint(e[i,t] == mdl.sum(y[i, k] for k in range(t)))

    #-----------------------------------------------------------------------------------------------------------------------------
    # w[i][t] = sum(x[i, tp] for tp in range(t - W + 1, t + 1))
    mdl.add_constraints((w[i, t] == mdl.sum(x[i, tp] for tp in range(max(0, t - W + 1), t + 1)) for i in range(P) for t in range(T)), names='sum_constraint')
    # w[i][t] <= 2 + z[i][t]
    mdl.add_constraints((w[i, t] <= 2 + z[i, t] for i in range(P) for t in range(T)), names='upper_bound_constraint')
    # w[i][t] >= 3 - 3 * (1 - z[i][t])
    mdl.add_constraints((w[i, t] >= 3 - 3 * (1 - z[i, t]) for i in range(P) for t in range(T)), names='lower_bound_constraint')

    #-----------------------------------------------------------------------------------------------------------------------------
    # f[i][t+k] >= x[i][t]  ∀i ∈ P, ∀t ∈ T-b, ∀k = 0, 1, ..., b
    mdl.add_constraints((f[i, t + k] >= x[i, t] for i in range(P) for t in range(T - b1) for k in range(b1)), names='f_constraint')
    
    # g[i][t+k] >= y[i][t]  ∀i ∈ P, ∀t ∈ T-b, ∀k = 0, 1, ..., b
    mdl.add_constraints((g[i, t + k] >= y[i, t] for i in range(P) for t in range(T - b2) for k in range(b2)), names='g_constraint')

    # make sure f[i, t] = 0 at other moments
    mdl.add_constraints((f[i, t] <= sum(x[i, tp] for tp in range(max(0, t - b1 + 1), t + 1)) for i in range(P) for t in range(T)), names='f_zero_constraint')
    # make sure g[i, t] = 0 at other moments
    mdl.add_constraints((g[i, t] <= sum(y[i, tp] for tp in range(max(0, t - b2 + 1), t + 1)) for i in range(P) for t in range(T)), names='g_zero_constraint')
    # mdl.add_constraints((g[i, t] + y[i, t] >= sum(y[i, tp] for tp in range(t, T)) for i in range(P) for t in range(T)), names='g_zero_constraint')

    mdl.add_constraints(x[i, t] + x[i, tp] <= 1 for i in range(P) for t in range(T) for tp in range(T) if (tp < t) and (tp + b1 >= t)) 
    # mdl.add_constraints(y[i, t] + y[i, tp] <= 1 for i in range(P) for t in range(T) for tp in range(T) if (tp < t) and (tp + b2 >= t))

    # define variable v
    for i in range(P):
        v[i, 0] = 0
    mdl.add_constraints((v[i,t] >= p[i,t-1] - p[i,t] for i in range(P) for t in range(1,T)), names='v_constraint')
    mdl.add_constraints((v[i,t] >= 0 for i in range(P) for t in range(1,T)), names='v_constraint')


    #-----------------------------------------------------------------------------------------------------------------------------
    # mdl.add_constraints(f[i, t] >= x[i, t] for t in range(T))
    # mdl.add_constraints(g[i, t] >= y[i, t] for t in range(T))
    # mdl.add_constraints(f[i, t] == f[i, t-1] - f[i, t-b1] for t in range(b1, T))
    # mdl.add_constraints(g[i, t] == g[i, t-1] - g[i, t-b2] for t in range(b2, T))
    

    # #-----------------------------------------------------------------------------------------------------------------------------
    
    # # Ensure if f[i, t-b1] == 1 then f[i, t] == 0, same for g
    # mdl.add_constraints((f[i, t] <= 1 - f[i, t - b1] for i in range(P) for t in range(b1, T)), names='f_zero_constraint')
    # mdl.add_constraints((g[i, t] <= 1 - g[i, t - b2] for i in range(P) for t in range(b2, T)), names='g_zero_constraint')

    #-------------------------------------------------------------------------------------------------------------------------------
    
    # mdl.add_constraint(sum((((3.5 - p[i,tp])*(10**5))*V/(8.314 * 293)*146.0)/1000 for i in range(P) for tp in range(t)) <= 100.0)
    # Total leakage constraint
    # # for i in range(P):
    #     leakageQuantity[i] = mdl.sum(K * (p[i, t-1] - p[i, t]) for t in range(1,T))
    # mdl.add_constraint(leakageQuantity[i] <= 0.08)

        
    leakageQuantity = mdl.sum(K * v[i, t] for i in range(P) for t in range(1,T))
    mdl.add_constraint(leakageQuantity <= 50.0)
    # mdl.add_constraint(leakageQuantity >= 0.0)


    #-----------------------------------------------------------------------------------------------------------------------------
    # OBJ FUNCTION
    # mdl.minimize(mdl.sum(couts1*x[i, t] + couts2*y[i, t] +k1*z[i,t]) for i in range(P) for t in range(T))
    cost = mdl.sum(couts1*x[i, t] + couts2*y[i, t] + k1*z[i,t] for i in range(P) for t in range(T))
    mdl.minimize(cost)

    # Solve the model
    si = mdl.solve()
    df = mdl.solution.as_df()

    st.dataframe(df)

    # Calculate total leakage from the solution 
    # total_leakage_value = round(sum(K * (si[p[i, t - 1]] - si[p[i, t]]) for i in range(P) for t in range(1, T)), 3)
    total_leakage_value = round(leakageQuantity.solution_value, 3)
    

    # 如果你需要总泄漏量
    st.write(f"Total leakage quantity: {si[leakageQuantity]}")


    sol = si.get_blended_objective_value_by_priority()[0]
    df_p, df_x_y = data(df)

    # if si:
    #     df = mdl.solution.as_df()
    #     sol = si.get_blended_objective_value_by_priority()[0]
    #     df_p, df_x_y = data(df)
    #     st.dataframe(df)
    # else:
    #     # Handle case where the model couldn't be solved
    #     print("Model could not be solved.")
    #     return None, None, None  # Or handle the failure gracefully

    # Return the solution
    return df_p, df_x_y, sol, total_leakage_value



def data_clustring(x):
    for i in range(x.shape[0]):
        r = x["name"].loc[i]
        e = r.split('_')[0]
        g = r.split('_')[1]
        h = r.split('_')[2]
        x.loc[i, 'cluster'] = h
        x.loc[i, 'OPERATION'] = g
        
    return x

def update_df_x_y(df_x_y):
    df_x_y = df_x_y.rename(columns={"CC": "center"})
    df_x_y["OPERATION"] = df_x_y["OPERATION"].replace({"x": "CR", "y": "RE"})
    df_x_y = df_x_y.drop("value", axis=1)
    df_x_y = df_x_y.drop("index", axis=1)
    return df_x_y
    
def vis_clustring(df_x_y, C):
    # Set the color palette and plot size
    # sns.color_palette("Paired")
    sns.set(rc={"figure.figsize":(12, 5)})
    sns.despine(left=True)
    
    # Get the number of caissons and data points
    x = len(df_x_y['CAISSON'].unique()) 
    y = df_x_y.shape[0]
  
    # Create a scatter plot with Plotly Express
    df_x_y['cluster'] = pd.factorize(df_x_y['CC'])[0]
    df_x_y["cluster"] = df_x_y["cluster"].astype(str)
    color_sequence = px.colors.qualitative.Dark24
    color_dict = {str(i): px.colors.qualitative.Plotly[i] for i in range(df_x_y['cluster'].nunique())}
    label_dict = {str(i): f'Category {chr(i + 65)}' for i in range(df_x_y['cluster'].nunique())}
    size_dict = {str(i): (i + 1) * 20 for i in range(df_x_y['cluster'].nunique())}

    centroid_data = df_x_y[df_x_y['TEMPS'] == df_x_y['CC']]

    df_x_y = df_x_y.rename(columns={"CC": "center"})
    df_x_y["OPERATION"] = df_x_y["OPERATION"].replace({"x": "CR", "y": "RE"})
    
    for i in range(df_x_y.shape[0]):
        if df_x_y.loc[i, 'OPERATION'] == "CR":
            df_x_y.loc[i, 'TOLERANCE'] = 7  
        if df_x_y.loc[i, 'OPERATION'] == "RE":
            df_x_y.loc[i, 'TOLERANCE'] = 1  
        


    # Create the scatter plot with background colors for each cluster
    chart = px.scatter(df_x_y.sort_values('CAISSON', ascending=False), x='TEMPS', y='CAISSON', color='cluster',color_discrete_map=color_dict,
                    
                     width=800, height=500, hover_data=['center', 'OPERATION', 'TOLERANCE'])
    
    # Add a background color to each cluster
    for i, color in color_dict.items():
        cluster_data = df_x_y[df_x_y['cluster'] == str(i)]
        cluster_center = pd.DataFrame(cluster_data.loc[(cluster_data['TEMPS'] == cluster_data['center'].astype(int)), ['TEMPS', 'CAISSON']].head(1))

        for index, row in cluster_data.iterrows():           
            x1 = row['TEMPS']
            y1 = row['CAISSON']
            try:
                x2 = cluster_center.iloc[0]['TEMPS']
                y2 = cluster_center.iloc[0]['CAISSON']
                if x1 != x2:
                    # Draw the lasso line
                    chart.add_shape(type='line',
                                x0=x1,
                                y0=y1,
                                x1=x2,
                                y1=y2,
                                line=dict(color=color, width=1, dash='dot'))
    
                else:
                    chart.add_shape(type='circle',
                            xref='x', yref='y',
                            x0=int(x1)-0.5, y0=int(y1)-0.2,
                            x1=int(x1)+0.5, y1=int(y1)+0.2,
                            line=dict(color=color, width=2))
            # chart.update_shapes(line_color=color)
            except:
                continue

    # Customize the plot
    chart.update_traces(marker=dict(sizemode='diameter', sizeref=120))
    chart.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=2, range=[0,45]),
                         yaxis=dict(tickmode='linear', tick0=-2, dtick=2, range=[-2,x+2]),
                         legend_title='', showlegend=True)
    chart.update_layout(plot_bgcolor='#fafafa')
    
    return chart
@st.cache_data
def df_x_y_mod(df, couts1, couts2):
    # df = df.drop(['index', 'value'], axis=1)

    # Drop 'index' and 'value' columns if they exist
    df = df.drop(['index', 'value'], axis=1, errors='ignore')

    # replace the values in the OPERATION column
    df['OPERATION'] = df['OPERATION'].replace({'x': 'CR', 'y': 'RE'})

    # increment the values in the CAISSON column
    
    P = int(df['CAISSON'].nunique())

    # price_map = {('CR', i): couts1[i] for i in range(P)}
    # price_map.update({('RE', i): couts2[i] for i in range(P)})
    price_map = {('CR'): couts1}
    price_map.update({('RE'): couts2})
    
    # create the Price column based on the OPERATION and CAISSON columns
    # df['Price'] = [price_map[(op, int(c))] for op, c in zip(df['OPERATION'], df['CAISSON'])]
    
    df['CAISSON'] = df['CAISSON'].astype('int') + 1 
    caisson_col = df.pop('CAISSON')
    df.insert(0, 'CAISSON', caisson_col)
    return df
def clusteringg(df_x_y):
    
    C = df_x_y.TEMPS.values

    D = np.array([i - np.array(df_x_y.TEMPS.values.astype(int)) for i in C]) # les distances entre les centroides et les points
    D[D[:] < 0 ] = +99999
#     print(f'the matrix D is : {D}')
    
    OCR = df_x_y[df_x_y['OPERATION'].str.contains('x')].first_valid_index() # les points de nuage DES CR
    OCRf = df_x_y[df_x_y['OPERATION'].str.contains('x')].last_valid_index() # les points de nuage DES CR
    try:
        ORE = df_x_y[df_x_y['OPERATION'].str.contains('y')].first_valid_index()  # les points de nuage DES CR
        OREf = df_x_y[df_x_y['OPERATION'].str.contains('y')].last_valid_index()  # les points de nuage DES CRF
    except:
        ORE = None
    

    O = df_x_y.shape[0]
    # df_x_y.to_excel('df.xlsx', index=False)
    K = O
    mdlgp = Model('Maintance_grouping')
    theta = mdlgp.binary_var_matrix(O, K, name = 'theta')
    M = 99999
   
    
    cluster = mdlgp.binary_var_list(K, name = 'cluster')
    mdlgp.minimize(mdlgp.sum(cluster[k] for k in range(K)))

    mdlgp.add_constraints(sum(theta[i, k] for i in range(O)) <= M*cluster[k] for k in range(K))
    mdlgp.add_constraints(sum(theta[i, k] for k in range(K)) == 1 for i in range(O))
   
    mdlgp.add_constraints(theta[i, k]*D[k, i] <= 7 for i in range(OCRf+1) for k in range(K))
    if ORE:
        mdlgp.add_constraints(theta[i, k]*D[k, i] <= 1 for i in range(ORE, OREf+1) for k in range(K))
            

    for i in range(O):
        for j in range(O):
            for k in range(O):
                if (df_x_y.loc[i, "CAISSON"] == df_x_y.loc[j, "CAISSON"]) & (i < j):
                    mdlgp.add_constraint(theta[i, k] + theta[j, k] <= 1)
    si2 = mdlgp.solve()
    val = si2.get_blended_objective_value_by_priority()[0]
    print(mdlgp.print_solution(log_output=False))

    df = mdlgp.solution.as_df()
    df = df[df['name'].str.contains('theta')]
    sol = si2.get_blended_objective_value_by_priority()[0]
    dfc = data_clustring(df)
    S = np.zeros(K) 
    
    dfc.to_csv("test1.csv")

    # original
    # Data = dfc.sort_values(by = 'cluster')['cluster'].value_counts().to_frame().reset_index().rename(columns={'index':'values', 'values':'count'})    
    # for v in Data['values']:
    #     S[int(v)] = Data[Data['values']== v]['cluster'].values[0]
    #version1
    # Data = dfc.sort_values(by='cluster')['cluster'].value_counts().to_frame().reset_index(drop=True)
    # for index, count in Data.iterrows():
    #     S[index] = count
    #version3
    Data = dfc.sort_values(by = 'cluster')['cluster'].value_counts().to_frame().reset_index()
    for idx in Data.index:
        v = idx
        S[int(v)] = Data[Data.index == idx]['cluster'].values[0]



    #original
    CC = C[dfc.cluster.values.astype('int')] 



    df_x_y['CC'] = CC
    chart = vis_clustring(df_x_y, CC)
    st.plotly_chart(chart)
    return sol
# Create the Streamlit app
def title_with_background(title, background_color):
    st.markdown(f"""
        <h1 style='background-color:{background_color};border-radius:10px;color:black;text-align:center;margin-bottom:20px'>{title}</h1>
    """, unsafe_allow_html=True)
def line_with_background(title, background_color):
    st.markdown(f"""
                <hr style='background-color:{background_color};border-left: 1px solid #ccc;height: 100%;position: absolute;left: 50%;'></h>
                """, unsafe_allow_html=True)

def app():
    # st.title('Maintenance Caisson')
    title_with_background("Maintenance Planification and Optimization: RTE Use Case", "#f0fdf4")
    st.markdown("<br>", unsafe_allow_html=True) 

    # st.title(':black[Maintenance Caisson] ')

    col1, col2, col3 = st.columns([1, 1.3, 2.4])
    def set_alpha(P):
        # alpha_low = np.random.uniform(0.01, 0.09, P)
        # alpha_high = np.random.uniform(0.004, 0.009, P)
        alpha_high = 0.005


        # PP  = int(P/2)
        # alpha = np.concatenate((alpha_low[:PP], alpha_high[PP:]))
        return alpha_high
    
    with col1:
        st.write("## Model Inputs")
        alert = """ <script>alert('This option is not available.')</script> """
        # deg = components.html(radio_button)  
        deg = st.radio("Degradation/Leakage Type:",
                                    ["Linear degradation", "Function of time", "Stochastic degradation", "Physical modelling"],
                                    index=0,
                                    key="degradation",
                                    )
        # if "Stochastic" or "Physical" in deg:
            # components.html(alert)
            # st.warning("This option is not available yet, please make another choice")
        P = st.slider('### Number of CSEM', min_value=1, max_value=20, step=1)
        T = st.slider('### Time horizon (days)', min_value=30, step=1)
        # Pinit = np.zeros((P, 1))
        # Pinit[:] = 3.5                 
        Pinit = np.random.uniform(3.21, 3.5001, (P, 1))    #//////

        Rcr = st.slider('### Available Technicians for CR', value=100)
        Rre = st.slider('### Available Technicians for REPAIR', value=100)

        # beta = st.slider('Beta', value=3, max_value=5)
        alpha = set_alpha(P)
        alpha_param = np.random.uniform(500, 1000, (P, 1))
                
        taux1 = 3
        taux2 = 3.2
        beta = 3
        gamma = 99999


        # r_cr = np.random.randint(1 , 3,(P,1))   #/////
        # r_re = np.random.randint(1 , 3,(P,1))    #////
        r_cr = 3   #/////
        r_re = 5    #////
        Dcr = np.random.randint(1 ,3,(P, 1))
        Dre = np.random.randint(1 ,3,(P, 1))  #////
        M = 3.6
        P_rempli = 3.5  #/////
        delta = np.random.randint(0, 1, P)
        deltay = np.random.randint(5, 18, P)

        # deltay = np.random.randint(0, 1, P)   # ///////
        # couts1 : cost of CR
        # couts1 = np.random.uniform(1, 10,  size=P)
        couts1 = 10
        # couts2 : cost of RE
        # couts2 = np.random.uniform(60, 80, size=P)
        couts2 = 100
        b1 = 3
        b2 = 4
        k1 = 2
        k2 = 4
        k3 = 20
        W = 15
        

        # constant for perfect equation coefficient = 10^5 * V/RT * 146 /1000
        # K = 2052.54
        # K = 299.67
        K = 5.993        # volume = 1
        # K = 59.93          # volume = 10

    

     
        if 'df_p' not in st.session_state:
            st.session_state.df_p = None

        if st.button('Run Optimization'):

            with st.spinner("Running function..."): 
                df_p, df_x_y, sol, total_leakage_value = create_model(P, T, Pinit, Rcr, Rre, r_cr, r_re, P_rempli, alpha, alpha_param, deg, taux1, taux2, gamma, M, couts1, couts2, b1, b2, k1, k2, k3, W, K)
                st.session_state.df_p = df_p
                st.session_state.df_x_y = df_x_y
                st.session_state.sol = np.around(sol, 2)
                st.session_state.total_leakage_value = total_leakage_value
            st.session_state.couts1 = couts1
            st.session_state.couts2 = couts2
            
       
    # st.markdown('<div class="vertical-line"></div>', unsafe_allow_html=True)
    
    # Create the Docplex model and get the solution
    with col2:
        if st.session_state.df_p is not None :
            dff = df_x_y_mod(st.session_state.df_x_y, st.session_state.couts1, st.session_state.couts2)
            dff.index = dff.index + 1  # 将索引改为从1开始
            # dff =st.session_state.df_x_y
            st.write('## Optimization Output')
            st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
            # st.write(dff.to_html(index=False), unsafe_allow_html=True)

            st.write('#### Total Cost (€):', st.session_state.sol)
            st.write('#### Total leakage quantity (kg):', st.session_state.total_leakage_value)
            st.write('#### Maintenance Actions:', dff.shape[0])
            st.write('##### Complément Remplissage (CR):', dff[dff['OPERATION'] == 'CR'].shape[0])
            st.write('##### Repair (RE):', dff[dff['OPERATION'] == 'RE'].shape[0])

            dff = dff.rename(columns={'TEMPS': 'TIME'})
            st.dataframe(dff, height=350)
            
    with col3:
        caissons = [f"Caisson {i+1}" for i in range(P)]

        if st.session_state.df_p is not None :
            st.write("## Data Visualization")
            c = st.radio('CSEM Visualization', caissons, horizontal=True)
            st.markdown("<hr>", unsafe_allow_html=True) 

            if st.button(f'Visualize all CSEMs'):   
                # Visualize the clustering
                fig2 = vis_all_caisson(st.session_state.df_x_y, st.session_state.df_p, taux1)
                st.plotly_chart(fig2)
            else:
                fig = vis_caisson(st.session_state.df_x_y, st.session_state.df_p, str(int(str(c)[-1]) - 1), taux1)
                st.plotly_chart(fig)


    st.markdown("<hr>", unsafe_allow_html=True) 
 
    col3, col4 = st.columns([2, 1])
    col11, col12 = st.columns([1, 0.01])
    col5, col6, col7 = st.columns([1, 1, 1])
    col8, col9, col10 = st.columns([1, 1, 1])
    
    if st.button('Maintenance Groupping'): 

        # with col11:
        #     comm = f"""
        #     <h1 style="box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);margin: 10px;padding:15px">Assuming that the cost  of traveling represent 40% of total cost of maintenance</h1>
        #     """
        #     components.html(comm)

        with col3:
            if st.session_state.df_x_y is not None :
                # Visualize the clustering
                sol = st.session_state.sol
                sol2 = clusteringg(st.session_state.df_x_y)
                # vis_clustring(st.session_state.df_x_y, (st.session_state.df_x_y))
                O = st.session_state.df_x_y.shape[0]
                gsol = round(0.6*sol + (0.4*sol)/sol2, 2)
                st.session_state.gsol = gsol
                gapc= round(float((sol - gsol)/sol)*100, 2)
                gapc = f"{gapc}%"

                gap= round(float((O - sol2)/O)*100, 2)
                gap = f"{gap}%"
            st.markdown("<style>.big-column{padding-right: 30px;}</style>", unsafe_allow_html=True)
        with col4:
            st.session_state.sol2 = sol2
            if st.session_state.df_p is not None :
                dfa = update_df_x_y(st.session_state.df_x_y)
                st.dataframe(dfa, height=450)
     
        fm = st.session_state.sol
        fm2 = st.session_state.gsol
        fm3 = st.session_state.sol2
                        
        with col5:
            com = f"""
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding-top:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>Maintenance Cost (Before grouping)</h2>
                        <h1>{fm} (€)</h1>
                        <p></p>
                    </div>
                    """
            components.html(com)
            # st.write('<style>  background-color: #FFFFFF;border-radius: 10px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding: 20px;text-align: center;}</style>', unsafe_allow_html=True)
        with col6:
            com = f"""
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding-top:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>Maintenace Cost (After grouping)</h2>
                        <h1>{fm2} (€)</h1>
                        <p></p>

                    </div>
                    """
            components.html(com)
        with col7:
            com = f"""
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding-top:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>GAP</h2>
                        <h1 style="color:#22c55e"><span><svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-arrow-big-up-line" width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">
                        <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
                        <path d="M9 12h-3.586a1 1 0 0 1 -.707 -1.707l6.586 -6.586a1 1 0 0 1 1.414 0l6.586 6.586a1 1 0 0 1 -.707 1.707h-3.586v6h-6v-6z"></path>
                        <path d="M9 21h6"></path>
                        </svg></span>  {gapc}</h1>

                    </div>
                    """
            components.html(com)
        with col8:
            com = f"""
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>Travels (Before grouping)</h2>
                        <h1>{O}</h1>

                    </div>
                    """
            components.html(com)                
        with col9:
            com = f"""
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>Travels (After grouping)</h2>
                        <h1>{int(sol2)}</h1>

                    </div>
                    """
            components.html(com)
        
        with col10:
            com = f"""
                <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding:2px; border-radius: 5px; text-align: center; height: 130px;">
                    <h2>GAP</h2>
                    <h1 style="color:#10b981"><span><svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-arrow-big-up-line" width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">
                        <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
                        <path d="M9 12h-3.586a1 1 0 0 1 -.707 -1.707l6.586 -6.586a1 1 0 0 1 1.414 0l6.586 6.586a1 1 0 0 1 -.707 1.707h-3.586v6h-6v-6z"></path>
                        <path d="M9 21h6"></path>
                        </svg></span>{gap}</h1>
                    <p></p>

                </div>
                """
            components.html(com)


# Run the app
if __name__ == '__main__':
    app()
    



# 2025/01/20
import sys
sys.path.append('C:\\Program Files\\IBM\\ILOG\\CPLEX_Studio210\\cplex\\python\\20.1\\x64_win64')
sys.path.append("C:\\ProgramData\\anaconda3\\Lib\\site-packages")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from docplex.mp.model import Model
import streamlit.components.v1 as components
import plotly.graph_objects as go


st.set_page_config(layout="wide")
hide_dataframe_row_index = """
            <style>
         div.c1{
                    background-color: #f5f5f5;
                    border: 2px solid;
                    padding: 20px 20px 20px 20px;
                    border-radius: 10px;
                    color: #ffc300;
                    box-shadow: 10px;
                    }
            </style>
            """


def data(x):
    df_p = x[x['value'] != 1].reset_index()
    df_x_y = x[x['value'].astype(int) == 1].reset_index()
    for i in range(df_p.shape[0]):
        r = df_p["name"].loc[i]
        e = r.split('_')[0]
        g = r.split('_')[1]
        h = r.split('_')[2]
        if (e == 'p'):
            df_p.loc[i, 'CAISSON'] = g
            df_p.loc[i, 'TEMPS'] = int(h)

    for i in range(df_x_y.shape[0]):
        r = df_x_y["name"].loc[i]
        e = r.split('_')[0]
        g = r.split('_')[1]
        h = r.split('_')[2]
        if(e == 'y') | (e == 'x'):
            df_x_y.loc[i, 'OPERATION'] = e
            df_x_y.loc[i, 'CAISSON'] = g
            df_x_y.loc[i, 'TEMPS'] = int(h)
        # else:
        #     df_x_y.loc[i, 'OPERATION'] = "NA"
        #     df_x_y.loc[i, 'CAISSON'] = g
        #     df_x_y.loc[i, 'TEMPS'] = 0
    df_x_y = df_x_y.dropna()[['value', 'OPERATION', 'CAISSON', 'TEMPS']].reset_index()
    

    
    return df_p, df_x_y


# Function to visualize the clustering
@st.cache_data
def vis_caisson(df_x_y, df_p, x, taux1):
    # Filter the data based on the selected caissons
    sns.color_palette("Paired")
    sns.set(rc={"figure.figsize":(12, 5)})
    sns.despine(left=True)
    x = [str(s) for s in x]
    df_p_caisson = df_p[df_p['CAISSON'].isin(x)]
    caisson = df_x_y[df_x_y['CAISSON'].isin(x)]


    chart = px.line(df_p_caisson, x='TEMPS', y='value', color='CAISSON', width=700, height=400)
    
    # Add the reference lines and vertical lines for the operations
    chart.add_hline(y=taux1, line_dash='dash', line_color='red', name='Taux 1')
    chart.add_hline(y=3.5, line_dash='dash', line_color='blue', name='3.5')
    
    for temp in caisson[caisson['OPERATION'] == 'x']['TEMPS'].values:
        chart.add_vline(x=temp, line_dash='dash', line_color='green', name='CR')
        chart.add_annotation(x=temp+0.5, y=2.95, text='CR', showarrow=False)
    
    for temp in caisson[caisson['OPERATION'] == 'y']['TEMPS'].values:
        chart.add_vline(x=temp, line_dash='dash', line_color='yellow', name='RE')
        chart.add_annotation(x=temp+0.5, y=2.95, text='RE', showarrow=False)
    
    
    # Set the x-axis tick angle and remove the legend title
    chart.update_layout(xaxis_tickangle=-45)
    
    # Set the y-axis range to start from 2.9
    chart.update_yaxes(range=[2.9, None])
    chart.update_xaxes(dtick='M1')

    # Remove legend
    chart.update_layout(showlegend=False)
   
   
    return chart


# newly editted function to show the legend index correctly
def vis_all_caisson(df_x_y, df_p, taux1):
    # Filter the data based on the selected caissons
    df_p['value'] = df_p['value'].astype(float)
    
    # Create a line chart of the pressure data using Plotly Express
    chart = px.line(df_p, x='TEMPS', y='value', color='CAISSON', width=700, height=400,
                    color_discrete_sequence=px.colors.qualitative.Plotly)
    
    # Add the reference lines
    chart.add_hline(y=taux1, line_dash='dash', line_color='green', name='Taux 1')
    chart.add_hline(y=3.5, line_dash='dash', line_color='blue', name='3.5')
    
    # Set the x-axis tick angle and legend title
    chart.update_layout(xaxis_tickangle=-45, legend_title='Caissons', plot_bgcolor='#f0fdf4')

    # Modify legend labels to start from 1
    for i, trace in enumerate(chart.data):
        chart.data[i].name = f'Caisson {i+1}'
    
    return chart


def create_model(P, T, Pinit, Rcr, Rre, r_cr, r_re, P_rempli, alpha,alpha_param,deg, taux1, taux2, gamma, M, couts1, couts2, b1, b2, k1, k2, k3, W, K):

    """ Model """
    mdl = Model('Maintenance_caisson')
    x = mdl.binary_var_matrix(P, T, name = 'x')  
    y = mdl.binary_var_matrix(P, T, name = 'y')
    
    f = mdl.binary_var_matrix(P, T, name = 'f')  
    g = mdl.binary_var_matrix(P, T, name = 'g')
    
    z = mdl.binary_var_matrix(P, T, name = 'z')
    w = mdl.binary_var_matrix(P, T, name = 'w')
    
    e = mdl.integer_var_matrix(P, T, name="e")

    p = mdl.continuous_var_matrix(P, T, name='p')

    d = mdl.binary_var_matrix(P, T, name = 'd')

    o = mdl.binary_var_matrix(P, T, name = 'o')

    v = mdl.continuous_var_matrix(P, T, name = 'v')

    # 计算递增速率
    slope = mdl.continuous_var_matrix(P, T, name='slope')



    # leakageQuantity = mdl.continuous_var_list(P, name='leakageQuantity')
    leakageQuantity = 0.0


    # Add constraints and objective function to the model
    # CONTRAINTES D'INITIALISATION DE PRESSION(OU DE PRESSIONS INITIAUX)
    mdl.add_constraints((p[i, 0] == (Pinit[i].item()) for i in range(P)), names = 'SPinit')
    mdl.add_constraints((p[i, t] >= 3.001 for i in range(P) for t in range(T)), names = 'SPmin')
    mdl.add_constraints((p[i, t] <= 3.5 for i in range(P) for t in range(T)), names = 'SPmax')

    
    # # CONTRAINTE DE RESOURCES 
    mdl.add_constraints((sum(r_cr*x[k, tp] for k in range(P) for tp in range(T) if (tp <= t) and (tp + b1 - 1>= t)) <= Rcr for t in range(T)))
    mdl.add_constraints((sum(r_re*y[k, tp] for k in range(P) for tp in range(T) if (tp <= t) and (tp + b2 - 1>= t)) <= Rre for t in range(T)))
    #-----------------------------------------------------------------------------------------------------------------------------

    if deg == 'Linear degradation':
        mdl.add_constraints((p[i, t+1] >= (1- alpha)*p[i, t] - M*(f[i, t] + (g[i, t] + e[i,t])) for i in range(P) for t in range(T-1)))
        mdl.add_constraints((p[i, t+1] <= (1- alpha)*p[i, t] + M*(f[i, t] + (g[i, t] + e[i,t])) for i in range(P) for t in range(T-1)))    
    else:
        mdl.add_constraints((p[i, t+1] >= (1- alpha_param[i][0]**(t/(t + 34) - 1))*p[i, t] - M*(f[i, t] + g[i, t]) for i in range(P) for t in range(1, T-1)))
        mdl.add_constraints((p[i, t+1] <= (1- alpha_param[i][0]**(t/(t + 34) - 1))*p[i, t] + M*(f[i, t] + g[i, t]) for i in range(P) for t in range(1, T-1)))
        
    #-----------------------------------------------------------------------------------------------------------------------------
    mdl.add_constraints((p[i, t+b1] >= P_rempli - M*(1 - x[i, t]) for i in range(P) for t in range(T-b1)), names = 'AFTER CR')
    mdl.add_constraints((p[i, t+b1] <= P_rempli + M*(1 - x[i, t]) for i in range(P) for t in range(T-b1)), names = 'AFTER CR')

    # Add linear interpolation constraints for CR
    for i in range(P):
        for t in range(T - b1):  # Ensure that time frames are not exceeded
            for k in range(1, b1):
                # Calculate the desired pressure value at time t+k
                expected_pressure = p[i, t] + (k / b1) * (P_rempli - p[i, t])
                mdl.add_constraint(p[i, t + k] >= expected_pressure - M * (1 - x[i, t]))
                mdl.add_constraint(p[i, t + k] <= expected_pressure + M * (1 - x[i, t]))

        for t in range(T - b1, T):
            for k in range(1, T - t):
                expected_pressure = p[i, t] + (k / (T-t)) * (P_rempli - p[i, t])
                mdl.add_constraint(p[i, t + k] >= expected_pressure - M * (1 - x[i, t]))
                mdl.add_constraint(p[i, t + k] <= expected_pressure + M * (1 - x[i, t]))

    # Add linear interpolation constraints for RE
    for i in range(P):
        for t in range(T - b2):  # Ensure that time frames are not exceeded
            for k in range(1, b2):
                # Calculate the desired pressure value at time t+k
                expected_pressure = p[i, t] + (k / b2) * (P_rempli - p[i, t])
                mdl.add_constraint(p[i, t + k] >= expected_pressure - M * (1 - y[i, t]))
                mdl.add_constraint(p[i, t + k] <= expected_pressure + M * (1 - y[i, t]))

    for t in range(T - b2, T):
        # for k in range(1, T - t):
        for k in range(0, T - t):
            expected_pressure = p[i, t] + (k / (T-t)) * (P_rempli - p[i, t])
            mdl.add_constraint(p[i, t + k] >= expected_pressure - M * (1 - y[i, t]))
            mdl.add_constraint(p[i, t + k] <= expected_pressure + M * (1 - y[i, t]))

    # if RE make pressure stay stable

    # for i in range(P):
    #     for t in range(T):
    #         mdl.add_constraints((p[i, tp] >= p[i, t] - M*(1 - y[i, t]) for tp in range(t+b2, T)), names = 'AFTER RE')
    #         mdl.add_constraints((p[i, tp] <= p[i, t] + M*(1 - y[i, t]) for tp in range(t+b2, T)), names = 'AFTER RE')
    
    # if RE make pressure come back to 3.5
    for i in range(P):
        for t in range(T):
            mdl.add_constraints((p[i, tp] >= P_rempli - M*(1 - y[i, t]) for tp in range(t+b2, T)), names = 'AFTER RE')
            mdl.add_constraints((p[i, tp] <= P_rempli + M*(1 - y[i, t]) for tp in range(t+b2, T)), names = 'AFTER RE')
    
        
                 
    mdl.add_constraints((y[i, tp] <= 1 - y[i, t] for i in range(P) for t in range(T-b2) for tp in range(t+b2, T)), names = 'AFTER RE')

    # mdl.add_constraints(x[i, t] + y[i, t] <= 1 for i in range(P) for t in range(T))

    # Additional constraints to define d[i, t]
    # mdl.add_constraints((p[i, t] - 3.2 <= (1 - d[i, t]) for i in range(P) for t in range(T)), names='d_upper_bound')
    # mdl.add_constraints((3.2 - p[i, t] <= d[i, t] for i in range(P) for t in range(T)), names='d_lower_bound')

    #calculate the action slope, if x[i, t] == 1，then slope = (3.5 - p[i, t]) / b
    # mdl.add_constraints((slope[i, t] == (3.5 - p[i, t])/b1 for i in range(P) for t in range(T)), names='slope_calculation')
    
            
    #-----------------------------------------------------------------------------------------------------------------------------
    # definition for cumulating the sum of RE action
    for t in range(T):
        for i in range(P):
                mdl.add_constraint(e[i,t] == mdl.sum(y[i, k] for k in range(t)))

    #-----------------------------------------------------------------------------------------------------------------------------
    # w[i][t] = sum(x[i, tp] for tp in range(t - W + 1, t + 1))
    mdl.add_constraints((w[i, t] == mdl.sum(x[i, tp] for tp in range(max(0, t - W + 1), t + 1)) for i in range(P) for t in range(T)), names='sum_constraint')
    # w[i][t] <= 2 + z[i][t]
    mdl.add_constraints((w[i, t] <= 2 + z[i, t] for i in range(P) for t in range(T)), names='upper_bound_constraint')
    # w[i][t] >= 3 - 3 * (1 - z[i][t])
    mdl.add_constraints((w[i, t] >= 3 - 3 * (1 - z[i, t]) for i in range(P) for t in range(T)), names='lower_bound_constraint')

    #-----------------------------------------------------------------------------------------------------------------------------
    # f[i][t+k] >= x[i][t]  ∀i ∈ P, ∀t ∈ T-b, ∀k = 0, 1, ..., b
    mdl.add_constraints((f[i, t + k] >= x[i, t] for i in range(P) for t in range(T - b1) for k in range(b1)), names='f_constraint')
    
    # g[i][t+k] >= y[i][t]  ∀i ∈ P, ∀t ∈ T-b, ∀k = 0, 1, ..., b
    mdl.add_constraints((g[i, t + k] >= y[i, t] for i in range(P) for t in range(T - b2) for k in range(b2)), names='g_constraint')

    # make sure f[i, t] = 0 at other moments
    mdl.add_constraints((f[i, t] <= sum(x[i, tp] for tp in range(max(0, t - b1 + 1), t + 1)) for i in range(P) for t in range(T)), names='f_zero_constraint')
    # make sure g[i, t] = 0 at other moments
    mdl.add_constraints((g[i, t] <= sum(y[i, tp] for tp in range(max(0, t - b2 + 1), t + 1)) for i in range(P) for t in range(T)), names='g_zero_constraint')
    # mdl.add_constraints((g[i, t] + y[i, t] >= sum(y[i, tp] for tp in range(t, T)) for i in range(P) for t in range(T)), names='g_zero_constraint')

    mdl.add_constraints(x[i, t] + x[i, tp] <= 1 for i in range(P) for t in range(T) for tp in range(T) if (tp < t) and (tp + b1 >= t)) 
    # mdl.add_constraints(y[i, t] + y[i, tp] <= 1 for i in range(P) for t in range(T) for tp in range(T) if (tp < t) and (tp + b2 >= t))

    # # define variable v - original way
    # for i in range(P):
    #     v[i, 0] = 0
    # mdl.add_constraints((v[i,t] >= p[i,t-1] - p[i,t] for i in range(P) for t in range(1,T)), names='v_constraint')
    # mdl.add_constraints((v[i,t] >= 0 for i in range(P) for t in range(1,T)), names='v_constraint')

    # define variable v - new way !!!!
    for i in range(P):
        for t in range(1, T):
            mdl.add_constraint(v[i, t] >= p[i, t-1] - p[i, t])
            mdl.add_constraint(v[i, t] >= 0)
            mdl.add_constraint(v[i, t] <= (p[i,t-1] - p[i, t]) + M * (1 - o[i, t]))
            mdl.add_constraint(v[i, t] <= M * o[i, t])
            mdl.add_constraint(M * o[i, t] >= p[i,t-1] - p[i, t])
            mdl.add_constraint(M * (1- o[i, t]) >= - (p[i,t-1] - p[i, t]))


    #-------------------------------------------------------------------------------------------------------------------------------
    
    # mdl.add_constraint(sum((((3.5 - p[i,tp])*(10**5))*V/(8.314 * 293)*146.0)/1000 for i in range(P) for tp in range(t)) <= 100.0)
    # separate leakage quantity for each caisson
    # for i in range(P):
    #     leakageQuantity[i] = mdl.sum(K * (p[i, t-1] - p[i, t]) for t in range(1,T))
    #     mdl.add_constraint(leakageQuantity[i] <= 0.08)

    # total leakage quantity constraint     
    leakageQuantity = mdl.sum(K * v[i, t] for i in range(P) for t in range(1,T))
    mdl.add_constraint(leakageQuantity <= 15.0)
    


    #-----------------------------------------------------------------------------------------------------------------------------
    # OBJ FUNCTION
    cost = mdl.sum(couts1*x[i, t] + couts2*y[i, t] + k1*z[i,t] for i in range(P) for t in range(T))
    mdl.minimize(cost)

    # Solve the model
    si = mdl.solve()
    df = mdl.solution.as_df()

    st.dataframe(df)

    # Calculate total leakage from the solution 
    total_leakage_value = round(leakageQuantity.solution_value, 3)
    

    # 如果你需要总泄漏量
    st.write(f"Total leakage quantity: {si[leakageQuantity]}")


    sol = si.get_blended_objective_value_by_priority()[0]
    df_p, df_x_y = data(df)

    # if si:
    #     df = mdl.solution.as_df()
    #     sol = si.get_blended_objective_value_by_priority()[0]
    #     df_p, df_x_y = data(df)
    #     st.dataframe(df)
    # else:
    #     # Handle case where the model couldn't be solved
    #     print("Model could not be solved.")
    #     return None, None, None  # Or handle the failure gracefully

    # Return the solution
    return df_p, df_x_y, sol, total_leakage_value



def data_clustring(x):
    for i in range(x.shape[0]):
        r = x["name"].loc[i]
        e = r.split('_')[0]
        g = r.split('_')[1]   # types of operations
        h = r.split('_')[2]   # cluster information
        x.loc[i, 'cluster'] = h
        x.loc[i, 'OPERATION'] = g
        
    return x

def update_df_x_y(df_x_y):
    df_x_y = df_x_y.rename(columns={"CC": "center"})
    df_x_y["OPERATION"] = df_x_y["OPERATION"].replace({"x": "CR", "y": "RE"})
    df_x_y = df_x_y.drop("value", axis=1)
    df_x_y = df_x_y.drop("index", axis=1)
    return df_x_y
    
def vis_clustring(df_x_y, C):
    # Set the color palette and plot size
    # sns.color_palette("Paired")
    sns.set(rc={"figure.figsize":(12, 5)})
    sns.despine(left=True)
    
    # Get the number of caissons and data points
    x = len(df_x_y['CAISSON'].unique()) 
    y = df_x_y.shape[0]
  
    # Create a scatter plot with Plotly Express
    df_x_y['cluster'] = pd.factorize(df_x_y['CC'])[0]
    df_x_y["cluster"] = df_x_y["cluster"].astype(str)
    color_sequence = px.colors.qualitative.Dark24
    color_dict = {str(i): px.colors.qualitative.Plotly[i] for i in range(df_x_y['cluster'].nunique())}
    label_dict = {str(i): f'Category {chr(i + 65)}' for i in range(df_x_y['cluster'].nunique())}
    size_dict = {str(i): (i + 1) * 20 for i in range(df_x_y['cluster'].nunique())}

    centroid_data = df_x_y[df_x_y['TEMPS'] == df_x_y['CC']]

    df_x_y = df_x_y.rename(columns={"CC": "center"})
    df_x_y["OPERATION"] = df_x_y["OPERATION"].replace({"x": "CR", "y": "RE"})
    
    for i in range(df_x_y.shape[0]):
        if df_x_y.loc[i, 'OPERATION'] == "CR":
            df_x_y.loc[i, 'TOLERANCE'] = 7  
        if df_x_y.loc[i, 'OPERATION'] == "RE":
            df_x_y.loc[i, 'TOLERANCE'] = 1  
        


    # Create the scatter plot with background colors for each cluster
    chart = px.scatter(df_x_y.sort_values('CAISSON', ascending=False), x='TEMPS', y='CAISSON', color='cluster',color_discrete_map=color_dict,
                    
                     width=800, height=500, hover_data=['center', 'OPERATION', 'TOLERANCE'])
    
    # Add a background color to each cluster
    for i, color in color_dict.items():
        cluster_data = df_x_y[df_x_y['cluster'] == str(i)]
        cluster_center = pd.DataFrame(cluster_data.loc[(cluster_data['TEMPS'] == cluster_data['center'].astype(int)), ['TEMPS', 'CAISSON']].head(1))

        for index, row in cluster_data.iterrows():           
            x1 = row['TEMPS']
            y1 = row['CAISSON']
            try:
                x2 = cluster_center.iloc[0]['TEMPS']
                y2 = cluster_center.iloc[0]['CAISSON']
                if x1 != x2:
                    # Draw the lasso line
                    chart.add_shape(type='line',
                                x0=x1,
                                y0=y1,
                                x1=x2,
                                y1=y2,
                                line=dict(color=color, width=1, dash='dot'))
    
                else:
                    chart.add_shape(type='circle',
                            xref='x', yref='y',
                            x0=int(x1)-0.5, y0=int(y1)-0.2,
                            x1=int(x1)+0.5, y1=int(y1)+0.2,
                            line=dict(color=color, width=2))
            # chart.update_shapes(line_color=color)
            except:
                continue

    # Customize the plot
    chart.update_traces(marker=dict(sizemode='diameter', sizeref=120))
    chart.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=2, range=[0,45]),
                         yaxis=dict(tickmode='linear', tick0=-2, dtick=2, range=[-2,x+2]),
                         legend_title='', showlegend=True)
    chart.update_layout(plot_bgcolor='#fafafa')
    
    return chart








@st.cache_data
def df_x_y_mod(df, couts1, couts2):
    # df = df.drop(['index', 'value'], axis=1)

    # Drop 'index' and 'value' columns if they exist
    df = df.drop(['index', 'value'], axis=1, errors='ignore')

    # replace the values in the OPERATION column
    df['OPERATION'] = df['OPERATION'].replace({'x': 'CR', 'y': 'RE'})

    # increment the values in the CAISSON column
    
    P = int(df['CAISSON'].nunique())

    # price_map = {('CR', i): couts1[i] for i in range(P)}
    # price_map.update({('RE', i): couts2[i] for i in range(P)})
    price_map = {('CR'): couts1}
    price_map.update({('RE'): couts2})
    
    # create the Price column based on the OPERATION and CAISSON columns
    # df['Price'] = [price_map[(op, int(c))] for op, c in zip(df['OPERATION'], df['CAISSON'])]
    
    df['CAISSON'] = df['CAISSON'].astype('int') + 1 
    caisson_col = df.pop('CAISSON')
    df.insert(0, 'CAISSON', caisson_col)
    return df

def clusteringg(df_x_y):
    
    C = df_x_y.TEMPS.values

    D = np.array([i - np.array(df_x_y.TEMPS.values.astype(int)) for i in C]) # les distances entre les points (time difference)
    D[D[:] < 0 ] = +99999
#     print(f'the matrix D is : {D}')
    
    OCR = df_x_y[df_x_y['OPERATION'].str.contains('x')].first_valid_index() # les points de nuage DES CR
    OCRf = df_x_y[df_x_y['OPERATION'].str.contains('x')].last_valid_index() # les points de nuage DES CR
    try:
        ORE = df_x_y[df_x_y['OPERATION'].str.contains('y')].first_valid_index()  # les points de nuage DES RE
        OREf = df_x_y[df_x_y['OPERATION'].str.contains('y')].last_valid_index()  # les points de nuage DES REF
    except:
        ORE = None

    O = df_x_y.shape[0]
    # df_x_y.to_excel('df.xlsx', index=False)
    K = O
    mdlgp = Model('Maintance_grouping')
    theta = mdlgp.binary_var_matrix(O, K, name = 'theta')
    M = 99999
   
    cluster = mdlgp.binary_var_list(K, name = 'cluster')
    mdlgp.minimize(mdlgp.sum(cluster[k] for k in range(K)))
    
    mdlgp.add_constraints(theta[i, k] <= cluster[k] for i in range(O) for k in range(K))
    mdlgp.add_constraints(sum(theta[i, k] for k in range(K)) == 1 for i in range(O))
    

    # mdlgp.add_constraints(theta[i, k]*D[k, i] <= 7 for i in range(OCRf+1) for k in range(K))
    if OCRf != None:
        for i in range(OCRf + 1):
            for k in range(K):
                    mdlgp.add_constraint(theta[i, k] * D[k, i] <= 7)
    if ORE != OREf:
        mdlgp.add_constraints(theta[i, k]*D[k, i] <= 1 for i in range(ORE, OREf+1) for k in range(K))

    for i in range(OCRf + 1):
        for j in range(ORE, OREf+1):
            for k in range(K):
                mdlgp.add_constraint(theta[i, k] + theta[j, k] <= 1)
    

    # # Maximum number of tasks limit (group capacity constraint) 
    # mdlgp.add_constraints(sum(theta[i, k] for i in range(OCRf+1)) <= 5 for k in range(K))
    # mdlgp.add_constraints(sum(theta[i, k] for i in range(ORE, OREf+1)) <= 3 for k in range(K))

    si2 = mdlgp.solve()
    val = si2.get_blended_objective_value_by_priority()[0]
    print(mdlgp.print_solution(log_output=False))

    df = mdlgp.solution.as_df()
    df = df[df['name'].str.contains('theta')]
    sol = si2.get_blended_objective_value_by_priority()[0]
    dfc = data_clustring(df)
    S = np.zeros(K) 
    
    dfc.to_csv("test1.csv")

    # original
    # Data = dfc.sort_values(by = 'cluster')['cluster'].value_counts().to_frame().reset_index().rename(columns={'index':'values', 'values':'count'})    
    # for v in Data['values']:
    #     S[int(v)] = Data[Data['values']== v]['cluster'].values[0]
    #version1
    # Data = dfc.sort_values(by='cluster')['cluster'].value_counts().to_frame().reset_index(drop=True)
    # for index, count in Data.iterrows():
    #     S[index] = count
    #version3
    Data = dfc.sort_values(by = 'cluster')['cluster'].value_counts().to_frame().reset_index()
    for idx in Data.index:
        v = idx
        S[int(v)] = Data[Data.index == idx]['cluster'].values[0]



    #original
    CC = C[dfc.cluster.values.astype('int')] 



    df_x_y['CC'] = CC
    chart = vis_clustring(df_x_y, CC)
    st.plotly_chart(chart)
    return sol

# Create the Streamlit app
def title_with_background(title, background_color):
    st.markdown(f"""
        <h1 style='background-color:{background_color};border-radius:10px;color:black;text-align:center;margin-bottom:20px'>{title}</h1>
    """, unsafe_allow_html=True)
def line_with_background(title, background_color):
    st.markdown(f"""
                <hr style='background-color:{background_color};border-left: 1px solid #ccc;height: 100%;position: absolute;left: 50%;'></h>
                """, unsafe_allow_html=True)

def app():
    # st.title('Maintenance Caisson')
    title_with_background("Maintenance Planification and Optimization: RTE Use Case", "#f0fdf4")
    st.markdown("<br>", unsafe_allow_html=True) 

    # st.title(':black[Maintenance Caisson] ')

    col1, col2, col3 = st.columns([1, 1.3, 2.4])
    def set_alpha(P):
        # alpha_low = np.random.uniform(0.01, 0.09, P)
        # alpha_high = np.random.uniform(0.004, 0.009, P)
        alpha_high = 0.005


        # PP  = int(P/2)
        # alpha = np.concatenate((alpha_low[:PP], alpha_high[PP:]))
        return alpha_high
    
    with col1:
        st.write("## Model Inputs")
        alert = """ <script>alert('This option is not available.')</script> """
        # deg = components.html(radio_button)  
        deg = st.radio("Degradation/Leakage Type:",
                                    ["Linear degradation", "Function of time", "Stochastic degradation", "Physical modelling"],
                                    index=0,
                                    key="degradation",
                                    )
        # if "Stochastic" or "Physical" in deg:
            # components.html(alert)
            # st.warning("This option is not available yet, please make another choice")
        P = st.slider('### Number of CSEM', min_value=1, max_value=20, step=1)
        T = st.slider('### Time horizon (days)', min_value=30, step=1)
        # Pinit = np.zeros((P, 1))
        # Pinit[:] = 3.5                 
        Pinit = np.random.uniform(3.21, 3.5001, (P, 1))    #//////

        Rcr = st.slider('### Available Technicians for CR', value=100)
        Rre = st.slider('### Available Technicians for REPAIR', value=100)

        # beta = st.slider('Beta', value=3, max_value=5)
        alpha = set_alpha(P)
        alpha_param = np.random.uniform(500, 1000, (P, 1))
                
        taux1 = 3
        taux2 = 3.2
        beta = 3
        gamma = 99999


        # r_cr = np.random.randint(1 , 3,(P,1))   #/////
        # r_re = np.random.randint(1 , 3,(P,1))    #////
        r_cr = 3   #/////
        r_re = 5    #////
        Dcr = np.random.randint(1 ,3,(P, 1))
        Dre = np.random.randint(1 ,3,(P, 1))  #////
        M = 3.6
        P_rempli = 3.5  #/////
        delta = np.random.randint(0, 1, P)
        deltay = np.random.randint(5, 18, P)

        # deltay = np.random.randint(0, 1, P)   # ///////
        # couts1 : cost of CR
        # couts1 = np.random.uniform(1, 10,  size=P)
        couts1 = 10
        # couts2 : cost of RE
        # couts2 = np.random.uniform(60, 80, size=P)
        couts2 = 100
        b1 = 1
        b2 = 5
        k1 = 1000
        k2 = 4
        k3 = 20
        W = 15

        C_crmax = 5
        C_crmax = 2
        

        # constant for perfect equation coefficient = 10^5 * V/RT * 146 /1000
        # K = 2052.54
        # K = 299.67
        K = 5.993        # volume = 1
        # K = 59.93          # volume = 10

    

     
        if 'df_p' not in st.session_state:
            st.session_state.df_p = None

        if st.button('Run Optimization'):

            with st.spinner("Running function..."): 
                df_p, df_x_y, sol, total_leakage_value = create_model(P, T, Pinit, Rcr, Rre, r_cr, r_re, P_rempli, alpha, alpha_param, deg, taux1, taux2, gamma, M, couts1, couts2, b1, b2, k1, k2, k3, W, K)
                st.session_state.df_p = df_p
                st.session_state.df_x_y = df_x_y
                st.session_state.sol = np.around(sol, 2)
                st.session_state.total_leakage_value = total_leakage_value
            st.session_state.couts1 = couts1
            st.session_state.couts2 = couts2
            
       
    # st.markdown('<div class="vertical-line"></div>', unsafe_allow_html=True)
    
    # Create the Docplex model and get the solution
    with col2:
        if st.session_state.df_p is not None :
            dff = df_x_y_mod(st.session_state.df_x_y, st.session_state.couts1, st.session_state.couts2)
            dff.index = dff.index + 1  # 将索引改为从1开始
            # dff =st.session_state.df_x_y
            st.write('## Optimization Output')
            st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
            # st.write(dff.to_html(index=False), unsafe_allow_html=True)

            st.write('#### Total Cost (€):', st.session_state.sol)
            st.write('#### Total leakage quantity (kg):', st.session_state.total_leakage_value)
            st.write('#### Maintenance Actions:', dff.shape[0])
            st.write('##### Complément Remplissage (CR):', dff[dff['OPERATION'] == 'CR'].shape[0])
            st.write('##### Repair (RE):', dff[dff['OPERATION'] == 'RE'].shape[0])

            dff = dff.rename(columns={'TEMPS': 'TIME'})
            st.dataframe(dff, height=350)
            
    with col3:
        caissons = [f"Caisson {i+1}" for i in range(P)]

        if st.session_state.df_p is not None :
            st.write("## Data Visualization")
            c = st.radio('CSEM Visualization', caissons, horizontal=True)
            st.markdown("<hr>", unsafe_allow_html=True) 

            if st.button(f'Visualize all CSEMs'):   
                # Visualize the clustering
                fig2 = vis_all_caisson(st.session_state.df_x_y, st.session_state.df_p, taux1)
                st.plotly_chart(fig2)
            else:
                fig = vis_caisson(st.session_state.df_x_y, st.session_state.df_p, str(int(str(c)[-1]) - 1), taux1)
                st.plotly_chart(fig)


    st.markdown("<hr>", unsafe_allow_html=True) 
 
    col3, col4 = st.columns([2, 1])
    col11, col12 = st.columns([1, 0.01])
    col5, col6, col7 = st.columns([1, 1, 1])
    col8, col9, col10 = st.columns([1, 1, 1])
    
    if st.button('Maintenance Groupping'): 

        # with col11:
        #     comm = f"""
        #     <h1 style="box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);margin: 10px;padding:15px">Assuming that the cost  of traveling represent 40% of total cost of maintenance</h1>
        #     """
        #     components.html(comm)

        with col3:
            if st.session_state.df_x_y is not None :
                # Visualize the clustering
                sol = st.session_state.sol
                sol2 = clusteringg(st.session_state.df_x_y)
                # vis_clustring(st.session_state.df_x_y, (st.session_state.df_x_y))
                O = st.session_state.df_x_y.shape[0]
                # gsol = round(0.6*sol + (0.4*sol)/sol2, 2)
                gsol = round(0.6*sol + (0.4*sol2)/O, 2)
                st.session_state.gsol = gsol
                gapc= round(float((sol - gsol)/sol)*100, 2)
                gapc = f"{gapc}%"

                gap= round(float((O - sol2)/O)*100, 2)
                gap = f"{gap}%"
            st.markdown("<style>.big-column{padding-right: 30px;}</style>", unsafe_allow_html=True)
        with col4:
            st.session_state.sol2 = sol2
            if st.session_state.df_p is not None :
                dfa = update_df_x_y(st.session_state.df_x_y)
                st.dataframe(dfa, height=450)
     
        fm = st.session_state.sol
        fm2 = st.session_state.gsol
        fm3 = st.session_state.sol2
                        
        with col5:
            com = f"""
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding-top:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>Maintenance Cost (Before grouping)</h2>
                        <h1>{fm} (€)</h1>
                        <p></p>
                    </div>
                    """
            components.html(com)
            # st.write('<style>  background-color: #FFFFFF;border-radius: 10px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding: 20px;text-align: center;}</style>', unsafe_allow_html=True)
        with col6:
            com = f"""
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding-top:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>Maintenace Cost (After grouping)</h2>
                        <h1>{fm2} (€)</h1>
                        <p></p>

                    </div>
                    """
            components.html(com)
        with col7:
            com = f"""
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding-top:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>GAP</h2>
                        <h1 style="color:#22c55e"><span><svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-arrow-big-up-line" width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">
                        <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
                        <path d="M9 12h-3.586a1 1 0 0 1 -.707 -1.707l6.586 -6.586a1 1 0 0 1 1.414 0l6.586 6.586a1 1 0 0 1 -.707 1.707h-3.586v6h-6v-6z"></path>
                        <path d="M9 21h6"></path>
                        </svg></span>  {gapc}</h1>

                    </div>
                    """
            components.html(com)
        with col8:
            com = f"""
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>Travels (Before grouping)</h2>
                        <h1>{O}</h1>

                    </div>
                    """
            components.html(com)                
        with col9:
            com = f"""
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>Travels (After grouping)</h2>
                        <h1>{int(sol2)}</h1>

                    </div>
                    """
            components.html(com)
        
        with col10:
            com = f"""
                <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding:2px; border-radius: 5px; text-align: center; height: 130px;">
                    <h2>GAP</h2>
                    <h1 style="color:#10b981"><span><svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-arrow-big-up-line" width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">
                        <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
                        <path d="M9 12h-3.586a1 1 0 0 1 -.707 -1.707l6.586 -6.586a1 1 0 0 1 1.414 0l6.586 6.586a1 1 0 0 1 -.707 1.707h-3.586v6h-6v-6z"></path>
                        <path d="M9 21h6"></path>
                        </svg></span>{gap}</h1>
                    <p></p>

                </div>
                """
            components.html(com)


# Run the app
if __name__ == '__main__':
    app()
    



2025.01.23 VM
import sys
sys.path.append('C:\\Program Files\\IBM\\ILOG\\CPLEX_Studio210\\cplex\\python\\20.1\\x64_win64')
sys.path.append("C:\\ProgramData\\anaconda3\\Lib\\site-packages")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from docplex.mp.model import Model
import streamlit.components.v1 as components
import plotly.graph_objects as go


st.set_page_config(layout="wide")
hide_dataframe_row_index = """
            <style>
         div.c1{
                    background-color: #f5f5f5;
                    border: 2px solid;
                    padding: 20px 20px 20px 20px;
                    border-radius: 10px;
                    color: #ffc300;
                    box-shadow: 10px;
                    }
            </style>
            """


def data(x):
    df_p = x[x['value'] != 1].reset_index()
    df_x_y = x[x['value'].astype(int) == 1].reset_index()
    for i in range(df_p.shape[0]):
        r = df_p["name"].loc[i]
        e = r.split('_')[0]
        g = r.split('_')[1]
        h = r.split('_')[2]
        if (e == 'p'):
            df_p.loc[i, 'CAISSON'] = g
            df_p.loc[i, 'TEMPS'] = int(h)

    for i in range(df_x_y.shape[0]):
        r = df_x_y["name"].loc[i]
        e = r.split('_')[0]
        g = r.split('_')[1]
        h = r.split('_')[2]
        if(e == 'y') | (e == 'x'):
            df_x_y.loc[i, 'OPERATION'] = e
            df_x_y.loc[i, 'CAISSON'] = g
            df_x_y.loc[i, 'TEMPS'] = int(h)
        # else:
        #     df_x_y.loc[i, 'OPERATION'] = "NA"
        #     df_x_y.loc[i, 'CAISSON'] = g
        #     df_x_y.loc[i, 'TEMPS'] = 0
    df_x_y = df_x_y.dropna()[['value', 'OPERATION', 'CAISSON', 'TEMPS']].reset_index()
    

    
    return df_p, df_x_y


# Function to visualize the clustering
@st.cache_data
def vis_caisson(df_x_y, df_p, x, taux1):
    # Filter the data based on the selected caissons
    sns.color_palette("Paired")
    sns.set(rc={"figure.figsize":(12, 5)})
    sns.despine(left=True)
    x = [str(s) for s in x]
    df_p_caisson = df_p[df_p['CAISSON'].isin(x)]
    caisson = df_x_y[df_x_y['CAISSON'].isin(x)]


    chart = px.line(df_p_caisson, x='TEMPS', y='value', color='CAISSON', width=700, height=400)
    
    # Add the reference lines and vertical lines for the operations
    chart.add_hline(y=taux1, line_dash='dash', line_color='red', name='Taux 1')
    chart.add_hline(y=3.5, line_dash='dash', line_color='blue', name='3.5')
    
    for temp in caisson[caisson['OPERATION'] == 'x']['TEMPS'].values:
        chart.add_vline(x=temp, line_dash='dash', line_color='green', name='CR')
        chart.add_annotation(x=temp+0.5, y=2.95, text='CR', showarrow=False)
    
    for temp in caisson[caisson['OPERATION'] == 'y']['TEMPS'].values:
        chart.add_vline(x=temp, line_dash='dash', line_color='yellow', name='RE')
        chart.add_annotation(x=temp+0.5, y=2.95, text='RE', showarrow=False)
    
    
    # Set the x-axis tick angle and remove the legend title
    chart.update_layout(xaxis_tickangle=-45)
    
    # Set the y-axis range to start from 2.9
    chart.update_yaxes(range=[2.9, None])
    chart.update_xaxes(dtick='M1')

    # Remove legend
    chart.update_layout(showlegend=False)
   
   
    return chart


# newly editted function to show the legend index correctly
def vis_all_caisson(df_x_y, df_p, taux1):
    # Filter the data based on the selected caissons
    df_p['value'] = df_p['value'].astype(float)
    
    # Create a line chart of the pressure data using Plotly Express
    chart = px.line(df_p, x='TEMPS', y='value', color='CAISSON', width=700, height=400,
                    color_discrete_sequence=px.colors.qualitative.Plotly)
    
    # Add the reference lines
    chart.add_hline(y=taux1, line_dash='dash', line_color='green', name='Taux 1')
    chart.add_hline(y=3.5, line_dash='dash', line_color='blue', name='3.5')
    
    # Set the x-axis tick angle and legend title
    chart.update_layout(xaxis_tickangle=-45, legend_title='Caissons', plot_bgcolor='#f0fdf4')

    # Modify legend labels to start from 1
    for i, trace in enumerate(chart.data):
        chart.data[i].name = 'Caisson {}'.format(i+1)


    
    return chart


def create_model(P, T, Pinit, Rcr, Rre, r_cr, r_re, P_rempli, alpha,alpha_param,deg, taux1, taux2, gamma, M, couts1, couts2, b1, b2, k1, k2, k3, W, K):

    """ Model """
    mdl = Model('Maintenance_caisson')
    x = mdl.binary_var_matrix(P, T, name = 'x')  
    y = mdl.binary_var_matrix(P, T, name = 'y')
    
    f = mdl.binary_var_matrix(P, T, name = 'f')  
    g = mdl.binary_var_matrix(P, T, name = 'g')
    
    z = mdl.binary_var_matrix(P, T, name = 'z')
    w = mdl.binary_var_matrix(P, T, name = 'w')
    
    e = mdl.integer_var_matrix(P, T, name="e")

    p = mdl.continuous_var_matrix(P, T, name='p')

    d = mdl.binary_var_matrix(P, T, name = 'd')

    o = mdl.binary_var_matrix(P, T, name = 'o')

    v = mdl.continuous_var_matrix(P, T, name = 'v')

    # 计算递增速率
    slope = mdl.continuous_var_matrix(P, T, name='slope')



    # leakageQuantity = mdl.continuous_var_list(P, name='leakageQuantity')
    leakageQuantity = 0.0


    # Add constraints and objective function to the model
    # CONTRAINTES D'INITIALISATION DE PRESSION(OU DE PRESSIONS INITIAUX)
    mdl.add_constraints((p[i, 0] == (Pinit[i].item()) for i in range(P)), names = 'SPinit')
    mdl.add_constraints((p[i, t] >= 3.001 for i in range(P) for t in range(T)), names = 'SPmin')
    mdl.add_constraints((p[i, t] <= 3.5 for i in range(P) for t in range(T)), names = 'SPmax')

    
    # # CONTRAINTE DE RESOURCES 
    mdl.add_constraints((sum(r_cr*x[k, tp] for k in range(P) for tp in range(T) if (tp <= t) and (tp + b1 - 1>= t)) <= Rcr for t in range(T)))
    mdl.add_constraints((sum(r_re*y[k, tp] for k in range(P) for tp in range(T) if (tp <= t) and (tp + b2 - 1>= t)) <= Rre for t in range(T)))
    #-----------------------------------------------------------------------------------------------------------------------------

    if deg == 'Linear degradation':
        mdl.add_constraints((p[i, t+1] >= (1- alpha)*p[i, t] - M*(f[i, t] + (g[i, t] + e[i,t])) for i in range(P) for t in range(T-1)))
        mdl.add_constraints((p[i, t+1] <= (1- alpha)*p[i, t] + M*(f[i, t] + (g[i, t] + e[i,t])) for i in range(P) for t in range(T-1)))    
    else:
        mdl.add_constraints((p[i, t+1] >= (1- alpha_param[i][0]**(t/(t + 34) - 1))*p[i, t] - M*(f[i, t] + g[i, t]) for i in range(P) for t in range(1, T-1)))
        mdl.add_constraints((p[i, t+1] <= (1- alpha_param[i][0]**(t/(t + 34) - 1))*p[i, t] + M*(f[i, t] + g[i, t]) for i in range(P) for t in range(1, T-1)))
        
    #-----------------------------------------------------------------------------------------------------------------------------
    mdl.add_constraints((p[i, t+b1] >= P_rempli - M*(1 - x[i, t]) for i in range(P) for t in range(T-b1)), names = 'AFTER CR')
    mdl.add_constraints((p[i, t+b1] <= P_rempli + M*(1 - x[i, t]) for i in range(P) for t in range(T-b1)), names = 'AFTER CR')

    # Add linear interpolation constraints for CR
    for i in range(P):
        for t in range(T - b1):  # Ensure that time frames are not exceeded
            for k in range(1, b1):
                # Calculate the desired pressure value at time t+k
                expected_pressure = p[i, t] + (k / b1) * (P_rempli - p[i, t])
                mdl.add_constraint(p[i, t + k] >= expected_pressure - M * (1 - x[i, t]))
                mdl.add_constraint(p[i, t + k] <= expected_pressure + M * (1 - x[i, t]))

        for t in range(T - b1, T):
            for k in range(1, T - t):
                expected_pressure = p[i, t] + (k / (T-t)) * (P_rempli - p[i, t])
                mdl.add_constraint(p[i, t + k] >= expected_pressure - M * (1 - x[i, t]))
                mdl.add_constraint(p[i, t + k] <= expected_pressure + M * (1 - x[i, t]))

    # Add linear interpolation constraints for RE
    for i in range(P):
        for t in range(T - b2):  # Ensure that time frames are not exceeded
            for k in range(1, b2):
                # Calculate the desired pressure value at time t+k
                expected_pressure = p[i, t] + (k / b2) * (P_rempli - p[i, t])
                mdl.add_constraint(p[i, t + k] >= expected_pressure - M * (1 - y[i, t]))
                mdl.add_constraint(p[i, t + k] <= expected_pressure + M * (1 - y[i, t]))

    for t in range(T - b2, T):
        # for k in range(1, T - t):
        for k in range(0, T - t):
            expected_pressure = p[i, t] + (k / (T-t)) * (P_rempli - p[i, t])
            mdl.add_constraint(p[i, t + k] >= expected_pressure - M * (1 - y[i, t]))
            mdl.add_constraint(p[i, t + k] <= expected_pressure + M * (1 - y[i, t]))

    # if RE make pressure stay stable

    # for i in range(P):
    #     for t in range(T):
    #         mdl.add_constraints((p[i, tp] >= p[i, t] - M*(1 - y[i, t]) for tp in range(t+b2, T)), names = 'AFTER RE')
    #         mdl.add_constraints((p[i, tp] <= p[i, t] + M*(1 - y[i, t]) for tp in range(t+b2, T)), names = 'AFTER RE')
    
    # if RE make pressure come back to 3.5
    for i in range(P):
        for t in range(T):
            mdl.add_constraints((p[i, tp] >= P_rempli - M*(1 - y[i, t]) for tp in range(t+b2, T)), names = 'AFTER RE')
            mdl.add_constraints((p[i, tp] <= P_rempli + M*(1 - y[i, t]) for tp in range(t+b2, T)), names = 'AFTER RE')
    
        
                 
    mdl.add_constraints((y[i, tp] <= 1 - y[i, t] for i in range(P) for t in range(T-b2) for tp in range(t+b2, T)), names = 'AFTER RE')

    # mdl.add_constraints(x[i, t] + y[i, t] <= 1 for i in range(P) for t in range(T))

    # Additional constraints to define d[i, t]
    # mdl.add_constraints((p[i, t] - 3.2 <= (1 - d[i, t]) for i in range(P) for t in range(T)), names='d_upper_bound')
    # mdl.add_constraints((3.2 - p[i, t] <= d[i, t] for i in range(P) for t in range(T)), names='d_lower_bound')

    #calculate the action slope, if x[i, t] == 1，then slope = (3.5 - p[i, t]) / b
    # mdl.add_constraints((slope[i, t] == (3.5 - p[i, t])/b1 for i in range(P) for t in range(T)), names='slope_calculation')
    
            
    #-----------------------------------------------------------------------------------------------------------------------------
    # definition for cumulating the sum of RE action
    for t in range(T):
        for i in range(P):
                mdl.add_constraint(e[i,t] == mdl.sum(y[i, k] for k in range(t)))

    #-----------------------------------------------------------------------------------------------------------------------------
    # w[i][t] = sum(x[i, tp] for tp in range(t - W + 1, t + 1))
    mdl.add_constraints((w[i, t] == mdl.sum(x[i, tp] for tp in range(max(0, t - W + 1), t + 1)) for i in range(P) for t in range(T)), names='sum_constraint')
    # w[i][t] <= 2 + z[i][t]
    mdl.add_constraints((w[i, t] <= 2 + z[i, t] for i in range(P) for t in range(T)), names='upper_bound_constraint')
    # w[i][t] >= 3 - 3 * (1 - z[i][t])
    mdl.add_constraints((w[i, t] >= 3 - 3 * (1 - z[i, t]) for i in range(P) for t in range(T)), names='lower_bound_constraint')

    #-----------------------------------------------------------------------------------------------------------------------------
    # f[i][t+k] >= x[i][t]  ∀i ∈ P, ∀t ∈ T-b, ∀k = 0, 1, ..., b
    mdl.add_constraints((f[i, t + k] >= x[i, t] for i in range(P) for t in range(T - b1) for k in range(b1)), names='f_constraint')
    
    # g[i][t+k] >= y[i][t]  ∀i ∈ P, ∀t ∈ T-b, ∀k = 0, 1, ..., b
    mdl.add_constraints((g[i, t + k] >= y[i, t] for i in range(P) for t in range(T - b2) for k in range(b2)), names='g_constraint')

    # make sure f[i, t] = 0 at other moments
    mdl.add_constraints((f[i, t] <= sum(x[i, tp] for tp in range(max(0, t - b1 + 1), t + 1)) for i in range(P) for t in range(T)), names='f_zero_constraint')
    # make sure g[i, t] = 0 at other moments
    mdl.add_constraints((g[i, t] <= sum(y[i, tp] for tp in range(max(0, t - b2 + 1), t + 1)) for i in range(P) for t in range(T)), names='g_zero_constraint')
    # mdl.add_constraints((g[i, t] + y[i, t] >= sum(y[i, tp] for tp in range(t, T)) for i in range(P) for t in range(T)), names='g_zero_constraint')

    mdl.add_constraints(x[i, t] + x[i, tp] <= 1 for i in range(P) for t in range(T) for tp in range(T) if (tp < t) and (tp + b1 >= t)) 
    # mdl.add_constraints(y[i, t] + y[i, tp] <= 1 for i in range(P) for t in range(T) for tp in range(T) if (tp < t) and (tp + b2 >= t))

    # # define variable v - original way
    # for i in range(P):
    #     v[i, 0] = 0
    # mdl.add_constraints((v[i,t] >= p[i,t-1] - p[i,t] for i in range(P) for t in range(1,T)), names='v_constraint')
    # mdl.add_constraints((v[i,t] >= 0 for i in range(P) for t in range(1,T)), names='v_constraint')

    # define variable v - new way !!!!
    for i in range(P):
        for t in range(1, T):
            mdl.add_constraint(v[i, t] >= p[i, t-1] - p[i, t])
            mdl.add_constraint(v[i, t] >= 0)
            mdl.add_constraint(v[i, t] <= (p[i,t-1] - p[i, t]) + M * (1 - o[i, t]))
            mdl.add_constraint(v[i, t] <= M * o[i, t])
            mdl.add_constraint(M * o[i, t] >= p[i,t-1] - p[i, t])
            mdl.add_constraint(M * (1- o[i, t]) >= - (p[i,t-1] - p[i, t]))


    #-------------------------------------------------------------------------------------------------------------------------------
    
    # mdl.add_constraint(sum((((3.5 - p[i,tp])*(10**5))*V/(8.314 * 293)*146.0)/1000 for i in range(P) for tp in range(t)) <= 100.0)
    # separate leakage quantity for each caisson
    # for i in range(P):
    #     leakageQuantity[i] = mdl.sum(K * (p[i, t-1] - p[i, t]) for t in range(1,T))
    #     mdl.add_constraint(leakageQuantity[i] <= 0.08)

    # total leakage quantity constraint     
    leakageQuantity = mdl.sum(K * v[i, t] for i in range(P) for t in range(1,T))
    mdl.add_constraint(leakageQuantity <= 40.0)
    


    #-----------------------------------------------------------------------------------------------------------------------------
    # OBJ FUNCTION
    cost = mdl.sum(couts1*x[i, t] + couts2*y[i, t] + k1*z[i,t] for i in range(P) for t in range(T))
    mdl.minimize(cost)

    # Solve the model
    si = mdl.solve()
    df = mdl.solution.as_df()

    st.dataframe(df)

    # Calculate total leakage from the solution 
    total_leakage_value = round(leakageQuantity.solution_value, 3)
    

    # 如果你需要总泄漏量
    st.write("Total leakage quantity: {}".format(si[leakageQuantity]))


    sol = si.get_blended_objective_value_by_priority()[0]
    df_p, df_x_y = data(df)

    # if si:
    #     df = mdl.solution.as_df()
    #     sol = si.get_blended_objective_value_by_priority()[0]
    #     df_p, df_x_y = data(df)
    #     st.dataframe(df)
    # else:
    #     # Handle case where the model couldn't be solved
    #     print("Model could not be solved.")
    #     return None, None, None  # Or handle the failure gracefully
    
    # create file name
    file_name = "P{}_T{}.csv".format(P, T)


    # save DataFrame to file csv
    df_x_y.to_csv(file_name, index=False, encoding='utf-8')


    print("Results saved to file: {}".format(file_name))

    # Return the solution
    return df_p, df_x_y, sol, total_leakage_value



def data_clustring(x):
    for i in range(x.shape[0]):
        r = x["name"].loc[i]
        e = r.split('_')[0]
        g = r.split('_')[1]   # types of operations
        h = r.split('_')[2]   # cluster information
        x.loc[i, 'cluster'] = h
        x.loc[i, 'OPERATION'] = g
        
    return x

def update_df_x_y(df_x_y):
    df_x_y = df_x_y.rename(columns={"CC": "center"})
    df_x_y["OPERATION"] = df_x_y["OPERATION"].replace({"x": "CR", "y": "RE"})
    df_x_y = df_x_y.drop("value", axis=1)
    df_x_y = df_x_y.drop("index", axis=1)
    return df_x_y
    
def vis_clustring(df_x_y, C):
    # Set the color palette and plot size
    # sns.color_palette("Paired")
    sns.set(rc={"figure.figsize":(12, 5)})
    sns.despine(left=True)
    
    # Get the number of caissons and data points
    x = len(df_x_y['CAISSON'].unique()) 
    y = df_x_y.shape[0]
  
    # Create a scatter plot with Plotly Express
    df_x_y['cluster'] = pd.factorize(df_x_y['CC'])[0]
    df_x_y["cluster"] = df_x_y["cluster"].astype(str)
    color_sequence = px.colors.qualitative.Dark24
    color_dict = {str(i): px.colors.qualitative.Plotly[i] for i in range(df_x_y['cluster'].nunique())}
    label_dict = {str(i): 'Category {}'.format(chr(i + 65)) for i in range(df_x_y['cluster'].nunique())}
    size_dict = {str(i): (i + 1) * 20 for i in range(df_x_y['cluster'].nunique())}

    centroid_data = df_x_y[df_x_y['TEMPS'] == df_x_y['CC']]

    df_x_y = df_x_y.rename(columns={"CC": "center"})
    df_x_y["OPERATION"] = df_x_y["OPERATION"].replace({"x": "CR", "y": "RE"})
    
    for i in range(df_x_y.shape[0]):
        if df_x_y.loc[i, 'OPERATION'] == "CR":
            df_x_y.loc[i, 'TOLERANCE'] = 7  
        if df_x_y.loc[i, 'OPERATION'] == "RE":
            df_x_y.loc[i, 'TOLERANCE'] = 1  
        


    # Create the scatter plot with background colors for each cluster
    chart = px.scatter(df_x_y.sort_values('CAISSON', ascending=False), x='TEMPS', y='CAISSON', color='cluster',color_discrete_map=color_dict,
                    
                     width=800, height=500, hover_data=['center', 'OPERATION', 'TOLERANCE'])
    
    # Add a background color to each cluster
    for i, color in color_dict.items():
        cluster_data = df_x_y[df_x_y['cluster'] == str(i)]
        cluster_center = pd.DataFrame(cluster_data.loc[(cluster_data['TEMPS'] == cluster_data['center'].astype(int)), ['TEMPS', 'CAISSON']].head(1))

        for index, row in cluster_data.iterrows():           
            x1 = row['TEMPS']
            y1 = row['CAISSON']
            try:
                x2 = cluster_center.iloc[0]['TEMPS']
                y2 = cluster_center.iloc[0]['CAISSON']
                if x1 != x2:
                    # Draw the lasso line
                    chart.add_shape(type='line',
                                x0=x1,
                                y0=y1,
                                x1=x2,
                                y1=y2,
                                line=dict(color=color, width=1, dash='dot'))
    
                else:
                    chart.add_shape(type='circle',
                            xref='x', yref='y',
                            x0=int(x1)-0.5, y0=int(y1)-0.2,
                            x1=int(x1)+0.5, y1=int(y1)+0.2,
                            line=dict(color=color, width=2))
            # chart.update_shapes(line_color=color)
            except:
                continue

    # Customize the plot
    chart.update_traces(marker=dict(sizemode='diameter', sizeref=120))
    chart.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=2, range=[0,45]),
                         yaxis=dict(tickmode='linear', tick0=-2, dtick=2, range=[-2,x+2]),
                         legend_title='', showlegend=True)
    chart.update_layout(plot_bgcolor='#fafafa')
    
    return chart








@st.cache_data
def df_x_y_mod(df, couts1, couts2):
    # df = df.drop(['index', 'value'], axis=1)

    # Drop 'index' and 'value' columns if they exist
    df = df.drop(['index', 'value'], axis=1, errors='ignore')

    # replace the values in the OPERATION column
    df['OPERATION'] = df['OPERATION'].replace({'x': 'CR', 'y': 'RE'})

    # increment the values in the CAISSON column
    
    P = int(df['CAISSON'].nunique())

    # price_map = {('CR', i): couts1[i] for i in range(P)}
    # price_map.update({('RE', i): couts2[i] for i in range(P)})
    price_map = {('CR'): couts1}
    price_map.update({('RE'): couts2})
    
    # create the Price column based on the OPERATION and CAISSON columns
    # df['Price'] = [price_map[(op, int(c))] for op, c in zip(df['OPERATION'], df['CAISSON'])]
    
    df['CAISSON'] = df['CAISSON'].astype('int') + 1 
    caisson_col = df.pop('CAISSON')
    df.insert(0, 'CAISSON', caisson_col)
    return df

def clusteringg(df_x_y):
    
    C = df_x_y.TEMPS.values

    D = np.array([i - np.array(df_x_y.TEMPS.values.astype(int)) for i in C]) # les distances entre les points (time difference)
    D[D[:] < 0 ] = +99999
#     print(f'the matrix D is : {D}')
    
    OCR = df_x_y[df_x_y['OPERATION'].str.contains('x')].first_valid_index() # les points de nuage DES CR
    OCRf = df_x_y[df_x_y['OPERATION'].str.contains('x')].last_valid_index() # les points de nuage DES CR
    try:
        ORE = df_x_y[df_x_y['OPERATION'].str.contains('y')].first_valid_index()  # les points de nuage DES RE
        OREf = df_x_y[df_x_y['OPERATION'].str.contains('y')].last_valid_index()  # les points de nuage DES REF
    except:
        ORE = None

    O = df_x_y.shape[0]
    # df_x_y.to_excel('df.xlsx', index=False)
    K = O
    mdlgp = Model('Maintance_grouping')
    theta = mdlgp.binary_var_matrix(O, K, name = 'theta')
    M = 99999
   
    cluster = mdlgp.binary_var_list(K, name = 'cluster')
    mdlgp.minimize(mdlgp.sum(cluster[k] for k in range(K)))
    
    mdlgp.add_constraints(theta[i, k] <= cluster[k] for i in range(O) for k in range(K))
    mdlgp.add_constraints(sum(theta[i, k] for k in range(K)) == 1 for i in range(O))
    

    # mdlgp.add_constraints(theta[i, k]*D[k, i] <= 7 for i in range(OCRf+1) for k in range(K))
    if OCRf != None:
        for i in range(OCRf + 1):
            for k in range(K):
                    mdlgp.add_constraint(theta[i, k] * D[k, i] <= 7)    # we consider here only the positive distances meaning that we don't look back
                    
    if ORE != OREf:
        mdlgp.add_constraints(theta[i, k]*D[k, i] <= 1 for i in range(ORE, OREf+1) for k in range(K))


    if OCRf != None and ORE != None:
        for k in range(K):
            for i in range(OCRf + 1):
                for j in range(ORE, OREf+1):
                    mdlgp.add_constraint(theta[i, k] + theta[j, k] <= 1)
                    
                    # Maximum number of tasks limit (group capacity constraint) 
                    mdlgp.add_constraints(sum(theta[i, k] for i in range(OCRf+1)) <= 5 for k in range(K))
                    mdlgp.add_constraints(sum(theta[i, k] for i in range(ORE, OREf+1)) <= 3 for k in range(K))



    si2 = mdlgp.solve()
    val = si2.get_blended_objective_value_by_priority()[0]
    print(mdlgp.print_solution(log_output=False))

    df = mdlgp.solution.as_df()
    df = df[df['name'].str.contains('theta')]
    sol = si2.get_blended_objective_value_by_priority()[0]
    dfc = data_clustring(df)
    S = np.zeros(K) 
    
    dfc.to_csv("test1.csv")

    # original
    # Data = dfc.sort_values(by = 'cluster')['cluster'].value_counts().to_frame().reset_index().rename(columns={'index':'values', 'values':'count'})    
    # for v in Data['values']:
    #     S[int(v)] = Data[Data['values']== v]['cluster'].values[0]
    #version1
    # Data = dfc.sort_values(by='cluster')['cluster'].value_counts().to_frame().reset_index(drop=True)
    # for index, count in Data.iterrows():
    #     S[index] = count
    #version3
    Data = dfc.sort_values(by = 'cluster')['cluster'].value_counts().to_frame().reset_index()
    for idx in Data.index:
        v = idx
        S[int(v)] = Data[Data.index == idx]['cluster'].values[0]


    #original
    CC = C[dfc.cluster.values.astype('int')] 

    df_x_y['CC'] = CC
    chart = vis_clustring(df_x_y, CC)
    st.plotly_chart(chart)
    return sol

# Create the Streamlit app
def title_with_background(title, background_color):
    st.markdown("""
                <h1 style='background-color:{};border-radius:10px;color:black;text-align:center;margin-bottom:20px'>{}</h1>
                """.format(background_color, title), unsafe_allow_html=True)

def line_with_background(title, background_color):
    st.markdown("""
                <hr style='background-color:{};border-left: 1px solid #ccc;height: 100%;position: absolute;left: 50%;'></h>
                """.format(background_color), unsafe_allow_html=True)


def app():
    # st.title('Maintenance Caisson')
    title_with_background("Maintenance Planification and Optimization: RTE Use Case", "#f0fdf4")
    st.markdown("<br>", unsafe_allow_html=True) 

    # st.title(':black[Maintenance Caisson] ')

    col1, col2, col3 = st.columns([1, 1.3, 2.4])
    def set_alpha(P):
        # alpha_low = np.random.uniform(0.01, 0.09, P)
        # alpha_high = np.random.uniform(0.004, 0.009, P)
        alpha_high = 0.005


        # PP  = int(P/2)
        # alpha = np.concatenate((alpha_low[:PP], alpha_high[PP:]))
        return alpha_high
    
    with col1:
        st.write("## Model Inputs")
        alert = """ <script>alert('This option is not available.')</script> """
        # deg = components.html(radio_button)  
        deg = st.radio("Degradation/Leakage Type:",
                                    ["Linear degradation", "Function of time", "Stochastic degradation", "Physical modelling"],
                                    index=0,
                                    key="degradation",
                                    )
        # if "Stochastic" or "Physical" in deg:
            # components.html(alert)
            # st.warning("This option is not available yet, please make another choice")
        P = st.slider('### Number of CSEM', min_value=1, max_value=20, step=1)
        T = st.slider('### Time horizon (days)', min_value=30, step=1)
        # Pinit = np.zeros((P, 1))
        # Pinit[:] = 3.5                 
        Pinit = np.random.uniform(3.21, 3.5001, (P, 1))    #//////

        Rcr = st.slider('### Available Technicians for CR', value=100)
        Rre = st.slider('### Available Technicians for REPAIR', value=100)

        # beta = st.slider('Beta', value=3, max_value=5)
        alpha = set_alpha(P)
        alpha_param = np.random.uniform(500, 1000, (P, 1))
                
        taux1 = 3
        taux2 = 3.2
        beta = 3
        gamma = 99999


        # r_cr = np.random.randint(1 , 3,(P,1))   #/////
        # r_re = np.random.randint(1 , 3,(P,1))    #////
        r_cr = 3   #/////
        r_re = 5    #////
        Dcr = np.random.randint(1 ,3,(P, 1))
        Dre = np.random.randint(1 ,3,(P, 1))  #////
        M = 3.6
        P_rempli = 3.5  #/////
        delta = np.random.randint(0, 1, P)
        deltay = np.random.randint(5, 18, P)

        # deltay = np.random.randint(0, 1, P)   # ///////
        # couts1 : cost of CR
        # couts1 = np.random.uniform(1, 10,  size=P)
        couts1 = 10
        # couts2 : cost of RE
        # couts2 = np.random.uniform(60, 80, size=P)
        couts2 = 100
        b1 = 1
        b2 = 5
        k1 = 1000
        k2 = 4
        k3 = 20
        W = 15

        C_crmax = 5
        C_crmax = 2
        

        # constant for perfect equation coefficient = 10^5 * V/RT * 146 /1000
        # K = 2052.54
        # K = 299.67
        K = 5.993        # volume = 1
        # K = 59.93          # volume = 10

    

     
        if 'df_p' not in st.session_state:
            st.session_state.df_p = None

        if st.button('Run Optimization'):

            with st.spinner("Running function..."): 
                df_p, df_x_y, sol, total_leakage_value = create_model(P, T, Pinit, Rcr, Rre, r_cr, r_re, P_rempli, alpha, alpha_param, deg, taux1, taux2, gamma, M, couts1, couts2, b1, b2, k1, k2, k3, W, K)
                st.session_state.df_p = df_p
                st.session_state.df_x_y = df_x_y
                st.session_state.sol = np.around(sol, 2)
                st.session_state.total_leakage_value = total_leakage_value
            st.session_state.couts1 = couts1
            st.session_state.couts2 = couts2
            
       
    # st.markdown('<div class="vertical-line"></div>', unsafe_allow_html=True)
    
    # Create the Docplex model and get the solution
    with col2:
        if st.session_state.df_p is not None :
            dff = df_x_y_mod(st.session_state.df_x_y, st.session_state.couts1, st.session_state.couts2)
            dff.index = dff.index + 1  # 将索引改为从1开始
            # dff =st.session_state.df_x_y
            st.write('## Optimization Output')
            st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
            # st.write(dff.to_html(index=False), unsafe_allow_html=True)

            st.write('#### Total Cost (€):', st.session_state.sol)
            st.write('#### Total leakage quantity (kg):', st.session_state.total_leakage_value)
            st.write('#### Maintenance Actions:', dff.shape[0])
            st.write('##### Complément Remplissage (CR):', dff[dff['OPERATION'] == 'CR'].shape[0])
            st.write('##### Repair (RE):', dff[dff['OPERATION'] == 'RE'].shape[0])

            dff = dff.rename(columns={'TEMPS': 'TIME'})
            st.dataframe(dff, height=350)
            
    with col3:
        caissons = ["Caisson {}".format(i + 1) for i in range(P)]

        if st.session_state.df_p is not None :
            st.write("## Data Visualization")
            c = st.radio('CSEM Visualization', caissons, horizontal=True)
            st.markdown("<hr>", unsafe_allow_html=True) 

            if st.button('Visualize all CSEMs'):  
                # Visualize the clustering
                fig2 = vis_all_caisson(st.session_state.df_x_y, st.session_state.df_p, taux1)
                st.plotly_chart(fig2)
            else:
                fig = vis_caisson(st.session_state.df_x_y, st.session_state.df_p, str(int(str(c)[-1]) - 1), taux1)
                st.plotly_chart(fig)


    st.markdown("<hr>", unsafe_allow_html=True) 
 
    col3, col4 = st.columns([2, 1])
    col11, col12 = st.columns([1, 0.01])
    col5, col6, col7 = st.columns([1, 1, 1])
    col8, col9, col10 = st.columns([1, 1, 1])
    
    if st.button('Maintenance Groupping'): 

        # with col11:
        #     comm = f"""
        #     <h1 style="box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);margin: 10px;padding:15px">Assuming that the cost  of traveling represent 40% of total cost of maintenance</h1>
        #     """
        #     components.html(comm)

        with col3:
            if st.session_state.df_x_y is not None :
                # Visualize the clustering
                sol = st.session_state.sol
                sol2 = clusteringg(st.session_state.df_x_y)
                # vis_clustring(st.session_state.df_x_y, (st.session_state.df_x_y))
                O = st.session_state.df_x_y.shape[0]
                # gsol = round(0.6*sol + (0.4*sol)/sol2, 2)
                gsol = round(0.6*sol + (0.4*sol2)/O, 2)
                st.session_state.gsol = gsol
                gapc= round(float((sol - gsol)/sol)*100, 2)
                gapc = "{}%".format(gapc)

                gap= round(float((O - sol2)/O)*100, 2)
                gap = "{}%".format(gap)
            st.markdown("<style>.big-column{padding-right: 30px;}</style>", unsafe_allow_html=True)
        with col4:
            st.session_state.sol2 = sol2
            if st.session_state.df_p is not None :
                dfa = update_df_x_y(st.session_state.df_x_y)
                st.dataframe(dfa, height=450)
     
        fm = st.session_state.sol
        fm2 = st.session_state.gsol
        fm3 = st.session_state.sol2
                        
        with col5:
            com = """
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding-top:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>Maintenance Cost (Before grouping)</h2>
                        <h1>{} (€)</h1>
                        <p></p>
                    </div>
                    """.format(int(fm))
            components.html(com)
            # st.write('<style>  background-color: #FFFFFF;border-radius: 10px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding: 20px;text-align: center;}</style>', unsafe_allow_html=True)
        with col6:
            com = """
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding-top:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>Maintenace Cost (After grouping)</h2>
                        <h1>{} (€)</h1>
                        <p></p>

                    </div>
                    """.format(int(fm2))
            components.html(com)
        with col7:
            com = """
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding-top:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>GAP</h2>
                        <h1 style="color:#22c55e"><span><svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-arrow-big-up-line" width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">
                        <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
                        <path d="M9 12h-3.586a1 1 0 0 1 -.707 -1.707l6.586 -6.586a1 1 0 0 1 1.414 0l6.586 6.586a1 1 0 0 1 -.707 1.707h-3.586v6h-6v-6z"></path>
                        <path d="M9 21h6"></path>
                        </svg></span>  {}</h1>

                    </div>
                    """.format(int(gapc))
            components.html(com)
        with col8:
            com = """
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>Travels (Before grouping)</h2>
                        <h1>{}</h1>
                    </div>
                    """.format(O)
            components.html(com)


        with col9:
            com = """
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>Travels (After grouping)</h2>
                        <h1>{}</h1>
                    </div>
                    """.format(int(sol2))
            components.html(com)
        
        with col10:
            com = """
                <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding:2px; border-radius: 5px; text-align: center; height: 130px;">
                    <h2>GAP</h2>
                    <h1 style="color:#10b981"><span><svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-arrow-big-up-line" width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">
                        <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
                        <path d="M9 12h-3.586a1 1 0 0 1 -.707 -1.707l6.586 -6.586a1 1 0 0 1 1.414 0l6.586 6.586a1 1 0 0 1 -.707 1.707h-3.586v6h-6v-6z"></path>
                        <path d="M9 21h6"></path>
                        </svg></span>{}</h1>
                    <p></p>

                </div>
                """.format(int(gap))
            components.html(com)


# Run the app
if __name__ == '__main__':
    app()
    






2025/01/27 all correct version
import sys
sys.path.append('C:\\Program Files\\IBM\\ILOG\\CPLEX_Studio210\\cplex\\python\\20.1\\x64_win64')
sys.path.append("C:\\ProgramData\\anaconda3\\Lib\\site-packages")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from docplex.mp.model import Model
import streamlit.components.v1 as components
import plotly.graph_objects as go
import time


st.set_page_config(layout="wide")
hide_dataframe_row_index = """
            <style>
         div.c1{
                    background-color: #f5f5f5;
                    border: 2px solid;
                    padding: 20px 20px 20px 20px;
                    border-radius: 10px;
                    color: #ffc300;
                    box-shadow: 10px;
                    }
            </style>
            """


def data(x):
    df_p = x[x['value'] != 1].reset_index()
    df_x_y = x[x['value'].astype(int) == 1].reset_index()
    for i in range(df_p.shape[0]):
        r = df_p["name"].loc[i]
        e = r.split('_')[0]
        g = r.split('_')[1]
        h = r.split('_')[2]
        if (e == 'p'):
            df_p.loc[i, 'CAISSON'] = g
            df_p.loc[i, 'TEMPS'] = int(h)

    for i in range(df_x_y.shape[0]):
        r = df_x_y["name"].loc[i]
        e = r.split('_')[0]
        g = r.split('_')[1]
        h = r.split('_')[2]
        if(e == 'y') | (e == 'x'):
            df_x_y.loc[i, 'OPERATION'] = e
            df_x_y.loc[i, 'CAISSON'] = g
            df_x_y.loc[i, 'TEMPS'] = int(h)
        # else:
        #     df_x_y.loc[i, 'OPERATION'] = "NA"
        #     df_x_y.loc[i, 'CAISSON'] = g
        #     df_x_y.loc[i, 'TEMPS'] = 0
    df_x_y = df_x_y.dropna()[['value', 'OPERATION', 'CAISSON', 'TEMPS']].reset_index()
    

    
    return df_p, df_x_y


# Function to visualize the clustering
@st.cache_data
def vis_caisson(df_x_y, df_p, x, taux1):
    # Filter the data based on the selected caissons
    sns.color_palette("Paired")
    sns.set(rc={"figure.figsize":(12, 5)})
    sns.despine(left=True)
    x = [str(s) for s in x]
    df_p_caisson = df_p[df_p['CAISSON'].isin(x)]
    caisson = df_x_y[df_x_y['CAISSON'].isin(x)]


    chart = px.line(df_p_caisson, x='TEMPS', y='value', color='CAISSON', width=700, height=400)
    
    # Add the reference lines and vertical lines for the operations
    chart.add_hline(y=taux1, line_dash='dash', line_color='red', name='Taux 1')
    chart.add_hline(y=3.5, line_dash='dash', line_color='blue', name='3.5')
    
    for temp in caisson[caisson['OPERATION'] == 'x']['TEMPS'].values:
        chart.add_vline(x=temp, line_dash='dash', line_color='green', name='CR')
        chart.add_annotation(x=temp+0.5, y=2.95, text='CR', showarrow=False)
    
    for temp in caisson[caisson['OPERATION'] == 'y']['TEMPS'].values:
        chart.add_vline(x=temp, line_dash='dash', line_color='yellow', name='RE')
        chart.add_annotation(x=temp+0.5, y=2.95, text='RE', showarrow=False)
    
    
    # Set the x-axis tick angle and remove the legend title
    chart.update_layout(xaxis_tickangle=-45)
    
    # Set the y-axis range to start from 2.9
    chart.update_yaxes(range=[2.9, None])
    chart.update_xaxes(dtick='M1')

    # Remove legend
    chart.update_layout(showlegend=False)
   
   
    return chart


# newly editted function to show the legend index correctly
def vis_all_caisson(df_x_y, df_p, taux1):
    # Filter the data based on the selected caissons
    df_p['value'] = df_p['value'].astype(float)
    
    # Create a line chart of the pressure data using Plotly Express
    chart = px.line(df_p, x='TEMPS', y='value', color='CAISSON', width=700, height=400,
                    color_discrete_sequence=px.colors.qualitative.Plotly)
    
    # Add the reference lines
    chart.add_hline(y=taux1, line_dash='dash', line_color='green', name='Taux 1')
    chart.add_hline(y=3.5, line_dash='dash', line_color='blue', name='3.5')
    
    # Set the x-axis tick angle and legend title
    chart.update_layout(xaxis_tickangle=-45, legend_title='Caissons', plot_bgcolor='#f0fdf4')

    # Modify legend labels to start from 1
    for i, trace in enumerate(chart.data):
        chart.data[i].name = f'Caisson {i+1}'
    
    return chart


def create_model(P, T, Pinit, Rcr, Rre, r_cr, r_re, P_rempli, alpha,alpha_param,deg, taux1, taux2, gamma, M, couts1, couts2, b1, b2, k1, k2, k3, W, K):

    """ Model """
    mdl = Model('Maintenance_caisson')
    x = mdl.binary_var_matrix(P, T, name = 'x')  
    y = mdl.binary_var_matrix(P, T, name = 'y')
    
    f = mdl.binary_var_matrix(P, T, name = 'f')  
    g = mdl.binary_var_matrix(P, T, name = 'g')
    
    z = mdl.binary_var_matrix(P, T, name = 'z')
    w = mdl.binary_var_matrix(P, T, name = 'w')
    
    e = mdl.integer_var_matrix(P, T, name="e")

    p = mdl.continuous_var_matrix(P, T, name='p')

    d = mdl.binary_var_matrix(P, T, name = 'd')

    o = mdl.binary_var_matrix(P, T, name = 'o')

    v = mdl.continuous_var_matrix(P, T, name = 'v')

    # 计算递增速率
    slope = mdl.continuous_var_matrix(P, T, name='slope')



    # leakageQuantity = mdl.continuous_var_list(P, name='leakageQuantity')
    leakageQuantity = 0.0


    # Add constraints and objective function to the model
    # CONTRAINTES D'INITIALISATION DE PRESSION(OU DE PRESSIONS INITIAUX)
    mdl.add_constraints((p[i, 0] == (Pinit[i].item()) for i in range(P)), names = 'SPinit')
    mdl.add_constraints((p[i, t] >= 3.001 for i in range(P) for t in range(T)), names = 'SPmin')
    mdl.add_constraints((p[i, t] <= 3.5 for i in range(P) for t in range(T)), names = 'SPmax')

    
    # # CONTRAINTE DE RESOURCES 
    mdl.add_constraints((sum(r_cr*x[k, tp] for k in range(P) for tp in range(T) if (tp <= t) and (tp + b1 - 1>= t)) <= Rcr for t in range(T)))
    mdl.add_constraints((sum(r_re*y[k, tp] for k in range(P) for tp in range(T) if (tp <= t) and (tp + b2 - 1>= t)) <= Rre for t in range(T)))
    #-----------------------------------------------------------------------------------------------------------------------------

    if deg == 'Linear degradation':
        mdl.add_constraints((p[i, t+1] >= (1- alpha)*p[i, t] - M*(f[i, t] + (g[i, t] + e[i,t])) for i in range(P) for t in range(T-1)))
        mdl.add_constraints((p[i, t+1] <= (1- alpha)*p[i, t] + M*(f[i, t] + (g[i, t] + e[i,t])) for i in range(P) for t in range(T-1)))    
    else:
        mdl.add_constraints((p[i, t+1] >= (1- alpha_param[i][0]**(t/(t + 34) - 1))*p[i, t] - M*(f[i, t] + g[i, t]) for i in range(P) for t in range(1, T-1)))
        mdl.add_constraints((p[i, t+1] <= (1- alpha_param[i][0]**(t/(t + 34) - 1))*p[i, t] + M*(f[i, t] + g[i, t]) for i in range(P) for t in range(1, T-1)))
        
    #-----------------------------------------------------------------------------------------------------------------------------
    mdl.add_constraints((p[i, t+b1] >= P_rempli - M*(1 - x[i, t]) for i in range(P) for t in range(T-b1)), names = 'AFTER CR')
    mdl.add_constraints((p[i, t+b1] <= P_rempli + M*(1 - x[i, t]) for i in range(P) for t in range(T-b1)), names = 'AFTER CR')

    # Add linear interpolation constraints for CR
    for i in range(P):
        for t in range(T - b1):  # Ensure that time frames are not exceeded
            for k in range(1, b1):
                # Calculate the desired pressure value at time t+k
                expected_pressure = p[i, t] + (k / b1) * (P_rempli - p[i, t])
                mdl.add_constraint(p[i, t + k] >= expected_pressure - M * (1 - x[i, t]))
                mdl.add_constraint(p[i, t + k] <= expected_pressure + M * (1 - x[i, t]))

        for t in range(T - b1, T):
            for k in range(1, T - t):
                expected_pressure = p[i, t] + (k / (T-t)) * (P_rempli - p[i, t])
                mdl.add_constraint(p[i, t + k] >= expected_pressure - M * (1 - x[i, t]))
                mdl.add_constraint(p[i, t + k] <= expected_pressure + M * (1 - x[i, t]))

    # Add linear interpolation constraints for RE
    for i in range(P):
        for t in range(T - b2):  # Ensure that time frames are not exceeded
            for k in range(1, b2):
                # Calculate the desired pressure value at time t+k
                expected_pressure = p[i, t] + (k / b2) * (P_rempli - p[i, t])
                mdl.add_constraint(p[i, t + k] >= expected_pressure - M * (1 - y[i, t]))
                mdl.add_constraint(p[i, t + k] <= expected_pressure + M * (1 - y[i, t]))

    for t in range(T - b2, T):
        # for k in range(1, T - t):
        for k in range(0, T - t):
            expected_pressure = p[i, t] + (k / (T-t)) * (P_rempli - p[i, t])
            mdl.add_constraint(p[i, t + k] >= expected_pressure - M * (1 - y[i, t]))
            mdl.add_constraint(p[i, t + k] <= expected_pressure + M * (1 - y[i, t]))

    # if RE make pressure stay stable

    # for i in range(P):
    #     for t in range(T):
    #         mdl.add_constraints((p[i, tp] >= p[i, t] - M*(1 - y[i, t]) for tp in range(t+b2, T)), names = 'AFTER RE')
    #         mdl.add_constraints((p[i, tp] <= p[i, t] + M*(1 - y[i, t]) for tp in range(t+b2, T)), names = 'AFTER RE')
    
    # if RE make pressure come back to 3.5
    for i in range(P):
        for t in range(T):
            mdl.add_constraints((p[i, tp] >= P_rempli - M*(1 - y[i, t]) for tp in range(t+b2, T)), names = 'AFTER RE')
            mdl.add_constraints((p[i, tp] <= P_rempli + M*(1 - y[i, t]) for tp in range(t+b2, T)), names = 'AFTER RE')
    
        
                 
    mdl.add_constraints((y[i, tp] <= 1 - y[i, t] for i in range(P) for t in range(T-b2) for tp in range(t+b2, T)), names = 'AFTER RE')

    # mdl.add_constraints(x[i, t] + y[i, t] <= 1 for i in range(P) for t in range(T))

    # Additional constraints to define d[i, t]
    # mdl.add_constraints((p[i, t] - 3.2 <= (1 - d[i, t]) for i in range(P) for t in range(T)), names='d_upper_bound')
    # mdl.add_constraints((3.2 - p[i, t] <= d[i, t] for i in range(P) for t in range(T)), names='d_lower_bound')

    #calculate the action slope, if x[i, t] == 1，then slope = (3.5 - p[i, t]) / b
    # mdl.add_constraints((slope[i, t] == (3.5 - p[i, t])/b1 for i in range(P) for t in range(T)), names='slope_calculation')
    
            
    #-----------------------------------------------------------------------------------------------------------------------------
    # definition for cumulating the sum of RE action
    for t in range(T):
        for i in range(P):
                mdl.add_constraint(e[i,t] == mdl.sum(y[i, k] for k in range(t)))

    #-----------------------------------------------------------------------------------------------------------------------------
    # w[i][t] = sum(x[i, tp] for tp in range(t - W + 1, t + 1))
    mdl.add_constraints((w[i, t] == mdl.sum(x[i, tp] for tp in range(max(0, t - W + 1), t + 1)) for i in range(P) for t in range(T)), names='sum_constraint')
    # w[i][t] <= 2 + z[i][t]
    mdl.add_constraints((w[i, t] <= 2 + z[i, t] for i in range(P) for t in range(T)), names='upper_bound_constraint')
    # w[i][t] >= 3 - 3 * (1 - z[i][t])
    mdl.add_constraints((w[i, t] >= 3 - 3 * (1 - z[i, t]) for i in range(P) for t in range(T)), names='lower_bound_constraint')

    #-----------------------------------------------------------------------------------------------------------------------------
    # f[i][t+k] >= x[i][t]  ∀i ∈ P, ∀t ∈ T-b, ∀k = 0, 1, ..., b
    mdl.add_constraints((f[i, t + k] >= x[i, t] for i in range(P) for t in range(T - b1) for k in range(b1)), names='f_constraint')
    
    # g[i][t+k] >= y[i][t]  ∀i ∈ P, ∀t ∈ T-b, ∀k = 0, 1, ..., b
    mdl.add_constraints((g[i, t + k] >= y[i, t] for i in range(P) for t in range(T - b2) for k in range(b2)), names='g_constraint')

    # make sure f[i, t] = 0 at other moments
    mdl.add_constraints((f[i, t] <= sum(x[i, tp] for tp in range(max(0, t - b1 + 1), t + 1)) for i in range(P) for t in range(T)), names='f_zero_constraint')
    # make sure g[i, t] = 0 at other moments
    mdl.add_constraints((g[i, t] <= sum(y[i, tp] for tp in range(max(0, t - b2 + 1), t + 1)) for i in range(P) for t in range(T)), names='g_zero_constraint')
    # mdl.add_constraints((g[i, t] + y[i, t] >= sum(y[i, tp] for tp in range(t, T)) for i in range(P) for t in range(T)), names='g_zero_constraint')

    mdl.add_constraints(x[i, t] + x[i, tp] <= 1 for i in range(P) for t in range(T) for tp in range(T) if (tp < t) and (tp + b1 >= t)) 
    # mdl.add_constraints(y[i, t] + y[i, tp] <= 1 for i in range(P) for t in range(T) for tp in range(T) if (tp < t) and (tp + b2 >= t))

    # # define variable v - original way
    # for i in range(P):
    #     v[i, 0] = 0
    # mdl.add_constraints((v[i,t] >= p[i,t-1] - p[i,t] for i in range(P) for t in range(1,T)), names='v_constraint')
    # mdl.add_constraints((v[i,t] >= 0 for i in range(P) for t in range(1,T)), names='v_constraint')

    # define variable v - new way !!!!
    for i in range(P):
        for t in range(1, T):
            mdl.add_constraint(v[i, t] >= p[i, t-1] - p[i, t])
            mdl.add_constraint(v[i, t] >= 0)
            mdl.add_constraint(v[i, t] <= (p[i,t-1] - p[i, t]) + M * (1 - o[i, t]))
            mdl.add_constraint(v[i, t] <= M * o[i, t])
            mdl.add_constraint(M * o[i, t] >= p[i,t-1] - p[i, t])
            mdl.add_constraint(M * (1- o[i, t]) >= - (p[i,t-1] - p[i, t]))


    #-------------------------------------------------------------------------------------------------------------------------------
    
    # mdl.add_constraint(sum((((3.5 - p[i,tp])*(10**5))*V/(8.314 * 293)*146.0)/1000 for i in range(P) for tp in range(t)) <= 100.0)
    # separate leakage quantity for each caisson
    # for i in range(P):
    #     leakageQuantity[i] = mdl.sum(K * (p[i, t-1] - p[i, t]) for t in range(1,T))
    #     mdl.add_constraint(leakageQuantity[i] <= 0.08)

    # total leakage quantity constraint     
    leakageQuantity = mdl.sum(K * v[i, t] for i in range(P) for t in range(1,T))
    mdl.add_constraint(leakageQuantity <= 18.75)
    


    #-----------------------------------------------------------------------------------------------------------------------------
    # OBJ FUNCTION
    cost = mdl.sum(couts1*x[i, t] + couts2*y[i, t] + k1*z[i,t] for i in range(P) for t in range(T))
    mdl.minimize(cost)

    # Solve the model
    si = mdl.solve()
    df = mdl.solution.as_df()

    st.dataframe(df)

    # Calculate total leakage from the solution 
    total_leakage_value = round(leakageQuantity.solution_value, 3)
    

    # # if want to print the total leakage quantity in the UI
    # st.write(f"Total leakage quantity: {si[leakageQuantity]}")


    sol = si.get_blended_objective_value_by_priority()[0]
    df_p, df_x_y = data(df)

    # if si:
    #     df = mdl.solution.as_df()
    #     sol = si.get_blended_objective_value_by_priority()[0]
    #     df_p, df_x_y = data(df)
    #     st.dataframe(df)
    # else:
    #     # Handle case where the model couldn't be solved
    #     print("Model could not be solved.")
    #     return None, None, None  # Or handle the failure gracefully
    
    # create file name
    file_name = f"P{P}_T{T}.csv"

    # save result DataFrameto file CSV
    df_x_y.to_csv(file_name, index=False, encoding='utf-8')


    print(f"Results saved to: {file_name}")

    # Return the solution
    return df_p, df_x_y, sol, total_leakage_value



def data_clustring(x):
    for i in range(x.shape[0]):
        r = x["name"].loc[i]
        e = r.split('_')[0]
        g = r.split('_')[1]   # types of operations
        h = r.split('_')[2]   # cluster information
        x.loc[i, 'cluster'] = h
        x.loc[i, 'OPERATION'] = g
        
    return x

def update_df_x_y(df_x_y):
    df_x_y = df_x_y.rename(columns={"CC": "center"})
    df_x_y["OPERATION"] = df_x_y["OPERATION"].replace({"x": "CR", "y": "RE"})
    df_x_y = df_x_y.drop("value", axis=1)
    df_x_y = df_x_y.drop("index", axis=1)
    return df_x_y
    
def vis_clustring(df_x_y, C):
    # Set the color palette and plot size
    # sns.color_palette("Paired")
    sns.set(rc={"figure.figsize":(12, 5)})
    sns.despine(left=True)
    
    # Get the number of caissons and data points
    x = len(df_x_y['CAISSON'].unique()) 
    y = df_x_y.shape[0]
  
    # Create a scatter plot with Plotly Express
    df_x_y['cluster'] = pd.factorize(df_x_y['CC'])[0]
    df_x_y["cluster"] = df_x_y["cluster"].astype(str)
    color_sequence = px.colors.qualitative.Dark24
    color_dict = {str(i): px.colors.qualitative.Plotly[i] for i in range(df_x_y['cluster'].nunique())}
    label_dict = {str(i): f'Category {chr(i + 65)}' for i in range(df_x_y['cluster'].nunique())}
    size_dict = {str(i): (i + 1) * 20 for i in range(df_x_y['cluster'].nunique())}

    centroid_data = df_x_y[df_x_y['TEMPS'] == df_x_y['CC']]

    df_x_y = df_x_y.rename(columns={"CC": "center"})
    df_x_y["OPERATION"] = df_x_y["OPERATION"].replace({"x": "CR", "y": "RE"})
    
    for i in range(df_x_y.shape[0]):
        if df_x_y.loc[i, 'OPERATION'] == "CR":
            df_x_y.loc[i, 'TOLERANCE'] = 7  
        if df_x_y.loc[i, 'OPERATION'] == "RE":
            df_x_y.loc[i, 'TOLERANCE'] = 1  
        


    # Create the scatter plot with background colors for each cluster
    chart = px.scatter(df_x_y.sort_values('CAISSON', ascending=False), x='TEMPS', y='CAISSON', color='cluster',color_discrete_map=color_dict,
                    
                     width=800, height=500, hover_data=['center', 'OPERATION', 'TOLERANCE'])
    
    # Add a background color to each cluster
    for i, color in color_dict.items():
        cluster_data = df_x_y[df_x_y['cluster'] == str(i)]
        cluster_center = pd.DataFrame(cluster_data.loc[(cluster_data['TEMPS'] == cluster_data['center'].astype(int)), ['TEMPS', 'CAISSON']].head(1))

        for index, row in cluster_data.iterrows():           
            x1 = row['TEMPS']
            y1 = row['CAISSON']
            try:
                x2 = cluster_center.iloc[0]['TEMPS']
                y2 = cluster_center.iloc[0]['CAISSON']
                if x1 != x2:
                    # Draw the lasso line
                    chart.add_shape(type='line',
                                x0=x1,
                                y0=y1,
                                x1=x2,
                                y1=y2,
                                line=dict(color=color, width=1, dash='dot'))
    
                else:
                    chart.add_shape(type='circle',
                            xref='x', yref='y',
                            x0=int(x1)-0.5, y0=int(y1)-0.2,
                            x1=int(x1)+0.5, y1=int(y1)+0.2,
                            line=dict(color=color, width=2))
            # chart.update_shapes(line_color=color)
            except:
                continue

    # Customize the plot
    chart.update_traces(marker=dict(sizemode='diameter', sizeref=120))
    chart.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=2, range=[0,45]),
                         yaxis=dict(tickmode='linear', tick0=-2, dtick=2, range=[-2,x+2]),
                         legend_title='', showlegend=True)
    chart.update_layout(plot_bgcolor='#fafafa')
    
    return chart








@st.cache_data
def df_x_y_mod(df, couts1, couts2):
    # df = df.drop(['index', 'value'], axis=1)

    # Drop 'index' and 'value' columns if they exist
    df = df.drop(['index', 'value'], axis=1, errors='ignore')

    # replace the values in the OPERATION column
    df['OPERATION'] = df['OPERATION'].replace({'x': 'CR', 'y': 'RE'})

    # increment the values in the CAISSON column
    
    P = int(df['CAISSON'].nunique())

    # price_map = {('CR', i): couts1[i] for i in range(P)}
    # price_map.update({('RE', i): couts2[i] for i in range(P)})
    price_map = {('CR'): couts1}
    price_map.update({('RE'): couts2})
    
    # create the Price column based on the OPERATION and CAISSON columns
    # df['Price'] = [price_map[(op, int(c))] for op, c in zip(df['OPERATION'], df['CAISSON'])]
    
    df['CAISSON'] = df['CAISSON'].astype('int') + 1 
    caisson_col = df.pop('CAISSON')
    df.insert(0, 'CAISSON', caisson_col)
    return df

def clusteringg(df_x_y):
    
    C = df_x_y.TEMPS.values

    D = np.array([i - np.array(df_x_y.TEMPS.values.astype(int)) for i in C]) # les distances entre les points (time difference)
    D[D[:] < 0 ] = +99999
#     print(f'the matrix D is : {D}')
    
    OCR = df_x_y[df_x_y['OPERATION'].str.contains('x')].first_valid_index() # les points de nuage DES CR
    OCRf = df_x_y[df_x_y['OPERATION'].str.contains('x')].last_valid_index() # les points de nuage DES CR
    try:
        ORE = df_x_y[df_x_y['OPERATION'].str.contains('y')].first_valid_index()  # les points de nuage DES RE
        OREf = df_x_y[df_x_y['OPERATION'].str.contains('y')].last_valid_index()  # les points de nuage DES REF
    except:
        ORE = None

    O = df_x_y.shape[0]
    # df_x_y.to_excel('df.xlsx', index=False)
    K = O
    mdlgp = Model('Maintance_grouping')
    theta = mdlgp.binary_var_matrix(O, K, name = 'theta')
    M = 99999
   
    cluster = mdlgp.binary_var_list(K, name = 'cluster')
    mdlgp.minimize(mdlgp.sum(cluster[k] for k in range(K)))
    
    mdlgp.add_constraints(theta[i, k] <= cluster[k] for i in range(O) for k in range(K))
    mdlgp.add_constraints(sum(theta[i, k] for k in range(K)) == 1 for i in range(O))
    

    # mdlgp.add_constraints(theta[i, k]*D[k, i] <= 7 for i in range(OCRf+1) for k in range(K))
    if OCRf != None:
        for i in range(OCRf + 1):
            for k in range(K):
                    mdlgp.add_constraint(theta[i, k] * D[k, i] <= 7)    # we consider here only the positive distances meaning that we don't look back
                    
    if ORE != OREf:
        mdlgp.add_constraints(theta[i, k]*D[k, i] <= 1 for i in range(ORE, OREf+1) for k in range(K))


    if OCRf != None and ORE != None:
        for k in range(K):
            for i in range(OCRf + 1):
                for j in range(ORE, OREf+1):
                    mdlgp.add_constraint(theta[i, k] + theta[j, k] <= 1)
                    
                    # Maximum number of tasks limit (group capacity constraint) 
                    mdlgp.add_constraints(sum(theta[i, k] for i in range(OCRf+1)) <= 5 for k in range(K))
                    mdlgp.add_constraints(sum(theta[i, k] for i in range(ORE, OREf+1)) <= 3 for k in range(K))



    si2 = mdlgp.solve()
    val = si2.get_blended_objective_value_by_priority()[0]
    print(mdlgp.print_solution(log_output=False))

    df = mdlgp.solution.as_df()
    df = df[df['name'].str.contains('theta')]
    sol = si2.get_blended_objective_value_by_priority()[0]
    dfc = data_clustring(df)
    S = np.zeros(K) 
    
    dfc.to_csv("test1.csv")

    # original
    # Data = dfc.sort_values(by = 'cluster')['cluster'].value_counts().to_frame().reset_index().rename(columns={'index':'values', 'values':'count'})    
    # for v in Data['values']:
    #     S[int(v)] = Data[Data['values']== v]['cluster'].values[0]
    #version1
    # Data = dfc.sort_values(by='cluster')['cluster'].value_counts().to_frame().reset_index(drop=True)
    # for index, count in Data.iterrows():
    #     S[index] = count
    #version3
    Data = dfc.sort_values(by = 'cluster')['cluster'].value_counts().to_frame().reset_index()
    for idx in Data.index:
        v = idx
        S[int(v)] = Data[Data.index == idx]['cluster'].values[0]


    #original
    CC = C[dfc.cluster.values.astype('int')] 

    df_x_y['CC'] = CC
    chart = vis_clustring(df_x_y, CC)
    st.plotly_chart(chart)
    return sol

# Create the Streamlit app
def title_with_background(title, background_color):
    st.markdown(f"""
        <h1 style='background-color:{background_color};border-radius:10px;color:black;text-align:center;margin-bottom:20px'>{title}</h1>
    """, unsafe_allow_html=True)
def line_with_background(title, background_color):
    st.markdown(f"""
                <hr style='background-color:{background_color};border-left: 1px solid #ccc;height: 100%;position: absolute;left: 50%;'></h>
                """, unsafe_allow_html=True)

def app():
    # st.title('Maintenance Caisson')
    title_with_background("Maintenance Planification and Optimization: RTE Use Case", "#f0fdf4")
    st.markdown("<br>", unsafe_allow_html=True) 

    # st.title(':black[Maintenance Caisson] ')

    col1, col2, col3 = st.columns([1, 1.3, 2.4])
    def set_alpha(P):
        # alpha_low = np.random.uniform(0.01, 0.09, P)
        # alpha_high = np.random.uniform(0.004, 0.009, P)
        alpha_high = 0.008


        # PP  = int(P/2)
        # alpha = np.concatenate((alpha_low[:PP], alpha_high[PP:]))
        return alpha_high
    
    with col1:
        st.write("## Model Inputs")
        alert = """ <script>alert('This option is not available.')</script> """
        # deg = components.html(radio_button)  
        deg = st.radio("Degradation/Leakage Type:",
                                    ["Linear degradation", "Function of time", "Stochastic degradation", "Physical modelling"],
                                    index=0,
                                    key="degradation",
                                    )
        # if "Stochastic" or "Physical" in deg:
            # components.html(alert)
            # st.warning("This option is not available yet, please make another choice")
        P = st.slider('### Number of CSEM', min_value=1, max_value=20, step=1)
        T = st.slider('### Time horizon (days)', min_value=15, step=1)
        # Pinit = np.zeros((P, 1))
        # Pinit[:] = 3.5
        # set seed to make sure the random values are the same each time
        np.random.seed(42)      
        Pinit = np.random.uniform(3.21, 3.5001, (P, 1))    #//////
        
        # create file name
        

        # tranform numpy list to Pandas DataFrame
        P_init = pd.DataFrame(Pinit)
        folder_name = "Pinit"
        file_path = f"{folder_name}/P{P}_T{T}.csv"
        P_init.to_csv(file_path, index=False, encoding='utf-8', header=False)  # header=False 表示不保存列名
        
        Rcr = st.slider('### Available Technicians for CR', value=100)
        Rre = st.slider('### Available Technicians for REPAIR', value=100)

        # beta = st.slider('Beta', value=3, max_value=5)
        alpha = set_alpha(P)
        alpha_param = np.random.uniform(500, 1000, (P, 1))
                
        taux1 = 3
        taux2 = 3.2
        beta = 3
        gamma = 99999


        # r_cr = np.random.randint(1 , 3,(P,1))   #/////
        # r_re = np.random.randint(1 , 3,(P,1))    #////
        r_cr = 3   #/////
        r_re = 5    #////
        Dcr = np.random.randint(1 ,3,(P, 1))
        Dre = np.random.randint(1 ,3,(P, 1))  #////
        M = 3.6
        P_rempli = 3.5  #/////
        delta = np.random.randint(0, 1, P)
        deltay = np.random.randint(5, 18, P)

        # deltay = np.random.randint(0, 1, P)   # ///////
        # couts1 : cost of CR
        # couts1 = np.random.uniform(1, 10,  size=P)
        couts1 = 10
        # couts2 : cost of RE
        # couts2 = np.random.uniform(60, 80, size=P)
        couts2 = 100
        b1 = 1
        b2 = 3
        k1 = 1000
        k2 = 4
        k3 = 20
        W = 15

        C_crmax = 5
        C_crmax = 2
        

        # constant for perfect equation coefficient = 10^5 * V/RT * 146 /1000
        # K = 2052.54
        # K = 299.67
        K = 5.993        # volume = 1
        # K = 59.93          # volume = 10

    

     
        if 'df_p' not in st.session_state:
            st.session_state.df_p = None

        if st.button('Run Optimization'):

            with st.spinner("Running function..."): 
                start_time = time.time()  # 记录开始时间
                df_p, df_x_y, sol, total_leakage_value = create_model(P, T, Pinit, Rcr, Rre, r_cr, r_re, P_rempli, alpha, alpha_param, deg, taux1, taux2, gamma, M, couts1, couts2, b1, b2, k1, k2, k3, W, K)
                
                end_time = time.time()  # 记录结束时间
                execution_time = end_time - start_time  # 计算运行时间
                
                st.session_state.df_p = df_p
                st.session_state.df_x_y = df_x_y
                st.session_state.sol = np.around(sol, 2)
                st.session_state.total_leakage_value = total_leakage_value
            st.session_state.couts1 = couts1
            st.session_state.couts2 = couts2
            
            # 显示运行时间       
            # st.success(f"Optimization completed in {execution_time:.2f} seconds!")
            st.write(f"Optimization completed in **{execution_time:.2f} seconds**.")
       
    # st.markdown('<div class="vertical-line"></div>', unsafe_allow_html=True)
    
    # Create the Docplex model and get the solution
    with col2:
        if st.session_state.df_p is not None :
            dff = df_x_y_mod(st.session_state.df_x_y, st.session_state.couts1, st.session_state.couts2)
            dff.index = dff.index + 1  # 将索引改为从1开始
            # dff =st.session_state.df_x_y
            st.write('## Optimization Output')
            st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
            # st.write(dff.to_html(index=False), unsafe_allow_html=True)

            st.write('#### Total Cost (€):', st.session_state.sol)
            st.write('#### Total leakage quantity (kg):', st.session_state.total_leakage_value)
            st.write('#### Maintenance Actions:', dff.shape[0])
            st.write('##### Complément Remplissage (CR):', dff[dff['OPERATION'] == 'CR'].shape[0])
            st.write('##### Repair (RE):', dff[dff['OPERATION'] == 'RE'].shape[0])

            dff = dff.rename(columns={'TEMPS': 'TIME'})
            st.dataframe(dff, height=350)
            
    with col3:
        caissons = [f"Caisson {i+1}" for i in range(P)]

        if st.session_state.df_p is not None :
            st.write("## Data Visualization")
            c = st.radio('CSEM Visualization', caissons, horizontal=True)
            st.markdown("<hr>", unsafe_allow_html=True) 

            if st.button(f'Visualize all CSEMs'):   
                # Visualize the clustering
                fig2 = vis_all_caisson(st.session_state.df_x_y, st.session_state.df_p, taux1)
                st.plotly_chart(fig2)
            else:
                fig = vis_caisson(st.session_state.df_x_y, st.session_state.df_p, str(int(str(c)[-1]) - 1), taux1)
                st.plotly_chart(fig)


    st.markdown("<hr>", unsafe_allow_html=True) 
 
    col3, col4 = st.columns([2, 1])
    col11, col12 = st.columns([1, 0.01])
    col5, col6, col7 = st.columns([1, 1, 1])
    col8, col9, col10 = st.columns([1, 1, 1])
    
    if st.button('Maintenance Groupping'): 

        # with col11:
        #     comm = f"""
        #     <h1 style="box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);margin: 10px;padding:15px">Assuming that the cost  of traveling represent 40% of total cost of maintenance</h1>
        #     """
        #     components.html(comm)

        with col3:
            if st.session_state.df_x_y is not None :
                # Visualize the clustering
                sol = st.session_state.sol
                sol2 = clusteringg(st.session_state.df_x_y)
                # vis_clustring(st.session_state.df_x_y, (st.session_state.df_x_y))
                O = st.session_state.df_x_y.shape[0]
                # gsol = round(0.6*sol + (0.4*sol)/sol2, 2)
                gsol = round(0.6*sol + (0.4*sol2)/O, 2)
                st.session_state.gsol = gsol
                gapc= round(float((sol - gsol)/sol)*100, 2)
                gapc = f"{gapc}%"

                gap= round(float((O - sol2)/O)*100, 2)
                gap = f"{gap}%"
            st.markdown("<style>.big-column{padding-right: 30px;}</style>", unsafe_allow_html=True)
        with col4:
            st.session_state.sol2 = sol2
            if st.session_state.df_p is not None :
                dfa = update_df_x_y(st.session_state.df_x_y)
                st.dataframe(dfa, height=450)
     
        fm = st.session_state.sol
        fm2 = st.session_state.gsol
        fm3 = st.session_state.sol2
                        
        with col5:
            com = f"""
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding-top:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>Maintenance Cost (Before grouping)</h2>
                        <h1>{fm} (€)</h1>
                        <p></p>
                    </div>
                    """
            components.html(com)
            # st.write('<style>  background-color: #FFFFFF;border-radius: 10px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding: 20px;text-align: center;}</style>', unsafe_allow_html=True)
        with col6:
            com = f"""
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding-top:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>Maintenace Cost (After grouping)</h2>
                        <h1>{fm2} (€)</h1>
                        <p></p>

                    </div>
                    """
            components.html(com)
        with col7:
            com = f"""
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding-top:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>GAP</h2>
                        <h1 style="color:#22c55e"><span><svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-arrow-big-up-line" width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">
                        <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
                        <path d="M9 12h-3.586a1 1 0 0 1 -.707 -1.707l6.586 -6.586a1 1 0 0 1 1.414 0l6.586 6.586a1 1 0 0 1 -.707 1.707h-3.586v6h-6v-6z"></path>
                        <path d="M9 21h6"></path>
                        </svg></span>  {gapc}</h1>

                    </div>
                    """
            components.html(com)
        with col8:
            com = f"""
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>Travels (Before grouping)</h2>
                        <h1>{O}</h1>

                    </div>
                    """
            components.html(com)                
        with col9:
            com = f"""
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>Travels (After grouping)</h2>
                        <h1>{int(sol2)}</h1>

                    </div>
                    """
            components.html(com)
        
        with col10:
            com = f"""
                <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding:2px; border-radius: 5px; text-align: center; height: 130px;">
                    <h2>GAP</h2>
                    <h1 style="color:#10b981"><span><svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-arrow-big-up-line" width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">
                        <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
                        <path d="M9 12h-3.586a1 1 0 0 1 -.707 -1.707l6.586 -6.586a1 1 0 0 1 1.414 0l6.586 6.586a1 1 0 0 1 -.707 1.707h-3.586v6h-6v-6z"></path>
                        <path d="M9 21h6"></path>
                        </svg></span>{gap}</h1>
                    <p></p>

                </div>
                """
            components.html(com)


# Run the app
if __name__ == '__main__':
    app()
    


2025.02.03    right version
import sys
sys.path.append('C:\\Program Files\\IBM\\ILOG\\CPLEX_Studio210\\cplex\\python\\20.1\\x64_win64')
sys.path.append("C:\\ProgramData\\anaconda3\\Lib\\site-packages")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from docplex.mp.model import Model
import streamlit.components.v1 as components
import plotly.graph_objects as go
import time


st.set_page_config(layout="wide")
hide_dataframe_row_index = """
            <style>
         div.c1{
                    background-color: #f5f5f5;
                    border: 2px solid;
                    padding: 20px 20px 20px 20px;
                    border-radius: 10px;
                    color: #ffc300;
                    box-shadow: 10px;
                    }
            </style>
            """


def data(x):
    df_p = x[x['value'] != 1].reset_index()
    df_x_y = x[x['value'].astype(int) == 1].reset_index()
    for i in range(df_p.shape[0]):
        r = df_p["name"].loc[i]
        e = r.split('_')[0]
        g = r.split('_')[1]
        h = r.split('_')[2]
        if (e == 'p'):
            df_p.loc[i, 'CAISSON'] = g
            df_p.loc[i, 'TEMPS'] = int(h)

    for i in range(df_x_y.shape[0]):
        r = df_x_y["name"].loc[i]
        e = r.split('_')[0]
        g = r.split('_')[1]
        h = r.split('_')[2]
        if(e == 'y') | (e == 'x'):
            df_x_y.loc[i, 'OPERATION'] = e
            df_x_y.loc[i, 'CAISSON'] = g
            df_x_y.loc[i, 'TEMPS'] = int(h)
        # else:
        #     df_x_y.loc[i, 'OPERATION'] = "NA"
        #     df_x_y.loc[i, 'CAISSON'] = g
        #     df_x_y.loc[i, 'TEMPS'] = 0
    df_x_y = df_x_y.dropna()[['value', 'OPERATION', 'CAISSON', 'TEMPS']].reset_index()
    

    
    return df_p, df_x_y


# Function to visualize the clustering
@st.cache_data
def vis_caisson(df_x_y, df_p, x, taux1):
    # Filter the data based on the selected caissons
    sns.color_palette("Paired")
    sns.set(rc={"figure.figsize":(12, 5)})
    sns.despine(left=True)
    x = [str(s) for s in x]
    df_p_caisson = df_p[df_p['CAISSON'].isin(x)]
    caisson = df_x_y[df_x_y['CAISSON'].isin(x)]


    chart = px.line(df_p_caisson, x='TEMPS', y='value', color='CAISSON', width=700, height=400)
    
    # Add the reference lines and vertical lines for the operations
    chart.add_hline(y=taux1, line_dash='dash', line_color='red', name='Taux 1')
    chart.add_hline(y=3.5, line_dash='dash', line_color='blue', name='3.5')
    
    for temp in caisson[caisson['OPERATION'] == 'x']['TEMPS'].values:
        chart.add_vline(x=temp, line_dash='dash', line_color='green', name='CR')
        chart.add_annotation(x=temp+0.5, y=2.95, text='CR', showarrow=False)
    
    for temp in caisson[caisson['OPERATION'] == 'y']['TEMPS'].values:
        chart.add_vline(x=temp, line_dash='dash', line_color='yellow', name='RE')
        chart.add_annotation(x=temp+0.5, y=2.95, text='RE', showarrow=False)
    
    
    # Set the x-axis tick angle and remove the legend title
    chart.update_layout(xaxis_tickangle=-45)
    
    # Set the y-axis range to start from 2.9
    chart.update_yaxes(range=[2.9, None])
    chart.update_xaxes(dtick='M1')

    # Remove legend
    chart.update_layout(showlegend=False)
   
   
    return chart


# newly editted function to show the legend index correctly
def vis_all_caisson(df_x_y, df_p, taux1):
    # Filter the data based on the selected caissons
    df_p['value'] = df_p['value'].astype(float)
    
    # Create a line chart of the pressure data using Plotly Express
    chart = px.line(df_p, x='TEMPS', y='value', color='CAISSON', width=700, height=400,
                    color_discrete_sequence=px.colors.qualitative.Plotly)
    
    # Add the reference lines
    chart.add_hline(y=taux1, line_dash='dash', line_color='green', name='Taux 1')
    chart.add_hline(y=3.5, line_dash='dash', line_color='blue', name='3.5')
    
    # Set the x-axis tick angle and legend title
    chart.update_layout(xaxis_tickangle=-45, legend_title='Caissons', plot_bgcolor='#f0fdf4')

    # Modify legend labels to start from 1
    for i, trace in enumerate(chart.data):
        chart.data[i].name = f'Caisson {i+1}'
    
    return chart


def create_model(P, T, Pinit, Rcr, Rre, r_cr, r_re, P_rempli, alpha,alpha_param,deg, taux1, taux2, gamma, M, couts1, couts2, b1, b2, k1, k2, k3, W, K):

    """ Model """
    mdl = Model('Maintenance_caisson', log_output=True)
    x = mdl.binary_var_matrix(P, T, name = 'x')  
    y = mdl.binary_var_matrix(P, T, name = 'y')
    
    f = mdl.binary_var_matrix(P, T, name = 'f')  
    g = mdl.binary_var_matrix(P, T, name = 'g')
    
    z = mdl.binary_var_matrix(P, T, name = 'z')
    w = mdl.binary_var_matrix(P, T, name = 'w')
    
    e = mdl.integer_var_matrix(P, T, name="e")

    p = mdl.continuous_var_matrix(P, T, name='p')

    d = mdl.binary_var_matrix(P, T, name = 'd')

    o = mdl.binary_var_matrix(P, T, name = 'o')

    v = mdl.continuous_var_matrix(P, T, name = 'v')

    # 计算递增速率
    slope = mdl.continuous_var_matrix(P, T, name='slope')



    # leakageQuantity = mdl.continuous_var_list(P, name='leakageQuantity')
    leakageQuantity = 0.0


    # Add constraints and objective function to the model
    # CONTRAINTES D'INITIALISATION DE PRESSION(OU DE PRESSIONS INITIAUX)
    mdl.add_constraints((p[i, 0] == (Pinit[i].item()) for i in range(P)), names = 'SPinit')
    mdl.add_constraints((p[i, t] >= 3.001 for i in range(P) for t in range(T)), names = 'SPmin')
    mdl.add_constraints((p[i, t] <= 3.5 for i in range(P) for t in range(T)), names = 'SPmax')

    
    # # CONTRAINTE DE RESOURCES 
    mdl.add_constraints((sum(r_cr*x[k, tp] for k in range(P) for tp in range(T) if (tp <= t) and (tp + b1 - 1>= t)) <= Rcr for t in range(T)))
    mdl.add_constraints((sum(r_re*y[k, tp] for k in range(P) for tp in range(T) if (tp <= t) and (tp + b2 - 1>= t)) <= Rre for t in range(T)))
    #-----------------------------------------------------------------------------------------------------------------------------

    if deg == 'Linear degradation':
        mdl.add_constraints((p[i, t+1] >= (1- alpha)*p[i, t] - M*(f[i, t] + (g[i, t] + e[i,t])) for i in range(P) for t in range(T-1)))
        mdl.add_constraints((p[i, t+1] <= (1- alpha)*p[i, t] + M*(f[i, t] + (g[i, t] + e[i,t])) for i in range(P) for t in range(T-1)))    
    else:
        mdl.add_constraints((p[i, t+1] >= (1- alpha_param[i][0]**(t/(t + 34) - 1))*p[i, t] - M*(f[i, t] + g[i, t]) for i in range(P) for t in range(1, T-1)))
        mdl.add_constraints((p[i, t+1] <= (1- alpha_param[i][0]**(t/(t + 34) - 1))*p[i, t] + M*(f[i, t] + g[i, t]) for i in range(P) for t in range(1, T-1)))
        
    #-----------------------------------------------------------------------------------------------------------------------------
    mdl.add_constraints((p[i, t+b1] >= P_rempli - M*(1 - x[i, t]) for i in range(P) for t in range(T-b1)), names = 'AFTER CR')
    mdl.add_constraints((p[i, t+b1] <= P_rempli + M*(1 - x[i, t]) for i in range(P) for t in range(T-b1)), names = 'AFTER CR')

    # Add linear interpolation constraints for CR
    for i in range(P):
        for t in range(T - b1):  # Ensure that time frames are not exceeded
            for k in range(1, b1):
                # Calculate the desired pressure value at time t+k
                expected_pressure = p[i, t] + (k / b1) * (P_rempli - p[i, t])
                mdl.add_constraint(p[i, t + k] >= expected_pressure - M * (1 - x[i, t]))
                mdl.add_constraint(p[i, t + k] <= expected_pressure + M * (1 - x[i, t]))

        for t in range(T - b1, T):
            for k in range(1, T - t):
                expected_pressure = p[i, t] + (k / (T-t)) * (P_rempli - p[i, t])
                mdl.add_constraint(p[i, t + k] >= expected_pressure - M * (1 - x[i, t]))
                mdl.add_constraint(p[i, t + k] <= expected_pressure + M * (1 - x[i, t]))

    # Add linear interpolation constraints for RE
    for i in range(P):
        for t in range(T - b2):  # Ensure that time frames are not exceeded
            for k in range(1, b2):
                # Calculate the desired pressure value at time t+k
                expected_pressure = p[i, t] + (k / b2) * (P_rempli - p[i, t])
                mdl.add_constraint(p[i, t + k] >= expected_pressure - M * (1 - y[i, t]))
                mdl.add_constraint(p[i, t + k] <= expected_pressure + M * (1 - y[i, t]))

    for t in range(T - b2, T):
        # for k in range(1, T - t):
        for k in range(0, T - t):
            expected_pressure = p[i, t] + (k / (T-t)) * (P_rempli - p[i, t])
            mdl.add_constraint(p[i, t + k] >= expected_pressure - M * (1 - y[i, t]))
            mdl.add_constraint(p[i, t + k] <= expected_pressure + M * (1 - y[i, t]))

    # if RE make pressure stay stable

    # for i in range(P):
    #     for t in range(T):
    #         mdl.add_constraints((p[i, tp] >= p[i, t] - M*(1 - y[i, t]) for tp in range(t+b2, T)), names = 'AFTER RE')
    #         mdl.add_constraints((p[i, tp] <= p[i, t] + M*(1 - y[i, t]) for tp in range(t+b2, T)), names = 'AFTER RE')
    
    # if RE make pressure come back to 3.5
    for i in range(P):
        for t in range(T):
            mdl.add_constraints((p[i, tp] >= P_rempli - M*(1 - y[i, t]) for tp in range(t+b2, T)), names = 'AFTER RE')
            mdl.add_constraints((p[i, tp] <= P_rempli + M*(1 - y[i, t]) for tp in range(t+b2, T)), names = 'AFTER RE')
    
        
                 
    mdl.add_constraints((y[i, tp] <= 1 - y[i, t] for i in range(P) for t in range(T-b2) for tp in range(t+b2, T)), names = 'AFTER RE')

    # mdl.add_constraints(x[i, t] + y[i, t] <= 1 for i in range(P) for t in range(T))


    
            
    #-----------------------------------------------------------------------------------------------------------------------------
    # definition for cumulating the sum of RE action
    for t in range(T):
        for i in range(P):
                mdl.add_constraint(e[i,t] == mdl.sum(y[i, k] for k in range(t)))

    #-----------------------------------------------------------------------------------------------------------------------------
    # w[i][t] = sum(x[i, tp] for tp in range(t - W + 1, t + 1))
    mdl.add_constraints((w[i, t] == mdl.sum(x[i, tp] for tp in range(max(0, t - W + 1), t + 1)) for i in range(P) for t in range(T)), names='sum_constraint')
    # w[i][t] <= 2 + z[i][t]
    mdl.add_constraints((w[i, t] <= 2 + z[i, t] for i in range(P) for t in range(T)), names='upper_bound_constraint')
    # w[i][t] >= 3 - 3 * (1 - z[i][t])
    mdl.add_constraints((w[i, t] >= 3 - 3 * (1 - z[i, t]) for i in range(P) for t in range(T)), names='lower_bound_constraint')

    #-----------------------------------------------------------------------------------------------------------------------------
    # f[i][t+k] >= x[i][t]  ∀i ∈ P, ∀t ∈ T-b, ∀k = 0, 1, ..., b
    mdl.add_constraints((f[i, t + k] >= x[i, t] for i in range(P) for t in range(T - b1) for k in range(b1)), names='f_constraint')
    
    # g[i][t+k] >= y[i][t]  ∀i ∈ P, ∀t ∈ T-b, ∀k = 0, 1, ..., b
    mdl.add_constraints((g[i, t + k] >= y[i, t] for i in range(P) for t in range(T - b2) for k in range(b2)), names='g_constraint')

    # make sure f[i, t] = 0 at other moments
    mdl.add_constraints((f[i, t] <= sum(x[i, tp] for tp in range(max(0, t - b1 + 1), t + 1)) for i in range(P) for t in range(T)), names='f_zero_constraint')
    # make sure g[i, t] = 0 at other moments
    mdl.add_constraints((g[i, t] <= sum(y[i, tp] for tp in range(max(0, t - b2 + 1), t + 1)) for i in range(P) for t in range(T)), names='g_zero_constraint')
    

    mdl.add_constraints(x[i, t] + x[i, tp] <= 1 for i in range(P) for t in range(T) for tp in range(T) if (tp < t) and (tp + b1 >= t)) 
    # mdl.add_constraints(y[i, t] + y[i, tp] <= 1 for i in range(P) for t in range(T) for tp in range(T) if (tp < t) and (tp + b2 >= t))

    # # define variable v - original way
    # for i in range(P):
    #     v[i, 0] = 0
    # mdl.add_constraints((v[i,t] >= p[i,t-1] - p[i,t] for i in range(P) for t in range(1,T)), names='v_constraint')
    # mdl.add_constraints((v[i,t] >= 0 for i in range(P) for t in range(1,T)), names='v_constraint')

    # define variable v - new way !!!!
    for i in range(P):
        for t in range(1, T):
            mdl.add_constraint(v[i, t] >= p[i, t-1] - p[i, t])
            mdl.add_constraint(v[i, t] >= 0)
            mdl.add_constraint(v[i, t] <= (p[i,t-1] - p[i, t]) + M * (1 - o[i, t]))
            mdl.add_constraint(v[i, t] <= M * o[i, t])
            mdl.add_constraint(M * o[i, t] >= p[i,t-1] - p[i, t])
            mdl.add_constraint(M * (1- o[i, t]) >= - (p[i,t-1] - p[i, t]))


    #-------------------------------------------------------------------------------------------------------------------------------
    
    # mdl.add_constraint(sum((((3.5 - p[i,tp])*(10**5))*V/(8.314 * 293)*146.0)/1000 for i in range(P) for tp in range(t)) <= 100.0)
    # separate leakage quantity for each caisson
    # for i in range(P):
    #     leakageQuantity[i] = mdl.sum(K * (p[i, t-1] - p[i, t]) for t in range(1,T))
    #     mdl.add_constraint(leakageQuantity[i] <= 0.08)

    # total leakage quantity constraint     
    leakageQuantity = mdl.sum(K * v[i, t] for i in range(P) for t in range(1,T))
    mdl.add_constraint(leakageQuantity <= 25.0)
    


    #-----------------------------------------------------------------------------------------------------------------------------
    # OBJ FUNCTION
    cost = mdl.sum(couts1*x[i, t] + couts2*y[i, t] + k1*z[i,t] for i in range(P) for t in range(T))
    mdl.minimize(cost)

    # Solve the model
    si = mdl.solve()
    print(mdl.solve_details.status)
    df = mdl.solution.as_df()

    st.dataframe(df)

    # Calculate total leakage from the solution 
    total_leakage_value = round(leakageQuantity.solution_value, 3)
    

    # # if want to print the total leakage quantity in the UI
    # st.write(f"Total leakage quantity: {si[leakageQuantity]}")


    sol = si.get_blended_objective_value_by_priority()[0]
    df_p, df_x_y = data(df)

    # if si:
    #     df = mdl.solution.as_df()
    #     sol = si.get_blended_objective_value_by_priority()[0]
    #     df_p, df_x_y = data(df)
    #     st.dataframe(df)
    # else:
    #     # Handle case where the model couldn't be solved
    #     print("Model could not be solved.")
    #     return None, None, None  # Or handle the failure gracefully
    
    # create file name
    file_name = f"P{P}_T{T}.csv"

    # save result DataFrameto file CSV
    df_x_y.to_csv(file_name, index=False, encoding='utf-8')


    print(f"Results saved to: {file_name}")

    # Return the solution
    return df_p, df_x_y, sol, total_leakage_value



def data_clustring(x):
    for i in range(x.shape[0]):
        r = x["name"].loc[i]
        e = r.split('_')[0]
        g = r.split('_')[1]   # types of operations
        h = r.split('_')[2]   # cluster information
        x.loc[i, 'cluster'] = h
        x.loc[i, 'OPERATION'] = g
        
    return x

def update_df_x_y(df_x_y):
    df_x_y = df_x_y.rename(columns={"CC": "center"})
    df_x_y["OPERATION"] = df_x_y["OPERATION"].replace({"x": "CR", "y": "RE"})
    df_x_y = df_x_y.drop("value", axis=1)
    df_x_y = df_x_y.drop("index", axis=1)
    return df_x_y
    
def vis_clustring(df_x_y, C):
    # Set the color palette and plot size
    # sns.color_palette("Paired")
    sns.set(rc={"figure.figsize":(12, 5)})
    sns.despine(left=True)
    
    # Get the number of caissons and data points
    x = len(df_x_y['CAISSON'].unique()) 
    y = df_x_y.shape[0]
  
    # Create a scatter plot with Plotly Express
    df_x_y['cluster'] = pd.factorize(df_x_y['CC'])[0]
    df_x_y["cluster"] = df_x_y["cluster"].astype(str)
    color_sequence = px.colors.qualitative.Dark24
    color_dict = {str(i): px.colors.qualitative.Plotly[i] for i in range(df_x_y['cluster'].nunique())}
    label_dict = {str(i): f'Category {chr(i + 65)}' for i in range(df_x_y['cluster'].nunique())}
    size_dict = {str(i): (i + 1) * 20 for i in range(df_x_y['cluster'].nunique())}

    centroid_data = df_x_y[df_x_y['TEMPS'] == df_x_y['CC']]

    df_x_y = df_x_y.rename(columns={"CC": "center"})
    df_x_y["OPERATION"] = df_x_y["OPERATION"].replace({"x": "CR", "y": "RE"})
    
    for i in range(df_x_y.shape[0]):
        if df_x_y.loc[i, 'OPERATION'] == "CR":
            df_x_y.loc[i, 'TOLERANCE'] = 7  
        if df_x_y.loc[i, 'OPERATION'] == "RE":
            df_x_y.loc[i, 'TOLERANCE'] = 1  
        


    # Create the scatter plot with background colors for each cluster
    chart = px.scatter(df_x_y.sort_values('CAISSON', ascending=False), x='TEMPS', y='CAISSON', color='cluster',color_discrete_map=color_dict,
                    
                     width=800, height=500, hover_data=['center', 'OPERATION', 'TOLERANCE'])
    
    # Add a background color to each cluster
    for i, color in color_dict.items():
        cluster_data = df_x_y[df_x_y['cluster'] == str(i)]
        cluster_center = pd.DataFrame(cluster_data.loc[(cluster_data['TEMPS'] == cluster_data['center'].astype(int)), ['TEMPS', 'CAISSON']].head(1))

        for index, row in cluster_data.iterrows():           
            x1 = row['TEMPS']
            y1 = row['CAISSON']
            try:
                x2 = cluster_center.iloc[0]['TEMPS']
                y2 = cluster_center.iloc[0]['CAISSON']
                if x1 != x2:
                    # Draw the lasso line
                    chart.add_shape(type='line',
                                x0=x1,
                                y0=y1,
                                x1=x2,
                                y1=y2,
                                line=dict(color=color, width=1, dash='dot'))
    
                else:
                    chart.add_shape(type='circle',
                            xref='x', yref='y',
                            x0=int(x1)-0.5, y0=int(y1)-0.2,
                            x1=int(x1)+0.5, y1=int(y1)+0.2,
                            line=dict(color=color, width=2))
            # chart.update_shapes(line_color=color)
            except:
                continue

    # Customize the plot
    chart.update_traces(marker=dict(sizemode='diameter', sizeref=120))
    chart.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=2, range=[0,45]),
                         yaxis=dict(tickmode='linear', tick0=-2, dtick=2, range=[-2,x+2]),
                         legend_title='', showlegend=True)
    chart.update_layout(plot_bgcolor='#fafafa')
    
    return chart








@st.cache_data
def df_x_y_mod(df, couts1, couts2):
    # df = df.drop(['index', 'value'], axis=1)

    # Drop 'index' and 'value' columns if they exist
    df = df.drop(['index', 'value'], axis=1, errors='ignore')

    # replace the values in the OPERATION column
    df['OPERATION'] = df['OPERATION'].replace({'x': 'CR', 'y': 'RE'})

    # increment the values in the CAISSON column
    
    P = int(df['CAISSON'].nunique())

    # price_map = {('CR', i): couts1[i] for i in range(P)}
    # price_map.update({('RE', i): couts2[i] for i in range(P)})
    price_map = {('CR'): couts1}
    price_map.update({('RE'): couts2})
    
    # create the Price column based on the OPERATION and CAISSON columns
    # df['Price'] = [price_map[(op, int(c))] for op, c in zip(df['OPERATION'], df['CAISSON'])]
    
    df['CAISSON'] = df['CAISSON'].astype('int') + 1 
    caisson_col = df.pop('CAISSON')
    df.insert(0, 'CAISSON', caisson_col)
    return df

def clusteringg(df_x_y):
    
    C = df_x_y.TEMPS.values

    D = np.array([i - np.array(df_x_y.TEMPS.values.astype(int)) for i in C]) # les distances entre les points (time difference)
    D[D[:] < 0 ] = +99999
#     print(f'the matrix D is : {D}')
    
    OCR = df_x_y[df_x_y['OPERATION'].str.contains('x')].first_valid_index() # les points de nuage DES CR
    OCRf = df_x_y[df_x_y['OPERATION'].str.contains('x')].last_valid_index() # les points de nuage DES CR
    try:
        ORE = df_x_y[df_x_y['OPERATION'].str.contains('y')].first_valid_index()  # les points de nuage DES RE
        OREf = df_x_y[df_x_y['OPERATION'].str.contains('y')].last_valid_index()  # les points de nuage DES REF
    except:
        ORE = None

    O = df_x_y.shape[0]
    # df_x_y.to_excel('df.xlsx', index=False)
    K = O
    mdlgp = Model('Maintance_grouping')
    theta = mdlgp.binary_var_matrix(O, K, name = 'theta')
    M = 99999
   
    cluster = mdlgp.binary_var_list(K, name = 'cluster')
    mdlgp.minimize(mdlgp.sum(cluster[k] for k in range(K)))
    
    mdlgp.add_constraints(theta[i, k] <= cluster[k] for i in range(O) for k in range(K))
    mdlgp.add_constraints(sum(theta[i, k] for k in range(K)) == 1 for i in range(O))
    

    # mdlgp.add_constraints(theta[i, k]*D[k, i] <= 7 for i in range(OCRf+1) for k in range(K))
    if OCRf != None:
        for i in range(OCRf + 1):
            for k in range(K):
                    mdlgp.add_constraint(theta[i, k] * D[k, i] <= 7)    # we consider here only the positive distances meaning that we don't look back
                    
    if ORE != OREf:
        mdlgp.add_constraints(theta[i, k]*D[k, i] <= 1 for i in range(ORE, OREf+1) for k in range(K))


    if OCRf != None and ORE != None:
        for k in range(K):
            for i in range(OCRf + 1):
                for j in range(ORE, OREf+1):
                    mdlgp.add_constraint(theta[i, k] + theta[j, k] <= 1)
                    
                    # Maximum number of tasks limit (group capacity constraint) 
                    mdlgp.add_constraints(sum(theta[i, k] for i in range(OCRf+1)) <= 5 for k in range(K))
                    mdlgp.add_constraints(sum(theta[i, k] for i in range(ORE, OREf+1)) <= 3 for k in range(K))



    si2 = mdlgp.solve()
    val = si2.get_blended_objective_value_by_priority()[0]
    print(mdlgp.print_solution(log_output=False))

    df = mdlgp.solution.as_df()
    df = df[df['name'].str.contains('theta')]
    sol = si2.get_blended_objective_value_by_priority()[0]
    dfc = data_clustring(df)
    S = np.zeros(K) 
    
    dfc.to_csv("test1.csv")

    # original
    # Data = dfc.sort_values(by = 'cluster')['cluster'].value_counts().to_frame().reset_index().rename(columns={'index':'values', 'values':'count'})    
    # for v in Data['values']:
    #     S[int(v)] = Data[Data['values']== v]['cluster'].values[0]
    #version1
    # Data = dfc.sort_values(by='cluster')['cluster'].value_counts().to_frame().reset_index(drop=True)
    # for index, count in Data.iterrows():
    #     S[index] = count
    #version3
    Data = dfc.sort_values(by = 'cluster')['cluster'].value_counts().to_frame().reset_index()
    for idx in Data.index:
        v = idx
        S[int(v)] = Data[Data.index == idx]['cluster'].values[0]


    #original
    CC = C[dfc.cluster.values.astype('int')] 

    df_x_y['CC'] = CC
    chart = vis_clustring(df_x_y, CC)
    st.plotly_chart(chart)
    return sol

# Create the Streamlit app
def title_with_background(title, background_color):
    st.markdown(f"""
        <h1 style='background-color:{background_color};border-radius:10px;color:black;text-align:center;margin-bottom:20px'>{title}</h1>
    """, unsafe_allow_html=True)
def line_with_background(title, background_color):
    st.markdown(f"""
                <hr style='background-color:{background_color};border-left: 1px solid #ccc;height: 100%;position: absolute;left: 50%;'></h>
                """, unsafe_allow_html=True)

def app():
    # st.title('Maintenance Caisson')
    title_with_background("Maintenance Planification and Optimization: RTE Use Case", "#f0fdf4")
    st.markdown("<br>", unsafe_allow_html=True) 

    # st.title(':black[Maintenance Caisson] ')

    col1, col2, col3 = st.columns([1, 1.3, 2.4])
    def set_alpha(P):
        # alpha_low = np.random.uniform(0.01, 0.09, P)
        # alpha_high = np.random.uniform(0.004, 0.009, P)
        alpha_high = 0.008


        # PP  = int(P/2)
        # alpha = np.concatenate((alpha_low[:PP], alpha_high[PP:]))
        return alpha_high
    
    with col1:
        st.write("## Model Inputs")
        alert = """ <script>alert('This option is not available.')</script> """
        # deg = components.html(radio_button)  
        deg = st.radio("Degradation/Leakage Type:",
                                    ["Linear degradation", "Function of time", "Stochastic degradation", "Physical modelling"],
                                    index=0,
                                    key="degradation",
                                    )
        # if "Stochastic" or "Physical" in deg:
            # components.html(alert)
            # st.warning("This option is not available yet, please make another choice")
        P = st.slider('### Number of CSEM', min_value=1, max_value=20, step=1)
        T = st.slider('### Time horizon (days)', min_value=15, step=1)
        # Pinit = np.zeros((P, 1))
        # Pinit[:] = 3.5

        # set seed to make sure the random values are the same each time
        np.random.seed(42)      
        # Pinit = np.random.uniform(3.21, 3.5001, (P, 1))    #//////
        Pinit = np.random.uniform(3.01, 3.5001, (P, 1))    #//////
        
        # store the initial pressure values in a CSV file
        # tranform numpy list to Pandas DataFrame
        P_init = pd.DataFrame(Pinit)
        folder_name = "Pinit2"
        file_path = f"{folder_name}/P{P}_T{T}.csv"
        P_init.to_csv(file_path, index=False, encoding='utf-8', header=False)  # header=False 表示不保存列名
        
        Rcr = st.slider('### Available Technicians for CR', value=100)
        Rre = st.slider('### Available Technicians for REPAIR', value=100)

        # beta = st.slider('Beta', value=3, max_value=5)
        alpha = set_alpha(P)
        alpha_param = np.random.uniform(500, 1000, (P, 1))
                
        taux1 = 3
        taux2 = 3.2
        beta = 3
        gamma = 99999


        # r_cr = np.random.randint(1 , 3,(P,1))   #/////
        # r_re = np.random.randint(1 , 3,(P,1))    #////
        r_cr = 3   #/////
        r_re = 5    #////
        Dcr = np.random.randint(1 ,3,(P, 1))
        Dre = np.random.randint(1 ,3,(P, 1))  #////
        M = 3.6
        P_rempli = 3.5  #/////
        delta = np.random.randint(0, 1, P)
        deltay = np.random.randint(5, 18, P)

        # deltay = np.random.randint(0, 1, P)   # ///////
        # couts1 : cost of CR
        # couts1 = np.random.uniform(1, 10,  size=P)
        couts1 = 10
        # couts2 : cost of RE
        # couts2 = np.random.uniform(60, 80, size=P)
        couts2 = 100
        b1 = 1
        b2 = 3
        k1 = 1000
        k2 = 4
        k3 = 20
        W = 15

        C_crmax = 5
        C_crmax = 2
        

        # constant for perfect equation coefficient = 10^5 * V/RT * 146 /1000
        # K = 2052.54
        # K = 299.67
        K = 5.993        # volume = 1
        # K = 59.93          # volume = 10

    

     
        if 'df_p' not in st.session_state:
            st.session_state.df_p = None

        if st.button('Run Optimization'):

            with st.spinner("Running function..."): 
                start_time = time.time()  # 记录开始时间
                df_p, df_x_y, sol, total_leakage_value = create_model(P, T, Pinit, Rcr, Rre, r_cr, r_re, P_rempli, alpha, alpha_param, deg, taux1, taux2, gamma, M, couts1, couts2, b1, b2, k1, k2, k3, W, K)
                
                end_time = time.time()  # 记录结束时间
                execution_time = end_time - start_time  # 计算运行时间
                
                st.session_state.df_p = df_p
                st.session_state.df_x_y = df_x_y
                st.session_state.sol = np.around(sol, 2)
                st.session_state.total_leakage_value = total_leakage_value
            st.session_state.couts1 = couts1
            st.session_state.couts2 = couts2
            
            # 显示运行时间       
            # st.success(f"Optimization completed in {execution_time:.2f} seconds!")
            st.write(f"Optimization completed in **{execution_time:.2f} seconds**.")
       
    # st.markdown('<div class="vertical-line"></div>', unsafe_allow_html=True)
    
    # Create the Docplex model and get the solution
    with col2:
        if st.session_state.df_p is not None :
            dff = df_x_y_mod(st.session_state.df_x_y, st.session_state.couts1, st.session_state.couts2)
            dff.index = dff.index + 1  # 将索引改为从1开始
            # dff =st.session_state.df_x_y
            st.write('## Optimization Output')
            st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
            # st.write(dff.to_html(index=False), unsafe_allow_html=True)

            st.write('#### Total Cost (€):', st.session_state.sol)
            st.write('#### Total leakage quantity (kg):', st.session_state.total_leakage_value)
            st.write('#### Maintenance Actions:', dff.shape[0])
            st.write('##### Complément Remplissage (CR):', dff[dff['OPERATION'] == 'CR'].shape[0])
            st.write('##### Repair (RE):', dff[dff['OPERATION'] == 'RE'].shape[0])

            dff = dff.rename(columns={'TEMPS': 'TIME'})
            st.dataframe(dff, height=350)
            
    with col3:
        caissons = [f"Caisson {i+1}" for i in range(P)]

        if st.session_state.df_p is not None :
            st.write("## Data Visualization")
            c = st.radio('CSEM Visualization', caissons, horizontal=True)
            st.markdown("<hr>", unsafe_allow_html=True) 

            if st.button(f'Visualize all CSEMs'):   
                # Visualize the clustering
                fig2 = vis_all_caisson(st.session_state.df_x_y, st.session_state.df_p, taux1)
                st.plotly_chart(fig2)
            else:
                fig = vis_caisson(st.session_state.df_x_y, st.session_state.df_p, str(int(str(c)[-1]) - 1), taux1)
                st.plotly_chart(fig)


    st.markdown("<hr>", unsafe_allow_html=True) 
 
    col3, col4 = st.columns([2, 1])
    col11, col12 = st.columns([1, 0.01])
    col5, col6, col7 = st.columns([1, 1, 1])
    col8, col9, col10 = st.columns([1, 1, 1])
    
    if st.button('Maintenance Groupping'): 

        # with col11:
        #     comm = f"""
        #     <h1 style="box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);margin: 10px;padding:15px">Assuming that the cost  of traveling represent 40% of total cost of maintenance</h1>
        #     """
        #     components.html(comm)

        with col3:
            if st.session_state.df_x_y is not None :
                # Visualize the clustering
                sol = st.session_state.sol
                sol2 = clusteringg(st.session_state.df_x_y)
                # vis_clustring(st.session_state.df_x_y, (st.session_state.df_x_y))
                O = st.session_state.df_x_y.shape[0]
                # gsol = round(0.6*sol + (0.4*sol)/sol2, 2)
                gsol = round(0.6*sol + (0.4*sol2)/O, 2)
                st.session_state.gsol = gsol
                gapc= round(float((sol - gsol)/sol)*100, 2)
                gapc = f"{gapc}%"

                gap= round(float((O - sol2)/O)*100, 2)
                gap = f"{gap}%"
            st.markdown("<style>.big-column{padding-right: 30px;}</style>", unsafe_allow_html=True)
        with col4:
            st.session_state.sol2 = sol2
            if st.session_state.df_p is not None :
                dfa = update_df_x_y(st.session_state.df_x_y)
                st.dataframe(dfa, height=450)
     
        fm = st.session_state.sol
        fm2 = st.session_state.gsol
        fm3 = st.session_state.sol2
                        
        with col5:
            com = f"""
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding-top:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>Maintenance Cost (Before grouping)</h2>
                        <h1>{fm} (€)</h1>
                        <p></p>
                    </div>
                    """
            components.html(com)
            # st.write('<style>  background-color: #FFFFFF;border-radius: 10px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding: 20px;text-align: center;}</style>', unsafe_allow_html=True)
        with col6:
            com = f"""
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding-top:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>Maintenace Cost (After grouping)</h2>
                        <h1>{fm2} (€)</h1>
                        <p></p>

                    </div>
                    """
            components.html(com)
        with col7:
            com = f"""
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding-top:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>GAP</h2>
                        <h1 style="color:#22c55e"><span><svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-arrow-big-up-line" width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">
                        <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
                        <path d="M9 12h-3.586a1 1 0 0 1 -.707 -1.707l6.586 -6.586a1 1 0 0 1 1.414 0l6.586 6.586a1 1 0 0 1 -.707 1.707h-3.586v6h-6v-6z"></path>
                        <path d="M9 21h6"></path>
                        </svg></span>  {gapc}</h1>

                    </div>
                    """
            components.html(com)
        with col8:
            com = f"""
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>Travels (Before grouping)</h2>
                        <h1>{O}</h1>

                    </div>
                    """
            components.html(com)                
        with col9:
            com = f"""
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>Travels (After grouping)</h2>
                        <h1>{int(sol2)}</h1>

                    </div>
                    """
            components.html(com)
        
        with col10:
            com = f"""
                <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding:2px; border-radius: 5px; text-align: center; height: 130px;">
                    <h2>GAP</h2>
                    <h1 style="color:#10b981"><span><svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-arrow-big-up-line" width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">
                        <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
                        <path d="M9 12h-3.586a1 1 0 0 1 -.707 -1.707l6.586 -6.586a1 1 0 0 1 1.414 0l6.586 6.586a1 1 0 0 1 -.707 1.707h-3.586v6h-6v-6z"></path>
                        <path d="M9 21h6"></path>
                        </svg></span>{gap}</h1>
                    <p></p>

                </div>
                """
            components.html(com)


# Run the app
if __name__ == '__main__':
    app()

2025.02.03 naive model right version
import numpy as np
import time

# 开始计时
start_time = time.time()

# 参数定义
alpha = 0.008  # 压力下降速率
T = 60         # 时间步数
P = 5

# np.random.seed(42)
# P_init = np.random.uniform(3.001, 3.5001, (P, 1))
# P_init = np.random.uniform(3.001, 3.5001, (P, 1))
# P_init = [3.19356, 3.47594, 3.36875, 3.30340, 3.08646, 3.08645, 3.03847, 3.43451, 3.30461]        # T=15
P_init = [3.19356, 3.47594, 3.36875, 3.30340, 3.08646]      
Pressure = np.zeros((P, T))       # 每个初始压力值对应一个压力数组
v = np.zeros((P, T))              # pressure difference with previous time step
x_action = np.zeros((P, T), dtype=int)  # 操作 x 的触发记录
y_action = np.zeros((P, T), dtype=int)  # 操作 y 的触发记录
cumulative_x_sum = np.zeros((P,), dtype=int)  # 累计 x 操作次数
leakage = 0.0                     # 泄漏量
cost = 0.0                        # 操作成本
K = 5.993

# 模拟压力变化和操作触发
for i in range(P):
    # Pressure[i, 0] = P_init[i][0]  # 设置每个初始压力的第一个时间步
    Pressure[i, 0] = P_init[i]  # 设置每个初始压力的第一个时间步

    # if P_init[i][0] > 3.2:  # 初始压力大于 3.2 的情况
    if P_init[i] > 3.2:  # 初始压力大于 3.2 的情况

        for t in range(1, T):
            if y_action[i, t - 1] == 1:  
                Pressure[i, t] = 3.5  # 如果 y_action 已经触发，后续压力固定为 3.5
                continue

            Pressure[i, t] = (1 - alpha) * Pressure[i, t - 1]  # 压力下降
            
            if 3.2 - Pressure[i, t] > 0.01:  
                if cumulative_x_sum[i] < 2:  
                    x_action[i, t] = 1  
                    Pressure[i, t:] = [3.5] * (T - t)  # 从当前时间步开始设置为3.5
                    cumulative_x_sum[i] += 1  
                else:  
                    y_action[i, t] = 1  
                    Pressure[i, t:] = [3.5] * (T - t)  
                    break  

    # elif P_init[i][0] < 3.2: 
    elif P_init[i] < 3.2: 
        y_action[i, 0] = 1  
        for t in range(1, T):
            Pressure[i, t] = 3.5  

for i in range(P):
    for t in range(T):
        v[i, t] = max(Pressure[i, t] - Pressure[i, t - 1],0) 
        leakage += K * v[i, t]
        cost += (10 * x_action[i, t]) + (100 * y_action[i, t])

# 输出结果
end_time = time.time()

for i in range(P):
    # print(f"Initial Pressure {i}: {P_init[i][0]:.4f}")
    print(f"Initial Pressure {i}: {P_init[i]:.4f}")
    print(f"Pressure over time: {Pressure[i]}")
    print(f"x actions: {x_action[i]}")
    print(f"y actions: {y_action[i]}")
    print("-" * 50)
print(f"Total leakage: {leakage:.4f}")
print(f"Total cost: {cost:.4f}")
print(f"Execution time:{end_time - start_time:.6f} seconds")





2025.02.24 正确最终版
import sys
sys.path.append('C:\\Program Files\\IBM\\ILOG\\CPLEX_Studio210\\cplex\\python\\20.1\\x64_win64')
sys.path.append("C:\\ProgramData\\anaconda3\\Lib\\site-packages")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from docplex.mp.model import Model
import streamlit.components.v1 as components
import plotly.graph_objects as go
import time


st.set_page_config(layout="wide")
hide_dataframe_row_index = """
            <style>
         div.c1{
                    background-color: #f5f5f5;
                    border: 2px solid;
                    padding: 20px 20px 20px 20px;
                    border-radius: 10px;
                    color: #ffc300;
                    box-shadow: 10px;
                    }
            </style>
            """


def data(x):
    df_p = x[x['value'] != 1].reset_index()
    df_x_y = x[x['value'].astype(int) == 1].reset_index()
    for i in range(df_p.shape[0]):
        r = df_p["name"].loc[i]
        e = r.split('_')[0]
        g = r.split('_')[1]
        h = r.split('_')[2]
        if (e == 'p'):
            df_p.loc[i, 'CAISSON'] = g
            df_p.loc[i, 'TEMPS'] = int(h)

    for i in range(df_x_y.shape[0]):
        r = df_x_y["name"].loc[i]
        e = r.split('_')[0]
        g = r.split('_')[1]
        h = r.split('_')[2]
        if(e == 'y') | (e == 'x'):
            df_x_y.loc[i, 'OPERATION'] = e
            df_x_y.loc[i, 'CAISSON'] = g
            df_x_y.loc[i, 'TEMPS'] = int(h)
        # else:
        #     df_x_y.loc[i, 'OPERATION'] = "NA"
        #     df_x_y.loc[i, 'CAISSON'] = g
        #     df_x_y.loc[i, 'TEMPS'] = 0
    df_x_y = df_x_y.dropna()[['value', 'OPERATION', 'CAISSON', 'TEMPS']].reset_index()
    
    return df_p, df_x_y


# Function to visualize the clustering
@st.cache_data
def vis_caisson(df_x_y, df_p, x, taux1):
    # Filter the data based on the selected caissons
    sns.color_palette("Paired")
    sns.set(rc={"figure.figsize":(12, 5)})
    sns.despine(left=True)
    x = [str(s) for s in x]
    df_p_caisson = df_p[df_p['CAISSON'].isin(x)]
    caisson = df_x_y[df_x_y['CAISSON'].isin(x)]


    chart = px.line(df_p_caisson, x='TEMPS', y='value', color='CAISSON', width=700, height=400)
    
    # Add the reference lines and vertical lines for the operations
    chart.add_hline(y=taux1, line_dash='dash', line_color='red', name='Taux 1')
    chart.add_hline(y=3.5, line_dash='dash', line_color='blue', name='3.5')
    
    for temp in caisson[caisson['OPERATION'] == 'x']['TEMPS'].values:
        chart.add_vline(x=temp, line_dash='dash', line_color='green', name='CR')
        chart.add_annotation(x=temp+0.5, y=2.95, text='CR', showarrow=False)
    
    for temp in caisson[caisson['OPERATION'] == 'y']['TEMPS'].values:
        chart.add_vline(x=temp, line_dash='dash', line_color='yellow', name='RE')
        chart.add_annotation(x=temp+0.5, y=2.95, text='RE', showarrow=False)
    
    
    # Set the x-axis tick angle and remove the legend title
    chart.update_layout(xaxis_tickangle=-45)
    
    # Set the y-axis range to start from 2.9
    chart.update_yaxes(range=[2.9, None])
    chart.update_xaxes(dtick='M1')

    # Remove legend
    chart.update_layout(showlegend=False)
    return chart


# newly editted function to show the legend index correctly
def vis_all_caisson(df_x_y, df_p, taux1):
    # Filter the data based on the selected caissons
    df_p['value'] = df_p['value'].astype(float)
    
    # Create a line chart of the pressure data using Plotly Express
    chart = px.line(df_p, x='TEMPS', y='value', color='CAISSON', width=700, height=400,
                    color_discrete_sequence=px.colors.qualitative.Plotly)
    
    # Add the reference lines
    chart.add_hline(y=taux1, line_dash='dash', line_color='green', name='Taux 1')
    chart.add_hline(y=3.5, line_dash='dash', line_color='blue', name='3.5')
    
    # Set the x-axis tick angle and legend title
    chart.update_layout(xaxis_tickangle=-45, legend_title='Caissons', plot_bgcolor='#f0fdf4')

    # Modify legend labels to start from 1
    for i, trace in enumerate(chart.data):
        chart.data[i].name = f'Caisson {i+1}'
    
    return chart

# first scheduling optimization model
def create_model(P, T, Pinit, Rcr, Rre, r_cr, r_re, P_rempli, alpha,alpha_param,deg, taux1, taux2, gamma, M, couts1, couts2, b1, b2, k1, W, K):

    """ Model """
    mdl = Model('Maintenance_caisson', log_output=True)
    x = mdl.binary_var_matrix(P, T, name = 'x')  
    y = mdl.binary_var_matrix(P, T, name = 'y')
    f = mdl.binary_var_matrix(P, T, name = 'f')  
    g = mdl.binary_var_matrix(P, T, name = 'g')
    z = mdl.binary_var_matrix(P, T, name = 'z')
    w = mdl.binary_var_matrix(P, T, name = 'w') 
    e = mdl.integer_var_matrix(P, T, name="e")
    p = mdl.continuous_var_matrix(P, T, name='p')
    d = mdl.binary_var_matrix(P, T, name = 'd')
    o = mdl.binary_var_matrix(P, T, name = 'o')
    v = mdl.continuous_var_matrix(P, T, name = 'v')
    slope = mdl.continuous_var_matrix(P, T, name='slope')
    leakageQuantity = 0.0

    # pressure evolution
    mdl.add_constraints((p[i, 0] == (Pinit[i].item()) for i in range(P)))
    mdl.add_constraints((p[i, t] >= 3.001 for i in range(P) for t in range(T)))
    mdl.add_constraints((p[i, t] <= 3.5 for i in range(P) for t in range(T)))

    # # CONTRAINTE DE RESOURCES 
    mdl.add_constraints((sum(r_cr*x[k, tp] for k in range(P) for tp in range(T) if (tp <= t) and (tp + b1 - 1>= t)) <= Rcr for t in range(T)))
    mdl.add_constraints((sum(r_re*y[k, tp] for k in range(P) for tp in range(T) if (tp <= t) and (tp + b2 - 1>= t)) <= Rre for t in range(T)))

    if deg == 'Linear degradation':
        mdl.add_constraints((p[i, t+1] >= (1- alpha)*p[i, t] - M*(f[i, t] + (g[i, t] + e[i,t])) for i in range(P) for t in range(T-1)))
        mdl.add_constraints((p[i, t+1] <= (1- alpha)*p[i, t] + M*(f[i, t] + (g[i, t] + e[i,t])) for i in range(P) for t in range(T-1)))    
    else:
        # not now useful but for other models
        mdl.add_constraints((p[i, t+1] >= (1- alpha_param[i][0]**(t/(t + 34) - 1))*p[i, t] - M*(f[i, t] + g[i, t]) for i in range(P) for t in range(1, T-1)))
        mdl.add_constraints((p[i, t+1] <= (1- alpha_param[i][0]**(t/(t + 34) - 1))*p[i, t] + M*(f[i, t] + g[i, t]) for i in range(P) for t in range(1, T-1)))
        
    # after a CR, pressure come back to 3.5 at the moment t+b1
    mdl.add_constraints((p[i, t+b1] >= P_rempli - M*(1 - x[i, t]) for i in range(P) for t in range(T-b1)))
    mdl.add_constraints((p[i, t+b1] <= P_rempli + M*(1 - x[i, t]) for i in range(P) for t in range(T-b1)))

    # Add linear interpolation constraints for CR during maintenance - for visualization
    for i in range(P):
        for t in range(T - b1):  
            for k in range(1, b1):
                # Calculate the desired pressure value at time t+k
                expected_pressure = p[i, t] + (k / b1) * (P_rempli - p[i, t])
                mdl.add_constraint(p[i, t + k] >= expected_pressure - M * (1 - x[i, t]))
                mdl.add_constraint(p[i, t + k] <= expected_pressure + M * (1 - x[i, t]))

        for t in range(T - b1, T):
            for k in range(1, T - t):
                expected_pressure = p[i, t] + (k / (T-t)) * (P_rempli - p[i, t])
                mdl.add_constraint(p[i, t + k] >= expected_pressure - M * (1 - x[i, t]))
                mdl.add_constraint(p[i, t + k] <= expected_pressure + M * (1 - x[i, t]))

    # Add linear interpolation constraints for RE during maintenance
    for i in range(P):
        for t in range(T - b2):  
            for k in range(1, b2):
                # Calculate the desired pressure value at time t+k
                expected_pressure = p[i, t] + (k / b2) * (P_rempli - p[i, t])
                mdl.add_constraint(p[i, t + k] >= expected_pressure - M * (1 - y[i, t]))
                mdl.add_constraint(p[i, t + k] <= expected_pressure + M * (1 - y[i, t]))

    for t in range(T - b2, T):
        for k in range(0, T - t):
            expected_pressure = p[i, t] + (k / (T-t)) * (P_rempli - p[i, t])
            mdl.add_constraint(p[i, t + k] >= expected_pressure - M * (1 - y[i, t]))
            mdl.add_constraint(p[i, t + k] <= expected_pressure + M * (1 - y[i, t]))
    
    # after RE, pressure come back to 3.5 from moment t+b2
    for i in range(P):
        for t in range(T):
            mdl.add_constraints((p[i, tp] >= P_rempli - M*(1 - y[i, t]) for tp in range(t+b2, T)))
            mdl.add_constraints((p[i, tp] <= P_rempli + M*(1 - y[i, t]) for tp in range(t+b2, T)))
        
    # only one RE is need for whole horizon             
    mdl.add_constraints((y[i, tp] <= 1 - y[i, t] for i in range(P) for t in range(T-b2) for tp in range(t+b2, T)))
    
    # definition for cumulating the sum of RE action
    for t in range(T):
        for i in range(P):
                mdl.add_constraint(e[i,t] == mdl.sum(y[i, k] for k in range(t)))

    # definition for cumulating the sum of CR action and definition of penalty term z
    mdl.add_constraints((w[i, t] == mdl.sum(x[i, tp] for tp in range(max(0, t - W + 1), t + 1)) for i in range(P) for t in range(T)))
    mdl.add_constraints((w[i, t] <= 2 + z[i, t] for i in range(P) for t in range(T)))
    mdl.add_constraints((w[i, t] >= 3 - 3 * (1 - z[i, t]) for i in range(P) for t in range(T)))

    # f definition
    mdl.add_constraints((f[i, t + k] >= x[i, t] for i in range(P) for t in range(T - b1) for k in range(b1)))
    # g definition
    mdl.add_constraints((g[i, t + k] >= y[i, t] for i in range(P) for t in range(T - b2) for k in range(b2)))
    mdl.add_constraints((f[i, t] <= sum(x[i, tp] for tp in range(max(0, t - b1 + 1), t + 1)) for i in range(P) for t in range(T)))
    mdl.add_constraints((g[i, t] <= sum(y[i, tp] for tp in range(max(0, t - b2 + 1), t + 1)) for i in range(P) for t in range(T)))
    mdl.add_constraints(x[i, t] + x[i, tp] <= 1 for i in range(P) for t in range(T) for tp in range(T) if (tp < t) and (tp + b1 >= t)) 

    # define variable v 
    for i in range(P):
        for t in range(1, T):
            mdl.add_constraint(v[i, t] >= p[i, t-1] - p[i, t])
            mdl.add_constraint(v[i, t] >= 0)
            mdl.add_constraint(v[i, t] <= (p[i,t-1] - p[i, t]) + M * (1 - o[i, t]))
            mdl.add_constraint(v[i, t] <= M * o[i, t])
            mdl.add_constraint(M * o[i, t] >= p[i,t-1] - p[i, t])
            mdl.add_constraint(M * (1- o[i, t]) >= - (p[i,t-1] - p[i, t]))

    # total leakage quantity constraint     
    leakageQuantity = mdl.sum(K * v[i, t] for i in range(P) for t in range(1,T))
    mdl.add_constraint(leakageQuantity <= 12.5)   #!!! here need to be changed accroding to different horizon and nb of CSEMs

    #-----------------------------------------------------------------------------------------------------------------------------
    # OBJ FUNCTION
    cost = mdl.sum(couts1*x[i, t] + couts2*y[i, t] + k1*z[i,t] for i in range(P) for t in range(T))
    mdl.minimize(cost)

    # Solve the model
    si = mdl.solve()
    print(mdl.solve_details.status)
    df = mdl.solution.as_df()

    st.dataframe(df)

    # Calculate total leakage from the solution 
    total_leakage_value = round(leakageQuantity.solution_value, 3)

    # # if want to print the total leakage quantity in the UI
    # st.write(f"Total leakage quantity: {si[leakageQuantity]}")

    sol = si.get_blended_objective_value_by_priority()[0]
    df_p, df_x_y = data(df)

    # create file name
    file_name = f"P{P}_T{T}.csv"

    # save result DataFrameto file CSV
    df_x_y.to_csv(file_name, index=False, encoding='utf-8')

    print(f"Results saved to: {file_name}")

    # Return the solution
    return df_p, df_x_y, sol, total_leakage_value


def data_clustring(x):
    for i in range(x.shape[0]):
        r = x["name"].loc[i]
        e = r.split('_')[0]
        g = r.split('_')[1]   # types of operations
        h = r.split('_')[2]   # cluster information
        x.loc[i, 'cluster'] = h
        x.loc[i, 'OPERATION'] = g
        
    return x

def update_df_x_y(df_x_y):
    df_x_y = df_x_y.rename(columns={"CC": "center"})
    df_x_y["OPERATION"] = df_x_y["OPERATION"].replace({"x": "CR", "y": "RE"})
    df_x_y = df_x_y.drop("value", axis=1)
    df_x_y = df_x_y.drop("index", axis=1)
    return df_x_y
    
def vis_clustring(df_x_y, C):
    # Set the color palette and plot size
    # sns.color_palette("Paired")
    sns.set(rc={"figure.figsize":(12, 5)})
    sns.despine(left=True)
    
    # Get the number of caissons and data points
    x = len(df_x_y['CAISSON'].unique()) 
    y = df_x_y.shape[0]
  
    # Create a scatter plot with Plotly Express
    df_x_y['cluster'] = pd.factorize(df_x_y['CC'])[0]
    df_x_y["cluster"] = df_x_y["cluster"].astype(str)
    color_sequence = px.colors.qualitative.Dark24
    color_dict = {str(i): px.colors.qualitative.Plotly[i] for i in range(df_x_y['cluster'].nunique())}
    label_dict = {str(i): f'Category {chr(i + 65)}' for i in range(df_x_y['cluster'].nunique())}
    size_dict = {str(i): (i + 1) * 20 for i in range(df_x_y['cluster'].nunique())}

    centroid_data = df_x_y[df_x_y['TEMPS'] == df_x_y['CC']]

    df_x_y = df_x_y.rename(columns={"CC": "center"})
    df_x_y["OPERATION"] = df_x_y["OPERATION"].replace({"x": "CR", "y": "RE"})
    
    for i in range(df_x_y.shape[0]):
        if df_x_y.loc[i, 'OPERATION'] == "CR":
            df_x_y.loc[i, 'TOLERANCE'] = 7  
        if df_x_y.loc[i, 'OPERATION'] == "RE":
            df_x_y.loc[i, 'TOLERANCE'] = 1  
        

    # Create the scatter plot with background colors for each cluster
    chart = px.scatter(df_x_y.sort_values('CAISSON', ascending=False), x='TEMPS', y='CAISSON', color='cluster',color_discrete_map=color_dict,
                    
                     width=800, height=500, hover_data=['center', 'OPERATION', 'TOLERANCE'])
    
    # Add a background color to each cluster
    for i, color in color_dict.items():
        cluster_data = df_x_y[df_x_y['cluster'] == str(i)]
        cluster_center = pd.DataFrame(cluster_data.loc[(cluster_data['TEMPS'] == cluster_data['center'].astype(int)), ['TEMPS', 'CAISSON']].head(1))

        for index, row in cluster_data.iterrows():           
            x1 = row['TEMPS']
            y1 = row['CAISSON']
            try:
                x2 = cluster_center.iloc[0]['TEMPS']
                y2 = cluster_center.iloc[0]['CAISSON']
                if x1 != x2:
                    # Draw the lasso line
                    chart.add_shape(type='line',
                                x0=x1,
                                y0=y1,
                                x1=x2,
                                y1=y2,
                                line=dict(color=color, width=1, dash='dot'))
    
                else:
                    chart.add_shape(type='circle',
                            xref='x', yref='y',
                            x0=int(x1)-0.5, y0=int(y1)-0.2,
                            x1=int(x1)+0.5, y1=int(y1)+0.2,
                            line=dict(color=color, width=2))
            # chart.update_shapes(line_color=color)
            except:
                continue

    # Customize the plot
    chart.update_traces(marker=dict(sizemode='diameter', sizeref=120))
    chart.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=2, range=[0,45]),
                         yaxis=dict(tickmode='linear', tick0=-2, dtick=2, range=[-2,x+2]),
                         legend_title='', showlegend=True)
    chart.update_layout(plot_bgcolor='#fafafa')
    
    return chart

@st.cache_data
def df_x_y_mod(df, couts1, couts2):

    # Drop 'index' and 'value' columns if they exist
    df = df.drop(['index', 'value'], axis=1, errors='ignore')

    # replace the values in the OPERATION column
    df['OPERATION'] = df['OPERATION'].replace({'x': 'CR', 'y': 'RE'})
    
    P = int(df['CAISSON'].nunique())

    price_map = {('CR'): couts1}
    price_map.update({('RE'): couts2})
    
    df['CAISSON'] = df['CAISSON'].astype('int') + 1 
    caisson_col = df.pop('CAISSON')
    df.insert(0, 'CAISSON', caisson_col)
    return df

# second clustering optimization model
def clusteringg(df_x_y):
    
    C = df_x_y.TEMPS.values

    D = np.array([i - np.array(df_x_y.TEMPS.values.astype(int)) for i in C]) # les distances entre les taches (time difference)
    D[D[:] < 0 ] = +99999
#     print(f'the matrix D is : {D}')
    
    OCR = df_x_y[df_x_y['OPERATION'].str.contains('x')].first_valid_index() # les points de nuage DES CR
    OCRf = df_x_y[df_x_y['OPERATION'].str.contains('x')].last_valid_index() # les points de nuage DES CR
    try:
        ORE = df_x_y[df_x_y['OPERATION'].str.contains('y')].first_valid_index()  # les points de nuage DES RE
        OREf = df_x_y[df_x_y['OPERATION'].str.contains('y')].last_valid_index()  # les points de nuage DES REF
    except:
        ORE = None

    O = df_x_y.shape[0]
    # df_x_y.to_excel('df.xlsx', index=False)
    K = O
    mdlgp = Model('Maintance_grouping')
    theta = mdlgp.binary_var_matrix(O, K, name = 'theta')
    M = 99999
   
    cluster = mdlgp.binary_var_list(K, name = 'cluster')
    mdlgp.minimize(mdlgp.sum(cluster[k] for k in range(K)))
    
    mdlgp.add_constraints(theta[i, k] <= cluster[k] for i in range(O) for k in range(K))
    mdlgp.add_constraints(sum(theta[i, k] for k in range(K)) == 1 for i in range(O))
    
    if OCRf != None:
        for i in range(OCRf + 1):
            for k in range(K):
                    mdlgp.add_constraint(theta[i, k] * D[k, i] <= 7)    # we consider here only the positive distances
                    
    if ORE != OREf:
        mdlgp.add_constraints(theta[i, k]*D[k, i] <= 1 for i in range(ORE, OREf+1) for k in range(K))


    if OCRf != None and ORE != None:
        for k in range(K):
            for i in range(OCRf + 1):
                for j in range(ORE, OREf+1):
                    mdlgp.add_constraint(theta[i, k] + theta[j, k] <= 1)
                    
                    # Maximum number of tasks limit (group capacity constraint) 
                    mdlgp.add_constraints(sum(theta[i, k] for i in range(OCRf+1)) <= 5 for k in range(K))
                    mdlgp.add_constraints(sum(theta[i, k] for i in range(ORE, OREf+1)) <= 3 for k in range(K))

    si2 = mdlgp.solve()
    val = si2.get_blended_objective_value_by_priority()[0]
    print(mdlgp.print_solution(log_output=False))

    df = mdlgp.solution.as_df()
    df = df[df['name'].str.contains('theta')]
    sol = si2.get_blended_objective_value_by_priority()[0]
    dfc = data_clustring(df)
    S = np.zeros(K) 
    
    # dfc.to_csv("test1.csv")

    Data = dfc.sort_values(by = 'cluster')['cluster'].value_counts().to_frame().reset_index()
    for idx in Data.index:
        v = idx
        S[int(v)] = Data[Data.index == idx]['cluster'].values[0]

    CC = C[dfc.cluster.values.astype('int')] 
    df_x_y['CC'] = CC
    chart = vis_clustring(df_x_y, CC)
    st.plotly_chart(chart)
    return sol

# Create the Streamlit app
def title_with_background(title, background_color):
    st.markdown(f"""
        <h1 style='background-color:{background_color};border-radius:10px;color:black;text-align:center;margin-bottom:20px'>{title}</h1>
    """, unsafe_allow_html=True)
def line_with_background(title, background_color):
    st.markdown(f"""
                <hr style='background-color:{background_color};border-left: 1px solid #ccc;height: 100%;position: absolute;left: 50%;'></h>
                """, unsafe_allow_html=True)

def app():
    # st.title('Maintenance Caisson')
    title_with_background("Maintenance Planification and Optimization: RTE Use Case", "#f0fdf4")
    st.markdown("<br>", unsafe_allow_html=True) 

    # st.title(':black[Maintenance Caisson] ')

    col1, col2, col3 = st.columns([1, 1.3, 2.4])
    
    def set_alpha(P):
        alpha_high = 0.008
        return alpha_high
    
    with col1:
        st.write("## Model Inputs")
        alert = """ <script>alert('This option is not available.')</script> """
        # deg = components.html(radio_button)  
        deg = st.radio("Degradation/Leakage Type:",
                                    ["Linear degradation", "Function of time", "Stochastic degradation", "Physical modelling"],
                                    index=0,
                                    key="degradation",
                                    )
        # if "Stochastic" or "Physical" in deg:
            # components.html(alert)
            # st.warning("This option is not available yet, please make another choice")
        P = st.slider('### Number of CSEM', min_value=1, max_value=20, step=1)
        T = st.slider('### Time horizon (days)', min_value=15, step=1)
        
        np.random.seed(42)      # set seed to make sure the random values are the same each time
        Pinit = np.random.uniform(3.01, 3.5001, (P, 1))    
        
        # store the initial pressure values in a CSV file
        # tranform numpy list to Pandas DataFrame
        P_init = pd.DataFrame(Pinit)
        folder_name = "Pinit2"
        file_path = f"{folder_name}/P{P}_T{T}.csv"
        P_init.to_csv(file_path, index=False, encoding='utf-8', header=False) 
        
        Rcr = st.slider('### Available Technicians for CR', value=100)     #available technicians for CR
        Rre = st.slider('### Available Technicians for REPAIR', value=100)    #available technicians for RE
        alpha = set_alpha(P)
        alpha_param = np.random.uniform(500, 1000, (P, 1))     
        taux1 = 3
        taux2 = 3.2
        gamma = 99999
        r_cr = 3   #nb of technicians for CR
        r_re = 5    #nb of technicians for RE
        M = 3.6     # big M
        P_rempli = 3.5  
        couts1 = 10    # couts1 : cost of CR
        couts2 = 100   # couts2 : cost of RE
        b1 = 1
        b2 = 3
        k1 = 1000
        W = 15
        C_crmax = 5       # group capacity constraint for CR
        C_remax = 2       # group capacity constraint for RE
        K = 5.993        # volume = 1m3, constant for transformation of pressure to masse
     
        if 'df_p' not in st.session_state:
            st.session_state.df_p = None

        if st.button('Run Optimization'):

            with st.spinner("Running function..."): 
                start_time = time.time()  # record start time
                df_p, df_x_y, sol, total_leakage_value = create_model(P, T, Pinit, Rcr, Rre, r_cr, r_re, P_rempli, alpha, alpha_param, deg, taux1, taux2, gamma, M, couts1, couts2, b1, b2, k1, W, K)
                
                end_time = time.time()  # record end time
                execution_time = end_time - start_time  # calculate execution time
                
                st.session_state.df_p = df_p
                st.session_state.df_x_y = df_x_y
                st.session_state.sol = np.around(sol, 2)
                st.session_state.total_leakage_value = total_leakage_value
            st.session_state.couts1 = couts1
            st.session_state.couts2 = couts2
            
            # Show execution time       
            st.write(f"Optimization completed in **{execution_time:.2f} seconds**.")
       
    # st.markdown('<div class="vertical-line"></div>', unsafe_allow_html=True)
    
    # Create the Docplex model and get the solution
    with col2:
        if st.session_state.df_p is not None :
            dff = df_x_y_mod(st.session_state.df_x_y, st.session_state.couts1, st.session_state.couts2)
            dff.index = dff.index + 1  # 将索引改为从1开始
            # dff =st.session_state.df_x_y
            st.write('## Optimization Output')
            st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
            # st.write(dff.to_html(index=False), unsafe_allow_html=True)

            st.write('#### Total Cost (€):', st.session_state.sol)
            st.write('#### Total leakage quantity (kg):', st.session_state.total_leakage_value)
            st.write('#### Maintenance Actions:', dff.shape[0])
            st.write('##### Complément Remplissage (CR):', dff[dff['OPERATION'] == 'CR'].shape[0])
            st.write('##### Repair (RE):', dff[dff['OPERATION'] == 'RE'].shape[0])

            dff = dff.rename(columns={'TEMPS': 'TIME'})
            st.dataframe(dff, height=350)
            
    with col3:
        caissons = [f"Caisson {i+1}" for i in range(P)]

        if st.session_state.df_p is not None :
            st.write("## Data Visualization")
            c = st.radio('CSEM Visualization', caissons, horizontal=True)
            st.markdown("<hr>", unsafe_allow_html=True) 

            if st.button(f'Visualize all CSEMs'):   
                # Visualize the clustering
                fig2 = vis_all_caisson(st.session_state.df_x_y, st.session_state.df_p, taux1)
                st.plotly_chart(fig2)
            else:
                fig = vis_caisson(st.session_state.df_x_y, st.session_state.df_p, str(int(str(c)[-1]) - 1), taux1)
                st.plotly_chart(fig)


    st.markdown("<hr>", unsafe_allow_html=True) 
 
    col3, col4 = st.columns([2, 1])
    col11, col12 = st.columns([1, 0.01])
    col5, col6, col7 = st.columns([1, 1, 1])
    col8, col9, col10 = st.columns([1, 1, 1])
    
    if st.button('Maintenance Groupping'): 

        # with col11:
        #     comm = f"""
        #     <h1 style="box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);margin: 10px;padding:15px">Assuming that the cost  of traveling represent 40% of total cost of maintenance</h1>
        #     """
        #     components.html(comm)

        with col3:
            if st.session_state.df_x_y is not None :
                # Visualize the clustering
                sol = st.session_state.sol
                sol2 = clusteringg(st.session_state.df_x_y)
                # vis_clustring(st.session_state.df_x_y, (st.session_state.df_x_y))
                O = st.session_state.df_x_y.shape[0]
                # gsol = round(0.6*sol + (0.4*sol)/sol2, 2)
                gsol = round(0.6*sol + (0.4*sol2)/O, 2)
                st.session_state.gsol = gsol
                gapc= round(float((sol - gsol)/sol)*100, 2)
                gapc = f"{gapc}%"

                gap= round(float((O - sol2)/O)*100, 2)
                gap = f"{gap}%"
            st.markdown("<style>.big-column{padding-right: 30px;}</style>", unsafe_allow_html=True)
        with col4:
            st.session_state.sol2 = sol2
            if st.session_state.df_p is not None :
                dfa = update_df_x_y(st.session_state.df_x_y)
                st.dataframe(dfa, height=450)
     
        fm = st.session_state.sol
        fm2 = st.session_state.gsol
        fm3 = st.session_state.sol2
                        
        with col5:
            com = f"""
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding-top:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>Maintenance Cost (Before grouping)</h2>
                        <h1>{fm} (€)</h1>
                        <p></p>
                    </div>
                    """
            components.html(com)
            # st.write('<style>  background-color: #FFFFFF;border-radius: 10px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding: 20px;text-align: center;}</style>', unsafe_allow_html=True)
        with col6:
            com = f"""
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding-top:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>Maintenace Cost (After grouping)</h2>
                        <h1>{fm2} (€)</h1>
                        <p></p>

                    </div>
                    """
            components.html(com)
        with col7:
            com = f"""
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding-top:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>GAP</h2>
                        <h1 style="color:#22c55e"><span><svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-arrow-big-up-line" width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">
                        <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
                        <path d="M9 12h-3.586a1 1 0 0 1 -.707 -1.707l6.586 -6.586a1 1 0 0 1 1.414 0l6.586 6.586a1 1 0 0 1 -.707 1.707h-3.586v6h-6v-6z"></path>
                        <path d="M9 21h6"></path>
                        </svg></span>  {gapc}</h1>

                    </div>
                    """
            components.html(com)
        with col8:
            com = f"""
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>Travels (Before grouping)</h2>
                        <h1>{O}</h1>

                    </div>
                    """
            components.html(com)                
        with col9:
            com = f"""
                    <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding:2px; border-radius: 5px; text-align: center; height: 130px;">
                        <h2>Travels (After grouping)</h2>
                        <h1>{int(sol2)}</h1>

                    </div>
                    """
            components.html(com)
        
        with col10:
            com = f"""
                <div style="overflow: visible;margin-buttom:100px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);padding:2px; border-radius: 5px; text-align: center; height: 130px;">
                    <h2>GAP</h2>
                    <h1 style="color:#10b981"><span><svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-arrow-big-up-line" width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">
                        <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
                        <path d="M9 12h-3.586a1 1 0 0 1 -.707 -1.707l6.586 -6.586a1 1 0 0 1 1.414 0l6.586 6.586a1 1 0 0 1 -.707 1.707h-3.586v6h-6v-6z"></path>
                        <path d="M9 21h6"></path>
                        </svg></span>{gap}</h1>
                    <p></p>

                </div>
                """
            components.html(com)


# Run the app
if __name__ == '__main__':
    app()







# ---------------------------------------------------------------------------
2025.03.24
# # calculate the new center and update the task time outside the window
# def update_task(df_x_y, tasks, task_groups):
#     updated_tasks = tasks[:]
    
#     for group in task_groups:
#         # calculate the new center
#         new_center = np.mean([task["time"] for task in group])
        
#         # update the cluster center of each group
#         for task in group:
#             task["new_center"] = new_center
        
#         # update the task time outside the window
#         for i, task in enumerate(updated_tasks):
#             if task in group:
#                 continue
#             if task["time"] > new_center:
#                 time_shift = new_center - task["time"]
#                 updated_tasks[i]["time"] += time_shift
#             else:
#                 time_shift = task["time"] - new_center
#                 updated_tasks[i]["time"] -= time_shift
    
#     return updated_tasks

# def update_overlapping(df_x_y):
#     # get all the CAISSONs
#     caissons = df_x_y['CAISSON'].unique()
    
#     # loop through all the CAISSONs
#     for caisson in caissons:
#         # Get current CAISSON's RE cluster center
#         re_centers = df_x_y[(df_x_y['CAISSON'] == caisson) & (df_x_y['OPERATION'].str.contains('y'))]['CC'].unique()
        
#         if len(re_centers) > 0:
#             # get the first RE center time
#             first_re_center = min(re_centers)
            
#             # Check if there are CR tasks after the first RE center time
#             cr_tasks = df_x_y[(df_x_y['CAISSON'] == caisson) & (df_x_y['OPERATION'].str.contains('x')) & (df_x_y['DATE'] >= first_re_center)]
            
#             if not cr_tasks.empty:
#                 # delete all the CR tasks after the first RE center time
#                 df_x_y = df_x_y[df_x_y['CAISSON'] != caisson]
#                 print(f"Deleted all the following tasks of CAISSON {caisson}")
    
#     return df_x_y

# def update_overlapping(df_x_y):
#     # get all the CAISSONs
#     caissons = df_x_y['CAISSON'].unique()
    
#     # iterate through all the CAISSONs
#     for caisson in caissons:
#         # get the RE cluster centers of the current CAISSON
#         re_centers = df_x_y[(df_x_y['CAISSON'] == caisson) & (df_x_y['OPERATION'].str.contains('y'))]['CC'].unique()
        
#         if len(re_centers) > 0:
#             # find the first RE center time
#             first_re_center = min(re_centers)
            
#             # find the CR tasks after the first RE center time
#             mask = (df_x_y['CAISSON'] == caisson) & (df_x_y['OPERATION'].str.contains('x')) & (df_x_y['DATE'] >= first_re_center)
            
#             # delete the CR tasks after the first RE center time
#             df_x_y = df_x_y[~mask]
#             print(f"Deleted tasks of CAISSON {caisson} after the first RE center time")
    
#     return df_x_y


# def update_tasks(df_x_y):
#     # get all the CAISSONs
#     caissons = df_x_y['CAISSON'].unique()
    
#     # iterate through all the CAISSONs
#     for caisson in caissons:
#         caisson_tasks = df_x_y[df_x_y['CAISSON'] == caisson]
        
#         # find the first task's CC and original time
#         first_task_cc = caisson_tasks[caisson_tasks['CC'].notna()]['CC'].iloc[0]
#         first_task_original_time = caisson_tasks[caisson_tasks['CC'].notna()]['DATE'].iloc[0]
        
#         # calculate the time difference between the first task's CC and original time
#         time_diff = first_task_cc - first_task_original_time
        
#         # update the time of all the tasks
#         updated_tasks = []
#         for index, task in caisson_tasks.iterrows():
#             if pd.isna(task['CC']):
#                 updated_task = task.copy()
#                 updated_task['DATE'] += time_diff
#                 updated_tasks.append(updated_task)
#             else:
#                 updated_tasks.append(task)
        
#         # update the tasks in the original dataframe
#         df_x_y.loc[df_x_y['CAISSON'] == caisson] = pd.DataFrame(updated_tasks)
    
#     return df_x_y



# -----------------------------------------------------------
2025/03/26 not all correct version
import sys
sys.path.append('C:\\Program Files\\IBM\\ILOG\\CPLEX_Studio210\\cplex\\python\\20.1\\x64_win64')
sys.path.append("C:\\ProgramData\\anaconda3\\Lib\\site-packages")
import numpy as np
import pandas as pd
from docplex.mp.model import Model
import matplotlib.pyplot as plt
import random

# define the parameters
PARAMS = {
    'P': 2,        # number of caissons
    'T': 14,      # horizon
    'Cmax': 3,      # maximum number of tasks of one cluster
    'window_size': 14,   # window size
    'advanced': 3          # advanced time
}   

# initialize the random values
random.seed(42)
alpha_values = [round(random.uniform(0.01, 0.05), 3) for _ in range(PARAMS['P'])]
initial_p_values = [round(random.uniform(3.2, 3.5), 2) for i in range(PARAMS['P'])]


# def clustering_CR(df_window, window_size, window_start):
#     """Grouping model"""
#     N = df_window.shape[0]

#     mdl = Model('Dynamic_Window_Clustering')
    
#     # define the variables
#     theta = mdl.binary_var_matrix(N, N, name='theta')
#     center = mdl.binary_var_matrix(N, window_size, name='center')  
#     cluster = mdl.binary_var_list(N, name='cluster')
#     o = mdl.integer_var_list(N, name='o', lb=0, ub=window_size-1) 
#     z = mdl.binary_var_matrix(N, window_size, name = 'z')
#     a = mdl.integer_var_list(N, name='a', lb=0, ub=window_size-1)
#     b = mdl.integer_var_list(N, name='b', lb=0, ub=window_size-1)

#     M1 = 1
#     M = 14

#     # objective function
#     mdl.minimize(mdl.sum(cluster[k] for k in range(N)))
    
   
#     # cluster constraint
#     mdl.add_constraints(theta[n, k] <= cluster[k] for n in range(N) for k in range(N))
#     mdl.add_constraints(mdl.sum(theta[n, k] for k in range(N)) == 1 for n in range(N))
    
#     # maximum capacity constraint
#     mdl.add_constraints(mdl.sum(theta[n, k] for n in range(N)) <= PARAMS['Cmax'] for k in range(N))
    
#     # only one center constraint
#     mdl.add_constraints(mdl.sum(center[k, t] for t in range(window_size)) == cluster[k] for k in range(N))

#     # task grouping and center determination
#     # definition of z
#     mdl.add_constraints(z[k, t] <= M * center[k, t] for k in range(N) for t in range(window_size))
#     mdl.add_constraints(z[k, t] >= o[k] - M * (1 - center[k, t]) for k in range(N) for t in range(window_size))
#     mdl.add_constraints(z[k, t] <= o[k] for k in range(N) for t in range(window_size))
    
#     # print(theta.keys())  # make sure (n, k) is in keys
#     # print(center.keys())  # make sure (k, t) is in keys



#     for n in range(N):
#         local_time = df_window.iloc[n]['window_time']
#         # print(local_time)
#         mdl.add_constraint(a[n] >= local_time - PARAMS['advanced'])
#         mdl.add_constraint(a[n] >= 0)
#         mdl.add_constraint(b[n] <= local_time + int((3.2 - 3.0)/alpha_values[int(df_window.iloc[n]["CSEM"])]))
#         mdl.add_constraint(b[n] <= window_size)
#         mdl.add_constraints(z[k, t] >= a[n] - M * (1 - theta[n, k]) - M * (1 - center[k, t]) for k in range(N) for t in range(window_size))
#         mdl.add_constraints(z[k, t] <= b[n] + M * (1 - theta[n, k]) for k in range(N) for t in range(window_size))

#         # mdl.add_constraints(z[k] >= local_time - PARAMS['advanced'] - M * (1 - theta[n, k]) - M * (1 - center[k, t]) for k in range(N) for t in range(window_size))
#         # mdl.add_constraints(z[k] <= local_time + int((3.2 - 3.0)/alpha_values[int(df_window.iloc[n]["CSEM"])]) + M * (1 - theta[n, k]) for k in range(N))
    
#     # solve the model
#     solution = mdl.solve()
#     if not solution:
#         print(f"Failed to solve the problem: {mdl.solve_details.status}")
#         return None
    
#     # result processing
#     df_window = df_window.copy()
#     centers = [None] * N  

#     # calculate the center time
#     for k in range(N):  # iterate over all tasks
#         for t in range(window_size):
#             if solution.get_value(center[k, t]) > 0.9:
#                 centers.append(window_start + t)  # calculate the original time
#                 break
                
    
#     # update the center to the original dataframe
#     # for n in range(N):
#     #     for k in centers:
#     #         if solution.get_value(theta[n,k]) > 0.9:
#     #             df_window.iloc[df_window.index[n]]['CENTER'] = centers[k]
#     #             break

#     # update the center to the original dataframe
#     for n in range(N):  # iterate over all tasks
#         for k in range(len(centers)):  # iterate over all centers
#             if solution.get_value(theta[n, k]) > 0.9:
#                 df_window.iloc[n, df_window.columns.get_loc("CENTER")] = centers[k]
#                 break  # oece the center is found, break the loop
#     df_window["CENTER"] = df_window["CENTER"].astype("float")  # make sure the column is float type

#     return df_window


def clustering_CR(df_window, window_size, window_start):
    """Grouping model"""
    N = df_window.shape[0]

    mdl = Model('Dynamic_Window_Clustering')
    
    # define the variables
    theta = mdl.binary_var_matrix(N, window_size, name='theta')
    center = mdl.binary_var_matrix(window_size, window_size, name='center')  
    cluster = mdl.binary_var_list(window_size, name='cluster')
    o = mdl.integer_var_list(window_size, name='o', lb=0, ub=window_size-1) 
    z = mdl.binary_var_matrix(window_size, window_size, name = 'z')
    a = mdl.integer_var_list(N, name='a', lb=0, ub=window_size-1)
    b = mdl.integer_var_list(N, name='b', lb=0, ub=window_size-1)

    M1 = 1
    M = 10

    # objective function
    mdl.minimize(mdl.sum(cluster[k] for k in range(window_size)))
    
   
    # cluster constraint
    mdl.add_constraints(theta[n, k] <= cluster[k] for n in range(N) for k in range(window_size))
    mdl.add_constraints(mdl.sum(theta[n, k] for k in range(window_size)) == 1 for n in range(N))
    
    # maximum capacity constraint
    mdl.add_constraints(mdl.sum(theta[n, k] for n in range(N)) <= PARAMS['Cmax'] for k in range(window_size))
    
    # only one center constraint
    mdl.add_constraints(mdl.sum(center[k, t] for t in range(window_size)) == cluster[k] for k in range(window_size))

    # task grouping and center determination
    # definition of z
    # mdl.add_constraints(z[k, t] <= M1 * center[k, t] for k in range(window_size) for t in range(window_size))
    # mdl.add_constraints(z[k, t] >= o[k] - M1 * (1 - center[k, t]) for k in range(window_size) for t in range(window_size))
    # mdl.add_constraints(z[k, t] <= o[k] for k in range(window_size) for t in range(window_size))
    
    # print(theta.keys())  # make sure (n, k) is in keys
    # print(center.keys())  # make sure (k, t) is in keys

    for n in range(N):
        local_time = df_window.iloc[n]['window_time']
    #     # print(local_time)
    #     mdl.add_constraint(a[n] >= local_time - PARAMS['advanced'])
    #     mdl.add_constraint(a[n] >= 0)
    #     mdl.add_constraint(b[n] <= local_time + int((3.2 - 3.0)/alpha_values[int(df_window.iloc[n]["CSEM"])]))
    #     mdl.add_constraint(b[n] <= window_size)
    #     mdl.add_constraints(z[k, t] >= a[n] - M * (1 - theta[n, k]) - M * (1 - center[k, t]) for k in range(window_size) for t in range(window_size))
    #     mdl.add_constraints(z[k, t] <= b[n] + M * (1 - theta[n, k]) for k in range(window_size) for t in range(window_size))

        mdl.add_constraints(o[k] >= local_time - PARAMS['advanced'] - M * (1 - theta[n, k]) for k in range(window_size))
        mdl.add_constraints(o[k] <= local_time + int((3.2 - 3.0)/alpha_values[int(df_window.iloc[n]["CSEM"])]) + M * (1 - theta[n, k]) for k in range(window_size))
    
    # solve the model
    solution = mdl.solve()
    if not solution:
        print(f"Failed to solve the problem: {mdl.solve_details.status}")
        return None
    
    # result processing
    df_window = df_window.copy()
    centers = [0] * window_size  

    # iterate over all tasks
    for n in range(N):
        for k in range(window_size):
            if solution.get_value(theta[n, k]) >= 0.9: # if this task is in the cluster k
                centers[n] = k  # center for this task is k
                df_window.iloc[n, df_window.columns.get_loc("CENTER")] = centers[k]
                break  
                
    df_window["CENTER"] = df_window["CENTER"].astype("float")  # make sure the column is float type

    return df_window


def main(window_size):
    """Main function"""
    df = pd.read_csv('maintenance_plan.csv')
    # df['CENTER'] = np.nan  # initialize the center column
    df['CENTER'] = 0  # initialize the center column
    window_start = 0
    while window_start <= PARAMS['T'] - window_size:
        # dynamic window selection
        window_end = window_start + window_size
        mask = (df['DATE'] >= window_start) & (df['DATE'] < window_end)
        window_df = df.loc[mask].copy()
        # print(window_df)
        # task local time conversion
        window_df['window_time'] = window_df['DATE'] - window_start
        # print(window_df)
        first_tasks = window_df.groupby('CSEM', as_index=False).first()
        print(first_tasks)
        print(alpha_values)
    
        # clustering
        clustered_df = clustering_CR(first_tasks, window_size, window_start)
        # print(clustered_df)
        
        if clustered_df is not None:
            # update the center column
            df.update(clustered_df[['CENTER']])
            
            # update the secondary tasks time
            update_secondary_tasks(df, clustered_df, window_start, window_end)
        
        # slide the window
        window_start += int(window_size/2)
    
    # result verification
    assert df['CENTER'].notna().all(), "Some tasks are not clustered"
    return df

def update_secondary_tasks(main_df, clustered_df, window_start, window_end):
    """Update the secondary tasks time"""
    for caisson in clustered_df["CSEM"].unique():
        # get the tasks in the same caisson
        tasks = main_df[
            (main_df['CSEM'] == caisson) & 
            (main_df['DATE'].between(window_start, window_end))
        ]
        
        if len(tasks) > 1:
            # get the second task index
            second_idx = tasks.index[1]
            
            # get the previous center time amd calculate the new time for the second task
            center_time = clustered_df.loc[clustered_df['CSEM'] == caisson, 'CENTER'].iloc[0]
            
            # calulate the new time
            alpha = alpha_values[caisson]
            print(center_time)
            print(alpha)
            new_date = int(center_time + 1 + (3.5-3.2)/alpha)
            print(new_date)
            # new_date = np.clip(new_date, 0, PARAMS['T']-1)
            
            # update the time
            main_df.loc[second_idx, 'DATE'] = new_date

if __name__ == "__main__":
    result_df = main(PARAMS['window_size'])
    
    # # plot the clustering center
    # plt.figure(figsize=(12,6))
    # plt.scatter(result_df['DATE'], result_df['CENTER'], alpha=0.7)
    # plt.title("cluster center distribution")
    # plt.xlabel("original time")
    # plt.ylabel("cluster center")
    # plt.grid(True)
    # plt.show()
    
    print("Results:")
    print(result_df.head())





# -----------------------------------------------------------------------------
2025.03.27 seem to be correct version
import sys
sys.path.append('C:\\Program Files\\IBM\\ILOG\\CPLEX_Studio210\\cplex\\python\\20.1\\x64_win64')
sys.path.append("C:\\ProgramData\\anaconda3\\Lib\\site-packages")
import numpy as np
import pandas as pd
from docplex.mp.model import Model
import matplotlib.pyplot as plt
import random

# define the parameters
PARAMS = {
    'P': 2,        # number of caissons
    'T': 14,      # horizon
    'Cmax': 3,      # maximum number of tasks of one cluster
    'window_size': 14,   # window size
    'advanced': 3          # advanced time
}   

# initialize the random values
random.seed(42)
alpha_values = [round(random.uniform(0.01, 0.05), 3) for _ in range(PARAMS['P'])]
initial_p_values = [round(random.uniform(3.2, 3.5), 2) for i in range(PARAMS['P'])]


def clustering_CR(df_window, window_size, window_start):
    """Grouping model"""
    N = df_window.shape[0]

    mdl = Model('Dynamic_Window_Clustering')
    
    # define the variables
    theta = mdl.binary_var_matrix(N, window_size, name='theta')
    center = mdl.binary_var_matrix(window_size, window_size, name='center')  
    cluster = mdl.binary_var_list(window_size, name='cluster')
    a = mdl.integer_var_list(N, name='a', lb=0, ub=window_size)
    b = mdl.integer_var_list(N, name='b', lb=0, ub=window_size)

    M1 = 1
    M = 10

    # objective function
    mdl.minimize(mdl.sum(cluster[k] for k in range(window_size)))
    
   
    # cluster constraint
    mdl.add_constraints(theta[n, k] <= cluster[k] for n in range(N) for k in range(window_size))
    mdl.add_constraints(mdl.sum(theta[n, k] for k in range(window_size)) == 1 for n in range(N))
    
    # maximum capacity constraint
    mdl.add_constraints(mdl.sum(theta[n, k] for n in range(N)) <= PARAMS['Cmax'] for k in range(window_size))
    
    # only one center constraint
    mdl.add_constraints(mdl.sum(center[k, t] for t in range(window_size)) == cluster[k] for k in range(window_size))

    # task grouping and center determination
    
    # print(theta.keys())  # make sure (n, k) is in keys
    # print(center.keys())  # make sure (k, t) is in keys

    for n in range(N):
        local_time = df_window.iloc[n]['window_time']
    #     # print(local_time)
        mdl.add_constraint(a[n] >= local_time - PARAMS['advanced'])
        mdl.add_constraint(a[n] >= 0)
        mdl.add_constraint(b[n] <= local_time + int((3.2 - 3.0)/alpha_values[int(df_window.iloc[n]["CSEM"])]))
        mdl.add_constraint(b[n] <= window_size)

        for t in range(window_size):
            mdl.add_constraints(t * center[k,t] >= a[n] - M * (1 - theta[n, k]) - M * (1 - center[k, t]) for k in range(window_size))
            mdl.add_constraints(t * center[k,t] <= b[n] + M * (1 - theta[n, k]) for k in range(window_size))
    
    # solve the model
    solution = mdl.solve()
    if not solution:
        print(f"Failed to solve the problem: {mdl.solve_details.status}")
        return None
    
    # result processing

    centers = {}
    for n in range(N):
        for k in range(window_size):
            for t in range(window_size):
                if solution.get_value(center[k, t]) >= 0.9:  # find the center for each cluster
                    print(f"Task {n} (CSEM={df_window.iloc[n]['CSEM']}) assigned to cluster {t + window_start}")
                    centers[n] = t + window_start  # change the time back to the original time
                    break
    second_value = list(centers.values())[1]  # 取第二个 value
    print("second value", second_value)
    # update the center column of the dataframe
    for n in range(N):
        for k in range(window_size):
            if solution.get_value(theta[n, k]) >= 0.9:
                df_window.at[n, 'CENTER'] = centers.get(n, 0)  

    return df_window


def main(window_size):
    """Main function"""
    df = pd.read_csv('maintenance_plan.csv')
    df['CENTER'] = 0  # initialize the center column
    window_start = 0
    while window_start <= PARAMS['T'] - window_size:
        # dynamic window selection
        window_end = window_start + window_size
        mask = (df['DATE'] >= window_start) & (df['DATE'] < window_end)
        window_df = df.loc[mask].copy()
        # print(window_df)
        # task local time conversion
        window_df['window_time'] = window_df['DATE'] - window_start
        # print(window_df)
        first_tasks = window_df.groupby('CSEM', as_index=False).first()
        # print(first_tasks)
        # print(alpha_values)
    
        # clustering
        clustered_df = clustering_CR(first_tasks, window_size, window_start)
        # print(clustered_df)
        
        if clustered_df is not None:
            # update the center column of original dataframe
            df['CENTER'] = df['CSEM'].map(clustered_df.set_index('CSEM')['CENTER'])
            # update the secondary tasks time
            update_secondary_tasks(df, clustered_df, window_start, window_end)
        
        # slide the window
        window_start += int(window_size/2)
    
    # result verification
    assert df['CENTER'].notna().all(), "Some tasks are not clustered"
    return df

def update_secondary_tasks(main_df, clustered_df, window_start, window_end):
    """Update the secondary tasks time"""
    for caisson in clustered_df["CSEM"].unique():
        # get the tasks in the same caisson
        tasks = main_df[
            (main_df['CSEM'] == caisson) & 
            (main_df['DATE'].between(window_start, window_end))
        ]
        
        if len(tasks) > 1:
            # get the second task index
            second_idx = tasks.index[1]
            main_df.loc[second_idx, 'CENTER'] = 0
            # get the previous center time amd calculate the new time for the second task
            center_time = clustered_df.loc[clustered_df['CSEM'] == caisson, 'CENTER'].iloc[0]
            
            # calulate the new time
            alpha = alpha_values[caisson]
            # print(center_time)
            # print(alpha)
            new_date = int(center_time + 1 + (3.5-3.2)/alpha)
            # print(new_date)
            
            # update the time
            main_df.loc[second_idx, 'DATE'] = new_date

if __name__ == "__main__":
    result_df = main(PARAMS['window_size'])
    
    # # plot the clustering center
    # plt.figure(figsize=(12,6))
    # plt.scatter(result_df['DATE'], result_df['CENTER'], alpha=0.7)
    # plt.title("cluster center distribution")
    # plt.xlabel("original time")
    # plt.ylabel("cluster center")
    # plt.grid(True)
    # plt.show()
    
    print("Results:")
    print(result_df.head())



#-----------------------------
new correct version
import sys
sys.path.append('C:\\Program Files\\IBM\\ILOG\\CPLEX_Studio210\\cplex\\python\\20.1\\x64_win64')
sys.path.append("C:\\ProgramData\\anaconda3\\Lib\\site-packages")
import numpy as np
import pandas as pd
from docplex.mp.model import Model
import matplotlib.pyplot as plt
import random

# define the parameters
PARAMS = {
    'P': 2,        # number of caissons
    'T': 14,      # horizon
    'Cmax': 3,      # maximum number of tasks of one cluster
    'window_size': 14,   # window size
    'advanced': 3          # advanced time
}   

# initialize the random values
random.seed(42)
alpha_values = [round(random.uniform(0.01, 0.05), 3) for _ in range(PARAMS['P'])]
initial_p_values = [round(random.uniform(3.2, 3.5), 2) for i in range(PARAMS['P'])]


def clustering_CR(df_window, window_size, window_start):
    """Grouping model"""
    # add preprocessing here
    if df_window.empty or window_size < 1:
        print(f"warning: window {window_start} has no tasks or window size is invalid")
        return df_window  # return the original dataframe if no tasks in the window


    N = df_window.shape[0]

    mdl = Model('Dynamic_Window_Clustering')
    
    # define the variables
    theta = mdl.binary_var_matrix(N, window_size, name='theta')
    center = mdl.binary_var_matrix(window_size, window_size, name='center')  
    cluster = mdl.binary_var_list(window_size, name='cluster')
    a = mdl.integer_var_list(N, name='a', lb=0, ub=window_size)
    b = mdl.integer_var_list(N, name='b', lb=0, ub=window_size)

    M1 = 1
    M = 10

    # objective function
    mdl.minimize(mdl.sum(cluster[k] for k in range(window_size)))
    
   
    # cluster constraint
    mdl.add_constraints(theta[n, k] <= cluster[k] for n in range(N) for k in range(window_size))
    mdl.add_constraints(mdl.sum(theta[n, k] for k in range(window_size)) == 1 for n in range(N))
    
    # maximum capacity constraint
    mdl.add_constraints(mdl.sum(theta[n, k] for n in range(N)) <= PARAMS['Cmax'] for k in range(window_size))
    
    # only one center constraint
    mdl.add_constraints(mdl.sum(center[k, t] for t in range(window_size)) == cluster[k] for k in range(window_size))

    # task grouping and center determination
    
    # print(theta.keys())  # make sure (n, k) is in keys
    # print(center.keys())  # make sure (k, t) is in keys

    for n in range(N):
        local_time = df_window.iloc[n]['window_time']
    #     # print(local_time)
        mdl.add_constraint(a[n] >= local_time - PARAMS['advanced'])
        mdl.add_constraint(a[n] >= 0)
        mdl.add_constraint(b[n] <= local_time + int((3.2 - 3.0)/alpha_values[int(df_window.iloc[n]["CSEM"])]))
        mdl.add_constraint(b[n] <= window_size)

        for t in range(window_size):
            mdl.add_constraints(t * center[k,t] >= a[n] - M * (1 - theta[n, k]) - M * (1 - center[k, t]) for k in range(window_size))
            mdl.add_constraints(t * center[k,t] <= b[n] + M * (1 - theta[n, k]) for k in range(window_size))
    
    # solve the model
    solution = mdl.solve()
    if not solution:
        print(f"Failed to solve the problem: {mdl.solve_details.status}")
        return None
    
    # result processing

    centers = {}
    for n in range(N):
        for k in range(window_size):
            for t in range(window_size):
                if solution.get_value(center[k, t]) >= 0.9:  # find the center for each cluster
                    # print(f"Task {n} (CSEM={df_window.iloc[n]['CSEM']}) assigned to cluster {t + window_start}")
                    centers[n] = t + window_start  # change the time back to the original time
                    break

    # update the center column of the dataframe
    for n in range(N):
        for k in range(window_size):
            if solution.get_value(theta[n, k]) >= 0.9:
                df_window.at[n, 'CENTER'] = centers.get(n, 0)  

    return df_window


def main(window_size):
    """Main function"""
    df = pd.read_csv('maintenance_plan.csv')
    df['CENTER'] = 0  # initialize the center column
    # df['CENTER'] = df['DATE'].copy()  # Initialize the center column with the original time
    window_start = 0
    while window_start <= PARAMS['T'] - window_size:
        # dynamic window selection
        window_end = window_start + window_size
        mask = (df['DATE'] >= window_start) & (df['DATE'] < window_end)
        window_df = df.loc[mask].copy()

        # task local time conversion
        window_df['window_time'] = window_df['DATE'] - window_start
        print("window", window_df)
        first_tasks = window_df.groupby('CSEM', as_index=False).first()
        print("first tasks", first_tasks)
        # print(alpha_values)
    
        # clustering
        clustered_df = clustering_CR(first_tasks, window_size, window_start)
        # print(clustered_df)
        
        if clustered_df is not None:
            # update the center column of original dataframe
            df.loc[mask, 'CENTER'] = df.loc[mask, 'CSEM'].map(clustered_df.set_index('CSEM')['CENTER']).fillna(df.loc[mask, 'CENTER'])
            
            # update the secondary tasks time
            update_secondary_tasks(df, clustered_df, window_start, window_end)
            
        # slide the window
        window_start += int(window_size/2)


    # result verification
    # assert df['CENTER'].notna().all(), "Some tasks are not clustered"
    return df

def update_secondary_tasks(main_df, clustered_df, window_start, window_end):
    """Update the secondary tasks time"""
    for caisson in clustered_df["CSEM"].unique():
        # get the tasks in the same caisson
        tasks = main_df[
            (main_df['CSEM'] == caisson) & 
            (main_df['DATE'].between(window_start, window_end))
        ]
        
        if len(tasks) > 1:
            # get the second task index
            second_idx = tasks.index[1]
            main_df.loc[second_idx, 'CENTER'] = 0
            # get the previous center time amd calculate the new time for the second task
            center_time = clustered_df.loc[clustered_df['CSEM'] == caisson, 'CENTER'].iloc[0]
            
            # calulate the new time
            alpha = alpha_values[caisson]
            # print(center_time)
            # print(alpha)
            new_date = int(center_time + 1 + (3.5-3.2)/alpha)
            # print(new_date)
            
            # update the time
            # main_df.loc[second_idx, 'DATE'] = new_date
            # main_df.loc[second_idx, 'CENTER'] = new_date
            main_df.loc[second_idx, ['DATE', 'CENTER']] = new_date, new_date


if __name__ == "__main__":
    result_df = main(PARAMS['window_size'])
    
    # # plot the clustering center
    # plt.figure(figsize=(12,6))
    # plt.scatter(result_df['DATE'], result_df['CENTER'], alpha=0.7)
    # plt.title("cluster center distribution")
    # plt.xlabel("original time")
    # plt.ylabel("cluster center")
    # plt.grid(True)
    # plt.show()
    
    print("Results:")
    print(result_df)


# ------------------------------------------------
2025/03/29 correct version without calculation of leakage
import sys
sys.path.append('C:\\Program Files\\IBM\\ILOG\\CPLEX_Studio210\\cplex\\python\\20.1\\x64_win64')
sys.path.append("C:\\ProgramData\\anaconda3\\Lib\\site-packages")
import numpy as np
import pandas as pd
from docplex.mp.model import Model
import matplotlib.pyplot as plt
import random

# define the parameters
PARAMS = {
    'P': 10,        # number of caissons
    'T': 60,      # horizon
    'Cmax': 3,      # maximum number of tasks of one cluster
    'window_size': 14,   # window size
    'advanced': 3          # advanced time
}   

# initialize the random values
random.seed(42)
alpha_values = [round(random.uniform(0.01, 0.05), 3) for _ in range(PARAMS['P'])]
initial_p_values = [round(random.uniform(3.2, 3.5), 2) for i in range(PARAMS['P'])]


def clustering_CR(df_window, window_size, window_start):
    """Grouping model"""
    # add preprocessing here
    if df_window.empty or window_size < 1:
        print(f"warning: window {window_start} has no tasks or window size is invalid")
        return df_window  # return the original dataframe if no tasks in the window

    N = df_window.shape[0]

    mdl = Model('Dynamic_Window_Clustering')
    
    # define the variables
    theta = mdl.binary_var_matrix(N, window_size, name='theta')
    center = mdl.binary_var_matrix(window_size, window_size, name='center')  
    cluster = mdl.binary_var_list(window_size, name='cluster')
    a = mdl.integer_var_list(N, name='a', lb=0, ub=window_size)
    b = mdl.integer_var_list(N, name='b', lb=0, ub=window_size)

    M1 = 1
    M = 10

    # objective function
    mdl.minimize(mdl.sum(cluster[k] for k in range(window_size)))
    
   
    # cluster constraint
    mdl.add_constraints(theta[n, k] <= cluster[k] for n in range(N) for k in range(window_size))
    mdl.add_constraints(mdl.sum(theta[n, k] for k in range(window_size)) == 1 for n in range(N))
    
    # maximum capacity constraint
    mdl.add_constraints(mdl.sum(theta[n, k] for n in range(N)) <= PARAMS['Cmax'] for k in range(window_size))
    
    # only one center constraint
    mdl.add_constraints(mdl.sum(center[k, t] for t in range(window_size)) == cluster[k] for k in range(window_size))

    # task grouping and center determination
    
    # print(theta.keys())  # make sure (n, k) is in keys
    # print(center.keys())  # make sure (k, t) is in keys

    for n in range(N):
        local_time = df_window.iloc[n]['window_time']
    #     # print(local_time)
        mdl.add_constraint(a[n] >= local_time - PARAMS['advanced'])
        mdl.add_constraint(a[n] >= 0)
        mdl.add_constraint(b[n] <= local_time + int((3.2 - 3.0)/alpha_values[int(df_window.iloc[n]["CSEM"])]))
        mdl.add_constraint(b[n] <= window_size)

        for t in range(window_size):
            mdl.add_constraints(t * center[k,t] >= a[n] - M * (1 - theta[n, k]) - M * (1 - center[k, t]) for k in range(window_size))
            mdl.add_constraints(t * center[k,t] <= b[n] + M * (1 - theta[n, k]) for k in range(window_size))
    
    # solve the model
    solution = mdl.solve()
    if not solution:
        print(f"Failed to solve the problem: {mdl.solve_details.status}")
        return None
    
    # result processing

    centers = {}
    for n in range(N):
        for k in range(window_size):
            for t in range(window_size):
                if solution.get_value(center[k, t]) >= 0.9:  # find the center for each cluster
                    # print(f"Task {n} (CSEM={df_window.iloc[n]['CSEM']}) assigned to cluster {t + window_start}")
                    centers[n] = t + window_start  # change the time back to the original time
                    break

    # update the center column of the dataframe
    for n in range(N):
        for k in range(window_size):
            if solution.get_value(theta[n, k]) >= 0.9:
                df_window.at[n, 'CENTER'] = centers.get(n, 0)  

    return df_window


def main(window_size):
    """Main function"""
    df = pd.read_csv('maintenance_plan.csv')
    df['CENTER'] = 0  # initialize the center column
    window_start = 0
    while window_start <= PARAMS['T'] - window_size:
        # dynamic window selection
        window_end = window_start + window_size
        mask = (df['DATE'] >= window_start) & (df['DATE'] < window_end) & (df['OPERATION']== 'x')
        window_df = df.loc[mask].copy()

        # task local time conversion
        window_df['window_time'] = window_df['DATE'] - window_start
        print("window", window_df)
        first_tasks = window_df.groupby('CSEM', as_index=False).first()
        print("first tasks", first_tasks)
        # print(alpha_values)
    
        # clustering
        clustered_df = clustering_CR(first_tasks, window_size, window_start)
        # print(clustered_df)
        
        if clustered_df is not None:
            # update the center column of original dataframe
            df.loc[mask, 'CENTER'] = df.loc[mask, 'CSEM'].map(clustered_df.set_index('CSEM')['CENTER']).fillna(df.loc[mask, 'CENTER'])
            
            # update the secondary tasks time
            update_secondary_tasks(df, clustered_df, window_start, window_end)
            
        # slide the window
        window_start += int(window_size/2)

    # Updates the center column of the RE operation, which will not be grouped
    df.loc[df['OPERATION'] == "y", 'CENTER'] = df.loc[df['OPERATION'] == "y", 'DATE']

    # Processing of the remaining tasks that are not assigned to a center and cannot be updated
    mask_remain = (df['OPERATION'] == "x") & (df['CENTER'] == 0)
    df.loc[mask_remain, 'CENTER'] = df.loc[mask_remain, 'DATE']

    # result verification
    # assert df['CENTER'].notna().all(), "Some tasks are not clustered"
    return df

def update_secondary_tasks(main_df, clustered_df, window_start, window_end):
    """Update the secondary tasks time"""
    for caisson in clustered_df["CSEM"].unique():
        # get the tasks in the same caisson
        tasks = main_df[
            (main_df['CSEM'] == caisson) & 
            (main_df['DATE'].between(window_start, window_end))
        ]
        
        if len(tasks) > 1:
            # get the second task index
            second_idx = tasks.index[1]
            main_df.loc[second_idx, 'CENTER'] = 0
            # get the previous center time amd calculate the new time for the second task
            center_time = clustered_df.loc[clustered_df['CSEM'] == caisson, 'CENTER'].iloc[0]
            
            # calulate the new time
            alpha = alpha_values[caisson]
            # print(center_time)
            # print(alpha)
            new_date = int(center_time + 1 + (3.5-3.2)/alpha)
            # print(new_date)
            
            # update the time
            main_df.loc[second_idx, ['DATE', 'CENTER']] = new_date, new_date


if __name__ == "__main__":
    result_df = main(PARAMS['window_size'])

    # 计算组合前成本
    operation_x_count = (result_df['OPERATION'] == 'x').sum()
    operation_y_count = (result_df['OPERATION'] == 'y').sum()
    before_cost = operation_x_count * 10 + operation_y_count * 100

    print(f"组合前成本: {before_cost}")

    # 计算组合后成本
    operation_x_center_unique_count = result_df[result_df['OPERATION'] == 'x']['CENTER'].nunique()
    operation_y_count = (result_df['OPERATION'] == 'y').sum()
    after_cost = operation_x_center_unique_count * 10 + operation_y_count * 100

    print(f"组合后成本: {after_cost}")


    # Plot1: cluster center vs original time
    plt.figure(figsize=(6, 6))
    # set colors：if OPERATION == "y"，red else blue
    colors = ['red' if op == 'y' else 'blue' for op in result_df['OPERATION']]
    plt.scatter(result_df['DATE'], result_df['CENTER'], alpha=0.7, c=colors)

    # title and labels
    plt.title("cluster center distribution")
    plt.xlabel("original time")
    plt.ylabel("cluster center")
    plt.grid(True)

    # set x and y axis limits
    max_range = max(result_df['DATE'].max(), result_df['CENTER'].max()+5)
    plt.xlim(0, max_range)
    plt.ylim(0, max_range)

    # add diagonal line
    plt.plot([0, max_range], [0, max_range], color='gray', linestyle='--', label='45° diagonal')
    # set aspect ratio to be equal
    plt.gca().set_aspect('equal')
    plt.show()


    # plot2:
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, csem in enumerate(sorted(result_df['CSEM'].unique())):
        # horizontal line for each CSEM
        ax.hlines(y=i, xmin=0, xmax=PARAMS['T'], color='gray', linewidth=1)
        
        # filter data for the current CSEM
        current_data = result_df[result_df['CSEM'] == csem]
        
        # scatter points for each task
        colors = ['blue' if op == 'x' else 'red' for op in current_data['OPERATION']]
        ax.scatter(current_data['DATE'], [i] * len(current_data), color=colors, s=50)
        
        # highlight the center point
        ax.scatter(current_data['CENTER'], [i] * len(current_data), color='orange', s=100)

    ax.set_yticks(range(len(result_df['CSEM'].unique())))
    ax.set_yticklabels([f"Caisson {csem}" for csem in sorted(result_df['CSEM'].unique())])
    ax.set_xlim(0, PARAMS['T'])
    ax.set_xlabel('Time')
    ax.set_title('Timeline Visualization')
    plt.show()

    
    print("Results:")
    print(result_df)
