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

@st.cache_data
# def vis_all_caisson(df_x_y, df_p, taux1):
#     # Filter the data based on the selected caissons
#     caissons = df_x_y['CAISSON'].unique()
#     df_p['value'] = df_p['value'].astype(float)
    
#     # Create a line chart of the pressure data
#     chart = px.line(df_p, x='TEMPS', y='value', color='CAISSON', width= 700, height=400)
    
#     # Add the reference lines
#     chart.add_hline(y=taux1, line_dash='dash', line_color='green', name='Taux 1')
#     chart.add_hline(y=3.5, line_dash='dash', line_color='blue', name='3.5')
    
#     # Set the x-axis tick angle and legend title
#     chart.update_layout(xaxis_tickangle=-45, legend_title='Caissons')
#     chart.update_layout(plot_bgcolor='#f0fdf4')

#     # Return the chart object
#     return chart

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

    # Add constraints and objective function to the model
    # CONTRAINTES D'INITIALISATION DE PRESSION(OU DE PRESSIONS INITIAUX)
    mdl.add_constraints((p[i, 0] == (Pinit[i].item()) for i in range(P)), names = 'SPinit')
    mdl.add_constraints((p[i, t] >= 3.001 for i in range(P) for t in range(T)), names = 'SPmin')
    mdl.add_constraints((p[i, t] <= 3.5 for i in range(P) for t in range(T)), names = 'SPmax')

    #-----------------------------------------------------------------------------------------------------------------------------
    # CONTRAINTE DE RESOURCES 
    mdl.add_constraints((sum(r_cr*f[k, tp] for k in range(P) for tp in range(T) if (tp <= t) and (tp + b1 - 1>= t)) <= Rcr for t in range(T)))
    mdl.add_constraints((sum(r_re*g[k, tp] for k in range(P) for tp in range(T) if (tp <= t) and (tp + b2 - 1>= t)) <= Rre for t in range(T)))

    #-----------------------------------------------------------------------------------------------------------------------------
    if deg == 'Linear degradation':
        mdl.add_constraints((p[i, t+1] >= (1- alpha[i])*p[i, t] - M*(f[i, t] + (g[i, t]+e[i,t])) for i in range(P) for t in range(T-1)))
        mdl.add_constraints((p[i, t+1] <= (1- alpha[i])*p[i, t] + M*(f[i, t] + (g[i, t]+e[i,t])) for i in range(P) for t in range(T-1)))    
    else:
        mdl.add_constraints((p[i, t+1] >= (1- alpha_param[i][0]**(t/(t + 34) - 1))*p[i, t] - M*(f[i, t] + g[i, t]) for i in range(P) for t in range(1, T-1)))
        mdl.add_constraints((p[i, t+1] <= (1- alpha_param[i][0]**(t/(t + 34) - 1))*p[i, t] + M*(f[i, t] + g[i, t]) for i in range(P) for t in range(1, T-1)))
        
    #-----------------------------------------------------------------------------------------------------------------------------
    mdl.add_constraints((p[i, t+b1] >= P_rempli - M*(1 - x[i, t]) for i in range(P) for t in range(T-b1)), names = 'AFTER CR')
    mdl.add_constraints((p[i, t+b1] <= P_rempli + M*(1 - x[i, t]) for i in range(P) for t in range(T-b1)), names = 'AFTER CR')
 
    mdl.add_constraints((p[i, tp] >= p[i, t] - M*(1 - y[i, t]) for i in range(P) for t in range(1, T-b2) for tp in range(t+b2, T-b2)), names = 'AFTER RE')
    mdl.add_constraints((p[i, tp] <= p[i, t] + M*(1 - y[i, t]) for i in range(P) for t in range(1, T-b2) for tp in range(t+b2, T-b2)), names = 'AFTER RE')

    # mdl.add_constraints(x[i, t] + y[i, t] <= 1 for i in range(P) for t in range(T))

    # Additional constraints to define d[i, t]
    # mdl.add_constraints((p[i, t] - 3.2 <= (1 - d[i, t]) for i in range(P) for t in range(T)), names='d_upper_bound')
    # mdl.add_constraints((3.2 - p[i, t] <= d[i, t] for i in range(P) for t in range(T)), names='d_lower_bound')

    # # we can only carry out maintenance after pressure is lower than 3.2
    # for i in range(P):
    #     for t in range(T):
    #         mdl.add_constraint(x[i, t] <= 1 + (1 - d[i, t]))
    #         mdl.add_constraint(x[i, t] <= d[i, t])
            # mdl.add_constraint(y[i, t] <= 1 + (1 - d[i, t]))
            # mdl.add_constraint(y[i, t] <= d[i, t])

    #-----------------------------------------------------------------------------------------------------------------------------
    # definition for cumulating the sum of RE action
    for t in range(T):
        for i in range(P):
            mdl.add_constraint(e[i,t] == mdl.sum(y[i, k] for k in range(t)))


    #-----------------------------------------------------------------------------------------------------------------------------
    # w[i][t] = sum(x[i, tp] for tp in range(t - W + 1, t + 1))
    mdl.add_constraints((w[i, t] == mdl.sum(x[i, tp] for tp in range(max(0, t - W + 1), t + 1)) for i in range(P) for t in range(1,T)), names='sum_constraint')
    # w[i][t] <= 2 + z[i][t]
    mdl.add_constraints((w[i, t] <= 2 + z[i, t] for i in range(P) for t in range(1,T)), names='upper_bound_constraint')
    # w[i][t] >= 3 - 3 * (1 - z[i][t])
    mdl.add_constraints((w[i, t] >= 3 - 3 * (1 - z[i, t]) for i in range(P) for t in range(1,T)), names='lower_bound_constraint')

    #-----------------------------------------------------------------------------------------------------------------------------
    # f[i][t+k] >= x[i][t]  ∀i ∈ P, ∀t ∈ T-b, ∀k = 0, 1, ..., b
    mdl.add_constraints((f[i, t + k] >= x[i, t] for i in range(P) for t in range(T - b1) for k in range(b1)), names='f_constraint')
    
    # g[i][t+k] >= y[i][t]  ∀i ∈ P, ∀t ∈ T-b, ∀k = 0, 1, ..., b
    mdl.add_constraints((g[i, t + k] >= y[i, t] for i in range(P) for t in range(T - b2) for k in range(b2)), names='g_constraint')
        
    #-----------------------------------------------------------------------------------------------------------------------------
    mdl.add_constraints((f[i, t]*b1 >= sum(x[i, tp] for tp in range(t - b1, t+1)) for i in range(P) for t in range(T) if t >= b1))
    mdl.add_constraints((f[i, t]   <= sum(x[i, tp] for tp in range(t - b1, t+1)) for i in range(P) for t in range(T) if t >= b1))

    mdl.add_constraints((f[i, t]*b1 >= sum(x[i, tp] for tp in range(t+1)) for i in range(P) for t in range(b1)))
    mdl.add_constraints((f[i, t]   <= sum(x[i, tp] for tp in range(t+1)) for i in range(P) for t in range(b1)))

    # mdl.add_constraints(x[i, t] + x[i, tp] <= 1 for i in range(P) for t in range(T) for tp in range(T) if (tp < t) and (tp + b1 >= t)) 
    
    mdl.add_constraints((g[i, t]*b2 >= sum( y[i, tp] for tp in range(t - b2, t+1)) for i in range(P) for t in range(T) if t >= b2))
    mdl.add_constraints((g[i, t]   <= sum( y[i, tp] for tp in range(t - b2, t+1)) for i in range(P) for t in range(T) if t >= b2))

    mdl.add_constraints((g[i, t]*b2 >= sum(y[i, tp] for tp in range(t+1)) for i in range(P) for t in range(b2)))
    mdl.add_constraints((g[i, t]   <= sum(y[i, tp] for tp in range(t+1)) for i in range(P) for t in range(b2)))

    # mdl.add_constraints(y[i, t] + y[i, tp] <= 1 for i in range(P) for t in range(T) for tp in range(T) if (tp < t) and (tp + b2 >= t))

    #-----------------------------------------------------------------------------------------------------------------------------
    # # mdl.add_constraints(f[i, t] >= x[i, t] for t in range(T))
    # # mdl.add_constraints(g[i, t] >= y[i, t] for t in range(T))
    # mdl.add_constraints(f[i, t] >= f[i, t-1] - f[i, t-b1] for t in range(b1, T))
    # mdl.add_constraints(g[i, t] >= g[i, t-1] - g[i, t-b2] for t in range(b2, T))

    #-----------------------------------------------------------------------------------------------------------------------------
    # mdl.add_constraint(sum((((3.5 - p[i,tp])*(10**5))*V/(8.314 * 293)*146.0)/1000 for i in range(P) for tp in range(t)) <= 100.0)
    # total cost of maintenance
    cost = sum(couts1*x[i, t] + couts2*y[i, t] + k1*z[i,t] for i in range(P) for t in range(T))
    mdl.add_constraint(cost <= 1000.0)
    
    #-----------------------------------------------------------------------------------------------------------------------------
    # OBJ FUNCTION
    total_leakage = mdl.sum(K*(p[i, t-1] - p[i, t]) for i in range(P) for t in range(1,T))
    mdl.add_constraint(total_leakage >= 1000.0)
    mdl.minimize(total_leakage)

    # Solve the model
    si = mdl.solve()
    df = mdl.solution.as_df()

    st.dataframe(df)

    # Calculate the total cost from the solution
    total_cost_value = round(sum(couts1 * si[x[i, t]] + couts2 * si[y[i, t]] + k1 * si[z[i, t]] for i in range(P) for t in range(T)), 3)

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
    return df_p, df_x_y, sol, total_cost_value



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
        alpha_high = np.random.uniform(0.004, 0.009, P)

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
        b1 = 2
        b2 = 4
        k1 = 2
        k2 = 4
        k3 = 20
        W = 15
        

        # constant for perfect equation coefficient = V/RT * 10^5
        K = 2052.54

    

     
        if 'df_p' not in st.session_state:
            st.session_state.df_p = None

        if st.button('Run Optimization'):

            with st.spinner("Running function..."): 
                df_p, df_x_y, sol, total_cost_value = create_model(P, T, Pinit, Rcr, Rre, r_cr, r_re, P_rempli, alpha, alpha_param, deg, taux1, taux2, gamma, M, couts1, couts2, b1, b2, k1, k2, k3, W, K)
                st.session_state.df_p = df_p
                st.session_state.df_x_y = df_x_y
                st.session_state.sol = np.around(sol, 2)
                st.session_state.total_cost_value = total_cost_value
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

            st.write('#### Total Cost (€):', st.session_state.total_cost_value)
            st.write('#### Total leakage quantity (kg):', st.session_state.sol)
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
    