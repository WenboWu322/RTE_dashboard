import sys
from mip import Model, xsum, BINARY, INTEGER, CONTINUOUS
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
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





def data_clustring(x):
    for i in range(x.shape[0]):
        r = x["name"].loc[i]
        e = r.split('_')[0]
        g = r.split('_')[1]   # types of operations
        h = r.split('_')[2]   # cluster information
        x.loc[i, 'cluster'] = h
        x.loc[i, 'OPERATION'] = g
        

def create_model(P, T, Pinit, Rcr, Rre, r_cr, r_re, P_rempli, alpha, alpha_param, deg, taux1, taux2, gamma, M, couts1, couts2, b1, b2, k1, W, K):
    # create the model
    m = Model('Maintenance_caisson', solver_name='CBC')

    # add variables
    x = [[m.add_var(name=f'x_{i}_{t}', var_type=BINARY) for t in range(T)] for i in range(P)]
    y = [[m.add_var(name=f'y_{i}_{t}', var_type=BINARY) for t in range(T)] for i in range(P)]
    f = [[m.add_var(name=f'f_{i}_{t}', var_type=BINARY) for t in range(T)] for i in range(P)]
    g = [[m.add_var(name=f'g_{i}_{t}', var_type=BINARY) for t in range(T)] for i in range(P)]
    z = [[m.add_var(name=f'z_{i}_{t}', var_type=BINARY) for t in range(T)] for i in range(P)]
    w = [[m.add_var(name=f'w_{i}_{t}', var_type=BINARY) for t in range(T)] for i in range(P)]
    e = [[m.add_var(name=f'e_{i}_{t}', var_type=INTEGER) for t in range(T)] for i in range(P)]
    p = [[m.add_var(name=f'p_{i}_{t}', var_type=CONTINUOUS) for t in range(T)] for i in range(P)]
    d = [[m.add_var(name=f'd_{i}_{t}', var_type=BINARY) for t in range(T)] for i in range(P)]
    o = [[m.add_var(name=f'o_{i}_{t}', var_type=BINARY) for t in range(T)] for i in range(P)]
    v = [[m.add_var(name=f'v_{i}_{t}', var_type=CONTINUOUS) for t in range(T)] for i in range(P)]
    slope = [[m.add_var(name=f'slope_{i}_{t}', var_type=CONTINUOUS) for t in range(T)] for i in range(P)]

    # pressure evolution
    for i in range(P):
        m.add_constr(p[i][0] == Pinit[i].item())
        for t in range(T):
            m.add_constr((p[i][t] >= 3.001))
            m.add_constr((p[i][t] <= 3.5))


    # CONTRAINTE DE RESOURCES 
    for t in range(T):
        m.add_constr(xsum(r_cr*x[k][tp] for k in range(P) for tp in range(T) if (tp <= t) and (tp + b1 - 1 >= t)) <= Rcr)
        m.add_constr(xsum(r_re*y[k][tp] for k in range(P) for tp in range(T) if (tp <= t) and (tp + b2 - 1 >= t)) <= Rre)

    if deg == 'Linear degradation':
        for i in range(P):
            for t in range(T-1):
                m.add_constr(p[i][t+1] >= (1-alpha)*p[i][t] - M*(f[i][t] + g[i][t] + e[i][t]))
                m.add_constr(p[i][t+1] <= (1-alpha)*p[i][t] + M*(f[i][t] + g[i][t] + e[i][t]))
    else:
        # not now useful but for other models
        for i in range(P):
            for t in range(1, T-1):
                m.add_constr(p[i][t+1] >= (1-alpha_param[i][0]**(t/(t + 34) - 1))*p[i][t] - M*(f[i][t] + g[i][t]))
                m.add_constr(p[i][t+1] <= (1-alpha_param[i][0]**(t/(t + 34) - 1))*p[i][t] + M*(f[i][t] + g[i][t]))

    # after a CR, pressure come back to 3.5 at the moment t+b1
    for i in range(P):
        for t in range(T-b1):
            m.add_constr(p[i][t+b1] >= P_rempli - M*(1 - x[i][t]))
            m.add_constr(p[i][t+b1] <= P_rempli + M*(1 - x[i][t]))

    # Add linear interpolation constraints for CR during maintenance - for visualization
    for i in range(P):
        for t in range(T - b1):  
            for k in range(1, b1):
                pressure_value = p[i][t].x if p[i][t].x is not None else 0.0  
                expected_pressure = pressure_value + (k / b1) * (P_rempli - pressure_value)
                m.add_constr(p[i][t + k] >= expected_pressure - M * (1 - x[i][t]))
                m.add_constr(p[i][t + k] <= expected_pressure + M * (1 - x[i][t]))

        for t in range(T - b1, T):
            for k in range(1, T - t):
                pressure_value = p[i][t].x if p[i][t].x is not None else 0.0
                expected_pressure = pressure_value + (k / (T-t)) * (P_rempli - pressure_value)
                m.add_constr(p[i][t + k] >= expected_pressure - M * (1 - x[i][t]))
                m.add_constr(p[i][t + k] <= expected_pressure + M * (1 - x[i][t]))

    # Add linear interpolation constraints for RE during maintenance
    for i in range(P):
        for t in range(T - b2):  
            for k in range(1, b2):
                pressure_value = p[i][t].x if p[i][t].x is not None else 0.0
                expected_pressure = pressure_value + (k / b2) * (P_rempli - pressure_value)
                m.add_constr(p[i][t + k] >= expected_pressure - M * (1 - y[i][t]))
                m.add_constr(p[i][t + k] <= expected_pressure + M * (1 - y[i][t]))

        for t in range(T - b2, T):
            for k in range(0, T - t):
                pressure_value = p[i][t].x if p[i][t].x is not None else 0.0
                expected_pressure = pressure_value + (k / (T-t)) * (P_rempli - pressure_value)
                m.add_constr(p[i][t + k] >= expected_pressure - M * (1 - y[i][t]))
                m.add_constr(p[i][t + k] <= expected_pressure + M * (1 - y[i][t]))

    # after RE, pressure come back to 3.5 from moment t+b2
    for i in range(P):
        for t in range(T):
            for tp in range(t+b2, T):
                m.add_constr(p[i][tp] >= P_rempli - M*(1 - y[i][t]))
                m.add_constr(p[i][tp] <= P_rempli + M*(1 - y[i][t]))

    # only one RE is need for whole horizon             
    for i in range(P):
        for t in range(T-b2):
            for tp in range(t+b2, T):
                m.add_constr(y[i][tp] <= 1 - y[i][t])

    # definition for cumulating the sum of RE action
    for t in range(T):
        for i in range(P):
            m.add_constr(e[i][t] == xsum(y[i][k] for k in range(t)))

    # definition for cumulating the sum of CR action and definition of penalty term z
    for i in range(P):
        for t in range(T):
            m.add_constr(w[i][t] == xsum(x[i][tp] for tp in range(max(0, t - W + 1), t + 1)))
            m.add_constr(w[i][t] <= 2 + z[i][t])
            m.add_constr(w[i][t] >= 3 - 3 * (1 - z[i][t]))

    # f definition
    for i in range(P):
        for t in range(T - b1):
            for k in range(b1):
                m.add_constr(f[i][t + k] >= x[i][t])
        for t in range(T):
            m.add_constr(f[i][t] <= xsum(x[i][tp] for tp in range(max(0, t - b1 + 1), t + 1)))

    # g definition
    for i in range(P):
        for t in range(T - b2):
            for k in range(b2):
                m.add_constr(g[i][t + k] >= y[i][t])
        for t in range(T):
            m.add_constr(g[i][t] <= xsum(y[i][tp] for tp in range(max(0, t - b2 + 1), t + 1)))

    for i in range(P):
        for t in range(T):
            for tp in range(T):
                if (tp < t) and (tp + b1 >= t):
                    m.add_constr(x[i][t] + x[i][tp] <= 1)

    # define variable v 
    for i in range(P):
        for t in range(1, T):
            m.add_constr(v[i][t] >= p[i][t-1] - p[i][t])
            m.add_constr(v[i][t] >= 0)
            m.add_constr(v[i][t] <= (p[i][t-1] - p[i][t]) + M * (1 - o[i][t]))
            m.add_constr(v[i][t] <= M * o[i][t])
            m.add_constr(M * o[i][t] >= p[i][t-1] - p[i][t])
            m.add_constr(M * (1-o[i][t]) >= - (p[i][t-1] - p[i][t]))

    # total leakage quantity constraint     
    leakageQuantity = xsum(K * v[i][t] for i in range(P) for t in range(1,T))
    m.add_constr(leakageQuantity <= 5.0)

    # OBJ FUNCTION
    cost = xsum(couts1*x[i][t] + couts2*y[i][t] + k1*z[i][t] for i in range(P) for t in range(T))
    m.objective = cost

    # # 设置容忍度
    # m.max_mip_gap = 1e-3  # 相对间隙容忍度，匹配 CPLEX 的默认值
    # m.max_mip_gap_abs = 1e-6  # 绝对间隙容忍度

    # Solve the model
    m.optimize()

    # Get solution details
    status = m.status

    # Prepare data for the data() function
    solution_data = []
    for var in m.vars:
        solution_data.append([var.name, var.x])

    df_solution = pd.DataFrame(solution_data, columns=['name', 'value'])

    # Call the data() function
    df_p, df_x_y = data(df_solution)

    # Calculate total leakage from the solution 
    total_leakage_value = round(leakageQuantity.x, 3)

    # Save results to CSV
    # file_name_p = f"P{P}_T{T}_pressure.csv"
    folder_name = "Results_MIP"
    file_name_xy = f"{folder_name}/P{P}_T{T}.csv"

    # df_p.to_csv(file_name_p, index=False, encoding='utf-8')
    df_x_y.to_csv(file_name_xy, index=False, encoding='utf-8')

    print(f"Results saved to: {file_name_xy}")

    # Return the solution
    return df_p, df_x_y, m.objective_value, total_leakage_value

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
            dff.index = dff.index + 1  # change the index to start from 1
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

    