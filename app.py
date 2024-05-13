import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import torch
from torch import nn


class TabularModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_size, dropout_prob=0.2, weight_decay=1e-5):
        super(TabularModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)  # Batch normalization
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)

        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)  # Batch normalization
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)

        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        self.bn3 = nn.BatchNorm1d(hidden_size3)  # Batch normalization
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_prob)

        self.linear4 = nn.Linear(hidden_size3, hidden_size4)
        self.bn4 = nn.BatchNorm1d(hidden_size4)  # Batch normalization
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout_prob)

        self.output = nn.Linear(hidden_size4, output_size)
        self.dropout_prob = dropout_prob
        self.weight_decay = weight_decay

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)  # Apply batch normalization
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        x = self.bn2(x)  # Apply batch normalization
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.linear3(x)
        x = self.bn3(x)  # Apply batch normalization
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.linear4(x)
        x = self.bn4(x)  # Apply batch normalization
        x = self.relu4(x)
        x = self.dropout4(x)

        output = self.output(x)
        return output

input_size = 8  # Number of input features
hidden_size1 = 128
hidden_size2 = 64
hidden_size3 = 32
hidden_size4 = 16
output_size = 9
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
model = TabularModel(input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_size).to(device).eval()
state_dict = torch.load('model.bin')
model.load_state_dict(state_dict)



scaler  = load('scaler.joblib')

columns = ['Mass_flowrate_Biomass',"Hot_side_Inlet_Temperature", 
          'Mass_flowrate_workingfluid','Maximum_heat_supplied',
          'Heat_Input_Heat_Exchanger','Net_Power_Output', 
          'Cycle_Thermal_Efficiency', 'Exergy_Efficiency']

#label = "Maximum_heat_supplied"
labels = ['Type_Biomass','Strength_of_mixture', 
           'Combustion_efficiency', 'Temperature_at_point1', 
           'Pressure_at_point1' , 'Temperature_at_point2', 'Pump_efficiency',
           'Evaporator_efficiency','Pinch_Point_Temperature'
           ]


def preview():
    pass
def show_data():
    #st.write(data)
    pass


def model_interface():
    st.title("Optimization of Biomass Thermal-Powered Organic Rankine Cycle using the Deep Learning Algorithm")
    st.write("What are our Decision Paramters based on your needs? Let's find out")
    with st.sidebar:
        st.subheader('Select Values')
        #strength_of_mixture = st.number_input(label="Stength of Mixture", min_value=0, max_value=10)
        mass_flowrate_Biomass = st.number_input(label="Mass Flowrate of Biomass", min_value=0.1, max_value=0.9)
        #combustion_efficiency = st.number_input(label="Combustion Efficiency", min_value=0.1, max_value=0.9)
        hot_side_Inlet_Temperature = st.number_input(label="Hot Side Inlet Temperature", min_value=440, max_value=1380)
        #temperature_at_point1 = st.number_input(label="Temperature at Point 1", min_value=0, max_value=500)
        #pressure_at_point1 = st.number_input(label="Pressure at Point 1", min_value=100000, max_value=10000000)
        mass_flowrate_workingfluid = st.number_input(label="Mass Flowrate of Working Fluid", min_value=1, max_value=11)
        #temperature_at_point2 = st.number_input(label="Temperature at Point 2", min_value=0, max_value=500)
        #evaporator_efficiency = st.number_input(label="Evaporator Efficiency", min_value=0.1, max_value=0.9)
        #pump_efficiency = st.number_input(label="Pump Efficiency", min_value=0.1, max_value=0.9)
        Maximum_heat_supplied = st.number_input(label="Maximum heat supplied", min_value=320, max_value=6900)
        Heat_Input_Heat_Exchanger = st.number_input(label="Heat Input into HeatExchanger", min_value=280, max_value=4020)
        Net_Power_Output = st.number_input(label="Net Power Output", min_value=85, max_value=1940)
        Cycle_Thermal_Efficiency = st.number_input(label="Cycle Thermal Efficiency", min_value=1.9, max_value=6.9)
        Exergy_Efficiency = st.number_input(label="Exergy Efficiency", min_value= 0.2, max_value= 1.4)
        values = [#strength_of_mixture,
                  mass_flowrate_Biomass,
                  #combustion_efficiency,
                  hot_side_Inlet_Temperature,
                  #temperature_at_point1,
                  #pressure_at_point1,
                  mass_flowrate_workingfluid,
                  #temperature_at_point2,
                  #evaporator_efficiency,
                  #pump_efficiency
                  Maximum_heat_supplied,
                  Heat_Input_Heat_Exchanger,
                  Net_Power_Output,
                  Cycle_Thermal_Efficiency,
                  Exergy_Efficiency
                  ]
        array = np.array(values).reshape(1, -1)
        array = scaler.transform(array)
    button = st.button("Make Inference")
    if button:
        prediction = model(torch.tensor(array).to(dtype = torch.float32))
        predictions = prediction.detach().tolist()[0]
        Type_Biomass = round(predictions[0])
        Strength_of_mixture = round(predictions[1])
        Combustion_efficiency = round(predictions[2],1)
        Temperature_at_point1 = int(predictions[3])
        Pressure_at_point1 = int(predictions[4])
        Temperature_at_point2 = int(predictions[5])
        Pump_efficiency = round(predictions[6],1)
        Evaporator_efficiency = round(predictions[7],1)
        Pinch_Point_Temperature = round(predictions[8], 1)

        st.write(f"Type of Biomass: {Type_Biomass}")
        st.write(f"Strength of mixture: {Strength_of_mixture}")
        st.write(f"Combustion efficiency: {Combustion_efficiency}")
        st.write(f"Temperature at point1: {Temperature_at_point1}")
        st.write(f"Pressure at point1: {Pressure_at_point1}")
        st.write(f"Temperature at point2: {Temperature_at_point2}")
        st.write(f"pump efficiency: {Pump_efficiency}")
        st.write(f"Evaporator efficiency: {Evaporator_efficiency}")
        st.write(f"Pinch Point Temperature: {Pinch_Point_Temperature}")

#st.title("Eniola's Project")

selected_option = st.sidebar.selectbox('Select an Option', ('Preview', 'Data', 'Interface'))
if selected_option == 'Preview':
    preview()
elif selected_option == 'Data':
    pass
elif selected_option == 'Interface':
    model_interface()
