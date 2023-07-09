import pandas as pd
import numpy as np



def construct_F_BLP(Model, Demand_parameter, C_total, num_tracts, num_current_stores, Quartile):

    Deltahat = Demand_parameter[0][0] + Demand_parameter[0][1] * np.log(C_total/1000)
    F_D_total = np.exp(Deltahat) / (1+np.exp(Deltahat))
    F_D_current = F_D_total[:,0:num_current_stores]

    F_DH_total = []
    for i in range(num_tracts):
                
        tract_quartile = Quartile[i]
                
        if tract_quartile == 1:
            deltahat = (Demand_parameter[1][0] + Demand_parameter[1][2]) + (Demand_parameter[1][1] + Demand_parameter[1][5]) * np.log(C_total[i,:]/1000)
            tract_willingness = np.exp(deltahat) / (1+np.exp(deltahat))
        elif tract_quartile == 2:
            deltahat = (Demand_parameter[1][0] + Demand_parameter[1][3]) + (Demand_parameter[1][1] + Demand_parameter[1][6]) * np.log(C_total[i,:]/1000)
            tract_willingness = np.exp(deltahat) / (1+np.exp(deltahat))
        elif tract_quartile == 3:
            deltahat = (Demand_parameter[1][0] + Demand_parameter[1][4]) + (Demand_parameter[1][1] + Demand_parameter[1][7]) * np.log(C_total[i,:]/1000)
            tract_willingness = np.exp(deltahat) / (1+np.exp(deltahat))
        elif tract_quartile == 4:
            deltahat = Demand_parameter[1][0] + Demand_parameter[1][1] * np.log(C_total[i,:]/1000)
            tract_willingness = np.exp(deltahat) / (1+np.exp(deltahat))
        else:
            deltahat = Demand_parameter[0][0] + Demand_parameter[0][1] * np.log(C_total[i,:]/1000)
            tract_willingness = np.exp(deltahat) / (1+np.exp(deltahat))
                
        F_DH_total.append(tract_willingness)
                
    F_DH_total = np.asarray(F_DH_total)
    F_DH_current = F_DH_total[:,0:num_current_stores]
    
    return F_D_current, F_D_total, F_DH_current, F_DH_total





def construct_F_LogLin(Model, Demand_parameter, C_total, num_tracts, num_current_stores, Quartile):

    F_D_total = Demand_parameter[0][0] + Demand_parameter[0][1] * np.log(C_total/1000)
    F_D_current = F_D_total[:,0:num_current_stores]

    F_DH_total = []
    for i in range(num_tracts):
                
        tract_quartile = Quartile[i]
                
        if tract_quartile == 1:
            tract_willingness = (Demand_parameter[1][0] + Demand_parameter[1][2]) + (Demand_parameter[1][1] + Demand_parameter[1][5]) * np.log(C_total[i,:]/1000)
        elif tract_quartile == 2:
            tract_willingness = (Demand_parameter[1][0] + Demand_parameter[1][3]) + (Demand_parameter[1][1] + Demand_parameter[1][6]) * np.log(C_total[i,:]/1000)
        elif tract_quartile == 3:
            tract_willingness = (Demand_parameter[1][0] + Demand_parameter[1][4]) + (Demand_parameter[1][1] + Demand_parameter[1][7]) * np.log(C_total[i,:]/1000)
        elif tract_quartile == 4:
            tract_willingness = Demand_parameter[1][0] + Demand_parameter[1][1] * np.log(C_total[i,:]/1000)
        else:
            tract_willingness = Demand_parameter[0][0] + Demand_parameter[0][1] * np.log(C_total[i,:]/1000)
                
        F_DH_total.append(tract_willingness)
                
    F_DH_total = np.asarray(F_DH_total)
    F_DH_current = F_DH_total[:,0:num_current_stores]

    return F_D_current, F_D_total, F_DH_current, F_DH_total