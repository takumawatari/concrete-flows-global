import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.stats import t
import random
import statistics
import pathlib
import sys
import math
import matplotlib.pyplot as plt

def stock_driven_MFA(in_func_con_matrix, 
                     in_func_mor_matrix,
                     ot_con_matrix, 
                     ot_mor_matrix,
                     st_func_con_matrix, 
                     st_func_mor_matrix,
                     life_matrix,
                     MI_con_matrix, 
                     MI_mor_matrix,
                     recycling_rate,
                     downcycling_rate,
                     hibernating_stock_rate,
                     reuse_rate,
                     construction_yield,
                     cement_content_concrete,
                     water_content_concrete,
                     fine_aggregates_content_concrete,
                     coarse_aggregates_content_concrete, 
                     slag_content_concrete,
                     ash_content_concrete,
                     cement_content_mortar,
                     water_content_mortar,
                     fine_aggregates_content_mortar,
                     coarse_aggregates_content_mortar, 
                     slag_content_mortar,
                     ash_content_mortar,
                     clinker_ratio,
                     gypsum_ratio,
                     slag_ratio,
                     ash_ratio,
                     limestone_ratio,
                     pozzolana_ratio,
                     CKD_generation_rate,
                     proportion_landfilled_CKD,
                     manufacturing_yield,
                     process_emission_factor,
                     thermal_energy,
                     carbon_intensity_fuel,
                     electricity_cement,
                     electricity_emission_factor,
                     aggregate_emission_factor,
                     electricity_slag,
                     electricity_ash,
                     mixing_emission_factor,
                     placement_emission_factor,
                     transportation_cement,
                     transportation_admixtures,
                     transportation_virgin_aggregate,
                     transportation_buried_aggregate,
                     transportation_recycled_aggregate,
                     year,
                     scenario_index):
    
    in_con_matrix = pd.DataFrame()
    in_mor_matrix = pd.DataFrame()
    
    for i in range(0,len(in_func_con_matrix.columns)):
        year_complete = np.arange(1900,2021)
        in_func_con_extended = in_func_con_matrix.iloc[:,i]
        in_func_mor_extended = in_func_mor_matrix.iloc[:,i]
        ot_con_extended = ot_con_matrix.iloc[:,i]
        ot_mor_extended = ot_mor_matrix.iloc[:,i]
        ot_con_extended = np.repeat(0,len(year_complete))
        ot_mor_extended = np.repeat(0,len(year_complete))
    
        for k in range(2021,2051):
            life_extended   = life_matrix.iloc[0:len(year_complete),i]
            MI_con = MI_con_matrix.iloc[0:len(year_complete),i]
            MI_mor = MI_mor_matrix.iloc[0:len(year_complete),i]
            
            # function outflow
            ot_func_con_list = in_func_con_extended * (norm.pdf(k-year_complete,life_extended,life_extended*0.28))
            ot_func_con      = sum(ot_func_con_list)
            ot_func_mor_list = in_func_mor_extended * (norm.pdf(k-year_complete,life_extended,life_extended*0.28))
            ot_func_mor      = sum(ot_func_mor_list)
            
            # material outflow            
            ot_con_list = (in_func_con_extended * MI_con) * (norm.pdf(k-year_complete,life_extended,life_extended*0.28))
            ot_con      = sum(ot_con_list)
            ot_con_extended = np.append(ot_con_extended,ot_con)
            ot_mor_list = (in_func_mor_extended * MI_mor) * (norm.pdf(k-year_complete,life_extended,life_extended*0.28))
            ot_mor      = sum(ot_mor_list)
            ot_mor_extended = np.append(ot_mor_extended,ot_mor)
            
            # function inflow
            in_func_con = st_func_con_matrix.iloc[k-1900,i] - st_func_con_matrix.iloc[k-1901,i] + ot_func_con
            in_func_con_extended = np.append(in_func_con_extended, in_func_con)
            in_func_mor = st_func_mor_matrix.iloc[k-1900,i] - st_func_mor_matrix.iloc[k-1901,i] + ot_func_mor
            in_func_mor_extended = np.append(in_func_mor_extended, in_func_mor)
            
            year_complete = np.append(year_complete,k)
        
        # material inflow matrix
        in_con = in_func_con_extended * MI_con_matrix.iloc[:,i]
        in_con_matrix = pd.concat([in_con_matrix, pd.Series(in_con)], axis = 1, ignore_index = True)
        in_mor = in_func_mor_extended * MI_mor_matrix.iloc[:,i]
        in_mor_matrix = pd.concat([in_mor_matrix, pd.Series(in_mor)], axis = 1, ignore_index = True)
        
        # material outflow matrix
        ot_con_matrix = pd.concat([ot_con_matrix, pd.Series(ot_con_extended)], axis = 1, ignore_index = True)
        ot_mor_matrix = pd.concat([ot_mor_matrix, pd.Series(ot_mor_extended)], axis = 1, ignore_index = True)
        
    ot_con_matrix=pd.concat([ot_con_matrix.fillna(0)[x]+ot_con_matrix[y] 
                             for x,y in zip(range(0,3),range(3,6))], axis = 1)
    ot_mor_matrix=pd.concat([ot_mor_matrix.fillna(0)[x]+ot_mor_matrix[y] 
                             for x,y in zip(range(0,3),range(3,6))], axis = 1)
    
    NAS_con_matrix = in_con_matrix - ot_con_matrix
    NAS_mor_matrix = in_mor_matrix - ot_mor_matrix
    st_con_matrix  = NAS_con_matrix.cumsum()
    st_mor_matrix  = NAS_mor_matrix.cumsum()
    
    in_con_matrix = in_con_matrix.rename(columns={0:"Residential buildings",
                                                  1:"Non-residential buildings",
                                                  2:"Civil engineering"})                   
    ot_con_matrix = ot_con_matrix.rename(columns={0:"Residential buildings",
                                                  1:"Non-residential buildings",
                                                  2:"Civil engineering"})
    st_con_matrix = st_con_matrix.rename(columns={0:"Residential buildings",
                                                  1:"Non-residential buildings",
                                                  2:"Civil engineering"})
    
    in_mor_matrix = in_mor_matrix.rename(columns={0:"Residential buildings",
                                                  1:"Non-residential buildings",
                                                  2:"Civil engineering"})
    ot_mor_matrix = ot_mor_matrix.rename(columns={0:"Residential buildings",
                                                  1:"Non-residential buildings",
                                                  2:"Civil engineering"})
    st_mor_matrix = st_mor_matrix.rename(columns={0:"Residential buildings",
                                                  1:"Non-residential buildings",
                                                  2:"Civil engineering"})
    
    # sum
    in_total_matrix = in_con_matrix + in_mor_matrix
    ot_total_matrix = ot_con_matrix + ot_mor_matrix
    st_total_matrix = st_con_matrix + st_mor_matrix

    in_con_total = in_con_matrix.sum(axis=1)
    ot_con_total = ot_con_matrix.sum(axis=1)
    st_con_total = st_con_matrix.sum(axis=1)
    in_mor_total = in_mor_matrix.sum(axis=1)
    ot_mor_total = ot_mor_matrix.sum(axis=1)
    st_mor_total = st_mor_matrix.sum(axis=1)

    in_total = in_con_total + in_mor_total
    ot_total = ot_con_total + ot_mor_total
    st_total = st_con_total + st_mor_total

    # eol management
    rec   = ot_total * recycling_rate
    down  = ot_total * downcycling_rate
    hiber = ot_total * hibernating_stock_rate
    reuse_Res  = ot_con_matrix["Residential buildings"] * reuse_rate
    reuse_NonR = ot_con_matrix["Non-residential buildings"] * reuse_rate
    reuse = reuse_Res + reuse_NonR
    land  = ot_total - rec - down - hiber - reuse
    EoL_mix = pd.DataFrame()
    EoL_mix = pd.concat([rec,down,hiber,land,reuse],axis=1)
    EoL_mix = EoL_mix.rename(columns={0:"Recycling",1:"Downcycling",2:"Hibernating stock",3:"Landfill",4:"Reuse"})
    
    # concrete and mortar production
    pro_con = (in_con_total/construction_yield) - reuse
    pro_mor = (in_mor_total/construction_yield)
    production_mix = pd.DataFrame()
    production_mix = pd.concat([pro_con,pro_mor],axis=1)
    production_mix = production_mix.rename(columns={0:"Concrete",1:"Mortar"})
    
    # construction loss
    concrete_loss = pro_con + reuse - in_con_total
    mortar_loss   = pro_mor - in_mor_total
    conloss_mix = pd.DataFrame()
    conloss_mix = pd.concat([concrete_loss,mortar_loss],axis=1)
    conloss_mix = conloss_mix.rename(columns={0:"Concrete loss",1:"Mortar loss"})
    
    # concrete mixture
    cement_con = pro_con * cement_content_concrete
    water_con  = pro_con * water_content_concrete
    fine_con   = pro_con * fine_aggregates_content_concrete
    coarse_con = pro_con * coarse_aggregates_content_concrete
    slag_con   = pro_con * slag_content_concrete
    ash_con    = pro_con * ash_content_concrete
    concrete_mix = pd.DataFrame()
    concrete_mix = pd.concat([cement_con,water_con,fine_con,coarse_con,slag_con,ash_con],axis=1)
    concrete_mix = concrete_mix.rename(columns={0:"Cement",1:"Water",2:"Fine aggregates",
                                                3:"Coarse aggregates",4:"Slag",5:"Fly ash"})
    
    # mortar mixture
    cement_mor = pro_mor * cement_content_mortar
    water_mor  = pro_mor * water_content_mortar
    fine_mor   = pro_mor * fine_aggregates_content_mortar
    coarse_mor = pro_mor * coarse_aggregates_content_mortar
    slag_mor   = pro_mor * slag_content_mortar
    ash_mor    = pro_mor * ash_content_mortar
    mortar_mix = pd.DataFrame()
    mortar_mix = pd.concat([cement_mor,water_mor,fine_mor,coarse_mor,slag_mor,ash_mor],axis=1)
    mortar_mix = mortar_mix.rename(columns={0:"Cement",1:"Water",2:"Fine aggregates",
                                            3:"Coarse aggregates",4:"Slag",5:"Fly ash"})
    
    # cement mixture
    cement_pro  = cement_con + cement_mor
    clinker_cem = cement_pro * clinker_ratio
    gypsum_cem  = cement_pro * gypsum_ratio
    slag_cem    = cement_pro * slag_ratio
    ash_cem     = cement_pro * ash_ratio
    lime_cem    = cement_pro * limestone_ratio
    pozzo_cem   = cement_pro * pozzolana_ratio
    cement_mix  = pd.DataFrame()
    cement_mix  = pd.concat([clinker_cem,gypsum_cem,slag_cem,ash_cem,lime_cem,pozzo_cem],axis=1)
    cement_mix  = cement_mix.rename(columns={0:"Clinker",1:"Gypsum",2:"Slag",3:"Fly ash",
                                             4:"Limestone",5:"Pozzolana"})
    
    # ckd
    CKD      = clinker_cem * CKD_generation_rate
    land_CKD = CKD * proportion_landfilled_CKD
    CKD_mix  = pd.DataFrame()
    CKD_mix  = pd.concat([CKD,land_CKD],axis=1)
    CKD_mix  = CKD_mix.rename(columns={0:"CKD generation",1:"Landfilled CKD"})
    
    # manufacturing loss
    fine_con_loss   = fine_con   / manufacturing_yield - fine_con
    coarse_con_loss = coarse_con / manufacturing_yield - coarse_con
    fine_mor_loss   = fine_mor   / manufacturing_yield - fine_mor
    manloss_mix = pd.DataFrame()
    manloss_mix = pd.concat([fine_con_loss,coarse_con_loss,fine_mor_loss],axis=1)
    manloss_mix = manloss_mix.rename(columns={0:"Fine aggregate loss in concrete manufacturing",
                                              1:"Coarse aggregate loss in concrete manufacturing",
                                              2:"Fine aggregate loss in mortar manufacturing"})
    
    # total aggregate production
    pro_fine_con   = fine_con + fine_con_loss
    pro_coarse_con = coarse_con + coarse_con_loss
    pro_fine_mor   = fine_mor + fine_mor_loss
    
    # natural aggregate production
    pro_fine_nat   = pro_fine_con + pro_fine_mor - (rec/2)
    pro_coarse_nat = pro_coarse_con - (rec/2)
    aggregate_mix = pd.DataFrame()
    aggregate_mix = pd.concat([pro_fine_nat,pro_coarse_nat,rec],axis=1)
    aggregate_mix = aggregate_mix.rename(columns={0:"Virgin fine aggregate production",
                                                  1:"Virgin coarse aggregate production",
                                                  2:"Recycled aggregate production"})
    
    # co2 emissions
    cem_process  = clinker_cem * process_emission_factor
    cem_thermal  = clinker_cem * thermal_energy * carbon_intensity_fuel
    cem_electric = cement_pro  * electricity_cement * electricity_emission_factor
    agg_pro      = (pro_fine_nat+pro_coarse_nat+rec) * aggregate_emission_factor
    slag_pre     = (slag_con + slag_mor) * electricity_slag * electricity_emission_factor
    ash_pre      = (ash_con + ash_mor) * electricity_ash * electricity_emission_factor
    adm_pre      = slag_pre + ash_pre
    mixing       = (pro_con + pro_mor) * mixing_emission_factor
    onsite       = (pro_con + pro_mor) * placement_emission_factor
    tra_cem      = cement_pro * transportation_cement
    tra_adm      = (slag_con + slag_mor + ash_con + ash_mor) * transportation_admixtures
    tra_agg_nat  = (pro_fine_nat+pro_coarse_nat) * transportation_virgin_aggregate
    tra_agg_land = land * transportation_buried_aggregate
    tra_agg_rec  = rec * transportation_recycled_aggregate
    tra_total    = tra_cem+tra_adm+tra_agg_nat+tra_agg_land+tra_agg_rec

    CO2_matrix = pd.DataFrame()
    CO2_matrix = pd.concat([cem_process,cem_thermal,cem_electric,agg_pro,adm_pre,mixing,onsite,tra_total],axis=1)
    CO2_matrix = CO2_matrix.rename(columns={0:"Cement production (carbonate calcination)",
                                            1:"Cement production (fuel combustion)",
                                            2:"Cement production (electricity use)",
                                            3:"Aggregate production",
                                            4:"Admixture preparation",
                                            5:"Mixing and batching",
                                            6:"On-site placement",
                                            7:"Transportation"})

    cement_mix     = cement_mix.rename(index=year)
    CKD_mix        = CKD_mix.rename(index=year)
    concrete_mix   = concrete_mix.rename(index=year)
    mortar_mix     = mortar_mix.rename(index=year)
    manloss_mix    = manloss_mix.rename(index=year)
    aggregate_mix  = aggregate_mix.rename(index=year)
    production_mix = production_mix.rename(index=year)
    conloss_mix    = conloss_mix.rename(index=year)
    in_con_matrix  = in_con_matrix.rename(index=year)
    in_mor_matrix  = in_mor_matrix.rename(index=year)
    ot_con_matrix  = ot_con_matrix.rename(index=year)
    ot_mor_matrix  = ot_mor_matrix.rename(index=year)
    st_con_matrix  = st_con_matrix.rename(index=year)
    st_mor_matrix  = st_mor_matrix.rename(index=year)
    in_total_matrix = in_total_matrix.rename(index=year)
    ot_total_matrix = ot_total_matrix.rename(index=year)
    st_total_matrix = st_total_matrix.rename(index=year)
    EoL_mix = EoL_mix.rename(index=year)
    CO2_matrix = CO2_matrix.rename(index=year)
    
    # results
    with pd.ExcelWriter('Results'+'/Scenario_'+str(scenario_index) + '.xlsx') as writer:
        cement_mix.to_excel(writer, sheet_name='Cement')
        CKD_mix.to_excel(writer, sheet_name='CKD')
        concrete_mix.to_excel(writer, sheet_name='Concrete')
        mortar_mix.to_excel(writer, sheet_name='Mortar')
        manloss_mix.to_excel(writer, sheet_name='Manloss')
        aggregate_mix.to_excel(writer, sheet_name='Aggregate')
        production_mix.to_excel(writer, sheet_name='Production')
        conloss_mix.to_excel(writer, sheet_name='Conloss')
        in_con_matrix.to_excel(writer, sheet_name='Inflow_con')
        in_mor_matrix.to_excel(writer, sheet_name='Inflow_mor')
        ot_con_matrix.to_excel(writer, sheet_name='Outflow_con')
        ot_mor_matrix.to_excel(writer, sheet_name='Outflow_mor')
        st_con_matrix.to_excel(writer, sheet_name='Stock_con')
        st_mor_matrix.to_excel(writer, sheet_name='Stock_mor')
        in_total_matrix.to_excel(writer, sheet_name='Inflow_total')
        ot_total_matrix.to_excel(writer, sheet_name='Outflow_total')
        st_total_matrix.to_excel(writer, sheet_name='Stock_total')
        EoL_mix.to_excel(writer, sheet_name='EoL')
        CO2_matrix.to_excel(writer, sheet_name='CO2')
    
    # figures
    ax=CO2_matrix.plot.area(figsize=(4, 4),lw=0)
    ax.set_title(str(scenario_index))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels),loc='upper left',bbox_to_anchor=(1, 0.8),fontsize=11)
    ax.grid(linewidth=0.5,linestyle = "--")