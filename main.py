from nodes_generator import NodeGenerator
from simulated_annealing import SimulatedAnnealing

import pandas as pd
import numpy as np


def main():
    temp = 1000
    stopping_temp = 1
    alpha = 0.723
    stopping_iter = 20000
    
    # df_0 = pd.read_excel('probarSimAnn.xlsx',sheet_name=0,header=0, skipfooter = 4, usecols='B:I')
    # df_1 = pd.read_excel('probarSimAnn.xlsx',sheet_name=1,header=0, skipfooter = 5, usecols='B:P')


    # '''run simulated annealing algorithm with 2-opt'''
    # sa = SimulatedAnnealing( df_0,temp, alpha, stopping_temp, stopping_iter, home=2)
    # sa.anneal()

    # '''show the improvement over time'''
    # sa.plotLearning()
    
    
    # '''run simulated annealing algorithm with 2-opt'''
    # sa = SimulatedAnnealing( df_1,temp, alpha, stopping_temp, stopping_iter, home= 2)
    # sa.anneal()

    # '''show the improvement over time'''
    # sa.plotLearning()
    
    df_3 = pd.read_excel('SA2.xlsx',sheet_name=0,header=0, usecols='B:BC')
    
    '''run simulated annealing algorithm with 2-opt'''
    sa = SimulatedAnnealing( df_3,temp, alpha, stopping_temp, stopping_iter, home = 10)
    sa.anneal()

    '''show the improvement over time'''
    sa.plotLearning()
    # min_list = []
    # for _ in range(10):
    #     '''set the simulated annealing algorithm params'''
    #     temp = 1000
    #     stopping_temp = 1
    #     alpha = 0.723
    #     stopping_iter = 25000
    #     '''run simulated annealing algorithm with 2-opt'''
    #     sa = SimulatedAnnealing( df_3,temp, alpha, stopping_temp, stopping_iter)
    #     min_list.append(sa.anneal())  

    # '''show the improvement over time'''
    # sa.plotLearning()
        
    # print("std dev: ", np.std(min_list))
    # print("mean: ", np.mean(min_list))


if __name__ == "__main__":
    main()