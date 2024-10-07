# -*- coding: utf-8 -*-
"""
Created on Sun Sep  25 08:26:49 2022

Python script to automatically extract all possible paths from a set of simulation parameters to the 
box-counting dimension of nanoparticles and compute the product of the motivation values of links 
involved in the path, given the causal links and their corresponding motivation values obtained from 
the process of interactive learning of Bayesian networks using OpenMarkov.
@author: Jonathan Yik Chang Ting
"""

import math
import csv
import xlsxwriter
import pandas as pd


synParams = ['T', 'tau', 'time']  # Synthesis parameters
coordFeats = ['N_atom_surface', 'MM_SCN_3', 'MM_SCN_4', 'MM_SCN_6', 'MM_SCN_7', 'MM_SCN_8', 'MM_SCN_10', 'MM_SCN_12', 'MM_TCN_11', 'MM_TCN_14', 'MM_TCN_15', 'MM_SCN_avg', 'MM_BCN_avg']  # Coordination features
crysFeats = ['q6q6_S_3', 'q6q6_S_12', 'q6q6_B_4', 'q6q6_B_7', 'q6q6_T_11', 'q6q6_T_13', 'q6q6_avg_surf', 'FCC', 'HCP', 'DECA']  # Crystallographic features
surfStructFeats = ['Curve_11-20', 'Curve_31-40', 'Curve_41-50', 'S_100', 'S_110', 'S_111', 'S_311']  # Surface structural features
wholeNPfeats = ['R_diff', 'R_skew', 'R_kurt', 'MM_BL_min', 'MM_BL_avg', 'MMM_BA1_avg', 'Volume']  # Nanoparticle topological features
dBox = ['DBoxEX']
allNodes = synParams + coordFeats + crysFeats + surfStructFeats + wholeNPfeats + dBox
motivationNumDict = {('N_atom_surface', 'q6q6_T_11'): 63.82, 
                     ('N_atom_surface', 'MM_SCN_10'): 60.80,
                     ('N_atom_surface', 'FCC'): 55.97, 
                     ('N_atom_surface', 'MM_SCN_12'): 46.88, 
                     ('N_atom_surface', 'q6q6_S_3'): 40.32, 
                     ('MM_SCN_10', 'HCP'): 39.86, 
                     ('q6q6_avg_surf', 'MM_BL_min'): 35.17, 
                     ('time', 'q6q6_avg_surf'): 33.74, 
                     ('MM_TCN_14', 'Curve_11-20'): 33.01, 
                     ('q6q6_avg_surf', 'MMM_BA1_avg'): 31.19, 
                     ('q6q6_T_11', 'Curve_41-50'): 27.62, 
                     ('MMM_BA1_avg', 'Volume'): 26.09, 
                     ('N_atom_surface', 'R_diff'): 25.27, 
                     ('q6q6_avg_surf', 'MM_BL_avg'): 25.21, 
                     ('N_atom_surface', 'MMM_BA1_avg'): 23.34, 
                     ('q6q6_T_11', 'Curve_31-40'): 21.22, 
                     ('MM_SCN_12', 'q6q6_T_13'): 19.54, 
                     ('q6q6_S_3', 'DECA'): 16.19, 
                     ('T', 'q6q6_avg_surf'): 15.36, 
                     ('tau', 'R_diff'): 14.76, 
                     ('Curve_41-50', 'R_skew'): 14.42, 
                     ('time', 'MM_BCN_avg'): 13.31, 
                     ('N_atom_surface', 'DBoxEX'): 13.31, 
                     ('Curve_31-40', 'S_110'): 13.10, 
                     ('q6q6_S_3', 'FCC'): 13.03, 
                     ('MM_SCN_12', 'MM_SCN_avg'): 12.85, 
                     ('tau', 'N_atom_surface'): 12.31, 
                     ('time', 'N_atom_surface'): 24.68, 
                     ('MM_SCN_4', 'q6q6_B_7'): 11.38, 
                     ('N_atom_surface', 'MM_SCN_6'): 11.29, 
                     ('N_atom_surface', 'MM_SCN_7'): 11.26, 
                     ('MM_SCN_12', 'q6q6_B_4'): 9.55, 
                     ('MM_SCN_avg', 'MM_BL_avg'): 9.24, 
                     ('q6q6_S_3', 'q6q6_avg_surf'): 9.10, 
                     ('MM_SCN_8', 'S_100'): 8.77, 
                     ('N_atom_surface', 'MM_TCN_14'): 8.11, 
                     ('MM_SCN_4', 'Curve_41-50'): 7.04, 
                     ('MM_TCN_11', 'MM_BCN_avg'): 6.96, 
                     ('MM_TCN_14', 'q6q6_B_7'): 6.92, 
                     ('N_atom_surface', 'MM_TCN_11'): 6.34, 
                     ('MM_SCN_10', 'q6q6_S_3'): 5.86, 
                     ('MM_SCN_10', 'Curve_11-20'): 5.80, 
                     ('MM_SCN_10', 'S_311'): 5.44, 
                     ('MM_TCN_15', 'FCC'): 4.91, 
                     ('time', 'MM_SCN_3'): 3.89, 
                     ('MM_SCN_8', 'DBoxEX'): 3.42, 
                     ('Volume', 'DBoxEX'): 3.35, 
                     ('S_111', 'R_diff'): 3.20, 
                     ('MM_SCN_6', 'q6q6_T_11'): 3.16, 
                     ('MM_SCN_12', 'q6q6_S_12'): 3.16, 
                     ('T', 'MM_BL_min'): 3.01, 
                     ('MM_SCN_3', 'MM_BL_min'): 6.38, 
                     ('N_atom_surface', 'MM_SCN_4'): 2.96, 
                     ('FCC', 'S_111'): 2.67, 
                     ('MM_TCN_15', 'q6q6_avg_surf'): 2.39, 
                     ('q6q6_T_13', 'Curve_11-20'): 2.26, 
                     ('DECA', 'MM_BL_avg'): 2.24, 
                     ('DECA', 'Curve_41-50'): 2.15, 
                     ('MM_TCN_15', 'R_skew'): 1.87, 
                     ('MM_TCN_11', 'q6q6_B_7'): 1.79, 
                     ('S_110', 'DBoxEX'): 1.50, 
                     ('S_311', 'DBoxEX'): 2.41, 
                     ('MM_SCN_6', 'HCP'): 1.15, 
                     ('MM_SCN_12', 'Curve_11-20'): 1.03, 
                     ('q6q6_S_3', 'Curve_31-40'): 1.02, 
                     ('MM_SCN_4', 'MM_BCN_avg'): 0.96, 
                     ('MM_SCN_4', 'MM_SCN_avg'): 0.57, 
                     ('MM_TCN_11', 'q6q6_B_4'): 0.52, 
                     ('MM_TCN_14', 'q6q6_B_4'): 1.73, 
                     ('MM_SCN_6', 'q6q6_S_3'): 0.35, 
                     ('MM_TCN_15', 'q6q6_B_4'): 0.28, 
                     ('S_311', 'MM_BL_avg'): 0.11, 
                     ('HCP', 'R_skew'): 0.02}
motivationNumNormDict = dict()
motivationNumSum = sum([value for value in motivationNumDict.values()])
for (variablePair, motivationNum) in motivationNumDict.items():
    motivationNumNormDict[variablePair] = motivationNum / motivationNumSum

rulesEncoding = {'T': 0, 'tau': 0, 'time': 0,
                 'MM_SCN_avg': 1, 'MM_SCN_3': 1, 'MM_SCN_4': 1, 'MM_SCN_6': 1, 'MM_SCN_7': 1, 'MM_SCN_8': 1, 'MM_SCN_10': 1, 'MM_SCN_12': 1, 'MM_BCN_avg': 1, 'MM_TCN_11': 1, 'MM_TCN_14': 1, 'MM_TCN_15': 1, 
                 'FCC': 2, 'HCP': 2, 'DECA': 2, 'q6q6_avg_surf': 2, 'q6q6_S_3': 2, 'q6q6_S_12': 2, 'q6q6_B_4': 2, 'q6q6_B_7': 2, 'q6q6_T_11': 2, 'q6q6_T_13': 2, 
                 'S_100': 3, 'S_110': 3, 'S_111': 3, 'S_311': 3, 'Curve_11-20': 3, 'Curve_31-40': 3, 'Curve_41-50': 3, 
                 'Volume': 4, 'N_atom_surface': 4, 'R_diff': 4, 'R_skew': 4, 'R_kurt': 4, 'MM_BL_avg': 4, 'MM_BL_min': 4, 'MMM_BA1_avg': 4, 
                 'DBoxEX': 5}
allPathProbDict = dict()
pathProbDictsList = [dict() for _ in dBox]
pathProbDFsList = []


def getLinks(nodeOfInterest, verbose=False):
    """
    Look for causal links directed from nodeOfInterest to other nodes
    
    Parameters:
        nodeOfInterest: string of start node
        verbose: Boolean indicating whether extra printings are required
    Outputs:
        targetLinks: list of tuples of causal links
    """
    if verbose: print("    Getting causal links directed from {0}...".format(nodeOfInterest))
    targetLinks = []
    for link in motivationNumNormDict.keys():
        if link[0] is nodeOfInterest:
            targetLinks.append(link)
            if verbose: print("        {0}".format(link))
    return targetLinks


def extendPath(nodeOfInterest, pathString, verbose=False):
    """
    Recursive function for path-searching
    
    Parameters:
        nodeOfInterest: string indicating the node from which causal links are searched from
        pathString: string containing nodes passed by so far
        verbose: Boolean indicating whether extra printings are required
    """
    if verbose: print("    CURRENT PATH:\t\t{0}".format(pathString))

    # Get all causal links directed from startNode
    targetLinks = getLinks(nodeOfInterest=nodeOfInterest, verbose=verbose)

    # Reset pathString if no link is found to be directed from nodeOfInterest
    if len(targetLinks) == 0:
        pathString = ""
        # if verbose: print("    * Path does not terminate at observation node!")

    for targetLink in targetLinks:
        if verbose: print("    Checking link: {0}".format(targetLink))

        # Extend the string of path
        if targetLink[0] in synParams: pathString = targetLink[0]
        pathString += " > {0}".format(targetLink[1])

        # Check if path is completed
        isCompletePath = True if targetLink[1] in dBox else False
        if isCompletePath:
            nodesInPath = pathString.split(" > ")
            startNodes = nodesInPath[:-1]
            endNodes = nodesInPath[1:]
            motivationNumNormList = [motivationNumNormDict[(startNodes[i], endNodes[i])] for i in range(len(startNodes))]
            pathProbability = math.prod(motivationNumNormList)
            allPathProbDict[pathString] = pathProbability
            print("FOUND a COMPLETE path:\t\t{0} (prob: {1:.2e})".format(pathString, pathProbability))
        else:
            extendPath(nodeOfInterest=targetLink[1], pathString=pathString, verbose=verbose)

        # Update pathString to continue searching for paths
        pathString =  " > ".join(pathString.split(" > ")[:-1])


def getPaths(verbose=False):
    """
    Identify all paths of a given Bayesian network (described in the form of 
    dictionary containing directed edges as keys) learnt using OpenMarkov
    
    Parameters:
        verbose
    """
    print("Finding all possible paths of learnt Bayesian network...")
    # Start each path with a coordination feature
    for ordParam in synParams:
        pathString = ""
        extendPath(nodeOfInterest=ordParam, pathString=pathString, verbose=verbose)


def splitDict():
    """
    Split allPathProbDict into separate dictionaries (pathProbDictsList) based 
    on the class each path ends on.
    """
    for i, (path, motivation) in enumerate(allPathProbDict.items()):
        for (classIdx, dBoxStr) in enumerate(dBox):
            if path.split(' > ')[-1] == dBoxStr:
                pathProbDictsList[classIdx][path] = motivation
                break
    
    
def writeToXLSX(datasetName='', metric='K2'):
    """
    Output all pathProbDict (dictionary containing paths as keys and motivation 
    values product as values) to pathMotivations.xlsx
    
    Parameters:
        metric: string indicating metric used to train Bayesian network, used 
        as worksheet name
    """
    writer = pd.ExcelWriter('pathMotivations_{}_{}.xlsx'.format(datasetName, metric), engine='xlsxwriter')
    print("\nWriting to pathMotivations_{}_{}.xlsx...".format(datasetName, metric))
    for (i, pathProbDict) in enumerate(pathProbDictsList):
        # Sort by Path Probability and compute all columns
        df = pd.DataFrame(pathProbDict, index=['Path Probability']).transpose()
        df.index.name = "Path"
        # df.columns = ["Path Probability"]
        df.sort_values(by="Path Probability", ascending=False, inplace=True)
        df["Path Percentage"] = df["Path Probability"] * 100
        pathProbDFsList.append(df)
        
        # Write to Excel sheet
        df.to_excel(writer, sheet_name=dBox[i])
        worksheet = writer.sheets[dBox[i]]
        worksheet.set_column('A:A', 92)  # Width
        worksheet.set_column('B:B', 14)
        worksheet.set_column('C:C', 14)
    writer.close()
    

def writeToTEX():
    """
    Output all pathProbDFs (DataFrames containing probability of each path) to 
    a form suitable for filling the tables in a .tex document
    """
    for df in pathProbDFsList:
        for i in range(len(df)):
            pathProb = df.iloc[i]["Path Probability"]
            nodes = df.iloc[i].name.split(" > ")
            for (i, node) in enumerate(nodes):
                if "_" in node: node = node.replace("_", "\_")
                nodes[i] = "\\texttt{" + node + "}"
            pathString = " $\\rightarrow$ ".join(nodes)
            isDirect = ""  # " & Direct" if len(nodes) == 2 else " & Indirect" Not relevant for this project
            probString = " & \\num{{{:.3E}}}  \\\\".format(pathProb)  # Overwrite for SI tables
            print(pathString + isDirect + probString)
        print()


def main():
    datasetName, metric = 'dBox', 'K2'
    getPaths(verbose=True)  # Fill up allPathProbDict
    splitDict()  # Split allPathProbDict into dictionaries based on dBox
    writeToXLSX(datasetName=datasetName, metric=metric)  # Output to .xlsx
    writeToTEX()  # Print out rows for tables to be inserted into a .tex document


if __name__ == "__main__":
    main()

                
