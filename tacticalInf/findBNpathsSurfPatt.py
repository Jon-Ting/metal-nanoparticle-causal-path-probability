# -*- coding: utf-8 -*-
"""
Created on Sun Sep  25 08:26:49 2022

Python script to automatically extract all possible paths from a set of structural parameters to the class of 
surface patterns and compute the product of the motivation values of links involved in the path, given the causal 
links and their corresponding motivation values obtained from the process of interactive learning of Bayesian 
networks using OpenMarkov.
@author: Jonathan Yik Chang Ting
"""

import math
import csv
import xlsxwriter
import pandas as pd


ordParams = ['scn', 'q6q6', 'gcn']  # Coordination features
angParams = ['ba1avg', 'ba1max', 'chi3', 'chi6']  # Angular features
booParams = ['q2', 'q6', 'q2avg', 'q4avg', 'q6avg', 'q12avg']  # Bond orientational order features
symParams = ['centSym']  # Symmetrical features
posParams = ['rad']  # Positional features
atomClass = ['class1', 'class2', 'class3', 'class4']  # Atom class
allNodes = posParams + symParams + ordParams + angParams + booParams + atomClass
motivationNumDict = {('scn', 'class1'): 432.18,
                     ('scn', 'q2avg'): 281.05,
                     ('scn', 'ba1avg'): 278.05, 
                     ('scn', 'centSym'): 271.72, 
                     ('scn', 'q2'): 262.38, 
                     ('scn', 'chi6'): 249.13, 
                     ('scn', 'class2'): 210.08, 
                     ('scn', 'class3'): 64.91, 
                     ('scn', 'q6avg'): 25.06, 
                     ('scn', 'chi3'): 18.69, 
                     ('scn', 'q12avg'): 18.49, 
                     ('scn', 'q6'): 4.90, 
                     ('scn', 'ba1max'): 4.51, 
                     ('scn', 'q4avg'): 3.40, 
                     ('scn', 'rad'): 0.61, 
                     ('q6q6', 'ba1max'): 98.49, 
                     ('q6q6', 'q6avg'): 73.20, 
                     ('q6q6', 'q12avg'): 49.32, 
                     ('q6q6', 'chi3'): 34.42, 
                     ('q6q6', 'chi6'): 14.91, 
                     ('q6q6', 'q4avg'): 8.86, 
                     ('gcn', 'rad'): 266.38, 
                     ('gcn', 'q2avg'): 19.93, 
                     ('gcn', 'ba1avg'): 4.08, 
                     ('ba1avg', 'q12avg'): 34.55, 
                     ('ba1avg', 'centSym'): 9.82, 
                     ('ba1avg', 'q4avg'): 4.62, 
                     ('ba1max', 'q6'): 15.83, 
                     ('chi3', 'q6'): 179.98, 
                     ('chi3', 'q6avg'): 101.94, 
                     ('chi3', 'q4avg'): 85.05, 
                     ('chi3', 'centSym'): 75.85, 
                     ('chi6', 'q2'): 35.67, 
                     ('q2avg', 'rad'): 49.52, 
                     ('q6avg', 'class2'): 49.23, 
                     ('q6avg', 'class3'): 49.09, 
                     ('centSym', 'class4'): 4.93}
motivationNumNormDict = dict()
motivationNumSum = sum([value for value in motivationNumDict.values()])
for (variablePair, motivationNum) in motivationNumDict.items():
    motivationNumNormDict[variablePair] = motivationNum / motivationNumSum

rulesEncoding = {'scn': 0, 'q6q6': 0, 'gcn': 0, 
                 'ba1avg': 1, 'ba1max': 1, 'chi3': 1, 'chi6': 1, 
                 'q2': 2, 'q6': 2, 'q2avg': 2, 'q4avg': 2, 'q6avg': 2, 'q12avg': 2, 
                 'centSym': 3, 
                 'rad': 4, 
                 'class1': 5, 'class2': 5, 'class3': 5, 'class4': 5}
allPathProbDict = dict()
pathProbDictsList = [dict() for _ in atomClass]
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
        if targetLink[0] in ordParams: pathString = targetLink[0]
        pathString += " > {0}".format(targetLink[1])

        # Check if path is completed
        isCompletePath = True if targetLink[1] in atomClass else False
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
    for ordParam in ordParams:
        pathString = ""
        extendPath(nodeOfInterest=ordParam, pathString=pathString, verbose=verbose)


def splitDict():
    """
    Split allPathProbDict into separate dictionaries (pathProbDictsList) based 
    on the class each path ends on.
    """
    for i, (path, motivation) in enumerate(allPathProbDict.items()):
        for (classIdx, atomClassStr) in enumerate(atomClass):
            if path.split(' > ')[-1] == atomClassStr:
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
        df.to_excel(writer, sheet_name=atomClass[i])
        worksheet = writer.sheets[atomClass[i]]
        worksheet.set_column('A:A', 35)  # Width
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
    datasetName, metric = 'surfPatt', 'K2'
    getPaths(verbose=True)  # Fill up allPathProbDict
    splitDict()  # Split allPathProbDict into dictionaries based on atomClass
    writeToXLSX(datasetName=datasetName, metric=metric)  # Output to .xlsx
    writeToTEX()  # Print out rows for tables to be inserted into a .tex document


if __name__ == "__main__":
    main()

                
