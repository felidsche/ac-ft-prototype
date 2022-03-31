## once run_simulation_cluster.sh is done and the model name is known
# 1. plug the model name into analyse_prediction.ipynb and run it
# plug the model name and the prediction time into tc_sensitivity_analysis.py and run it
# this will generate 5 plots in out/analysis/sa/plots/bar
python3 tc_sensitivity_analysis.py
# this will generate the feature importance plot and the tree structure json file
python3 visualize_GBT.py
# this will display the json structure
decision_tree.html
# analyse_decision_tree.ipynb has more granular plots on decision trees