all: topo

topo:
	python ./Code/experiments/topo/SVM.py > ./Code/experiments/results/topo_SVM.txt &
	python ./Code/experiments/topo/FRF.py > ./Code/experiments/results/topo_RFF.txt &
	python ./Code/experiments/topo/RBF.py > ./Code/experiments/results/topo_RBF.txt &
	
forest_covertypes:
	python ./Code/experiments/forest_covertypes/FRF.py > ./Code/experiments/results/forest_covertypes_FRF.txt &
	python ./Code/experiments/forest_covertypes/SVM.py > ./Code/experiments/results/forest_covertypes_SVM.txt &