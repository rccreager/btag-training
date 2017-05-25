

echo "assuming you have conda..."
conda create --name btagFresh
source activate btagFresh
conda install python=3.6.1
conda install h5py
conda install keras
conda install -c conda-forge keras=2.0.2
conda install -c conda-forge matplotlib=2.0.2
conda install scikit-learn

