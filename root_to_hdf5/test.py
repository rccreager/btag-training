from convert_utils import root2pandas,flatten 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#import deepdish.io as io
 
filepath = '/Users/rcreager/Dropbox/Documents/bTagging_and_ML/HLT-btag-studies/samples/group.perf-flavtag.11010669.Akt4EMTo._000124.root'

df = root2pandas(filepath,'bTag_AntiKt4EMTopoJets', stop=100)
print(df.shape)

#test saving and loading from hd5
df.to_hdf('test_pd.h5', 'data')
new_df = pd.read_hdf('test_pd.h5', 'data')

#get only the jetfitter variables
jf_df = df[[key for key in df.keys() if (key.startswith('jet_jf') and '_vtx_' not in key)]]
print(jf_df.keys())

#flatten events into objects
jf_df_flat = pd.DataFrame({k: flatten(c) for k, c in jf_df.iteritems()})
print(jf_df_flat[jf_df.keys()])

#get everything ML ready
X = jf_df_flat.as_matrix()
le = LabelEncoder()
flavor = flatten(df['jet_LabDr_HadF'])
y = le.fit_transform(flavor)

ix = range(X.shape[0]) # array of indices, just to keep track of them for safety reasons and future checks
X_train, X_test, y_train, y_test, ix_train, ix_test = train_test_split(X, y, ix, train_size=0.6)

#scale inputs to avoid weird small/big weights
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)






#io.save('test_deepdish.h5', df)
#io.load('test_deepdish.h5')

#data = root2array(filepath)
#branches = data.dtype.names
#numBranches = len(data.dtype.names)
#eventShape = data.shape

#df = pd.DataFrame(data)
#print(df.shape)
#rootfile = ROOT.TFile(filepath)
#roottree = rootfile.Get("bTag_AntiKt4EMTopoJets")

#gROOT.Reset()
#c1 = TCanvas( 'c1', 'Example with Formula', 200, 10, 700, 500 )
 
#
# Create a one dimensional function and draw it
#
#fun1 = TF1( 'fun1', 'abs(sin(x)/x)', 0, 10 )
#c1.SetGridx()
#c1.SetGridy()
#fun1.Draw()
#c1.Update()
