from convert_utils import root2pandas,flatten
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

matplotlib.rcParams.update({'font.size': 16})

filepath = '/Users/rcreager/Dropbox/Documents/bTagging_and_ML/root_to_hdf5/samples/group.perf-flavtag.11010669.Akt4EMTo._000124.root'
df = root2pandas(filepath,'bTag_AntiKt4EMTopoJets', stop=100)
flavor = flatten(df['jet_LabDr_HadF'])
flavor_pids = np.unique(flavor)

jf_df = df[[key for key in df.keys() if (key.startswith('jet_jf') and '_vtx_' not in key)]]
jf_df_flat = pd.DataFrame({k: flatten(c) for k, c in jf_df.iteritems()})

for key in jf_df_flat.keys(): # plot the various variables one by one on different graphs
   
    # set up your figures
    fig = plt.figure(figsize=(8, 6), dpi=100)
    # specify ranges and binning strategies that make sense
    bins = np.linspace(
        min(jf_df_flat[key][jf_df_flat[key]!= -99]), # min
        max(jf_df_flat[key]), # max
        50 # number of bins
    )
    # select your favorite matplotlib color palette
    color = iter(cm.hsv(np.linspace(0, 0.8, len(flavor_pids))))
    # plot the histogram for each flavor using a different color
    for k in flavor_pids:
        c = next(color)
        _ = plt.hist(jf_df_flat[key][flavor == k][jf_df_flat[key]!= -99],
                    bins=bins, histtype='step', label='Flavor = {}'.format(k), color=c,
                    normed=True)

    # prettify your histograms
    plt.xlabel(key)
    plt.ylabel('Arbitrary Units')
    plt.legend()
    plt.show()


