import matplotlib as plt
import sklearn


def save_plot(fig, plot_name):
    fig.savefig(plot_name)

def evaluate_roc(model, test_data):
    '''
    calculate some interesting stuff for evaluation/plotting
    input: trained model, testing data with true labels
    output true pos, false pos, auc 
    '''
    #once the format of the test_data is known, change this!
    pred_labels = model.predict(test_data)
    fpr, tpr, _ = roc_curve(true_labels, pred_labels)
    auc = auc(fpr,tpr)
    return tpr, fpr, auc

def make_roc(model, test_data):
    '''
    Make a basic ROC curve with AUC
    input: true positive rate, false positive rate, area under curve
    calculated by the sklearn roc_curve function 
    '''
    tpr, fpr, auc = evaluate_roc(model, test_data)
    fig = plt.figure(1)
    plt.title('ROC Curve')
    plt.plot(fpr, trp, 'b', label='AUC = %0.2f'% auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    save_plot(fig,'ROC.pdf')



