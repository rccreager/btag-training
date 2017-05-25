import matplotlib as plt
import sklearn


fpr, tpr, _ = roc_curve(true_labels, pred_labels)
auc = auc(fpr,tpr)

def make_roc(tpr, fpr, auc):
    '''
    Make a basic ROC curve with AUC
    input: true positive rate, false positive rate, area under curve
    calculated by the sklearn roc_curve function 
    '''
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
    fig.savefig('ROC.pdf')


