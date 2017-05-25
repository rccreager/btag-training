import matplotlib as plt
import sklearn


def evaluate_performance(model, test_data):
    '''
    calculate some interesting stuff for evaluation/plotting
    input: trained model, testing data with true labels
    output true pos, false pos, auc 
    '''
    pred_labels = model.evaluate(test_data)
    fpr, tpr, _ = roc_curve(true_labels, pred_labels)
    auc = auc(fpr,tpr)
    return tpr, fpr, auc

def make_roc(model, test_data):
    '''
    Make a basic ROC curve with AUC
    input: true positive rate, false positive rate, area under curve
    calculated by the sklearn roc_curve function 
    '''
    tpr, fpr, auc = evaluate_performance(model, test_data)
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


