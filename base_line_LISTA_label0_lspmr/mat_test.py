import scipy.io as scio
import numpy as np

data_file = r'validations_dataset_test_split1_0.75_mp.mat'
data = scio.loadmat(data_file)

mAP = np.mean(data['AP_all'])

mclassification_rate = np.mean(data['classification_rate_all'])

true_classify = np.sum(data['classification_rate_all'] == 1)

print(mAP, mclassification_rate, true_classify)



