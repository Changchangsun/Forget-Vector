import numpy as np
import torch
import torch.utils.data
from sklearn.svm import SVC

from .base import EvalMetric
from .utils import register
# from .dataloader_base import UnlearnLoader
from torch.utils.data import DataLoader, ConcatDataset, Subset
import delta_utils

from tqdm import tqdm
import datetime
class MIA(EvalMetric):
    """Metric for evaluating the Membership Inference Attacks (MIA).

    Attributes:
        criterion (str): Criterion to be used in MIA.
    """

    def __init__(self, criterion, is_retrain_standard=True):
        """Initialize the MIA metric.

        Args:
            criterion (str): Criterion for MIA.
            is_retrain_standard (bool): Flag for retraining.

        """

        super().__init__(is_retrain_standard)
        self.criterion = criterion

    @staticmethod
    def collect_prob(loader, model,device):
        """Collect probabilities from the given model and data loader.

        Args:
            loader (DataLoader): Data loader.
            model (object): Model to evaluate.

        Returns:
            tuple: Probabilities and corresponding labels.

        """

        if loader is None:
            return torch.zeros([0, 10]), torch.zeros([0])

        model.eval()
        pred_prob = []
        targets = []
        for batch in tqdm(loader):
            batch = [data.to(device) for data in batch]
            input = batch[:-1]
            label = batch[-1]

            with torch.no_grad():
                output = model(*input)
            pred_prob.append(
                torch.clone(torch.nn.functional.softmax(output, dim=-1).data)
            )
            targets.append(label)

        return torch.cat(pred_prob, dim=0), torch.cat(targets, dim=0)

    @staticmethod
    def SVC_fit_predict(shadow_train, shadow_test, target_train, target_test):
        
        """Train a classifier on shadow data and test on target data.

        Args:
            shadow_train (tensor): Shadow training data.
            shadow_test (tensor): Shadow test data.
            target_train (tensor): Target training data.
            target_test (tensor): Target test data.

        Returns:
            float: Average accuracy of the classifier.

        """
        n_shadow_train = shadow_train.shape[0]
        n_shadow_test = shadow_test.shape[0]
        n_target_train = target_train.shape[0]
        n_target_test = target_test.shape[0]

        X_shadow = (
            torch.cat([shadow_train, shadow_test])
            .cpu()
            .numpy()
            .reshape(n_shadow_train + n_shadow_test, -1)
        )
        Y_shadow = np.concatenate([np.ones(n_shadow_train), np.zeros(n_shadow_test)])
        clf = SVC(C=3, gamma="auto", kernel="rbf")
        clf.fit(X_shadow, Y_shadow)

        accs = []

        if n_target_train > 0:
            print("n_target_train > 0")
            X_target_train = target_train.cpu().numpy().reshape(n_target_train, -1)
            acc_train = clf.predict(X_target_train).mean()
            accs.append(acc_train)

        if n_target_test > 0:
            
            X_target_test = target_test.cpu().numpy().reshape(n_target_test, -1)
            acc_test = 1 - clf.predict(X_target_test).mean()
            accs.append(acc_test)
        return np.mean(accs)

    @staticmethod
    def entropy(p, dim=-1, keepdim=False):
        """Calculate the entropy of a tensor.

        Args:
            p (tensor): Probability tensor.
            dim (int, optional): The dimension to reduce.
            keepdim (bool, optional): Whether the output tensor has dim retained or not.

        Returns:
            tensor: Entropy of p.

        """

        return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(
            dim=dim, keepdim=keepdim
        )

    @staticmethod
    def m_entropy(p, labels, dim=-1, keepdim=False):
        """Calculate the modified entropy of a tensor.

        Args:
            p (tensor): Probability tensor.
            labels (tensor): Class labels.
            dim (int, optional): The dimension to reduce.
            keepdim (bool, optional): Whether the output tensor has dim retained or not.

        Returns:
            tensor: Modified entropy of p.

        """

        log_prob = torch.where(p > 0, p.log(), torch.tensor(1e-30).to(p.device).log())
        reverse_prob = 1 - p
        log_reverse_prob = torch.where(
            p > 0, p.log(), torch.tensor(1e-30).to(p.device).log()
        )
        modified_probs = p.clone()
        labels = labels.long()
        modified_probs[:, labels] = reverse_prob[:, labels]
        modified_log_probs = log_reverse_prob.clone()
        modified_log_probs[:, labels] = log_prob[:, labels]
        return -torch.sum(modified_probs * modified_log_probs, dim=dim, keepdim=keepdim)

    def prob_to_data(self, prob, labels):
        """Convert probabilities to data based on criterion.

        Args:
            prob (tensor): Probability tensor.
            labels (tensor): Class labels.

        Returns:
            tensor: Data converted from probabilities.

        Raises:
            NotImplementedError: If criterion is not implemented.

        """

        if self.criterion == "correctness":
            return (torch.argmax(prob, axis=1) == labels).int()
        elif self.criterion == "confidence":
            return torch.gather(prob, 1, labels[:, None])
        elif self.criterion == "entropy":
            return self.entropy(prob)
        elif self.criterion == "modified_entropy":
            return self.m_entropy(prob, labels)
        else:
            raise NotImplementedError(
                "criterion {} not implemented".format(self.criterion)
            )
    def SVC_MIA(self, train_pos, train_neg, test_pos, test_neg, model,device):
        """Perform Membership Inference Attacks using Support Vector Classifier.

        Args:
            train_pos (tensor): Positive training set.
            train_neg (tensor): Negative training set.
            test_pos (tensor): Positive test set.
            test_neg (tensor): Negative test set.
            model (object): Model to evaluate.

        Returns:
            float: MIA success rate.

        """
        train_pos_prob, train_pos_labels = self.collect_prob(train_pos, model,device)
        train_neg_prob, train_neg_labels = self.collect_prob(train_neg, model,device)
        test_pos_prob, test_pos_labels = self.collect_prob(test_pos, model,device)
        test_neg_prob, test_neg_labels = self.collect_prob(test_neg, model,device)

        train_pos_data = self.prob_to_data(train_pos_prob, train_pos_labels)
        train_neg_data = self.prob_to_data(train_neg_prob, train_neg_labels)
        test_pos_data = self.prob_to_data(test_pos_prob, test_pos_labels)
        test_neg_data = self.prob_to_data(test_neg_prob, test_neg_labels)
        return self.SVC_fit_predict(
            train_pos_data, train_neg_data, test_pos_data, test_neg_data
        )


@register("mia_efficacy")
class MIAEfficacy(MIA):
    """
    Class for evaluating the efficacy of Membership Inference Attacks (MIA).

    Attributes:
        Inherited from the MIA class.
    """

    def evaluate(self, model, loaders, iteration,device,pattern,seed):#########
        """
        Evaluate the efficacy of MIA on a given model.

        Args:
            model: The model to evaluate.
            loaders (UnlearnLoader): A dictionary containing data loaders for testing, remaining and forgetting.
            iteration (int): The current iteration number.

        Returns:
            float: The efficacy score calculated using Support Vector Classification (SVC).
        """
    
        delta_utils.setup_seed(seed)
        
        if pattern == "classwise_with_adv":
            test_dataset = loaders["val_retain_adv"].dataset 
            remain_dataset = loaders["retain_adv"].dataset
            forget_loader = loaders["forget_adv"]
            
        elif pattern == "classwise":
            test_dataset = loaders["val_retain"].dataset
            remain_dataset = loaders["retain"].dataset
            forget_loader = loaders["forget"]
            
        elif pattern == "datawise_with_adv": 
            test_dataset = loaders["val_adv"].dataset 
            remain_dataset = loaders["retain_adv"].dataset
            forget_loader = loaders["forget_adv"]    
        elif pattern == "datawise":
            test_dataset = loaders["val"].dataset 
            remain_dataset = loaders["retain"].dataset
            forget_loader = loaders["forget"]
    
        elif pattern == "classwise_with_delta":
            test_dataset = loaders["val_retain_delta"].dataset 
            remain_dataset = loaders["retain_delta"].dataset
            forget_loader = loaders["forget_delta"]
            
        elif pattern == "datawise_with_delta":
            test_dataset = loaders["val_delta"].dataset ##
            remain_dataset = loaders["retain_delta"].dataset
            forget_loader = loaders["forget_delta"]
        
        
        test_len = min(len(test_dataset), len(remain_dataset))
        
        indexes = range(len(remain_dataset))
        rng = np.random.RandomState(seed)
        indexes = rng.choice(indexes, size=test_len, replace=False)
        train_pos = Subset(remain_dataset, indexes)
        
        indexes = range(len(test_dataset))
        rng = np.random.RandomState(seed)
        indexes = rng.choice(indexes, size=test_len, replace=False)
        train_neg = Subset(test_dataset, indexes)

        train_pos_loader = torch.utils.data.DataLoader(
            train_pos, batch_size=forget_loader.batch_size, shuffle=False
        )
        train_neg_loader = torch.utils.data.DataLoader(
            train_neg, batch_size=forget_loader.batch_size, shuffle=False
        )

        return self.SVC_MIA(
            train_pos=train_pos_loader,
            train_neg=train_neg_loader,
            test_pos=None,
            test_neg=forget_loader,
            model=model,
            device = device,
        )