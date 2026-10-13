import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy import signal
from torch import nn


def _format_progress_value(value):
    if isinstance(value, (float, np.floating)):
        return f'{float(value):.4f}'
    return str(value)



def require_gradient(term, name, weight=None):
    """Fail loudly when a weighted loss term cannot move the parameters.

    Two separate bugs in this codebase took the same shape: a term was computed from a
    tensor that had been detached -- once by a reference forward overwriting a cached
    attribute, once by a sampling helper decorated with no_grad -- so the term became a
    constant and its weight silently stopped doing anything. Nothing raised, nothing
    diverged, and the only symptom was a hyper-parameter that appeared to have no effect.
    Any term entering the objective with a non-zero weight is checked here.
    """
    if weight is not None and float(weight) == 0.0:
        return term
    if not torch.is_tensor(term):
        raise TypeError(f'Loss term {name!r} is {type(term).__name__}, not a tensor.')
    if not term.requires_grad:
        raise RuntimeError(
            f'Loss term {name!r} carries a non-zero weight but has no gradient path. '
            f'It was most likely computed under torch.no_grad() or from a detached '
            f'tensor, which makes its weight a no-op.'
        )
    return term

def print_progress_line(prefix, epoch, total_epochs, **metrics):
    parts = [f'[{prefix}]', f'{epoch}/{total_epochs}']
    for key, value in metrics.items():
        if value is None:
            continue
        parts.append(f'{key}={_format_progress_value(value)}')
    print('\r' + ' '.join(parts), end='', flush=True)


def finish_progress_line(summary=None):
    print()
    if summary:
        print(summary)
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, X):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(X)

def Spectral_Normalization(m):
    for name, layer in m.named_children():
        m.add_module(name, Spectral_Normalization(layer))
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        return nn.utils.spectral_norm(m)
    else:
        return m

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

class CELoss_Marginal_Smooth(nn.Module):

    def __init__(self, class_num, alpha=0.6, stimulus_type=12):
        super(CELoss_Marginal_Smooth, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.stimulus_matrix_4 = [[0, 1],
                                  [2, 3]]

        self.stimulus_matrix_12 = [[0, 1, 2, 3],
                                   [4, 5, 6, 7],
                                   [8, 9, 10, 11]]
        self.stimulus_matrix_40 = [
            [0, 1, 2, 3, 4],
            [5,6,7,8,9,10],
            [10,11,12,13,14],
            [15,16,17,18,19],
            [20,21,22,23,24],
            [25,26,27,28,29],
            [30,31,32,33,34],
            [35,36,37,38,39]]


        if stimulus_type == 4:
            self.stimulus_matrix = self.stimulus_matrix_4

        elif stimulus_type == 12:
            self.stimulus_matrix = self.stimulus_matrix_12

        elif stimulus_type == 40:
            self.stimulus_matrix = self.stimulus_matrix_40


        self.rows = len(self.stimulus_matrix[:])
        self.cols = len(self.stimulus_matrix[0])


        self.attention_lst = [[1.0 / (int(0 <= (i // self.cols - 1) <= self.rows - 1) +
                                      int(0 <= (i // self.cols + 1) <= self.rows - 1) +
                                      int(0 <= (i % self.cols - 1) <= self.cols - 1) +
                                      int(0 <= (i % self.cols + 1) <= self.cols - 1) +
                          int(0 <= (i // self.cols - 1) <= self.rows - 1 and 0 <= i % self.cols - 1 <= self.cols - 1) +
                          int(0 <= (i // self.cols - 1) <= self.rows - 1 and 0 <= i % self.cols + 1 <= self.cols - 1) +
                          int(0 <= (i // self.cols + 1) <= self.rows - 1 and 0 <= i % self.cols - 1 <= self.cols - 1) +
                          int(0 <= (i // self.cols + 1) <= self.rows - 1 and 0 <= i % self.cols + 1 <= self.cols - 1))
                               for j in range(class_num)] for i in range(class_num)]

        self.attention_lst = np.asarray(self.attention_lst)


    def forward(self, outputs, targets):
        '''
        :param outputs: predictive results, shape: (batch_size, class_num)
        :param targets: ground truth, shape: (batch_size,)
        :return:
        '''
        batch_size, class_num = outputs.shape
        # Obtain target labels and create smooth labels
        targets_data = targets.cpu().data
        smoothed_labels = torch.zeros(size=(batch_size, class_num), device=outputs.device)

        # Fill in the remaining parts of the attention matrix
        for i in range(smoothed_labels.shape[0]):
            label = targets_data[i]
            smoothed_labels[i] = torch.from_numpy(self.attention_lst[label])

        # Fill the smooth label with a position assignment of 1.0
        smoothed_labels[torch.arange(batch_size), targets_data] = 1.0

        # Calculate loss
        log_prob = F.log_softmax(outputs, dim=1)
        att_loss = -torch.sum(log_prob * smoothed_labels) / batch_size
        ce_loss = nn.CrossEntropyLoss()(outputs, targets)
        loss_add = self.alpha * ce_loss + (1 - self.alpha) * att_loss

        return loss_add



def normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return (sp.eye(adj.shape[0]) - d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)).tocoo()


def laplacian(adj):
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1)).flatten()
    d_mat = sp.diags(row_sum)
    return (d_mat - adj).tocoo()


def gcn(adj):
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return (sp.eye(adj.shape[0]) + d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)).tocoo()


def aug_normalized_adjacency(adj):
    adj = adj + torch.eye(adj.shape[0]).cuda()
    row_sum = torch.sum(adj, dim=1)
    d_inv_sqrt = torch.flatten(torch.pow(row_sum, -0.5))
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)


def bingge_norm_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return (d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt) + sp.eye(adj.shape[0])).tocoo()


def normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return (d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)).tocoo()


def random_walk_laplacian(adj):
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv = np.power(row_sum, -1.0).flatten()
    d_mat = sp.diags(d_inv)
    return (sp.eye(adj.shape[0]) - d_mat.dot(adj)).tocoo()


def aug_random_walk(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv = np.power(row_sum, -1.0).flatten()
    d_mat = sp.diags(d_inv)
    return (d_mat.dot(adj)).tocoo()


def random_walk(adj):
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv = np.power(row_sum, -1.0).flatten()
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).tocoo()


def no_norm(adj):
    adj = sp.coo_matrix(adj)
    return adj


def i_norm(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    return adj


def fetch_normalization(type):
    switcher = {
        'NormLap': normalized_laplacian,  # A' = I - D^-1/2 * A * D^-1/2
        'Lap': laplacian,  # A' = D - A
        'RWalkLap': random_walk_laplacian,  # A' = I - D^-1 * A
        'FirstOrderGCN': gcn,  # A' = I + D^-1/2 * A * D^-1/2
        'AugNormAdj': aug_normalized_adjacency,
        # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2          !!!!!!!!!!!!!!!!!!!!
        'BingGeNormAdj': bingge_norm_adjacency,  # A' = I + (D + I)^-1/2 * (A + I) * (D + I)^-1/2
        'NormAdj': normalized_adjacency,  # D^-1/2 * A * D^-1/2
        'RWalk': random_walk,  # A' = D^-1*A
        'AugRWalk': aug_random_walk,  # A' = (D + I)^-1*(A + I)
        'NoNorm': no_norm,  # A' = A
        'INorm': i_norm,  # A' = A + I
    }
    func = switcher.get(type, lambda: "Invalid normalization technique.")
    return func


def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def cal_Parameters(net, name):
    '''
    :param net: input neural network
    :return: number of parameters of network
    '''
    total = sum([param.nelement() for param in net.parameters()])
    print(f"Number of parameters of {name}: {total / 1e6:.2f}M")


def norm_Data(X):
    '''
    :param X: input EEG Data
    :return: Normalizaed EEG
    '''
    X = X.numpy()
    Nh = X.shape[0]
    Nc = X.shape[2]
    for h in range(Nh):
        for c in range(Nc):
            X[h, 0, c, :] = (X[h, 0, c, :] - np.mean(X[h, 0, c, :])) / np.std(X[h, 0, c, :])

    norm_X = torch.from_numpy(X)
    return norm_X


def filter_Data(X, Fs, low, high, N=4, type="bandpass"):
    '''
    :param X: the eeg data to be filtered
    :param Fs: sample rate
    :param low: lowest bound frequency
    :param high: highest bound frequency
    :param N: filter-order
    :return: filtered eeg data
    '''
    b, a = 0, 0
    if type == "bandpass":
        b, a = signal.butter(N, [2 * low / Fs, 2 * high / Fs], type)

    elif type == "highpass":
        b, a = signal.butter(N, 2 * low / Fs, type)

    elif type == "lowpass":
        b, a = signal.butter(N, 2 * high / Fs, type)

    X = signal.filtfilt(b, a, X, axis=-1)

    return X


def get_Template_Signal(X, Nf):
    '''
    :param X: input eeg data (Nh × 1 × Nc × Nt)
    :param Nf: number of flicker stimulus
    :return: template eeg signal (Nf × 1 × Nc × Nt)
    '''
    X = X.numpy()
    reference_signals = []
    num_per_cls = X.shape[0] // Nf
    for cls_num in range(Nf):
        reference_f = X[cls_num * num_per_cls:(cls_num + 1) * num_per_cls]
        reference_f = np.mean(reference_f, axis=0)
        reference_signals.append(reference_f)
    reference_signals = np.asarray(reference_signals)
    reference_signals = torch.from_numpy(reference_signals)
    return reference_signals


def EEG_Data_Segment(eeg_data, label_data, Ns):
    '''
    :param eeg_data: input eeg data
    :param label_data: input label data
    :param Ns: number of segmented sample points
    :return: segmented eeg data
    '''
    Nh = eeg_data.shape[0]
    Nc = eeg_data.shape[2]
    Nt = eeg_data.shape[3]
    num_segments = Nt // Ns
    segment_eeg_data = torch.zeros((Nh * num_segments, 1, Nc, Ns))
    segment_label_data = torch.zeros((Nh * num_segments, 1))
    for h in range(Nh):
        for s in range(num_segments):
            segment_eeg_data[h * num_segments + s, :, :, :] = eeg_data[h, :, :, s * Ns: (s + 1) * Ns]
            segment_label_data[h * num_segments + s, :] = label_data[h]

    print("segment success!")
    return segment_eeg_data, segment_label_data

def normalize_A(A, device='cuda', symmetry=False):
    A = F.relu(A).to(device)
    if symmetry:
        A = A + torch.transpose(A, 0, 1)
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    else:
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    return L


def generate_cheby_adj(A, K,device='cuda'):
    support = []
    for i in range(K):
        if i == 0:
            support.append(torch.eye(A.shape[1]).to(device))
        elif i == 1:
            support.append(A.to(device))
        else:
            temp = torch.matmul(support[-1].cuda(), A.to(device))
            support.append(temp.to(device))
    return support


def randomedge_drop(percent, adj, normalization):
    """
    Randomly drop edge and drop percent% edges.
    """
    "Opt here"
    coo_a = adj.to_sparse()
    indices = coo_a._indices()
    nnz = coo_a._nnz()
    perm = torch.randperm(nnz)
    preserve_nnz = int(nnz * (1.-percent))
    perm = perm[:preserve_nnz]
    i = indices[:, perm]
    v = coo_a._values()[perm]
    r_adj = torch.sparse_coo_tensor(i, v, adj.shape).to_dense()
    r_adj = preprocess_adj(normalization, r_adj)
    return r_adj


def preprocess_adj(normalization, adj):
    adj_normalizer = fetch_normalization(normalization)
    r_adj = adj_normalizer(adj)
    return r_adj
