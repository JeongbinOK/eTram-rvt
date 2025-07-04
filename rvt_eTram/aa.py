import numpy as np

npz = np.load(
    "/home/oeoiewt/eTraM/rvt_eTram/data/gen_processed/train/train_day_0001/labels_v2/labels.npz",
    allow_pickle=True,
)
print("NPZ keys :", list(npz.keys()))  # ['labels', 'objframe_idx_2_label_idx']

labels_arr = npz["labels"]  # structured array
print(
    "dtype of labels:", labels_arr.dtype
)  # e.g. [('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('cls', '<u1')]
print("first row :", labels_arr[0])
