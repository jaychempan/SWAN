

from utils import load_from_npy

l_f = load_from_npy('./local_feature/rsitmd_local.npy')[()]

print(l_f.shape)
