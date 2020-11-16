
from typing import NamedTuple, Optional, List, Any

import torch
import torch_struct

import numpy as np

START = np.ones((1, 768)) * 10

class Pack(dict):
    def __getattr__(self, name):
        return self[name]

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def copy(self):
        pack = Pack()
        for k, v in self.items():
            if type(v) is list:
                pack[k] = list(v)
            else:
                pack[k] = v
        return pack

class Alignment(NamedTuple):
    alignment: np.array
    consensus: Optional[np.array]
    gaps: Optional[np.array]
    dists: Optional[np.array]
    ys: np.array
    cols: np.array


class AlignmentInfo(NamedTuple):
    # algo keys
    algo: str
    subalgo: Optional[str]
    G: float
    Gx: float
    Gz: float
    Gxz: float

    # alignment keys
    alignment: np.array
    ys: np.array
    consensus: Optional[np.array]
    gaps: Optional[np.array]
    cols: Optional[np.array]

    # distance keys
    steiner_dists: np.array
    sp_dists: Optional[np.array]
    star_dists: Optional[np.array]
    pairwise_dists: Optional[np.array]

    # multiple alignment
    var: np.array
    sse: np.array

    # meta info
    cluster: np.array
    score_history: List[float]

def maybe_np(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu()
    return np.array(x) if x is not None else None

def convert_align(
    x, G, Gx, Gz, Gxz, cluster, score_history, algo, subalgo,
    star_dists=None,
    pairwise_dists=None,
):
    var, sse = col_var(x.ys, x.gaps)
    sp_dists = SP(x.ys, x.gaps, G)
    return AlignmentInfo(
        algo = algo,
        subalgo = subalgo,
        G = G,
        Gx = Gx,
        Gz = Gz,
        Gxz = Gxz,
        alignment = maybe_np(x.alignment),
        ys = np.array(x.ys),
        consensus = maybe_np(x.consensus),
        gaps = maybe_np(x.gaps),
        cols = maybe_np(x.cols),
        steiner_dists = maybe_np(x.dists),
        sp_dists = sp_dists,
        star_dists = star_dists,
        pairwise_dists = pairwise_dists,
        var = maybe_np(var),
        sse = maybe_np(sse),
        cluster = cluster,
        score_history = score_history,
    )


def alignment_seq_hard_single(alignment, x, gaps):
    # TODO: loop over batches to handle length mismatches, or use padding
    N = alignment.shape[0]
     
    # x: T2 x H

    # build potentials
    # match_scores: batch x length x length
    # expand alignment and x to N x T1 x T2 x H
    # then sum over H and N.
    match_scores = -(alignment[:,:,None] - x[None,None]).pow(2).sum(-1)
    # account for aligning to existing gaps
    if G != 0:
        match_scores[gaps[:,:,None].expand_as(match_scores).bool()] = -G
    all_match_scores = match_scores.sum(0)
    pots = all_match_scores[:,:,None].repeat(1,1,3)

    # account for new gaps
    if G != 0:
        # new gaps in alignment must align with new symbols,
        # meaning a new symbol aligns to N gaps
        pots[:,:,0] = -N*G

        # gaps in sequence could align to gaps (no penalty),
        # but we only want to penalize introducing NEW gaps
        pots[:,:,2] = -(N-gaps.sum(0)[:,None]) * G

    # no padding needed because no batching.
    T1, T2 = all_match_scores.shape
    print(T1, T2)
    if T1 > T2:
        pots = pots.transpose(0,1)
    # get paths
    # unsqueeze Batch dim
    dist = ts.AlignmentCRF(pots.to(device)[None])
    # WARNING: FAILS IF min(T1, T2) < 5 / length mismatch is too large
    #print(dist.max)
    path = dist.argmax.squeeze(0)
    if T1 > T2:
        path = path.transpose(0,1).contiguous()

    # build new alignment
    parts = path.nonzero()

    parts_a = parts[:, 0]
    parts_b = parts[:, 1]
    gap_info = parts[:,2]

    a = alignment[:, parts_a]
    b = x[None, parts_b]

    alignment = torch.cat([a,b], 0)
    alignment_string = torch.cat([
        alignment_string[:,parts_a],
        parts_b[None],
    ], 0)
    # handle new gaps
    new_gaps = torch.zeros(alignment_string.shape, dtype=torch.float)
    s = 0
    D1 = 0 if T1 <= T2 else 2
    D2 = 2 if T1 <= T2 else 0
    for t in range(len(gap_info)):
        if gap_info[t] == D1:
            new_gaps[:-1,t] = True
        else:
            new_gaps[:-1,t] = gaps[:,s]
            s += 1
        if gap_info[t] == D2:
            new_gaps[-1,t] = True
    return alignment, alignment_string, new_gaps.bool()


def steiner_dist(
    xs, z,
    xs_no_gaps=None, z_no_gaps=None, # unused
    G=200,
    MAX_SEQ_LEN=25,
    device=torch.device("cuda:0"),
):
    raise NotImplementedError("Deprecated")

    if xs_no_gaps or z_no_gaps:
        raise NotImplementedError

    # z: T1 x H
    # xs: [Tx x H] (N)
    N = len(xs)
    lz = z.shape[0]

    transposed = np.zeros(N, dtype=np.bool)
     
    # build potentials
    # match_scores: batch x length x length
    # expand alignment and x to N x T1 x T2 x H
    # then sum over H and N.
    #pots = torch.empty(N, 2*MAX_SEQ_LEN, 2*MAX_SEQ_LEN, 3).fill_(-1e5)
    pots = torch.empty(N, 2*MAX_SEQ_LEN, 5*MAX_SEQ_LEN, 3).fill_(-1e5)

    for n in range(N):
        lx = xs[n].shape[0]
        match_scores = -(xs[n][:,None] - z[None]).pow(2).sum(-1)

        if lx > lz:
            l1 = lz
            l2 = lx
            match_scores = match_scores.transpose(0,1)
            transposed[n] = True
        else:
            l1 = lx
            l2 = lz

        pots[n, :l1, :l2, 1] = match_scores
        if G != 0:
            pots[n, :l1, :l2, 0] = -G
            pots[n, :l1, :l2, 2] = -G
        else:
            pots[n, :l1, :l2, 0] = match_scores
            pots[n, :l1, :l2, 2] = match_scores

        # padding
        pots[n, l1-1, l2:, 0] = 0
        pots[n, l1:, -1, 2] = 0

    dist = torch_struct.AlignmentCRF(pots.to(device))

    # unnormalized sum of squared euclidean distances along path
    dists = -dist.max.cpu()
    # one hot paths
    paths = dist.argmax.cpu()

    parts_list = []
    for n in range(N):
        GAP = 2 if transposed[n] else 0
        lx = xs[n].shape[0]
        path = paths[n]
        if transposed[n]:
            path = path.transpose(0, 1)
        parts = path.nonzero()
        cutoff = min(
            i for i, (x, y, d) in enumerate(parts.tolist())
            if not (x < lx and y < lz)
        )
        parts_list.append(parts[:cutoff])

    # find alignment positions
    # first compute number of aligned elements for each index in consensus sequence
    max_counts = np.zeros(lz, dtype=np.int32)
    counts = np.zeros(lz, dtype=np.int32)
    for parts in parts_list:
        counts.fill(0)
        for idx in parts[:,1]:
            counts[idx] += 1
        max_counts = np.maximum(max_counts, counts)
    starts = max_counts.cumsum()
    L = max_counts.sum()

    alignments = np.empty((N, L), dtype=np.int32)
    alignments.fill(-1) # initialize to gaps
    repeat_alignments = np.empty((N, L), dtype=np.int32)
    no_gaps = np.empty((N, L), dtype=np.bool)
    no_gaps.fill(False)
    for n, parts in enumerate(parts_list):
        is_gap = 0 if not transposed[n] else 2
        last_zi = 0
        x_idx = 0
        start_idx = 0
        for (xi, zi, di) in parts.numpy():
            if zi != last_zi:
                # track x_idx
                x_idx = starts[start_idx]
                start_idx += 1
            repeat_alignments[n,x_idx] = xi
            if di != is_gap:
                alignments[n,x_idx] = xi
                x_idx += 1
            last_zi = zi
    no_gaps = alignments >= 0
    #import pdb; pdb.set_trace()

    H = z.shape[-1]
    ys = torch.zeros(N, L, H)
    alignments_torch = torch.tensor(alignments)
    no_gaps_torch = torch.tensor(no_gaps)

    for n in range(N):
        ys[n] = xs[n][alignments[n]]
    # zero out gap representations
    ys[~no_gaps_torch] = 0

    """
    # deprecated
    alignments = []
    no_gaps = []
    for parts in parts_list:
        alignments.append([x[0].item() for x in parts if x[0] < lx and x[1] < lz])
        no_gaps.append([x[2].item() != GAP for x in parts if x[0] < lx and x[1] < lz])

    L = max(len(y) for y in alignments)
    H = z.shape[-1]
    ys = torch.zeros(N, L, H)
    alignments_torch = torch.empty(N, L, dtype=torch.int32).fill_(-1)
    no_gaps_torch = torch.zeros(N, L, dtype=torch.bool)
    for n in range(N):
        lx = len(alignments[n])
        ys[n, :lx] = xs[n][alignments[n]]
        notagap = torch.tensor(no_gaps[n], dtype=torch.bool)
        ys[n, :lx][~notagap] = 0 # zero out gaps
        no_gaps_torch[n, :lx] = notagap
        alignments_torch[n, :lx] = torch.tensor(alignments[n], dtype=torch.int32)
    """
    return alignments_torch, ys, no_gaps_torch, dists, starts

def aggregate_ys_naive(ys, no_gaps, G=200):
    n = no_gaps.sum(0)
    z = ys.sum(0) / n[:,None]
    return z[n != 0]

def aggregate_ys_vote(ys, no_gaps, Gx=200, Gz=0, Gxz=0):
    n = no_gaps.sum(0)
    z = ys.sum(0) / n[:,None]

    se = ((ys - z[None]) ** 2).sum(-1)
    se[~no_gaps] = Gx
    gap_cost = no_gaps.sum(0) * Gz + (~no_gaps).sum(0) * Gxz

    singletons = no_gaps.sum(0) == 1
    mask = se.sum(0) < gap_cost

    var, err = col_var(ys, ~no_gaps)
    #import pdb; pdb.set_trace()

    return z[mask], mask

def aggregate_ys_lenient(ys, no_gaps, G=200):
    n = no_gaps.sum(0)
    z = ys.sum(0) / n[:,None]

    se = (ys - z[None]).pow(2).sum(-1)
    se[~no_gaps] = G
    gap_cost = no_gaps.sum(0) * G

    #singletons = no_gaps.sum(0) == 1
    mask = se.sum(0) < gap_cost
    var, err = col_var(ys, ~no_gaps)
    mask_var = var > G

    #if mask.sum() == 0:
        #import pdb; pdb.set_trace()

    return z[mask | mask_var]

def col_var(ys, gaps):
    z = ys.sum(0) / (~gaps).sum(0)[...,None]
    se = ((ys - z[None]) ** 2).sum(-1) * ~gaps
    mse = se.sum(0) / (~gaps).sum(0)
    return mse, se


def SP(ys, gaps, G):
    N, T, H = ys.shape
    xy_index = [(x,y) for x in range(N) for y in range(x+1, N)]
    index = torch.tensor(xy_index)
    dists = ((ys[index[:,0]] - ys[index[:,1]]) ** 2).sum(-1)
    flat_gaps = gaps[index]
    gaps0 = flat_gaps[:,0]
    gaps1 = flat_gaps[:,1]

    # gap-not
    dists[gaps0] = G
    dists[gaps1] = G

    # gap-gap
    dists[gaps0 & gaps1] = 0

    return dists

# assymetric steiner distance
def steiner_dist_as(
    xs, z,
    xs_no_gaps=None, z_no_gaps=None, # unused
    Gx=100, Gz=150,
    MAX_SEQ_LEN=25,
    device=torch.device("cuda:0"),
):
    if xs_no_gaps or z_no_gaps:
        raise NotImplementedError

    # z: T1 x H
    # xs: [Tx x H] (N)
    N = len(xs)
    lz = z.shape[0]

    transposed = np.zeros(N, dtype=np.bool)
     
    # build potentials
    # match_scores: batch x length x length
    # expand alignment and x to N x T1 x T2 x H
    # then sum over H and N.
    #pots = torch.empty(N, 2*MAX_SEQ_LEN, 2*MAX_SEQ_LEN, 3).fill_(-1e5)
    pots = torch.empty(N, 2*MAX_SEQ_LEN+1, 5*MAX_SEQ_LEN+1, 3).fill_(-1e5)

    for n in range(N):
        lx = xs[n].shape[0]
        match_scores = -(xs[n][:,None] - z[None]).pow(2).sum(-1)

        if lx > lz:
            l1 = lz
            l2 = lx
            g1 = Gz
            g2 = Gx
            match_scores = match_scores.transpose(0,1)
            transposed[n] = True
        else:
            l1 = lx
            l2 = lz
            g1 = Gx
            g2 = Gz

        pots[n, :l1, :l2, 1] = match_scores
        if Gx != 0:
            pots[n, :l1, :l2, 0] = -g1
            pots[n, :l1, :l2, 2] = -g2
        else:
            pots[n, :l1, :l2, 0] = match_scores
            pots[n, :l1, :l2, 2] = match_scores

        # padding
        pots[n, l1-1, l2:, 0] = 0
        pots[n, l1:, -1, 2] = 0

    dist = torch_struct.AlignmentCRF(pots.to(device))

    # unnormalized sum of squared euclidean distances along path
    dists = -dist.max.cpu()
    # one hot paths
    paths = dist.argmax.cpu()

    parts_list = []
    for n in range(N):
        GAP = 2 if transposed[n] else 0
        lx = xs[n].shape[0]
        path = paths[n]
        if transposed[n]:
            path = path.transpose(0, 1)
        parts = path.nonzero()
        cutoff = min(
            i for i, (x, y, d) in enumerate(parts.tolist())
            if not (x < lx and y < lz)
        )
        parts_list.append(parts[:cutoff])

    # break here

    # find alignment positions
    # first compute number of aligned elements for each index in consensus sequence
    max_counts = np.zeros(lz, dtype=np.int32)
    counts = np.zeros(lz, dtype=np.int32)
    for parts in parts_list:
        counts.fill(0)
        for idx in parts[:,1]:
            counts[idx] += 1
        max_counts = np.maximum(max_counts, counts)
    starts = max_counts.cumsum()
    L = max_counts.sum()

    alignments = np.empty((N, L), dtype=np.int32)
    alignments.fill(-1) # initialize to gaps
    repeat_alignments = np.empty((N, L), dtype=np.int32)
    no_gaps = np.empty((N, L), dtype=np.bool)
    no_gaps.fill(False)
    for n, parts in enumerate(parts_list):
        is_gap = 0 if not transposed[n] else 2
        last_zi = 0
        x_idx = 0
        start_idx = 0
        for (xi, zi, di) in parts.numpy():
            if zi != last_zi:
                # track x_idx
                x_idx = starts[start_idx]
                start_idx += 1
            repeat_alignments[n,x_idx] = xi
            if di != is_gap:
                alignments[n,x_idx] = xi
                x_idx += 1
            last_zi = zi
    no_gaps = alignments >= 0

    H = z.shape[-1]
    ys = torch.zeros(N, L, H)
    alignments_torch = torch.tensor(alignments)
    no_gaps_torch = torch.tensor(no_gaps)

    for n in range(N):
        ys[n] = xs[n][alignments[n]]
    # zero out gap representations
    ys[~no_gaps_torch] = 0

    return alignments_torch, ys, no_gaps_torch, dists, starts

def pairwise_dist_matrix(Xs, G):
    N = len(Xs)
    D = np.zeros((N, N), dtype=np.float32)
    pairs = [(Xs[n+1:], Xs[n]) for n in range(N-1)]
    for n, (xs, x) in enumerate(pairs):
        _, _, _, dists, _ = steiner_dist_as(xs, x, Gx=G, Gz=G)
        D[n, n+1:] = dists
    return D

