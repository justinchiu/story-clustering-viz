
import pickle
import pathlib
from textwrap import wrap

import numpy as np

import streamlit as st
import plotly.graph_objects as go

from msa_utils import Alignment, AlignmentInfo


def find_best_scores(alignments):
    # story 3167 + 5 buckets.
     
    scores = np.zeros(len(alignments), dtype=np.float32)
    centroids = np.zeros(len(alignments), dtype=np.int32)
    for i, alignment in enumerate(alignments):
        scores[i] = alignment.sp_dists.sum()
        centroids[i] = alignment.cluster[0]

    idxs = np.argsort(scores)
    bucket_width = int((len(alignments) - 1) / 5)
    bucket_idxs = []
    for start in range(1, len(alignments), bucket_width):
        b_idxs = idxs[(start <= idxs) * (idxs < start + bucket_width)]
        bucket_idxs.append(b_idxs)

    # big bucket first
    bucket_idxs = list(reversed(bucket_idxs))

    # sort alignments by ascending sp score within buckets
    #return [alignments[0]] + [alignments[i] for bidxs in bucket_idxs for i in bidxs[1:3]]
    return [alignments[0]] + [alignments[i] for i in bucket_idxs[0][1:20]]

def intro():
    st.markdown("# Multiple Sequence Alignment")
    st.write("We present examples of multiple sequence alignments obtained from our progressive alignment algorithm. The rows of each heatmap correspond to stories in a cluster, and each column corresponds to an index in the multiple alignment. The color of cell shows the original index of the sentence in the original story. Hover over cells to read the sentences.")
    st.write("The first sentence in all stories gives the Story ID, the distance to the centroid story, and the writing prompt.")
    st.write("We show the full text below the heatmaps, so that it is easier to compare across stories in the alignment.")

def display_alignments(alignments, prompts, sents):
    def write_alignment(cluster, alignments, dist, prompts, sents):
        N, T = alignments.shape
        table = [[None for _ in range(N)] for _ in range(T + 1)]
        last = None
        # each story is a column
        for i, col in enumerate(alignments):
            story_idx = cluster[i]
            # TODO: add this back in after bringing back steiner distance
            #table[0][i] = f"{story_idx} ({dist[order[i]]:.2f}): {prompts[story_idx]}"
            table[0][i] = f"{story_idx} ({dist[i]:,.2f}): {prompts[i]}"
            for j, row in enumerate(col):
                if row >= 0:
                    table[j+1][i] = f"{row}: {sents[i][row]}"
                else:
                    table[j+1][i] = "-"
        st.table(table)

    def get_alignment(cluster, alignments, dist, prompts, sents):
        N, T = alignments.shape
        table = [[None for _ in range(T+1)] for _ in range(N)]
        last = None
        # each story is a column
        for i, col in enumerate(alignments):
            story_idx = cluster[i]
            # TODO: add this back in after bringing back steiner distance
            #table[0][i] = f"{story_idx} ({dist[order[i]]:.2f}): {prompts[story_idx]}"
            table[i][0] = f"{story_idx} ({dist[i]:,.2f}): {'<br>'.join(wrap(prompts[i]))}"
            for j, row in enumerate(col):
                if row >= 0:
                    table[i][j+1] = f"{'<br>'.join(wrap(sents[i][row]))}"
                else:
                    table[i][j+1] = "-"
        return table

    align_idx = st.slider("Use this slider to choose the centroid story", 0, len(alignments)-1, 0)

    x = alignments[align_idx]
    ps = prompts[align_idx]
    ss = sents[align_idx]

    st.write(f"### Centroid Story: {x.cluster[0]} (Steiner distance {x.steiner_dists.sum().item():,.1f} | SP score {x.sp_dists.sum():,.1f})")

    x.alignment[x.gaps] = -1

    # Plot alignment heatmap with pyplot
    N, T = x.alignment.shape

    # Plot alignment heatmap with plotly
    hover = get_alignment(x.cluster, x.alignment[:,1:]-1, x.star_dists, ps, ss)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z = x.alignment,
        type = "heatmap",
        colorscale = "viridis",
        text = hover,
        colorbar=dict(title="Original Index"),
        hovertemplate = "Story: %{y} | Index: %{x}<br>%{text}<extra></extra>",
    ))
    layout = go.Layout(
        xaxis = go.layout.XAxis(
            title = go.layout.xaxis.Title(
                text='Index',
            ),
        ),
        yaxis=go.layout.YAxis(
            title = go.layout.yaxis.Title(
                text='Story',
            )
        )
    )
    fig.update_layout(layout)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    write_alignment(x.cluster, x.alignment[:,1:]-1, x.star_dists, ps, ss)


def viz(alignments, prompts, sents):
    st.set_page_config(
        # Alternate names: setup_page, page, layout
        layout="wide",
    )
    intro()
    st.markdown("## Progressive alignments with G=125")
    display_alignments(alignments, prompts, sents)

def _resave(alignments):
    data_dir = "viz_data/"
    prompts = open(data_dir + "train.wp_source", "r").readlines()
    story2sent = pickle.load(open(data_dir + "story2sent.0.pkl", "rb"))
    sents = open(data_dir + "sent_text.0.txt", "r").readlines()

    alignments = find_best_scores(alignments)
    # convert sp_dists from torch to numpy (mistake in earlier code)
    for i, alignment in enumerate(alignments):
        alignments[i] = alignment._replace(sp_dists=alignment.sp_dists.numpy())

    # get prompts
    a_prompts = [
        [prompts[story_idx] for story_idx in alignment.cluster]
        for alignment in alignments
    ]
    # get sentences
    a_sents = [
        [
            [sents[x] for x in story2sent[story_idx]]
            for story_idx in alignment.cluster
        ]
        for alignment in alignments
    ]

    # resave
    out_dir = "viz_data_small/"
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    with open(out_dir + "alignments.pkl", "wb") as f:
        pickle.dump(alignments, f)
    with open(out_dir + "prompts.pkl", "wb") as f:
        pickle.dump(a_prompts, f)
    with open(out_dir + "sents.pkl", "wb") as f:
        pickle.dump(a_sents, f)

    return alignments, a_prompts, a_sents


if __name__ == "__main__":
    # only run this if working from large data in viz_data.
    #alignments = load_stuff("viz_data/alignments.pkl")
    #alignments, prompts, sents = _resave(alignments)

    out_dir = "viz_data_small/"
    with open(out_dir + "alignments.pkl", "rb") as f:
        alignments = pickle.load(f)
    with open(out_dir + "prompts.pkl", "rb") as f:
        prompts = pickle.load(f)
    with open(out_dir + "sents.pkl", "rb") as f:
        sents = pickle.load(f)

    viz(alignments, prompts, sents)
