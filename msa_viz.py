
import pickle
from textwrap import wrap

import numpy as np

import streamlit as st
#import matplotlib.pyplot as plt
import plotly.graph_objects as go

from msa_utils import Alignment, AlignmentInfo

#plt.rcParams.update({'font.size': 10})

def load_stuff(savefile):
    with open(savefile, "rb") as f:
        alignments = pickle.load(f)
    return alignments

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
    # ... manually choose all indices?
    #return [alignments[0]] + [alignments[i] for i in bucket_idxs[1]]

def sentence_lens(alignments, story2sent, sents):
    pass

def find_dense_columns(alignments):
    colcounts = np.zeros(len(alignments), dtype=np.int32)
    for i, alignment in enumerate(alignments):
        if alignment.algo == "greedy":
            gaps = alignment.gaps
            N, T = gaps.shape
            notagapsum = (~gaps).sum(0)
            counts = (notagapsum[1:] >= N-1).sum()
            colcounts[i] = counts
    sorted_idxs = np.argsort(colcounts)[::-1]
    sorted_counts = colcounts[sorted_idxs]

    k = 10
    topk_counts = sorted_counts[:k]
    topk_clusters = [alignments[idx].cluster for idx in sorted_idxs[:k]]

    st.write("Clusters with the most dense columns (K-1 nongaps)")
    for cluster, counts in zip(topk_clusters, topk_counts):
        st.write(cluster)
        st.write(f"Cols: {counts}")

def plot_column_counts(alignments0, alignments1=None):
    numcols = alignments0[0].gaps.shape[0]+1

    def get_col_counts(alignments):
        colcounts = np.zeros((len(alignments), numcols), dtype=np.int32)
        for i, alignment in enumerate(alignments):
            if alignment.algo == "greedy":
                gaps = alignment.gaps
                N, T = gaps.shape
                notagapsum = (~gaps).sum(0)
                np.add.at(colcounts[i], notagapsum, 1)
        return colcounts[:,1:]

    fig, ax = plt.subplots()
    plt.title(f"Column densities for progressive alignment")

    if alignments1 is not None:
        width=0.35
        xs = np.arange(1, numcols)
        colcounts0 = get_col_counts(alignments0)
        ax.bar(xs-width/2, colcounts0.sum(0), width, label="G=125")
        colcounts1 = get_col_counts(alignments1)
        ax.bar(xs+width/2, colcounts1.sum(0), width, label="G=150")
        ax.legend()
    else:
        colcounts0 = get_col_counts(alignments0)
        plt.bar(range(numcols), colcounts0.sum(0))

    #fig.tight_layout()
    ax.set_ylabel("Number of occurrences")
    ax.set_xlabel("Column density")
    st.pyplot(fig)
    plt.close()

    # caption
    st.write("This graph shows the column density counts for all multiple alignments obtained via progress alignment with G=125 and 150. For each column of every multiple alignment, its density is given by the number of nongap tokens in that column. Each column is of height 5. We see that the lower gap penalty results in longer alignments (number of columns increases) and less dense columns.")

def plot_column_counts_prog_hc(alignments):
    numcols = alignments[0].gaps.shape[0]+1

    def get_col_counts(alignments, algo, subalgo=None):
        colcounts = np.zeros((len(alignments), numcols), dtype=np.int32)
        for i, alignment in enumerate(alignments):
            if alignment.algo == algo and alignment.subalgo == subalgo:
                gaps = alignment.gaps
                N, T = gaps.shape
                notagapsum = (~gaps).sum(0)
                np.add.at(colcounts[i], notagapsum, 1)
        return colcounts[:,1:]

    fig, ax = plt.subplots()
    plt.title(f"Column densities for progressive alignment vs HC(IA)")

    width=0.35
    xs = np.arange(1, numcols)
    colcounts0 = get_col_counts(alignments, "greedy")
    ax.bar(xs-width/2, colcounts0.sum(0), width, label="Progressive")
    colcounts1 = get_col_counts(alignments, "mm", "starinit")
    ax.bar(xs+width/2, colcounts1.sum(0), width, label="HC(IA)")
    ax.legend()

    #fig.tight_layout()
    ax.set_ylabel("Number of occurrences")
    ax.set_xlabel("Column density")
    st.pyplot(fig)
    plt.close()

    # caption
    st.write("This graph shows the column density counts for all multiple alignments obtained via progress alignment with G=125 and Hill Climbing initialized with the iterative averaging mean sequence. For each column of every multiple alignment, its density is given by the number of nongap tokens in that column. Each column is of height 5. We see that the HC algorithm returns much shorter alignments with denser columns.")

def get_scores(alignments):
    greedy_score = 0
    mm_score = 0
    mm_star_score = 0
    star_score = 0
    for alignment in alignments:
        dist = alignment.steiner_dists.sum()
        if alignment.algo == "greedy":
            greedy_score += dist
        elif alignment.algo == "mm" and alignment.subalgo == "greedyinit":
            mm_score += dist
        elif alignment.algo == "mm" and alignment.subalgo == "starinit":
            mm_star_score += dist
        elif alignment.algo == "star":
            star_score += dist
        else:
            raise KeyError
    string = f"greedy {greedy_score:,.2f} | mm {mm_score:,.2f} | mmstar {mm_star_score:,.2f} | star {star_score:,.2f}"
    print("Full data scores: " + string)
    st.write("Full data scores")
    st.write(string)

def get_sp_scores(alignments):
    greedy_score = 0
    mm_score = 0
    mm_star_score = 0
    star_score = 0
    for alignment in alignments:
        dist = alignment.sp_dists.sum()
        if alignment.algo == "greedy":
            greedy_score += dist
        elif alignment.algo == "mm" and alignment.subalgo == "greedyinit":
            mm_score += dist
        elif alignment.algo == "mm" and alignment.subalgo == "starinit":
            mm_star_score += dist
        elif alignment.algo == "star":
            star_score += dist
        else:
            raise KeyError
    string = f"greedy {greedy_score:,.2f} | mm {mm_score:,.2f} | mmstar {mm_star_score:,.2f} | star {star_score:,.2f}"
    print("Full data sp scores: " + string)
    st.write("Full data sp scores")
    st.write(string)

def score_curves(alignments):
    curves = []
    for alignment in alignments:
        if alignment.algo == "star":
            curves.append(alignment.score_history)

    fig, ax = plt.subplots()
    plt.title(f"Score curve for consensus alignment")

    num_increases = 0
    for i, curve in enumerate(curves):
        x = np.array(curve)
        plt.plot(x)
        num_inc = (x[:-1] < x[1:]).sum().item()
        if num_inc > 0:
            num_increases += 1
            print(i)
            print(curve)
            
    st.write(f"{num_increases} / {len(curves)} examples saw curve go back up")

    #fig.tight_layout()
    ax.set_ylabel("Steiner Score")
    ax.set_xlabel("Iterations")
    st.pyplot(fig)
    plt.close()
    st.write(f"Each line in the graph corresponds to a unique cluster")

def intro():
    st.markdown("# Multiple Sequence Alignment")
    st.write("We present examples of multiple sequence alignments obtained from our progressive alignment algorithm. The rows of each heatmap correspond to stories in a cluster, and each column corresponds to an index in the multiple alignment. The color of cell shows the original index of the sentence in the original story. Hover over cells to read the sentences.")
    st.write("The first sentence in all stories gives the Story ID, the distance to the centroid story, and the writing prompt.")
    st.write("We show the full text below the heatmaps, so that it is easier to compare across stories in the alignment.")

def display_alignments(alignments, story2sent, prompts, sents, algo="greedy", subalgo=None):
    def write_alignment(cluster, alignments, dist):
        N, T = alignments.shape
        table = [[None for _ in range(N)] for _ in range(T + 1)]
        last = None
        # each story is a column
        for i, col in enumerate(alignments):
            story_idx = cluster[i]
            story_sents = story2sent[story_idx]
            # TODO: add this back in after bringing back steiner distance
            #table[0][i] = f"{story_idx} ({dist[order[i]]:.2f}): {prompts[story_idx]}"
            table[0][i] = f"{story_idx} ({dist[i]:,.2f}): {prompts[story_idx]}"
            for j, row in enumerate(col):
                if row >= 0:
                    table[j+1][i] = f"{row}: {sents[story_sents[row]]}"
                else:
                    table[j+1][i] = "-"
        st.table(table)

    def get_alignment(cluster, alignments, dist):
        N, T = alignments.shape
        table = [[None for _ in range(T+1)] for _ in range(N)]
        last = None
        # each story is a column
        for i, col in enumerate(alignments):
            story_idx = cluster[i]
            story_sents = story2sent[story_idx]
            # TODO: add this back in after bringing back steiner distance
            #table[0][i] = f"{story_idx} ({dist[order[i]]:.2f}): {prompts[story_idx]}"
            table[i][0] = f"{story_idx} ({dist[i]:,.2f}): {'<br>'.join(wrap(prompts[story_idx]))}"
            for j, row in enumerate(col):
                if row >= 0:
                    table[i][j+1] = f"{'<br>'.join(wrap(sents[story_sents[row]]))}"
                else:
                    table[i][j+1] = "-"
        return table

    for iter, x in enumerate(alignments):
        # dbg
        #if x.cluster[0] != 35266:
        #if 210226 not in cluster:
            #continue
        if x.algo != algo and x.subalgo == subalgo:
            continue

        # min length 5
        #lens = [len(story2sent[story]) < 5 for story in x.cluster]
        #if any(lens):
            #continue

        st.write(f"### Centroid Story: {x.cluster[0]} (Steiner distance {x.steiner_dists.sum().item():,.1f} | SP score {x.sp_dists.sum():,.1f})")

        #st.write(f"{x.algo} {x.subalgo} (G={x.G} Gx={x.Gx} Gz={x.Gz} Gxz={x.Gxz})")
        # augment alignment_string with gaps

        x.alignment[x.gaps] = -1
        #st.write(x.alignment.T)

        # Plot alignment heatmap with pyplot
        N, T = x.alignment.shape

        # Print this one for the paper
        #fig, ax = plt.subplots()
        #ax.imshow(x.alignment, aspect="equal")
        #ax.set_ylabel("Story")
        #ax.set_xlabel("Index")
        #ax.set_yticks([0,N-1])
        #ax.set_yticklabels([1,N])
        #plt.tight_layout()

        #st.pyplot(fig)
        #plt.close()

        # Plot alignment heatmap with plotly
        hover = get_alignment(x.cluster, x.alignment[:,1:]-1, x.star_dists)

        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z = x.alignment,
            #y = np.arange(1, x.alignment.shape[0]+1),
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
        #import pdb; pdb.set_trace()

        #st.write("Consensus string columns (only for star, figure out consensus string for greedy based on col variance)")
        #st.write(x.cols)

        #st.write("Pairwise DTW distances")
        #st.write(x.pairwise_dists)

        #st.write("Average sentence lengths")
        #average_lengths = []
        #for story_idx in x.cluster:
            #sentence_lengths = [len(sents[sent_idx].split()) for sent_idx in story2sent[story_idx]]
            #average_lengths.append(sum(sentence_lengths) / len(sentence_lengths))
        #st.write(np.array(average_lengths))

        write_alignment(x.cluster, x.alignment[:,1:]-1, x.star_dists)
        #exit()


def viz(alignments):
    st.set_page_config(
        # Alternate names: setup_page, page, layout
        layout="wide",
    )

    intro()

    alignments = find_best_scores(alignments)

    data_dir = "/home/jtc257/code/story-clustering/output/sbert_train/"

    #prompts = open("/home/jtc257/code/stories/data/writingPrompts/train.wp_source", "r").readlines()
    #story2sent = pickle.load(open(data_dir + "story2sent.0.pkl", "rb"))
    #sents = open(data_dir + "sent_text.0.txt", "r").readlines()
    prompts = open("viz_data/train.wp_source", "r").readlines()
    story2sent = pickle.load(open("viz_data/story2sent.0.pkl", "rb"))
    sents = open("viz_data/sent_text.0.txt", "r").readlines()


    st.markdown("## Progressive alignments with G=125")
    display_alignments(alignments, story2sent, prompts, sents, algo="greedy")


if __name__ == "__main__":
    # cp-ed to below
    #alignments = load_stuff("msa_output_greedy/msa_savefile.Gc125.0.G125.0.Gx50.0.Gz250.0.Gxz0.size5.min_len10.min_d130.0.n100000.k256.pkl")
    alignments = load_stuff("viz_data/alignments.pkl")
    viz(alignments)
