import bokeh
import httpx
import numpy as np
import pandas as pd
import streamlit as st
import umap

from bokeh.models import CategoricalColorMapper, ColumnDataSource, HoverTool
from bokeh.palettes import Spectral10
from bokeh.plotting import figure, output_notebook, show
from sklearn.metrics.pairwise import cosine_similarity


st.title("S2 API")

doi = st.text_input("DOI", "10.1101/444398")
num_papers = st.number_input("Number of papers", min_value=1, max_value=100, value=30)
query_term = st.text_input("Text query", "distant supervision biomedical text mining")
n = st.number_input("Number of recommendations", min_value=1, max_value=100, value=5)

# TODO actually select these from initial list of papers
liked_ids = [
    "0824e6f75e18325a79b11e3e4a118409e3297f97",
    "d0c5f901868f6e2cb126fd51b155f631372a9669",
]
disliked_ids = [
    "13d51ef5fbf4447bd9d58283387f1610a4fcfce4",
    "b6407952a59dd1664e44e3a6336f91e8599aa30f",
    "b9c8fed348084c5b31722f5433ea805299d006aa",
    "48de1a31cca6631bd73a5d0854acfda5e5195d66",
    "0e908cfd65ebd60c690e2aabcb9e0d67bdcbfb81",
]


def print_paper(paper: dict):
    print(paper["title"])
    print(paper["url"])
    print(paper["abstract"])
    print(paper["authors"])


def get_paper_batch_info(ids):
    payload = {"ids": ids}
    r = httpx.post(
        "https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,abstract,authors,year,venue,embedding,tldr",
        json=payload,
        timeout=50.0,
    )
    return r.json()


def print_paper_info(paper_ids, paper_info_df):
    for i in paper_ids:
        print(i)
        print(paper_info_df.loc[paper_info_df.loc[:, "id_"] == i, "title"])
        print()


def parse_paper_batch_info(json_response):
    id_to_vector = {}
    id_info = []
    for p in json_response:
        id_ = p["paperId"]
        assert p["embedding"]["model"] == "specter@v0.1.1"
        id_to_vector[id_] = p["embedding"]["vector"]
        tldr = p["tldr"]["text"] if p["tldr"] is not None else ""
        id_info.append(
            [
                id_,
                p["title"],
                p["year"],
                p["venue"],
                p["abstract"],
                tldr,
                [a["name"] for a in p["authors"]],
            ]
        )
    id_info_df = pd.DataFrame(
        id_info,
        columns=["id_", "title", "year", "venue", "abstract", "tldr", "authors"],
    )
    return id_info_df, id_to_vector


def embed_matrix(ids, id_to_vector, embed_dim):
    return np.fromiter(
        (id_to_vector[i] for i in ids), dtype=np.dtype((float, embed_dim))
    )


def embed_umap(in_df):
    reducer = umap.UMAP(random_state=42)
    reducer.fit(in_df)
    out_df = pd.DataFrame(reducer.transform(in_df))
    out_df.columns = ["x", "y"]
    return out_df


def d(a, m):
    return 1 - cosine_similarity(a, m)


def bokeh_vis(in_df):
    datasource = ColumnDataSource({str(c): v.values for c, v in in_df.items()})
    tooltips = [
        ("(x,y)", "($x, $y)"),
        ("id", "@id_"),
        ("title", "@title"),
    ]
    plot_figure = figure(
        title="UMAP projection of papers",
        width=800,
        height=800,
        tooltips=tooltips,
    )
    color_map = CategoricalColorMapper(
        palette=["black", "orange", "blue", "red", "grey"],
        factors=["query", "liked", "disliked", "recommend", "value"],
    )
    plot_figure.circle(
        "x",
        "y",
        source=datasource,
        size=20,
        color={"field": "type", "transform": color_map},
        alpha=0.5,
    )
    return plot_figure


r1 = httpx.get(
    f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=title,url,abstract,authors"
)
j1 = r1.json()
paper_id = j1["paperId"]

r2 = httpx.get(
    f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{paper_id}?limit={num_papers}&fields=title,url,abstract,authors"
)
j2 = r2.json()

ids = [p["paperId"] for p in j2["recommendedPapers"]] + [paper_id]
j3 = get_paper_batch_info(ids)
id_info_df, id_to_vector = parse_paper_batch_info(j3)

attractor_ids = liked_ids + [paper_id]
detractor_ids = disliked_ids.copy()
assert all((i in id_to_vector for i in attractor_ids))
assert all((i in id_to_vector for i in detractor_ids))

embed_dim = 768
attractor_ids_mat = embed_matrix(attractor_ids, id_to_vector, embed_dim)
detractor_ids_mat = embed_matrix(detractor_ids, id_to_vector, embed_dim)
assert attractor_ids_mat.shape == (len(attractor_ids), embed_dim)
assert detractor_ids_mat.shape == (len(detractor_ids), embed_dim)

r4 = httpx.get(
    f"https://api.semanticscholar.org/graph/v1/paper/search?query={query_term}&fields=title,url,abstract,authors&limit=50"
)
j4 = r4.json()

new_query_ids = [p["paperId"] for p in j4["data"]]
j5 = get_paper_batch_info(new_query_ids)
new_query_id_df, new_query_id_to_vector = parse_paper_batch_info(j5)

new_query_ids_mat = embed_matrix(new_query_ids, new_query_id_to_vector, embed_dim)
assert new_query_ids_mat.shape == (len(new_query_ids), embed_dim)

attractor_d = d(new_query_ids_mat, attractor_ids_mat)
detractor_d = d(new_query_ids_mat, detractor_ids_mat)
loss = attractor_d.min(axis=1) - detractor_d.min(axis=1)

keep_query_id_idx = loss.argsort()[:n]
keep_query_ids = [
    new_query_ids[i] for i in keep_query_id_idx
]

vis_df = pd.concat([id_info_df, new_query_id_df])
vis_df = vis_df.drop_duplicates(subset="id_")
vis_df.reset_index(drop=True, inplace=True)

id_types = []
for i in vis_df.loc[:, "id_"]:
    if i == paper_id:
        id_types.append("query")
    elif i in liked_ids:
        id_types.append("liked")
    elif i in disliked_ids:
        id_types.append("disliked")
    elif i in keep_query_ids:
        id_types.append("recommend")
    else:
        id_types.append("value")
vis_df.insert(0, "type", id_types)

ids = vis_df.loc[:, "id_"].tolist()
embeddings = []
for i in ids:
    if i in id_to_vector:
        embeddings.append(id_to_vector[i])
    elif i in new_query_id_to_vector:
        embeddings.append(new_query_id_to_vector[i])
embed_df = pd.DataFrame(embeddings, index=ids)
embed_df_t = embed_umap(embed_df)
embed_df_t["id_"] = ids
df_t = embed_df_t.merge(vis_df, on="id_", validate="one_to_one")

st.bokeh_chart(bokeh_vis(df_t), use_container_width=True)
st.dataframe(vis_df)