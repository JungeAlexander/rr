import httpx
import numpy as np
import pandas as pd
import streamlit as st
import tenacity
import umap

from bokeh.models import CategoricalColorMapper, ColumnDataSource, HoverTool
from bokeh.palettes import Spectral10
from bokeh.plotting import figure, output_notebook, show
from sklearn.metrics.pairwise import cosine_similarity


st.title("Paper recommender")

with st.expander("What is this? :thinking_face: "):
    st.markdown(
        "See [this blog post](https://www.alexanderjunge.net/blog/s2-recommender/) for more details."
    )
doi = "10.1101/444398"
text_query = "distant supervision biomedical text mining"

# doi = "10.1016/j.mce.2018.04.002"
# text_query = "gip glp1 obesity diabetes"

doi = st.text_input("DOI", doi)
num_papers = st.number_input("Number of papers", min_value=1, max_value=100, value=10)


# TODO actually select these from initial list of papers
# liked_ids = [
#     "0824e6f75e18325a79b11e3e4a118409e3297f97",
#     "d0c5f901868f6e2cb126fd51b155f631372a9669",
# ]
# disliked_ids = [
#     "13d51ef5fbf4447bd9d58283387f1610a4fcfce4",
#     "b6407952a59dd1664e44e3a6336f91e8599aa30f",
#     "b9c8fed348084c5b31722f5433ea805299d006aa",
#     "48de1a31cca6631bd73a5d0854acfda5e5195d66",
#     "0e908cfd65ebd60c690e2aabcb9e0d67bdcbfb81",
# ]


def print_paper(paper: dict):
    print(paper["title"])
    print(paper["url"])
    print(paper["abstract"])
    print(paper["authors"])


@tenacity.retry(
    wait=tenacity.wait.wait_exponential(multiplier=1, min=4, max=10),
    stop=tenacity.stop_after_attempt(3),
    retry=tenacity.retry_if_exception_type(httpx.HTTPError),
)
def make_request(url, type, json=None, timeout=50.0):
    if type == "get":
        r = httpx.get(url, timeout=timeout)
    elif type == "post":
        r = httpx.post(url, json=json, timeout=timeout)
    j = r.json()
    try:
        r.raise_for_status()
    except httpx.HTTPError as exc:
        st.experimental_show(json)
        st.experimental_show(j)
        print(
            f"Error response {exc.response.status_code} while requesting {exc.request.url!r}."
        )
        raise exc
    return j


def get_paper_batch_info(ids):
    payload = {"ids": ids}
    return make_request(
        url="https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,abstract,authors,year,venue,embedding,tldr",
        type="post",
        json=payload,
        timeout=50.0,
    )


def print_paper_info(paper_ids, paper_info_df):
    for i in paper_ids:
        print(i)
        print(paper_info_df.loc[paper_info_df.loc[:, "id_"] == i, "title"])
        print()


def parse_paper_batch_info(json_response, query_paper_id):
    id_to_vector = {}
    id_info = []
    for p in json_response:
        id_ = p["paperId"]
        assert p["embedding"]["model"] == "specter@v0.1.1"
        id_to_vector[id_] = p["embedding"]["vector"]
        tldr = p["tldr"]["text"] if p["tldr"] is not None else ""
        id_info.append(
            [
                False,
                False,
                p["title"],
                p["year"],
                p["venue"],
                p["abstract"],
                tldr,
                [a["name"] for a in p["authors"]],
                id_,
            ]
        )
    id_info_df = pd.DataFrame(
        id_info,
        columns=[
            "like",
            "dislike",
            "title",
            "year",
            "venue",
            "abstract",
            "tldr",
            "authors",
            "id_",
        ],
    )
    # always like the query paper
    id_info_df.loc[id_info_df.loc[:, "id_"] == query_paper_id, "like"] = True
    return id_info_df, id_to_vector


def assert_ids_in_vector(id_to_vector, ids):
    for i in ids:
        if i not in id_to_vector:
            st.experimental_show(ids)
            st.experimental_show(id_to_vector.keys())
            raise ValueError(f"{i} not in id_to_vector")


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
        factors=["query", "liked", "disliked", "recommended", "other"],
    )
    plot_figure.circle(
        "x",
        "y",
        source=datasource,
        size=20,
        color={"field": "type", "transform": color_map},
        alpha=0.5,
        legend_group="type"
    )
    return plot_figure


j1 = make_request(
    url=f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=title,url,abstract,authors",
    type="get",
)
paper_id = j1["paperId"]

j2 = make_request(
    url=f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{paper_id}?limit={num_papers}&fields=title,url,abstract,authors",
    type="get",
)
ids = [p["paperId"] for p in j2["recommendedPapers"]] + [paper_id]
j3 = get_paper_batch_info(ids)
id_info_df, id_to_vector = parse_paper_batch_info(j3, paper_id)


id_info_df_selected = st.experimental_data_editor(id_info_df)
liked_ids = id_info_df_selected.loc[id_info_df_selected["like"], "id_"].values
disliked_ids = id_info_df_selected.loc[id_info_df_selected["dislike"], "id_"].values

st.header("Recommendations :sparkles: :sparkles: :sparkles:")
query_term = st.text_input("Text query", text_query)
n = st.number_input("Number of recommendations", min_value=1, max_value=100, value=5)
go = st.button("GO")
if not go:
    st.stop()

attractor_ids = list(liked_ids)
detractor_ids = list(disliked_ids)

if not attractor_ids:
    st.warning("No liked papers - try liking at least one paper.")
    st.stop()

if not detractor_ids:
    st.warning("No disliked papers - try disliking at least one paper.")
    st.stop()

if set(attractor_ids).intersection(detractor_ids):
    st.warning("Make sure to either like or dislike each paper.")
    st.stop()

assert_ids_in_vector(id_to_vector, attractor_ids)
assert_ids_in_vector(id_to_vector, detractor_ids)

embed_dim = 768
attractor_ids_mat = embed_matrix(attractor_ids, id_to_vector, embed_dim)
detractor_ids_mat = embed_matrix(detractor_ids, id_to_vector, embed_dim)
assert attractor_ids_mat.shape == (len(attractor_ids), embed_dim)
assert detractor_ids_mat.shape == (len(detractor_ids), embed_dim)

j4 = make_request(
    url=f"https://api.semanticscholar.org/graph/v1/paper/search?query={query_term}&fields=title,url,abstract,authors&limit=30",
    type="get",
)

new_query_ids = [p["paperId"] for p in j4["data"]]
j5 = get_paper_batch_info(new_query_ids)
new_query_id_df, new_query_id_to_vector = parse_paper_batch_info(j5, paper_id)

new_query_ids_mat = embed_matrix(new_query_ids, new_query_id_to_vector, embed_dim)
assert new_query_ids_mat.shape == (len(new_query_ids), embed_dim)

attractor_d = d(new_query_ids_mat, attractor_ids_mat)
detractor_d = d(new_query_ids_mat, detractor_ids_mat)
loss = attractor_d.min(axis=1) - detractor_d.min(axis=1)

keep_query_id_idx = loss.argsort()
keep_query_ids = []
for i in keep_query_id_idx:
    new_id = new_query_ids[i]
    if (
        (new_id != paper_id)
        and (new_id not in liked_ids)
        and (new_id not in disliked_ids)
    ):
        keep_query_ids.append(new_id)
    if len(keep_query_ids) >= n:
        break

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
        id_types.append("recommended")
    else:
        id_types.append("other")
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
st.markdown("### All papers:")
st.dataframe(vis_df)
