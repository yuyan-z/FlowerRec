import streamlit as st

from query_pipeline import load_local_dataset, load_collection, do_query, format_query_result
from rag_pipeline import ResponseGenerator


DEFAULT_QUERY = "soft pink and white flowers for Mother's Day"


@st.cache_resource
def init_resources():
    ds = load_local_dataset()
    collection = load_collection()
    return ds, collection

st.title("Flower Recommender System")

model_name = st.selectbox("Choose a model", ["llava", "gpt-4o"])

query_text = st.text_area("Enter your query", placeholder=DEFAULT_QUERY)

if st.button("Start"):
    if not query_text.strip():
        query_text = DEFAULT_QUERY

    status = st.status("Starting...", expanded=True)

    try:
        status.update(label="Loading resources...", state="running")
        ds, collection = init_resources()

        status.update(label="Querying database...", state="running")
        result = do_query(collection, query_text, 2)
        result_formatted = format_query_result(ds, query_text, result)

        status.update(label=f"Generating response...", state="running")
        generator = ResponseGenerator(model_name)
        response = generator.generate_response(result_formatted)

        status.update(label="Finished", state="complete")

        # Show results
        st.subheader("📄 Results")
        st.markdown(response)

        # Show images
        if result_formatted["images"]:
            st.subheader("🖼️ Images")
            for idx, img_b64 in enumerate(result_formatted["images"]):
                st.image(f"data:image/jpeg;base64,{img_b64}", caption=f"Image {idx + 1}", use_container_width=True)

    except Exception as e:
        status.update(label=f"❌ Error：{e}", state="error")