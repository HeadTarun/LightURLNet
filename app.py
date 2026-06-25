import pandas as pd
import streamlit as st

from inference import predict_url


st.set_page_config(page_title="LightURLNet URL Safety", page_icon="search", layout="centered")

st.title("LightURLNet URL Safety")
st.caption("Run local ONNX inference to classify a URL as safe or dangerous.")

url = st.text_input("URL", placeholder="https://example.com/login")
scan = st.button("Scan URL", type="primary", use_container_width=True)

if scan:
    try:
        result = predict_url(url)
    except Exception as exc:
        st.error(str(exc))
    else:
        is_danger = result["label"] == "Danger"
        st.metric("Verdict", result["label"], f"{result['confidence']:.1%} confidence")

        if is_danger:
            st.error("Dangerous URL detected")
        else:
            st.success("URL appears safe")

        st.progress(result["danger_probability"], text="Danger probability")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Signal": "Safe", "Probability": result["safe_probability"]},
                    {"Signal": "Danger", "Probability": result["danger_probability"]},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

st.divider()
st.subheader("Batch test")
batch_urls = st.text_area(
    "URLs, one per line",
    placeholder="https://example.com\nhttp://suspicious-login.example/path",
    height=120,
)

if st.button("Scan Batch", use_container_width=True):
    rows = []
    for line in batch_urls.splitlines():
        if not line.strip():
            continue
        try:
            item = predict_url(line)
            rows.append(
                {
                    "URL": item["url"],
                    "Verdict": item["label"],
                    "Safe": item["safe_probability"],
                    "Danger": item["danger_probability"],
                    "Confidence": item["confidence"],
                }
            )
        except Exception as exc:
            rows.append({"URL": line.strip(), "Verdict": f"Error: {exc}"})

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.warning("Add at least one URL.")
