import streamlit as st

# Center the content using HTML and Markdown
st.markdown(
    """
    <div style="display: flex; justify-content: center; align-items: center; height: 80vh; text-align: center;">
        <div>
            <h1 style="font-size: 3em;">Handwritten Digit Classifier</h1>
            <p style="font-size: 1.5em;">Welcome!</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
