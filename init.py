import streamlit as st
import ollama
from PIL import Image
import io
from typing import Optional

# Constants
ACCEPTED_FORMATS = ['png', 'jpg', 'jpeg']
VISION_MODEL = 'llama3.2-vision'  # For image recognition
MATH_MODEL = 'qwen2-math:1.5b'    # For solving equations

PROMPT_TEMPLATE = """Extract and convert mathematical equations from the image provided into LaTeX code.
Rules you MUST follow:
1. Output raw LaTeX code only
2. No explanatory text
3. No dollar signs ($) or delimiters
4. No equation simplification
5. No LaTeX preamble or document structure
6. No symbol explanations"""

def init_session_state():
    """Initialize session state variables."""
    if 'ocr_result' not in st.session_state:
        st.session_state.ocr_result = None
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'solution_result' not in st.session_state:
        st.session_state.solution_result = None

def setup_page():
    """Configure the page layout and title."""
    st.set_page_config(
        page_title="LaTeX OCR & Solver",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("LaTeX OCR & Solver")

def create_clear_button():
    """Create a clear button in the top right corner."""
    _, col2 = st.columns([6, 1])
    with col2:
        if st.button("Clear"):
            st.session_state.ocr_result = None
            st.session_state.uploaded_image = None
            st.session_state.solution_result = None
            st.experimental_rerun()

def process_image(image_file) -> Optional[str]:
    """Process the uploaded image and return LaTeX code."""
    try:
        response = ollama.chat(
            model=VISION_MODEL,
            messages=[{
                'role': 'user',
                'content': PROMPT_TEMPLATE,
                'images': [image_file.getvalue()]
            }]
        )
        return response['message']['content']
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def solve_equation(latex_code) -> Optional[str]:
    """Solve the equation using qwen2-math."""
    try:
        response = ollama.chat(
            model=MATH_MODEL,
            messages=[{
                'role': 'user',
                'content': f"Реши уравнение {latex_code} и представь ответ в LaTeX."
            }]
        )
        return response['message']['content']
    except Exception as e:
        st.error(f"Error solving equation: {str(e)}")
        return None

def sidebar_content():
    """Handle sidebar content and image upload."""
    with st.sidebar:
        st.header("Upload Image")

        # Drag-and-drop file upload
        uploaded_file = st.file_uploader(
            "Supported formats: PNG, JPG, JPEG",
            type=ACCEPTED_FORMATS,
            help="Drag and drop a file here or select it manually."
        )

        # User guide
        st.markdown("---")
        st.markdown("### How to use:")
        st.markdown("1. Take a screenshot using `Win + Shift + S`.")
        st.markdown("2. Save the screenshot as a file (e.g., `screenshot.png`).")
        st.markdown("3. Upload the file above.")

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.session_state.uploaded_image = image

        if st.session_state.uploaded_image:
            st.image(st.session_state.uploaded_image, caption="Uploaded Image")

            if st.button("Extract LaTeX code", type="primary"):
                with st.spinner("Processing image..."):
                    # Convert image to bytes
                    img_byte_arr = io.BytesIO()
                    st.session_state.uploaded_image.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()

                    # Process image
                    result = process_image(io.BytesIO(img_byte_arr))
                    if result:
                        st.session_state.ocr_result = result
                        st.session_state.solution_result = None  # Reset previous solution

            if st.session_state.ocr_result and st.button("Solve Equation", type="primary"):
                with st.spinner("Solving equation..."):
                    solution = solve_equation(st.session_state.ocr_result)
                    if solution:
                        st.session_state.solution_result = solution

def display_results():
    """Display LaTeX code, rendered equation, and solution."""
    if st.session_state.ocr_result:
        st.markdown("### LaTeX Code")
        st.code(st.session_state.ocr_result, language='latex')

        st.markdown("### LaTeX Rendered")
        cleaned_latex = st.session_state.ocr_result.replace(r"\[", "").replace(r"\]", "")
        st.latex(cleaned_latex)

        if st.session_state.solution_result:
            st.markdown("### Solution")
            st.code(st.session_state.solution_result, language='latex')
            st.latex(st.session_state.solution_result)
    else:
        st.info("Upload an image and click 'Extract LaTeX code' to see the results.")

def main():
    """Main application function."""
    init_session_state()
    setup_page()
    create_clear_button()

    st.markdown('<p style="margin-top: -20px;">Extract LaTeX code from images and solve equations</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    sidebar_content()
    display_results()

    st.markdown("---")
    st.markdown(f"Vision model: {VISION_MODEL} | Math model: {MATH_MODEL}")

if __name__ == "__main__":
    main()
