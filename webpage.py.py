import streamlit as st
import subprocess

# Set the page title and icon
st.set_page_config(
    page_title="Program Output Display",
    page_icon="📊",
)

# Title and description
st.title("Program Output Display")
st.write("This app displays the output of three external programs.")


# Run external programs and capture their output
def run_program(command):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        return result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        return str(e)


# Run your three programs and capture their output
output1 = run_program(r"C:\Users\lalit\OneDrive\Desktop\DTL\ECG_MAIN\Raw Data\Single Modal\ECG\p1.py.py")
output2 = run_program(r"C:\Users\lalit\OneDrive\Desktop\DTL\ECG_MAIN\Raw Data\Single Modal\ECG\p2.py.py")
output3 = run_program(r"C:\Users\lalit\OneDrive\Desktop\DTL\ECG_MAIN\Raw Data\Single Modal\ECG\p3.py.py")

# Apply formatting and styling to the output
st.subheader("Output of Program 1")

# Use Markdown formatting with syntax highlighting
st.markdown(f"**Output:**\n```python\n{output1}\n```")

# Apply custom CSS styling
st.subheader("Output of Program 2")
st.markdown(f"<div style='color: blue; background-color: lightgray; padding: 10px;'>{output2}</div>", unsafe_allow_html=True)

# Apply code syntax highlighting
st.subheader("Output of Program 3")
st.code(output3, language='python')
