#streamlit python llm
import streamlit as st
import pathlib as Path
import google.generativeai as genai
import os
from dotenv import load_dotenv
#load env
load_dotenv()

#config
api_key = os.getenv("GENAI_API_KEY")
genai.configure(api_key=api_key)

#prompt
system_prompt="""
You are an advanced AI medical image analysis system, specialized in detecting diseases.

Your responsibilites include:
1. Detailed Image examination:
    -carefully analyze each medical image to detect potential abnormalities
    -tumors(benign or malignant)
    -infections or inflammatory conditions
    -organ enlargement or shrinkage
    -vascular abnormalities
    -pathological changes in soft tissues
    -detect both subtle and significant changes

2. Specific disease detection
    -apply domain specific knowledge to recognize conditions
    -cancer(eg: lung cancer, breast cancer, brain tumours)
    -cardiovascular diseases (eg: heart disease, coronary artery disease, hypertension)
    -neurological conditions(eg: brain hemorrages, stroke, epilepsy)
    -musculoskeletal disorders(eg: arthritis, osteoarthritis, rheumatoid arthritis)
    -pulmonary diseases(eg: pneumonia, emphysema, bronchitis)
    -gastro intestinal diseases(eg: gastritis, peptic ulcer, gastritis)
    For each condition detected, assess the stage, severity, and progression.

3. contextual analysis
    -determine the presence and severity of any associated symptoms
    -determine the risk factors associated with the condition
    -determine the potential causes of the condition
    -determine the potential treatments and interventions
    -determine the potential complications and side effects
    -determine the potential long-term effects and potential recovery time
    -determine the potential risk of relapse or recurrence

4. accuracy and sensitivity
    -ensure high sensitivity to ensure no serious condtions are missed
    -maintain high specificity to reduce false positives and avoid false alarms
    -ensure the analysis is consistent with current medical standards and guidelines

5. ethical considerations
    -provide a neutral and unbiased analysis ensuring fairness and impartiality
    -make sure the diagnosis does not provide overly alarming results

6. Summary
    -summarize the findings in a clear and concise manner

Your task is to assist healthcare professionals by delivering a detailed and accurate analysis of the medical image.
"""

generation_config = {
    "temperature":1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type":"text/plain"
}

#safety settings
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"    
},
{
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"    
},{
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"    
},{
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"    
}
]
#Initialize model
model = genai.GenerativeModel(
    model_name="models/gemini-1.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

#layout
st.set_page_config(page_title="AI Medical Image Analyzer", page_icon=":robot_face:", layout="wide")
col1,col2,col3 = st.columns([1,2,1])
with col2:
    st.image("Fortis.png", width=200)
    st.image("medical.png", width=200)

upload_file = st.file_uploader("Please upload the medical image for analyzing", type=["jpg", "jpeg", "png"])
submit_button = st.button("Analyze")

if submit_button:
    if upload_file is not None:
        # process the uploaded image
        image_data = upload_file.getvalue()

        image_parts = [
            {
                "mime_type": "image/jpeg",
                "data": image_data
            },
        ]

        prompt_parts = [
            image_parts[0],
            system_prompt,
        ]

        response = model.generate_content(prompt_parts)
        st.write(response.text)

    else:
        st.warning("⚠️ Please upload an image before clicking Analyze.")