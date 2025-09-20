import gradio as gr
from KAN_app import predict_cin_probability


def infer(
    maggic_score, hgb, plt, neutrophils, mo, ly, rdw, crp, troponin_i,
    albumin, ldl, hdl, triglyceride, stent_diameter, pain_to_balloon_time,
    contrast_volume, plasma_osmolality,
    previous_cad, hypertension, multiple_lesion, procedure,
):
    patient = {
        "MAGGIC score": maggic_score,
        "Hemoglobin": hgb,
        "Platelets": plt,
        "Neutrophils": neutrophils,
        "Monocyte": mo,
        "Lymphocyte": ly,
        "RDW": rdw,
        "CRP": crp,
        "Troponin I": troponin_i,
        "Albumin": albumin,
        "LDL": ldl,
        "HDL": hdl,
        "Triglyceride": triglyceride,
        "Stent diameter": stent_diameter,
        "Pain-to-balloon time": pain_to_balloon_time,
        "Contrast volume": contrast_volume,
        "Plasma osmolality": plasma_osmolality,
        "Previous CAD": previous_cad,
        "Hypertension": hypertension,
        "Multiple lesion": multiple_lesion,
        "Procedure": procedure,
    }
    prob = predict_cin_probability(patient)
    return f"{prob * 100:.1f}%"


with gr.Blocks(title="Risk prediction tool for contrast-induced nephropathy in patients with ST segment elevation myocardial infarction") as demo:
    gr.Markdown("## Risk prediction tool for contrast-induced nephropathy in patients with ST segment elevation myocardial infarction")
    with gr.Row():
        with gr.Column():
            maggic_score = gr.Number(label="MAGGIC score", value=25)
            hgb = gr.Number(label="Hemoglobin (g/dL)", value=13.0)
            plt = gr.Number(label="Platelets (10^3/µL)", value=250)
            neutrophils = gr.Number(label="Neutrophils (10^3/µL)", value=4.0)
            mo = gr.Number(label="Monocyte (10^3/µL)", value=0.5)
            ly = gr.Number(label="Lymphocyte (10^3/µL)", value=2.0)
            rdw = gr.Number(label="RDW (%)", value=13.5)
            crp = gr.Number(label="CRP (mg/L)", value=5.0)
            troponin_i = gr.Number(label="Troponin I (ng/mL)", value=0.02)
            albumin = gr.Number(label="Albumin (g/dL)", value=3.8)
            ldl = gr.Number(label="LDL (mg/dL)", value=110)
            hdl = gr.Number(label="HDL (mg/dL)", value=45)
            triglyceride = gr.Number(label="Triglyceride (mg/dL)", value=150)
            stent_diameter = gr.Number(label="Stent diameter (mm)", value=3.0)
            pain_to_balloon_time = gr.Number(label="Pain-to-balloon time (min)", value=60)
            contrast_volume = gr.Number(label="Contrast volume (mL)", value=120)
            plasma_osmolality = gr.Number(label="Plasma osmolality (mOsm/kg)", value=285)

        with gr.Column():
            previous_cad = gr.Dropdown(["Yes", "No"], value="No", label="Previous CAD")
            hypertension = gr.Dropdown(["Yes", "No"], value="No", label="Hypertension")
            multiple_lesion = gr.Dropdown(["Yes", "No"], value="No", label="Multiple lesion")
            procedure = gr.Dropdown(["Yes", "No"], value="Yes", label="Procedure")
            out = gr.Textbox(label="CIN+ probability (%)", interactive=False)
            btn = gr.Button("Predict")

    btn.click(
        infer,
        inputs=[
            maggic_score, hgb, plt, neutrophils, mo, ly, rdw, crp, troponin_i,
            albumin, ldl, hdl, triglyceride, stent_diameter, pain_to_balloon_time,
            contrast_volume, plasma_osmolality,
            previous_cad, hypertension, multiple_lesion, procedure,
        ],
        outputs=[out],
    )

if __name__ == "__main__":
    demo.launch()


