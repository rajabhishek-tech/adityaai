from fpdf import FPDF

def generate_pdf(df, file_name="AdityaAI_EDA_Report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "AdityaAI - Automated EDA Report", ln=True)
    pdf.ln(5)

    pdf.cell(0, 10, f"Rows: {df.shape[0]} | Columns: {df.shape[1]}", ln=True)
    pdf.ln(5)

    pdf.cell(0, 10, "Columns Overview:", ln=True)
    pdf.ln(3)

    for col in df.columns:
        pdf.cell(0, 8, f"- {col} ({df[col].dtype})", ln=True)

    pdf.output(file_name)
    return file_name
