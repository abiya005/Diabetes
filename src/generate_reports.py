"""
src/generate_reports.py
Generates four academic research PDF reports using ReportLab:
  1. reports/preprocessing_report.pdf
  2. reports/model_comparison_report.pdf
  3. reports/fairness_report.pdf
  4. reports/final_research_report.pdf
"""

import os

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

os.makedirs("reports", exist_ok=True)


def create_base_styles():
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=10,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=15,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=14,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=6,
    )
    callout_style = ParagraphStyle(
        "Callout",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1A202C"),
        backColor=colors.HexColor("#EDF2F7"),
        borderColor=colors.HexColor("#CBD5E0"),
        borderWidth=1,
        borderPadding=8,
        spaceBefore=8,
        spaceAfter=10,
    )
    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=1,
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        alignment=1,
    )
    return {
        "title": title_style,
        "subtitle": subtitle_style,
        "heading": heading_style,
        "body": body_style,
        "callout": callout_style,
        "th": table_header_style,
        "td": table_cell_style,
    }


def df_to_rl_table(df: pd.DataFrame, styles, col_widths=None) -> Table:
    """Convert pandas DataFrame to styled ReportLab Table."""
    data = []
    # Header
    header_row = [Paragraph(str(col), styles["th"]) for col in df.columns]
    data.append(header_row)
    # Rows
    for _, row in df.iterrows():
        r = []
        for val in row:
            if isinstance(val, float):
                formatted = f"{val:.4f}" if abs(val) < 1 else f"{val:.2f}"
            elif isinstance(val, int):
                formatted = f"{val:,}"
            else:
                formatted = str(val)
            r.append(Paragraph(formatted, styles["td"]))
        data.append(r)

    t = Table(data, colWidths=col_widths)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#F7FAFC")],
                ),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


# ==========================================================
# 1. PREPROCESSING REPORT
# ==========================================================
def generate_preprocessing_report():
    pdf_path = "reports/preprocessing_report.pdf"
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = create_base_styles()
    story = []

    story.append(
        Paragraph("Data Preprocessing & Leakage Prevention Audit", styles["title"])
    )
    story.append(
        Paragraph(
            "Study: Evaluating Robustness and Fairness of Diabetes Risk Prediction Models",
            styles["subtitle"],
        )
    )

    story.append(
        Paragraph("1. Dataset Ingestion & Target Harmonization", styles["heading"])
    )
    story.append(
        Paragraph(
            "The project utilizes the CDC Behavioral Risk Factor Surveillance System (BRFSS 2015) Diabetes Health Indicators "
            "cohort containing N = 253,680 survey records and 21 input features. In compliance with clinical risk screening objectives, "
            "the original multi-class outcome (0 = no diabetes, 1 = prediabetes, 2 = diabetes) was harmonized into a binary screening target: "
            "Class 0 represents negative observations (N = 213,703, 84.24%), and Class 1 represents positive diabetes or prediabetes risk "
            "(N = 39,977, 15.76%).",
            styles["body"],
        )
    )

    profile_df = pd.read_csv("results/dataset_profile.csv")
    story.append(
        Paragraph("Table 1: Dataset Profile & Cohort Demographics", styles["heading"])
    )
    story.append(df_to_rl_table(profile_df, styles, col_widths=[280, 220]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("2. Feature Taxonomy & Transformations", styles["heading"]))
    story.append(
        Paragraph(
            "Features were systematically categorized to avoid blind normalization and preserve clinical semantics:<br/>"
            "• <b>Binary Features (13 variables):</b> HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, "
            "PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk. Imputed using most frequent values and preserved as 0/1 indicators.<br/>"
            "• <b>Numerical Continuous Features (3 variables):</b> BMI, MentHlth (days poor mental health), PhysHlth (days illness). Imputed with median and standard-scaled (mean=0, std=1).<br/>"
            "• <b>Ordinal Clinical Features (3 variables):</b> GenHlth (1-5), Education (1-6), Income (1-8). Imputed with median and standard-scaled to ensure linear/margin stability.<br/>"
            "• <b>Sensitive Demographics:</b> Sex (0=Female, 1=Male) and Age (13 categorical tiers). Retained unscaled in separate audit containers for demographic fairness audits.",
            styles["body"],
        )
    )

    story.append(Paragraph("3. Leakage Prevention Protocol", styles["heading"]))
    story.append(
        Paragraph(
            "A strict 80:20 stratified train-test split (random_state=42) was conducted. To eliminate data leakage:<br/>"
            "1. All feature scalers and imputers were fitted <b>strictly on the training split (N = 202,944)</b>.<br/>"
            "2. The test partition (N = 50,736) was held out untouched until final blind model evaluation.<br/>"
            "3. Cross-validation transformations were applied strictly within each training fold pipeline.",
            styles["callout"],
        )
    )

    doc.build(story)
    print(f"Generated {pdf_path}")


# ==========================================================
# 2. MODEL COMPARISON REPORT
# ==========================================================
def generate_model_comparison_report():
    pdf_path = "reports/model_comparison_report.pdf"
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = create_base_styles()
    story = []

    story.append(
        Paragraph(
            "Machine Learning Model Development & Performance Report", styles["title"]
        )
    )
    story.append(
        Paragraph(
            "Comparative Assessment of Baseline, Ensemble, Margin, and Boosting Classifiers",
            styles["subtitle"],
        )
    )

    story.append(
        Paragraph(
            "1. Overall Predictive Performance on Held-Out Test Set (N = 50,736)",
            styles["heading"],
        )
    )
    perf_df = pd.read_csv("results/overall_performance.csv")
    display_df = perf_df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "Specificity",
            "F1",
            "ROC_AUC",
            "False Negative Rate",
        ]
    ]
    story.append(
        df_to_rl_table(display_df, styles, col_widths=[90, 50, 50, 50, 55, 45, 55, 80])
    )
    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            "2. Clinical Healthcare Implications: Minimizing False Negatives",
            styles["heading"],
        )
    )
    story.append(
        Paragraph(
            "In chronic metabolic disease screening, the clinical cost of errors is asymmetric. A <b>False Negative (FN)</b> represents "
            "a missed diagnosis of prediabetes or diabetes, leading to unmonitored progressive beta-cell dysfunction, retinopathy, "
            "nephropathy, and cardiovascular mortality. In contrast, a <b>False Positive (FP)</b> merely triggers an inexpensive, non-invasive "
            "fasting plasma glucose or HbA1c blood assay. Consequently, models prioritizing high Sensitivity (Recall) and low False Negative Rates "
            "(XGBoost at 21.50% FNR, Random Forest at 23.59% FNR) are clinically superior to models optimizing raw accuracy alone.",
            styles["callout"],
        )
    )

    story.append(Paragraph("3. Algorithm-Specific Findings", styles["heading"]))
    story.append(
        Paragraph(
            "• <b>XGBoost:</b> Achieved the strongest discriminative capability across all metrics (ROC-AUC = 0.8247, Recall = 78.50%, FNR = 21.50%).<br/>"
            "• <b>Random Forest:</b> Delivered robust ensemble predictions (ROC-AUC = 0.8220, Recall = 76.41%), closely matching gradient boosting.<br/>"
            "• <b>Logistic Regression:</b> Highly competitive interpretable baseline (ROC-AUC = 0.8176, Recall = 76.05%), indicating strong linear separation.<br/>"
            "• <b>Linear SVM:</b> Achieved the highest raw accuracy (84.81%) and specificity (97.12%), but produced a clinically unacceptable False Negative Rate (80.98%) "
            "at the standard 0.5 decision threshold due to class imbalance probability compression.",
            styles["body"],
        )
    )

    story.append(Paragraph("4. Statistical Significance Testing", styles["heading"]))
    stat_df = pd.read_csv("results/statistical_results.csv")
    stat_auc = stat_df[stat_df["Metric"] == "ROC_AUC"][
        [
            "Model_1",
            "Model_2",
            "Mean_1",
            "Mean_2",
            "Mean_Difference",
            "Paired_t_Statistic",
            "t_test_p_value",
        ]
    ]
    story.append(
        df_to_rl_table(stat_auc, styles, col_widths=[80, 80, 50, 50, 65, 75, 75])
    )

    doc.build(story)
    print(f"Generated {pdf_path}")


# ==========================================================
# 3. FAIRNESS REPORT
# ==========================================================
def generate_fairness_report():
    pdf_path = "reports/fairness_report.pdf"
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = create_base_styles()
    story = []

    story.append(
        Paragraph("Demographic Fairness & Health Equity Audit Report", styles["title"])
    )
    story.append(
        Paragraph(
            "Evaluating Disparities Across Age and Biological Sex Cohorts",
            styles["subtitle"],
        )
    )

    story.append(Paragraph("1. Mathematical Fairness Metrics", styles["heading"]))
    story.append(
        Paragraph(
            "Four formal fairness criteria were evaluated across demographic groups:<br/>"
            "• <b>Demographic Parity Difference (DPD):</b> |P(Y_hat=1 | A=a) - P(Y_hat=1 | A=b)|<br/>"
            "• <b>Equal Opportunity Difference (EOD):</b> |Recall_a - Recall_b| (True Positive Rate parity)<br/>"
            "• <b>Equalized Odds Difference (EODD):</b> max(|TPR_a - TPR_b|, |FPR_a - FPR_b|)<br/>"
            "• <b>False Negative Rate Difference (FNRD):</b> |FNR_a - FNR_b| (Missed case disparity)",
            styles["body"],
        )
    )

    fairness_df = pd.read_csv("results/fairness_metrics.csv")
    story.append(
        Paragraph(
            "Table 1: Fairness Disparity Metrics Across Models", styles["heading"]
        )
    )
    display_fair = fairness_df[
        [
            "Model",
            "Attribute",
            "Demographic_Parity_Difference",
            "Equal_Opportunity_Difference",
            "False_Negative_Rate_Difference",
        ]
    ]
    story.append(df_to_rl_table(display_fair, styles, col_widths=[90, 110, 95, 95, 85]))
    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            "2. Age-Based Equity Findings (Critical Disparity)", styles["heading"]
        )
    )
    story.append(
        Paragraph(
            "A severe systemic disparity was observed across age cohorts. In younger adults (18–34 years), diabetes prevalence is low (~2.8%). "
            "Because age is heavily weighted in the model decision boundary, models miss the vast majority of young prediabetic/diabetic patients: "
            "<b>False Negative Rates in 18–34 year-olds reached 67.2% for Logistic Regression, 77.4% for Random Forest, and 78.8% for XGBoost</b>. "
            "Conversely, older adults (65+) achieved FNRs below 16%. This results in an Equal Opportunity Difference exceeding 0.51–0.64 across models.",
            styles["callout"],
        )
    )

    story.append(
        Paragraph(
            "3. Sex-Based Equity Findings (Near-Perfect Parity)", styles["heading"]
        )
    )
    story.append(
        Paragraph(
            "In contrast to age, all four machine learning algorithms demonstrated remarkable sex equity. "
            "The Equal Opportunity Difference between female and male cohorts was negligible (0.0019 for Logistic Regression, 0.0030 for XGBoost). "
            "This demonstrates that models do not introduce diagnostic bias between biological sexes.",
            styles["body"],
        )
    )

    age_df = pd.read_csv("results/age_fairness.csv")
    xgb_age = age_df[age_df["Model"] == "XGBoost"][
        [
            "Age_Group",
            "Sample_Size",
            "Actual_Prevalence",
            "Recall",
            "False_Negative_Rate",
            "Recall_CI_Lower",
            "Recall_CI_Upper",
        ]
    ]
    story.append(
        Paragraph(
            "Table 2: XGBoost Performance by Age Bracket (with 95% Bootstrap CIs)",
            styles["heading"],
        )
    )
    story.append(
        df_to_rl_table(xgb_age, styles, col_widths=[70, 65, 75, 65, 75, 75, 75])
    )

    doc.build(story)
    print(f"Generated {pdf_path}")


# ==========================================================
# 4. FINAL COMPREHENSIVE RESEARCH REPORT
# ==========================================================
def generate_final_research_report():
    pdf_path = "reports/final_research_report.pdf"
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = create_base_styles()
    story = []

    # Title & Metadata
    story.append(
        Paragraph(
            "Evaluating the Robustness and Fairness of Diabetes Risk Prediction Models Across Demographic Groups",
            styles["title"],
        )
    )
    story.append(
        Paragraph(
            "Full Academic Research Report • CDC BRFSS 2015 Cohort Study",
            styles["subtitle"],
        )
    )

    # Abstract
    story.append(Paragraph("Executive Abstract", styles["heading"]))
    story.append(
        Paragraph(
            "<b>Background:</b> Machine learning models for diabetes risk assessment are increasingly deployed in population health, "
            "yet their demographic fairness across age and sex, and their robustness to data degradation, remain understudied.<br/>"
            "<b>Methods:</b> Four algorithms—Logistic Regression, Random Forest, Support Vector Machine, and XGBoost—were trained "
            "on 202,944 observations and evaluated on a held-out test partition of 50,736 records. Fairness was audited using Fairlearn "
            "metrics and 95% bootstrap confidence intervals. Robustness was evaluated under 5%–20% missing data (MCAR), input corruption, "
            "and clinical subpopulation distribution shifts.<br/>"
            "<b>Results:</b> XGBoost achieved the highest overall discriminative performance (ROC-AUC = 0.8247, Recall = 78.50%, FNR = 21.50%). "
            "While models exhibited outstanding sex parity (EOD < 0.025), a severe age disparity was discovered: younger patients (18–34) "
            "experienced False Negative Rates of 67%–79%, compared to 14%–17% in patients aged 65+ (EOD = 0.645). Under 20% input corruption, "
            "Random Forest proved most resilient (AUC drop = 0.0404).<br/>"
            "<b>Conclusion:</b> High accuracy is deceptive in medical screening; models must be tuned with demographic-aware thresholds "
            "to prevent catastrophic diagnostic neglect in younger populations.",
            styles["callout"],
        )
    )

    # Research Questions & Synthesis
    story.append(Paragraph("1. Answers to Core Research Questions", styles["heading"]))
    story.append(
        Paragraph(
            "1. <b>Which model performed best overall?</b> XGBoost Classifier demonstrated the superior predictive capability with the highest "
            "ROC-AUC (0.8247) and sensitivity (78.50%), significantly outperforming all other models in paired statistical tests (p < 10^-8).<br/>"
            "2. <b>Which demographic groups experienced higher false-negative rates?</b> Younger adults aged 18–34 suffered the highest FNR "
            "(67.2% in LR, 77.4% in RF, 78.8% in XGBoost, 97.1% in SVM). In contrast, sex cohorts exhibited near-identical FNRs (~21%–24%).<br/>"
            "3. <b>Which model demonstrated the smallest fairness disparities?</b> Linear SVM exhibited the lowest raw demographic disparity, "
            "but this was an artifact of severe under-prediction (FNR > 80%). Among clinically viable models, Logistic Regression demonstrated "
            "the smallest age disparity (EOD = 0.513 vs 0.645 for XGBoost).<br/>"
            "4. <b>Which model maintained performance best under missing and noisy data?</b> Logistic Regression had the smallest drop under "
            "missing data (AUC drop = 0.0264 at 20%), while Random Forest was most robust under input feature noise (AUC drop = 0.0404 at 20%).<br/>"
            "5. <b>Is the highest-accuracy model also the fairest and most robust?</b> No. Linear SVM achieved 84.81% accuracy but failed as a clinical "
            "screening tool due to an 80.98% False Negative Rate. High accuracy obscured acute diagnostic bias.",
            styles["body"],
        )
    )

    # Consolidated Matrix
    combined_df = pd.read_csv("results/combined_comparison.csv")
    story.append(
        Paragraph(
            "Table 1: Three-Dimensional Research Comparison Matrix", styles["heading"]
        )
    )
    disp_comb = combined_df[
        [
            "Model",
            "Accuracy",
            "Recall",
            "ROC_AUC",
            "Sex_Equal_Opportunity_Diff",
            "Age_Equal_Opportunity_Diff",
            "Noisy_Data_20pct_AUC_Drop",
        ]
    ]
    story.append(
        df_to_rl_table(disp_comb, styles, col_widths=[90, 50, 50, 55, 80, 80, 75])
    )
    story.append(Spacer(1, 10))

    # Limitations
    story.append(
        Paragraph(
            "2. Research Limitations & Governance Recommendations", styles["heading"]
        )
    )
    story.append(
        Paragraph(
            "• <b>Self-Reported Data:</b> BRFSS relies on telephonic surveys without biochemical validation (HbA1c / OGTT).<br/>"
            "• <b>Fixed Decision Boundary:</b> Uniform 0.5 threshold causes young adult underdiagnosis; clinical deployments should employ subgroup-specific operating thresholds.<br/>"
            "• <b>Simulated Covariate Shift:</b> Cross-sectional validation approximates subpopulation divergence rather than longitudinal temporal validation.",
            styles["body"],
        )
    )

    doc.build(story)
    print(f"Generated {pdf_path}")


if __name__ == "__main__":
    generate_preprocessing_report()
    generate_model_comparison_report()
    generate_fairness_report()
    generate_final_research_report()
    print("All 4 PDF reports generated successfully.")
