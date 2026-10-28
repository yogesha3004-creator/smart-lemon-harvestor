import os
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

import pptx
from pptx.util import Inches as PptInches, Pt as PptPt
from pptx.dml.color import RGBColor as PptRGBColor
from pptx.enum.text import PP_ALIGN

def set_cell_border(cell, **kwargs):
    """Set cell borders for docx table."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(r'''
        <w:tcBorders %s>
            <w:top w:val="single" w:sz="4" w:space="0" w:color="000000"/>
            <w:left w:val="single" w:sz="4" w:space="0" w:color="000000"/>
            <w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/>
            <w:right w:val="single" w:sz="4" w:space="0" w:color="000000"/>
        </w:tcBorders>
    ''' % nsdecls('w'))
    tcPr.append(tcBorders)

def generate_report():
    doc = docx.Document()
    
    # Page setup - 1 inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1.25)
        section.right_margin = Inches(1)

    # Base style
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(12)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)
    normal_style.paragraph_format.line_spacing = 1.5

    def add_p(text, align=WD_ALIGN_PARAGRAPH.JUSTIFY, bold=False, italic=False, size=12, space_after=6, space_before=0, color=None):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.line_spacing = 1.5
        run = p.add_run(text)
        run.bold = bold
        run.italic = italic
        run.font.name = 'Times New Roman'
        run.font.size = Pt(size)
        if color:
            run.font.color.rgb = color
        return p

    def add_heading_1(text):
        doc.add_page_break()
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(12)
        p.paragraph_format.space_before = Pt(18)
        run = p.add_run(text)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(14)
        return p

    def add_heading_2(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.space_before = Pt(12)
        run = p.add_run(text)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(12)
        return p

    # ==================== PAGE 1: TITLE PAGE ====================
    add_p("SMART MONITORING SYSTEM FOR AGRICULTURAL FIELDS USING AI", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=16, space_after=24, space_before=18)
    add_p("A PROJECT REPORT", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=13, space_after=18)
    add_p("Submitted by", align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, size=12, space_after=12)

    # Students Table
    table = doc.add_table(rows=3, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    students = [
        ("Yogesh A", "25MTR123"),
        ("SUSHMITA B", "25MTR106"),
        ("SOPIA S", "25MTR094")
    ]
    for i, (name, roll) in enumerate(students):
        row = table.rows[i]
        c1, c2 = row.cells[0], row.cells[1]
        c1.text = name
        c2.text = roll
        set_cell_border(c1)
        set_cell_border(c2)
        c1.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        c2.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        c1.paragraphs[0].runs[0].font.bold = True
        c2.paragraphs[0].runs[0].font.bold = True
        c1.paragraphs[0].runs[0].font.name = 'Times New Roman'
        c2.paragraphs[0].runs[0].font.name = 'Times New Roman'

    add_p("\nin partial fulfilment of the requirements for the award of the degree\nof", align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, size=11, space_before=36, space_after=12)
    add_p("BACHELOR OF ENGINEERING\nIN\nMECHATRONICS ENGINEERING", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=13, space_after=18)
    add_p("DEPARTMENT OF MECHATRONICS ENGINEERING", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=12, space_after=36)
    add_p("KONGU ENGINEERING COLLEGE\n(Autonomous)\nPERUNDURAI ERODE – 638 060", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=13, space_after=0)

    # ==================== PAGE 2: BONAFIDE CERTIFICATE ====================
    doc.add_page_break()
    add_p("NOVEMBER 2026", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=12, space_after=18, color=RGBColor(180, 0, 0))
    add_p("BONAFIDE CERTIFICATE", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=14, space_after=24)
    
    cert_text = (
        "This is to certify that the Project report entitled SMART MONITORING SYSTEM "
        "FOR AGRICULTURAL FIELDS USING AI is the bonafide record of project work "
        "done by Yogesh A (25MTR123), SUSHMITA B (25MTR106), SOPIA S (25MTR094), "
        "in partial fulfillment of the requirements for the award of the Degree of "
        "Bachelor of Engineering in Mechatronics Engineering during the year 2026 - 2027."
    )
    add_p(cert_text, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=40)

    sig_table = doc.add_table(rows=1, cols=2)
    sig_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    sig_table.autofit = False

    cell_sup = sig_table.rows[0].cells[0]
    cell_hod = sig_table.rows[0].cells[1]

    p_sup = cell_sup.paragraphs[0]
    p_sup.paragraph_format.line_spacing = 1.3
    r = p_sup.add_run("SUPERVISOR\n")
    r.bold = True
    r = p_sup.add_run("Dr. S. K. THANGARASU, B.E., M.E., Ph.D.\n")
    r.bold = True
    r.font.color.rgb = RGBColor(180, 0, 0)
    p_sup.add_run("Associate Professor\nMechatronics Engineering,\nKongu Engineering College,\nPerundurai - 638060.\nErode, Tamilnadu.")

    p_hod = cell_hod.paragraphs[0]
    p_hod.paragraph_format.line_spacing = 1.3
    r = p_hod.add_run("HEAD OF THE DEPARTMENT\n")
    r.bold = True
    r = p_hod.add_run("Dr. C. JEGADHEESAN, M.Tech., Ph.D.,\n")
    r.bold = True
    r.font.color.rgb = RGBColor(180, 0, 0)
    p_hod.add_run("Professor and Head\nMechatronics Engineering,\nKongu Engineering College,\nPerundurai - 638060.\nErode, Tamilnadu.")

    add_p("\n\nDate:", align=WD_ALIGN_PARAGRAPH.LEFT, space_before=24, space_after=18)
    add_p("Submitted for the end semester viva voce examination held on ____________________", align=WD_ALIGN_PARAGRAPH.CENTER, space_after=36)
    
    exam_table = doc.add_table(rows=1, cols=2)
    exam_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    c1, c2 = exam_table.rows[0].cells[0], exam_table.rows[0].cells[1]
    p1 = c1.paragraphs[0]
    p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p1.add_run("INTERNAL EXAMINER").bold = True
    p2 = c2.paragraphs[0]
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.add_run("EXTERNAL EXAMINER").bold = True

    # ==================== PAGE 3: DECLARATION ====================
    doc.add_page_break()
    add_p("DEPARTMENT OF MECHATRONICS ENGINEERING\nKONGU ENGINEERING COLLEGE\n(Autonomous)\nPERUNDURAI ERODE – 638060", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=12, space_after=12)
    add_p("November 2026", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=12, space_after=18, color=RGBColor(180, 0, 0))
    add_p("DECLARATION", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=14, space_after=24)

    decl_text = (
        "We affirm that the Project Report titled SMART MONITORING SYSTEM FOR "
        "AGRICULTURAL FIELDS USING AI being submitted in partial fulfillment of the "
        "requirements for the award of Bachelor of Engineering is the original work carried out by "
        "us. It has not formed part of any other project report or dissertation on the basis of which a "
        "degree or award was conferred on an earlier occasion on this or any other candidate."
    )
    add_p(decl_text, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=30)

    add_p("1. Yogesh A (25MTR123)", align=WD_ALIGN_PARAGRAPH.RIGHT, bold=True, space_after=6)
    add_p("2. SUSHMITA B (25MTR106)", align=WD_ALIGN_PARAGRAPH.RIGHT, bold=True, space_after=6)
    add_p("3. SOPIA S (25MTR094)", align=WD_ALIGN_PARAGRAPH.RIGHT, bold=True, space_after=36)

    add_p("I certify that the declaration made by the above candidates is true to the best of my knowledge.", align=WD_ALIGN_PARAGRAPH.LEFT, space_after=36)
    add_p("Name and Signature of the Supervisor with seal\n(Dr. S. K. Thangarasu, Associate Professor)", align=WD_ALIGN_PARAGRAPH.RIGHT, bold=True, space_after=18)
    add_p("Date:", align=WD_ALIGN_PARAGRAPH.LEFT)

    # ==================== PAGE 4: ABSTRACT ====================
    add_heading_1("ABSTRACT")
    abstract_text = (
        "Agricultural fields plantations require regular monitoring to identify insect attacks and crop "
        "damage early. Conventional visual inspection is labour-intensive and may not cover large fields "
        "consistently. This mini project proposes a smart monitoring system that combines drone-mounted "
        "RGB imaging, artificial intelligence (AI), GPS location association and a mobile application to "
        "support agricultural fields crop monitoring. The planned classifier assigns plant images to two "
        "classes: Normal or Suspected Insect-Affected. Images are organized and labelled for model "
        "development and evaluation, and each prediction can be linked to an image identifier, timestamp "
        "and GPS coordinates where available. The mobile interface is intended to present suspected "
        "detections and their locations so that farmers or field workers can prioritize follow-up inspection. "
        "The system is a decision-support aid, not a replacement for expert agricultural diagnosis. Its "
        "expected benefits include faster plantation coverage, more systematic records and reduced effort in "
        "locating potentially affected plants. The supplied presentation does not provide verified dataset "
        "counts, exact equipment specifications or numerical model performance; these must be documented "
        "after implementation. Evaluation should use an independent test set and suitable measures such as "
        "accuracy, precision, recall, F1-score and a confusion matrix."
    )
    add_p(abstract_text, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=18)
    add_p("Keywords: Agricultural fields, precision agriculture, artificial intelligence, RGB imaging, drone monitoring, GPS mapping, crop health.", bold=True, space_before=12)

    # ==================== PAGE 5: ACKNOWLEDGEMENT ====================
    add_heading_1("ACKNOWLEDGEMENT")
    add_p("We thank our beloved Correspondent Thiru. E. R. K. KRISHNAN, M. Com., and all the members of Kongu Vellalar Institute of Technology Trust at this high time for providing us with plethora of facilities of complete my project successfully.", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_p("We thank our beloved Principal Dr. R. PARAMESHWARAN M.E., Ph.D., who has been a bastion of moral strength and a source of incessant encouragement to us.", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_p("We take immense pleasure to express our heartfelt thanks to our Head of the Department, Mechatronics Engineering, Dr. C. JEGADHEESAN, M.Tech., Ph.D., for his constant support provided all through the course of our project.", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_p("This work would not have been materialized without the great guidance given by our guide Dr. S. K. THANGARASU, B.E., M.E., Ph.D., Associate Professor, who had been a constant source of ideas and inspiration.", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_p("We thank our project coordinators Mr. T. KARTHIK M.E., (Ph.D.)., for their valuable suggestions for completion of the project successfully.", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_p("We also thank the teaching and non-teaching staff members of Mechatronics Engineering Department and all our fellow students who stood with us to complete out project successfully.", align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    # ==================== PAGE 6: SDG ====================
    add_heading_1("JUSTIFICATION FOR RELEVANT SUSTAINABLE DEVELOPMENT GOALS")
    sdg_table = doc.add_table(rows=5, cols=2)
    sdg_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    rows_data = [
        ("PROJECT TITLE", "SMART MONITORING SYSTEM FOR AGRICULTURAL FIELDS USING AI"),
        ("ALIGNED SDG", "SDG 2 – Zero Hunger\nSDG 9 – Industry, Innovation and Infrastructure\nSDG 12 – Responsible Consumption and Production"),
        ("SDG PICTURE", "[SDG 2: Zero Hunger | SDG 9: Industry, Innovation & Infrastructure | SDG 12: Responsible Consumption and Production]"),
        ("JUSTIFICATION\n(150 to 200 words)", 
         "This project supports sustainable agriculture by proposing a way to screen agricultural fields plants "
         "for suspected insect damage using drone-based RGB images and AI. Early identification may help farmers "
         "and field workers prioritize inspections and reduce the time spent searching across large plantation "
         "areas. GPS association can connect observations to locations, allowing suspected plants to be revisited "
         "and checked. The mobile reporting concept can organize image classifications and location information "
         "in a single interface. The project aligns with SDG 2 because improved crop monitoring may support "
         "agricultural productivity and food-related livelihoods; SDG 9 because it applies digital technologies "
         "and AI to an agricultural monitoring problem; and SDG 12 because targeted inspection may help make "
         "crop-management decisions more informed. These are expected contributions, not measured outcomes. "
         "The system's practical value will depend on the quality and diversity of labelled images, model "
         "performance on independent data, GPS accuracy, field conditions and confirmation by agricultural experts. "
         "The proposed system is a decision-support aid and should not be treated as a definitive diagnosis of plant "
         "disease or insect attack."),
        ("SUPERVISOR'S SIGN WITH NAME", "Dr. S. K. Thangarasu, Associate Professor")
    ]

    for idx, (h, c) in enumerate(rows_data):
        row = sdg_table.rows[idx]
        c1, c2 = row.cells[0], row.cells[1]
        c1.text = h
        c2.text = c
        set_cell_border(c1)
        set_cell_border(c2)
        c1.paragraphs[0].runs[0].font.bold = True
        c1.paragraphs[0].runs[0].font.name = 'Times New Roman'
        c2.paragraphs[0].runs[0].font.name = 'Times New Roman'

    # ==================== PAGE 7: TABLE OF CONTENTS ====================
    add_heading_1("TABLE OF CONTENTS")
    toc_data = [
        ("CHAPTER NO.", "TITLE", "PAGE NO."),
        ("", "ABSTRACT", "5"),
        ("", "ACKNOWLEDGEMENT", "6"),
        ("", "JUSTIFICATION FOR RELEVANT SUSTAINABLE DEVELOPMENT GOALS", "7"),
        ("", "LIST OF TABLES", "9"),
        ("", "LIST OF FIGURES", "10"),
        ("", "LIST OF SYMBOLS AND ABBREVIATIONS", "11"),
        ("1", "INTRODUCTION", "12"),
        ("", "1.1 Background", "12"),
        ("", "1.2 Need for the System", "12"),
        ("", "1.3 Objectives", "12"),
        ("2", "LITERATURE REVIEW", "14"),
        ("", "2.1 Summary of Literature Review", "14"),
        ("3", "PROBLEM DEFINITION AND OBJECTIVES", "16"),
        ("", "3.1 Problem Identification", "16"),
        ("", "3.2 Objectives", "17"),
        ("4", "PROPOSED SYSTEM", "18"),
        ("", "4.1 Overview of the Proposed System", "18"),
        ("", "4.2 Workflow", "18"),
        ("5", "EQUIPMENT DETAILS", "19"),
        ("", "5.1 Dataset Description", "19"),
        ("", "5.2 Equipment Used", "19"),
        ("6", "HARDWARE AND SOFTWARE DETAILS", "20"),
        ("", "6.1 Hardware Details", "20"),
        ("", "6.2 Software Details", "20"),
        ("7", "RESULTS AND DISCUSSION", "21"),
        ("", "7.1 Metrics Used", "21"),
        ("8", "CONCLUSION AND FUTURE WORK", "23"),
        ("", "REFERENCES", "24"),
    ]
    toc_table = doc.add_table(rows=len(toc_data), cols=3)
    toc_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, (c1_txt, c2_txt, c3_txt) in enumerate(toc_data):
        row = toc_table.rows[i]
        row.cells[0].text = c1_txt
        row.cells[1].text = c2_txt
        row.cells[2].text = c3_txt
        for cell in row.cells:
            set_cell_border(cell)
            if i == 0 or c1_txt != "":
                cell.paragraphs[0].runs[0].font.bold = True
            cell.paragraphs[0].runs[0].font.name = 'Times New Roman'

    # ==================== PAGE 8: LIST OF TABLES & FIGURES & SYMBOLS ====================
    add_heading_1("LIST OF TABLES")
    lot_data = [
        ("TABLE NO.", "NAME OF THE TABLE", "PAGE NO."),
        ("1", "Evaluation metrics used for AI classification", "21"),
        ("2", "Results recording template for model evaluation", "22")
    ]
    lot_table = doc.add_table(rows=len(lot_data), cols=3)
    lot_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, (c1, c2, c3) in enumerate(lot_data):
        row = lot_table.rows[i]
        row.cells[0].text, row.cells[1].text, row.cells[2].text = c1, c2, c3
        for cell in row.cells:
            set_cell_border(cell)
            if i == 0: cell.paragraphs[0].runs[0].font.bold = True
            cell.paragraphs[0].runs[0].font.name = 'Times New Roman'

    add_p("\nLIST OF FIGURES", bold=True, size=14, space_before=18, space_after=12)
    lof_data = [
        ("FIGURE NO.", "NAME OF THE FIGURE", "PAGE NO."),
        ("1", "Conceptual architecture of the proposed monitoring system", "18"),
        ("2", "Workflow of drone-based AI monitoring", "18")
    ]
    lof_table = doc.add_table(rows=len(lof_data), cols=3)
    lof_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, (c1, c2, c3) in enumerate(lof_data):
        row = lof_table.rows[i]
        row.cells[0].text, row.cells[1].text, row.cells[2].text = c1, c2, c3
        for cell in row.cells:
            set_cell_border(cell)
            if i == 0: cell.paragraphs[0].runs[0].font.bold = True
            cell.paragraphs[0].runs[0].font.name = 'Times New Roman'

    add_p("\nLIST OF SYMBOLS AND ABBREVIATIONS", bold=True, size=14, space_before=18, space_after=12)
    abbrevs = [
        ("AI", "Artificial Intelligence"),
        ("GPS", "Global Positioning System"),
        ("RGB", "Red, Green and Blue"),
        ("SDG", "Sustainable Development Goal"),
        ("UAV", "Unmanned Aerial Vehicle"),
        ("UI", "User Interface")
    ]
    for abb, desc in abbrevs:
        add_p(f"{abb:<8} -    {desc}", space_after=3)

    # ==================== CHAPTER 1 ====================
    add_heading_1("CHAPTER-1\nINTRODUCTION")
    add_heading_2("1.1 Background")
    add_p("Agricultural fields is a valuable plantation crop, and maintaining plant health is important for yield and quality. Insect attacks and other visible crop-health problems can reduce productivity when they are not identified early. Manual inspection can be slow across large or uneven plantations and may depend on the experience of individual observers. Drone-based RGB imaging offers a way to collect repeatable visual observations over a wider area. When combined with AI image classification, GPS tagging and a mobile interface, these observations can be organized to support timely field inspection.")
    add_heading_2("1.2 Need for the System")
    add_p("The system is motivated by the need for quicker field coverage, early identification of suspected insect attacks, reduced manual effort and reliable location tracking. An automated image-screening workflow can help prioritize field visits, while GPS-linked records make it easier to revisit the relevant area. The system is intended to assist inspection rather than replace the judgement of farmers or agricultural experts.")
    add_heading_2("1.3 Objectives")
    add_p("The objectives are to capture plant images using a drone-mounted RGB camera; prepare labelled images for model development and validation; classify images into Normal and Suspected Insect-Affected categories; associate observations with GPS coordinates where available; display results through a mobile application concept; and evaluate model behaviour using suitable classification metrics and field verification.")

    # ==================== CHAPTER 2 ====================
    add_heading_1("CHAPTER-2\nLITERATURE SURVEY")
    add_p("The proposed project draws on four related technical areas: drone-based agricultural monitoring, RGB image acquisition, AI-based image classification, and GPS/mobile reporting. UAVs can collect images from areas that are difficult to inspect quickly on foot. RGB images provide visible information about plant colour, leaf shape and surface symptoms. A supervised AI classifier can learn patterns from labelled examples, while GPS coordinates and timestamps help connect predictions to locations in the field. The mobile application provides a practical way to view suspected detections and prioritize follow-up. These technologies support screening, but reliable performance depends on representative data, correct labels, consistent image capture and field verification.")
    add_heading_2("2.1 SUMMARY OF LITERATURE REVIEW")
    add_p("The concepts reviewed indicate that drone imaging can improve the speed of agricultural data collection, AI can support initial screening of plant images, and GPS with a mobile interface can make observations easier to locate and review. These components should be designed as one workflow. The system must be evaluated on images that were not used for training, and predictions should be verified by a farmer or agricultural expert before decisions are made.")

    # ==================== CHAPTER 3 ====================
    add_heading_1("CHAPTER-3\nPROBLEM DEFINITION AND OBJECTIVES")
    add_heading_2("3.1 PROBLEM IDENTIFICATION")
    add_p("Current agricultural fields-field monitoring can rely on visual inspection by farmers or field workers. Large farms require substantial time and manpower, and symptoms may be missed because of distance, occlusion, lighting or inconsistent inspection intervals. The project addresses the need for a systematic method to capture plant images, screen them for suspected insect damage and associate findings with field locations. Problem statement: to design a drone-assisted monitoring workflow that captures agricultural fields plant images, uses an AI model to classify them as Normal or Suspected Insect-Affected, associates observations with GPS coordinates where available, and presents results for follow-up inspection.")
    add_p("Conceptual 3D model contents: show a drone frame with propulsion units, an RGB camera mounted beneath the frame, a GPS module or location source, and a processing/mobile reporting workflow. Label the camera field of view and indicate how captured images are associated with location data. The model is conceptual unless a physical prototype and exact components are confirmed.")
    add_heading_2("3.2 OBJECTIVES")
    add_p("The project proposes a smart monitoring workflow for agricultural fields fields using drone-mounted RGB imaging, AI classification, GPS mapping and a mobile application. It aims to classify images as Normal or Suspected Insect-Affected, show locations that may need inspection, and support more systematic crop monitoring. Model performance and field usefulness must be confirmed through testing. The main objective of this project is to develop a smart monitoring system for agricultural fields using artificial intelligence and drone technology, with a particular focus on agricultural fields plantations. The system aims to capture aerial images of agricultural fields and analyse them using AI-based image processing techniques to identify plant health conditions and detect visible signs of diseases, pest attacks, and other abnormalities. GPS technology is intended to identify and record the locations of affected plants or plantation areas, while a mobile application can be used to display monitoring results and location information. The proposed system aims to reduce manual inspection, save time and labour, improve the efficiency of plantation monitoring, and provide timely information to farmers for better decision-making. It also aims to support precision agriculture by identifying areas that require further inspection or treatment, maintaining plantation monitoring records, and encouraging sustainable farming practices through the integration of drone technology, artificial intelligence, GPS, and mobile communication.")

    # ==================== CHAPTER 4 ====================
    add_heading_1("CHAPTER-4\nPROPOSED SYSTEM")
    add_heading_2("4.1 Overview of the Proposed System")
    add_p("The proposed system combines aerial image acquisition, AI-based screening, GPS tagging and mobile reporting. The drone captures RGB images of agricultural fields plants. Images are organized and passed to an AI classifier, which predicts whether each image is Normal or Suspected Insect-Affected. The predicted class is stored with the image identifier, timestamp and location information where available. The mobile application concept presents flagged observations so that the user can review them and arrange a field inspection.")
    add_heading_2("4.1.1 System Architecture")
    add_p("The system can be organized into five connected modules: image acquisition, image preparation, AI-based classification, GPS association and result visualization. The image-acquisition module captures visible-spectrum images. The preparation module checks image quality and formats images for the model. The classifier generates a class prediction. The location module associates the image with coordinates and time, and the reporting module displays results through a mobile interface. A unique image identifier should be maintained throughout the workflow so that the prediction can be traced back to its source image.")
    add_heading_2("4.2 Workflow")
    add_p("1. Plan a safe flight path and capture images of agricultural fields plants using an RGB camera.")
    add_p("2. Organize images and assign verified labels for model training and testing.")
    add_p("3. Pre-process images consistently and submit them to the trained AI model.")
    add_p("4. Receive the predicted class: Normal or Suspected Insect-Affected.")
    add_p("5. Attach GPS coordinates and observation details to each result, where available.")
    add_p("6. Display flagged results and locations through the mobile application.")
    add_p("7. Ask a farmer or agricultural expert to inspect suspected plants and confirm the condition.")
    add_heading_2("4.3 Expected Benefits and Limitations")
    add_p("Expected benefits include faster coverage of plantation areas, reduced dependence on repeated manual searching, location-based records for follow-up inspection and earlier attention to potentially affected plants. The system can only screen the images it receives. Occlusion, motion blur, changing light, unfamiliar symptoms and poor-quality labels may affect predictions. GPS accuracy and correspondence between the image and location must also be checked. The system should support, not replace, expert field judgement.")

    # ==================== CHAPTER 5 ====================
    add_heading_1("CHAPTER-5\nEQUIPMENT DETAILS")
    add_heading_2("5.1 Dataset Description")
    add_p("The presentation specifies a labelled dataset for improving the AI model but does not provide dataset size, source, class counts or train/test split. These values should be filled in after the dataset is assembled. The planned classes are Normal and Suspected Insect-Affected. The dataset should cover different plant conditions, lighting, camera distances and plantation locations so that the model is not trained on a narrow set of examples.")
    add_heading_2("5.1.1 Dataset Preparation and Labelling")
    add_p("Dataset preparation should begin by removing unreadable or duplicate images and checking whether labels match the visible plant condition. Images should represent normal plants and a range of suspected insect-related symptoms. Labels should be checked by a knowledgeable reviewer where possible. To reduce evaluation bias, images of the same plant or nearly identical frames should not be spread across training and test subsets. The final report should state the number of images in each class and how the dataset was divided.")
    add_heading_2("5.2 Equipment Used")
    add_p("The project presentation identifies drone technology, an RGB camera and GPS as the key sensing elements. Exact hardware models and specifications were not supplied and should be added after the selected equipment is confirmed.")
    add_p("• Drone/UAV: carries the camera and supports aerial image collection.")
    add_p("• RGB camera: captures visible-spectrum images of agricultural fields plants.")
    add_p("• GPS receiver or drone location system: provides coordinates associated with observations.")
    add_p("• Computer or compatible processing device: runs image preparation and AI inference, depending on implementation.")
    add_p("• Smartphone/mobile application: displays detection labels, alerts and location information.")
    add_heading_2("5.3 Equipment Selection Considerations")
    add_p("Hardware should be selected according to plantation size, flight endurance, camera resolution, payload capacity, safe operating requirements, GPS quality, environmental conditions and the processing workload. Camera mounting should minimize vibration and ensure that images are sufficiently sharp. Equipment models, specifications and cost should be documented only after they are finalized.")

    # ==================== CHAPTER 6 ====================
    add_heading_1("CHAPTER-6\nHARDWARE AND SOFTWARE DETAILS")
    add_heading_2("6.1 Hardware Details")
    add_p("The hardware concept consists of an aerial platform, image sensor and location source. The drone provides mobility over the plantation, the RGB camera collects plant images, and GPS supplies location information when available. A computer or compatible processing device is used for dataset preparation and model inference, while a smartphone or mobile application interface presents the results. Exact component models and specifications were not stated in the supplied presentation.")
    add_heading_2("6.1.1 Image Capture and Location Association")
    add_p("During image capture, the camera should be mounted securely and oriented to obtain clear views of the plant canopy. Flight height, speed, image overlap and capture timing should be selected to provide consistent images while maintaining safe operation. Location association may use GPS metadata recorded by the drone or a synchronized position log. The implementation should verify that timestamps and coordinates correspond to the captured image, and records with missing or unreliable location data should be marked accordingly.")
    add_heading_2("6.2 Software Details")
    add_p("The software workflow includes image collection and organization, labelled dataset management, AI model training or inference, GPS metadata handling and a mobile display. The supplied presentation does not identify a specific AI framework, model architecture, mobile platform or database, so these should be documented once selected. The implementation should record software versions and model settings to make testing reproducible.")
    add_heading_2("6.3 Data Processing Pipeline")
    add_p("1. Import images and inspect image quality.")
    add_p("2. Resize or normalize images as required by the selected model.")
    add_p("3. Use verified labels to train and validate the classifier.")
    add_p("4. Evaluate predictions on an independent test set.")
    add_p("5. Store each prediction with its class, timestamp and associated coordinates.")
    add_p("6. Present suspected locations in the mobile interface and verify them in the field.")
    add_heading_2("6.4 Data Management and Safety")
    add_p("Maintain a clear record of dataset versions, label changes, model versions and test results. Drone flights must follow applicable local rules and be conducted safely, with attention to people, property, weather and plantation conditions. Images and location records should be stored securely and used only for the intended monitoring purpose.")

    # ==================== CHAPTER 7 ====================
    add_heading_1("CHAPTER-7\nRESULTS AND DISCUSSION")
    add_p("Results Overview\nThe supplied presentation describes the intended AI model and prototype workflow, but it does not provide numerical test results, a confusion matrix, dataset counts or measured accuracy. Therefore, no numerical performance result is asserted in this report. The results section should be completed after the model has been tested using labelled data and the field workflow has been checked.")
    add_heading_2("7.1 Metrics Used")
    add_p("For a two-class classifier, the following metrics can be calculated from the independent test set:")
    add_p("• Accuracy: proportion of all test images classified correctly.")
    add_p("• Precision for Suspected Insect-Affected: proportion of images flagged as suspected that are confirmed as suspected by the reference labels.")
    add_p("• Recall for Suspected Insect-Affected: proportion of reference-labelled suspected images that the model successfully flags.")
    add_p("• F1-score: harmonic mean of precision and recall.")
    add_p("• Confusion matrix: counts of true and false predictions for both classes.")
    add_p("For agricultural screening, recall is important because missed affected plants may delay intervention; precision is also important because excessive false alarms can increase inspection workload.")
    add_heading_2("7.1.1 Evaluation Procedure")
    add_p("The evaluation should use a held-out test set that was not used to train or tune the model. Predictions should be compared with verified reference labels, and the confusion matrix should be recorded before calculating metrics. If possible, testing should include images from different plants, dates and lighting conditions. Field trials should separately record whether suspected detections are confirmed by an expert or farmer.")
    add_heading_2("7.2 Comparison of Results")
    add_p("No measured results were included in the supplied project presentation, so a numerical comparison cannot yet be reported. Once testing is complete, the team can compare candidate models or settings using the same independent test set. The comparison should report class-wise precision and recall in addition to overall accuracy, because a high overall accuracy may hide poor detection of the suspected class.")
    add_heading_2("7.2.1 Results Recording Template")
    add_p("Record the date, test-set size, number of images in each class, model version, evaluation device, accuracy, class-wise precision and recall, F1-score and confusion matrix. Also record the number of detections verified during field inspection and any failures caused by blur, shadows, occlusion or incorrect GPS association. Do not enter estimated values as measured results.")
    add_heading_2("7.3 Discussion")
    add_p("The proposed workflow is intended to reduce the time needed to survey a plantation and connect suspected detections with locations for follow-up. Whether these benefits are achieved depends on model quality, image coverage, GPS reliability, usability of the mobile interface and the time required for verification. The project should be assessed through both classification metrics and practical field observations.")

    # ==================== CHAPTER 8 ====================
    add_heading_1("CHAPTER-8\nCONCLUSION AND FUTURE WORK")
    add_heading_2("8.1 Conclusion")
    add_p("This mini project proposes a smart monitoring system for agricultural fields agricultural fields using drone-based RGB imaging, AI classification, GPS mapping and a mobile application. The intended workflow captures plant images, classifies them as Normal or Suspected Insect-Affected, associates results with location information and presents suspected detections for follow-up. The design may support more systematic monitoring, but its accuracy and practical benefits must be established through dataset-based evaluation and field verification. Dataset details, exact equipment specifications and measured results should be added after implementation.")
    add_heading_2("8.2 Future Work")
    add_p("• Collect a larger, balanced and expert-verified agricultural fields image dataset across seasons and plantation conditions.")
    add_p("• Evaluate suitable AI architectures and report class-wise metrics on an independent test set.")
    add_p("• Improve robustness to shadows, occlusion, blur and changes in camera distance.")
    add_p("• Integrate GPS metadata with images and test location accuracy in the field.")
    add_p("• Develop a mobile interface with a map, image preview, timestamps and alert history.")
    add_p("• Conduct field trials with farmers or agricultural experts to verify suspected detections.")
    add_p("• Investigate additional classes or symptom categories only when sufficient labelled data are available.")
    add_p("• Document hardware specifications, power use, flight coverage and processing time.")

    # ==================== REFERENCES ====================
    add_heading_1("REFERENCES")
    refs = [
        "1. Bouguettaya, A., Zarzour, H., Kechida, A., and Taberkit, A. M., “A Survey on Deep Learning-Based Identification of Plant and Crop Diseases from UAV-Based Aerial Images,” Cluster Computing, vol. 26, pp. 1297–1317, 2023. DOI: 10.1007/s10586-022-03627-x.",
        "2. Kouadio, L., El Jarroudi, M., Belabess, Z., et al., “A Review on UAV-Based Applications for Plant Disease Detection and Monitoring,” Remote Sensing, vol. 15, no. 17, article 4273, 2023. DOI: 10.3390/rs15174273.",
        "3. Zhu, H., Lin, C., Liu, G., et al., “Intelligent Agriculture: Deep Learning in UAV-Based Remote Sensing Imagery for Crop Diseases and Pests Detection,” Frontiers in Plant Science, vol. 15, 2024. DOI: 10.3389/fpls.2024.1435016.",
        "4. Agrawal, J., and Arafat, M. Y., “Transforming Farming: A Review of AI-Powered UAV Technologies in Precision Agriculture,” Drones, vol. 8, no. 11, article 664, 2024. DOI: 10.3390/drones8110664.",
        "5. Wang, D., Zhao, M., Li, Z., et al., “A Survey of Unmanned Aerial Vehicles and Deep Learning in Precision Agriculture,” European Journal of Agronomy, article 127477, 2025. DOI: 10.1016/j.eja.2024.127477.",
        "6. Guebsi, R., Mami, S., and Chokmani, K., “Drones in Precision Agriculture: A Comprehensive Review of Applications, Technologies, and Challenges,” Drones, vol. 8, no. 11, article 686, 2024. DOI: 10.3390/drones8110686.",
        "7. Radočaj, D., Radočaj, P., Plaščak, I., and Jurišić, M., “Evolution of Deep Learning Approaches in UAV-Based Crop Leaf Disease Detection: A Web of Science Review,” Applied Sciences, vol. 15, no. 19, article 10778, 2025. DOI: 10.3390/app151910778."
    ]
    for r in refs:
        add_p(r, space_after=8)

    output_path = "docs/SMART_MONITORING_SYSTEM_PROJECT_REPORT.docx"
    doc.save(output_path)
    print(f"Report saved to: {output_path}")


def generate_presentation():
    prs = pptx.Presentation()
    prs.slide_width = PptInches(13.333)
    prs.slide_height = PptInches(7.5)
    blank_layout = prs.slide_layouts[6]

    DARK_BG = PptRGBColor(15, 23, 42)      # Slate 900
    CARD_BG = PptRGBColor(30, 41, 59)      # Slate 800
    EMERALD = PptRGBColor(16, 185, 129)    # Emerald 500
    GOLD = PptRGBColor(245, 158, 11)       # Amber 500
    WHITE = PptRGBColor(248, 250, 252)
    MUTED = PptRGBColor(148, 163, 184)

    def style_slide(slide, title_text, category="MECHATRONICS ENGINEERING"):
        # Background
        bg = slide.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height) # 1 = msoShapeRectangle
        bg.fill.solid()
        bg.fill.fore_color.rgb = DARK_BG
        bg.line.fill.background()

        # Top Category & Title
        txBox = slide.shapes.add_textbox(PptInches(0.8), PptInches(0.4), PptInches(11.7), PptInches(1.2))
        tf = txBox.text_frame
        tf.word_wrap = True
        
        p0 = tf.paragraphs[0]
        r0 = p0.add_run()
        r0.text = category.upper()
        r0.font.bold = True
        r0.font.size = PptPt(11)
        r0.font.color.rgb = GOLD

        p1 = tf.add_paragraph()
        r1 = p1.add_run()
        r1.text = title_text
        r1.font.bold = True
        r1.font.size = PptPt(24)
        r1.font.color.rgb = WHITE

    # ---------- Slide 1: Title Slide ----------
    slide1 = prs.slides.add_slide(blank_layout)
    bg = slide1.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = DARK_BG
    bg.line.fill.background()

    tb = slide1.shapes.add_textbox(PptInches(1), PptInches(0.8), PptInches(11.3), PptInches(2.5))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "KONGU ENGINEERING COLLEGE (AUTONOMOUS)\nDEPARTMENT OF MECHATRONICS ENGINEERING"
    r.font.bold = True
    r.font.size = PptPt(14)
    r.font.color.rgb = GOLD

    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run()
    r2.text = "\nSMART MONITORING SYSTEM FOR AGRICULTURAL FIELDS USING AI"
    r2.font.bold = True
    r2.font.size = PptPt(26)
    r2.font.color.rgb = WHITE

    # Details Cards
    card1 = slide1.shapes.add_shape(1, PptInches(1.5), PptInches(3.8), PptInches(4.8), PptInches(2.8))
    card1.fill.solid()
    card1.fill.fore_color.rgb = CARD_BG
    card1.line.color.rgb = EMERALD
    tf1 = card1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    r = p.add_run()
    r.text = "SUBMITTED BY:\n\n"
    r.font.bold = True
    r.font.color.rgb = GOLD
    r.font.size = PptPt(13)
    
    students = [
        "1. Yogesh A  (25MTR123)",
        "2. SUSHMITA B  (25MTR106)",
        "3. SOPIA S  (25MTR094)"
    ]
    for s in students:
        p_s = tf1.add_paragraph()
        r_s = p_s.add_run()
        r_s.text = s
        r_s.font.size = PptPt(14)
        r_s.font.color.rgb = WHITE
        r_s.font.bold = True

    card2 = slide1.shapes.add_shape(1, PptInches(7.0), PptInches(3.8), PptInches(4.8), PptInches(2.8))
    card2.fill.solid()
    card2.fill.fore_color.rgb = CARD_BG
    card2.line.color.rgb = EMERALD
    tf2 = card2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    r = p.add_run()
    r.text = "PROJECT SUPERVISOR:\n\n"
    r.font.bold = True
    r.font.color.rgb = GOLD
    r.font.size = PptPt(13)

    p_sup = tf2.add_paragraph()
    r_sup = p_sup.add_run()
    r_sup.text = "Dr. S. K. THANGARASU, B.E., M.E., Ph.D.\n"
    r_sup.font.size = PptPt(15)
    r_sup.font.bold = True
    r_sup.font.color.rgb = WHITE

    p_des = tf2.add_paragraph()
    r_des = p_des.add_run()
    r_des.text = "Associate Professor\nDepartment of Mechatronics Engineering\nKongu Engineering College, Perundurai"
    r_des.font.size = PptPt(13)
    r_des.font.color.rgb = MUTED

    # ---------- Helper to create 2 column content slides ----------
    def create_2col_slide(title, left_title, left_bullets, right_title, right_bullets):
        s = prs.slides.add_slide(blank_layout)
        style_slide(s, title)

        # Left Card
        c1 = s.shapes.add_shape(1, PptInches(0.8), PptInches(1.8), PptInches(5.6), PptInches(5.0))
        c1.fill.solid()
        c1.fill.fore_color.rgb = CARD_BG
        c1.line.color.rgb = EMERALD
        tf1 = c1.text_frame
        tf1.word_wrap = True
        p1 = tf1.paragraphs[0]
        r1 = p1.add_run()
        r1.text = left_title.upper() + "\n\n"
        r1.font.bold = True
        r1.font.size = PptPt(15)
        r1.font.color.rgb = GOLD
        for b in left_bullets:
            pb = tf1.add_paragraph()
            rb = pb.add_run()
            rb.text = "• " + b + "\n"
            rb.font.size = PptPt(13)
            rb.font.color.rgb = WHITE

        # Right Card
        c2 = s.shapes.add_shape(1, PptInches(6.9), PptInches(1.8), PptInches(5.6), PptInches(5.0))
        c2.fill.solid()
        c2.fill.fore_color.rgb = CARD_BG
        c2.line.color.rgb = EMERALD
        tf2 = c2.text_frame
        tf2.word_wrap = True
        p2 = tf2.paragraphs[0]
        r2 = p2.add_run()
        r2.text = right_title.upper() + "\n\n"
        r2.font.bold = True
        r2.font.size = PptPt(15)
        r2.font.color.rgb = GOLD
        for b in right_bullets:
            pb = tf2.add_paragraph()
            rb = pb.add_run()
            rb.text = "• " + b + "\n"
            rb.font.size = PptPt(13)
            rb.font.color.rgb = WHITE
        return s

    # Slide 2: Abstract & Need
    create_2col_slide(
        "Project Abstract & System Motivation",
        "Project Abstract",
        [
            "Regular agricultural field inspection is vital to identify insect attacks and crop damage early.",
            "Conventional visual inspection is labour-intensive, costly, and difficult to cover large fields consistently.",
            "Combines drone-mounted RGB imaging, AI classification, GPS tagging, and mobile application reporting.",
            "Classifies plant health into: Normal vs Suspected Insect-Affected.",
            "Acts as an intelligent decision-support aid for prioritized field inspection."
        ],
        "Motivation & Problem Need",
        [
            "Quicker coverage of large and uneven plantation areas.",
            "Early identification of pest attacks prevents extensive crop loss.",
            "GPS-linked records allow field workers to revisit exact coordinates.",
            "Reduces human fatigue and dependence on observer bias.",
            "Systematic digital documentation of agricultural health records."
        ]
    )

    # Slide 3: Objectives
    create_2col_slide(
        "Project Objectives & Scope",
        "Primary Objectives",
        [
            "Capture aerial RGB plant images using a drone-mounted camera.",
            "Prepare and curate a labelled dataset for model training and validation.",
            "Classify images into Normal and Suspected Insect-Affected categories using AI.",
            "Associate visual observations with GPS coordinates and timestamps.",
            "Display detection results and flagged alerts via a mobile interface."
        ],
        "Scope & Impact",
        [
            "Target Crop Focus: Agricultural fields plantations.",
            "Screening Automation: Prioritizes expert visits to high-risk zones.",
            "Precision Agriculture: Enhances resource management and targeted treatment.",
            "Sustainable Farming: Contributes to reduced pesticide overuse.",
            "Mechatronics Integration: Merges aerial robotics, computer vision, and IoT."
        ]
    )

    # Slide 4: System Architecture & Workflow
    create_2col_slide(
        "System Architecture & 5-Stage Workflow",
        "5 Core System Modules",
        [
            "1. Image Acquisition Module: Drone-mounted visible-spectrum camera captures high-resolution imagery.",
            "2. Image Preparation Module: Quality checks, normalization, resizing, and preprocessing.",
            "3. AI Classification Module: Deep learning model predicts plant condition.",
            "4. Location Module: Associates GPS coordinates and observation timestamps.",
            "5. Mobile Reporting Module: Visualizes flagged targets on operator UI."
        ],
        "Operational Workflow Steps",
        [
            "Step 1: Plan safe flight path and capture aerial crop images.",
            "Step 2: Preprocess images and feed into trained AI model.",
            "Step 3: Model outputs class prediction (Normal vs Insect-Affected).",
            "Step 4: Attach GPS coordinates to observations.",
            "Step 5: Transmit alerts to mobile interface for field verification."
        ]
    )

    # Slide 5: Equipment & Hardware Details
    create_2col_slide(
        "Hardware Platform & Sensing Equipment",
        "Hardware Sensing Stack",
        [
            "Drone / UAV: Aerial mobility platform with vibration-damped gimbal.",
            "RGB Camera: High-resolution visible spectrum imaging for leaf/crop inspection.",
            "GPS Module: Precision spatial localization for geo-tagging detections.",
            "Processing Device: Onboard or base station computer for AI model inference.",
            "Mobile Device: Field smartphone / tablet for operator monitoring."
        ],
        "Hardware Selection Criteria",
        [
            "Plantation Size & Flight Endurance: Battery optimization for broad field coverage.",
            "Payload Capacity: Balanced weight distribution for stable flight.",
            "Optical Resolution: High pixel density to capture micro-level leaf symptoms.",
            "Environmental Durability: Wind resistance and field weather readiness.",
            "Fail-Safe Operation: Return-to-home and obstacle avoidance protocols."
        ]
    )

    # Slide 6: AI Classification & Software Architecture
    create_2col_slide(
        "AI Architecture & Software Stack",
        "AI Image Processing Pipeline",
        [
            "Input Image Normalization: Standardized resolution and ImageNet color stats.",
            "Deep Learning Feature Extractor: Convolutional neural networks extract visual leaf patterns.",
            "Multi-Class Discrimination: Distinguishes between healthy foliage and insect damage.",
            "Independent Validation: Model evaluated on strictly held-out test splits.",
            "Decision Thresholding: Calibrated confidence metrics for actionable alerts."
        ],
        "Software Ecosystem",
        [
            "Operating System: Windows / Linux workstation environment.",
            "Programming Framework: Python with PyTorch / OpenCV / Ultralytics.",
            "Data Pipeline: Automated image augmentation, bounding box parsing, and scaling.",
            "User Interface: Interactive Cockpit HUD with live telemetry and GPS overlay.",
            "Security & Safety Interlock: Protected operator safety protocols."
        ]
    )

    # Slide 7: Evaluation Metrics
    create_2col_slide(
        "Evaluation Metrics & Verification Plan",
        "Quantitative Metrics",
        [
            "Accuracy: Overall proportion of correctly classified test images.",
            "Precision: Proportion of flagged insect detections that are true positives (reduces false alarms).",
            "Recall: Proportion of affected plants successfully identified (critical for pest containment).",
            "F1-Score: Harmonic mean balancing precision and recall.",
            "Confusion Matrix: Detailed distribution of True/False predictions."
        ],
        "Field Verification Procedure",
        [
            "Held-Out Testing: Testing exclusively on unseen plantation datasets.",
            "Variable Environmental Conditions: Verification across sunlight, shadows, and leaf angles.",
            "Ground Truth Validation: On-field inspection by agricultural experts.",
            "Location Accuracy: Verifying GPS correspondence with actual plant coordinates.",
            "Continuous Improvement: Error analysis feed for model retraining."
        ]
    )

    # Slide 8: SDG Alignment
    create_2col_slide(
        "Sustainable Development Goals (SDGs)",
        "Aligned Goals",
        [
            "SDG 2: Zero Hunger",
            "Supports crop yield protection, minimizes pest-driven food losses, and safeguards agricultural livelihoods.",
            "SDG 9: Industry, Innovation & Infrastructure",
            "Applies cutting-edge artificial intelligence and autonomous robotics to modernize precision agriculture.",
            "SDG 12: Responsible Consumption & Production",
            "Enables targeted pesticide application rather than blanket spraying, protecting soil health and water ecosystems."
        ],
        "Sustainable Impact",
        [
            "Reduces Chemical Runoff: Precision hotspot spraying prevents chemical wastage.",
            "Resource Efficiency: Saves energy, inspection time, and farm operational expenditure.",
            "Food Security: Early insect detection prevents rapid catastrophic infestations.",
            "Data-Driven Decisions: Builds long-term digital crop history for informed farm policies."
        ]
    )

    # Slide 9: Conclusion & Future Scope
    create_2col_slide(
        "Conclusion & Future Enhancements",
        "Conclusion",
        [
            "Proposed a comprehensive AI and drone-assisted smart monitoring workflow.",
            "Combines high-resolution RGB aerial imaging with automated classification.",
            "Bridges the gap between drone scouting and ground-level action via GPS and mobile alerts.",
            "Significantly reduces manual labour while improving systematic crop monitoring records."
        ],
        "Future Enhancements",
        [
            "Multi-Spectral & Thermal Sensors: Detection of water stress and sub-visual disease signatures.",
            "Expanded Crop & Pest Taxonomies: Multi-class pest and disease identification.",
            "Edge AI Deployment: Real-time onboard drone inference (NVIDIA Jetson / Edge TPU).",
            "Autonomous Swarm Scouting: Multi-drone coordinated fleet coverage for mega-plantations."
        ]
    )

    output_pptx = "docs/SMART_MONITORING_SYSTEM_PRESENTATION.pptx"
    prs.save(output_pptx)
    print(f"Presentation saved to: {output_pptx}")

if __name__ == "__main__":
    generate_report()
    generate_presentation()
