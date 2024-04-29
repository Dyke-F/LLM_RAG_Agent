__all__ = "cases"


xing = (
    """Ms. Xing, a 33-year-old patient, was diagnosed with cholangiocellular carcinoma in early 2023. Following her diagnosis, she underwent a complete surgical resection (R0) and was treated with adjuvant chemotherapy using capecitabine for six months. Subsequent imaging showed no evidence of disease recurrence until September of the same year, when MRI imaging ("September2023.png") revealed a new, solitary metastasis.
    In response, a treatment regimen of gemcitabine and cisplatin was initiated. Considering Ms. Xing's young age, a liver biopsy was performed for comprehensive panel diagnostics. Both imaging results and panel findings are made available for review:

***
Radiology Report, MRI Abdomen September 2023
Mrs Xing, *02/03/1987
Procedure: MRI Abdomen with iv Contrast
Clinical Question: Post Cholangiocellular carcinoma, new metastasis? Recurrence? Staging.
Technique: MRI Abdomen
Findings:
Liver: There is one single hypointense lesion in the left lobe of the liver, measuring approximately 0.4 cm in its greatest dimension (Location: [475, 250, 490, 275]). Highly suggestive for recurrence of the known cholangiocellular carcinoma.
Biliary Tree: The intrahepatic and extrahepatic bile ducts are not dilated. 
Pancreas: The pancreas appears regular in size, shape, and enhancement, showing no abnormalities such as masses, cysts, or dilation of ducts.
Spleen: The spleen is within normal limits regarding size and density, with no detected lesions.
Adrenal Glands: Both adrenal glands present a normal appearance in terms of size and structure.
Kidneys: The kidneys are normal in all aspects of size, contour, and functionality, with no signs of masses, stones, or hydronephrosis.
Bowel: No obstructions or abnormal thickening is observed in the bowel. 
Vessels: The abdominal aorta and its main branches are unobstructed, with no aneurysms or significant narrowing observed.
Lymph Nodes: There is no enlargement of the abdominal or pelvic lymph nodes.
Peritoneum: No evidence of peritoneal implants or significant fluid accumulation is present.
Impression: 
Strong suspicion of a recurrence of the known CCC due to a solitary metastasis in the left liver.

Molecular Report Summary:
Microsatellite-Instability High, Mutation in BRAF V600E and CD74-ROS1 variant fusion.
No alterations in NTRK or FGFR2.
***

Currently, the patient experiences recurrent ascites with notable abdominal tension.
Also, a new MRI-scan was performed yesterday (“February2024.png”) by our in house radiologists.
According to the latest radiology report, the MRI scan performed on the patient with the system ID X-09.22 and documented as "February2024.png" revealed a single metastatic lesion in the left lobe of the liver, delineated by the coordinates 455, 270, 505, 320. The report emphasizes the importance of a comprehensive review by comparing these findings with the images from a previous examination in September 2023 during the forthcoming tumor board meeting for a detailed analysis. Also, the imaging findings suggest the presence of peritoneal carcinomatosis. Her system ID is X-09.22.""",

"Please investigate the current state of the disease in detail. What does the new MRI scan show? Did the metastasis grow? In case of progress, what targeted or chemotherapy options could we suggest for the tumor board tomorrow? What treatment options does the patient still have according to the official guidelines for cholangiocellular cancer given her medical history. Be aware that the patient is extremely young and we need all options we could find. Also check any options you can find on google or pubmed."
)


all_vars = globals().copy()
cases = {
    str(k): v
    for k, v in all_vars.items()
    if isinstance(v, tuple) and not k.startswith("__")
}
