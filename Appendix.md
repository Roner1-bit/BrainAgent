# BrainAgent — Reproducibility Record (long form)

> **Note.** This is the long-form markdown rendering of the reproducibility
> record and extended methods. The authoritative appendix, whose section
> lettering (Appendix A-B, B-C, C-G …) matches the cross-references in the
> paper, is [`appendix.pdf`](appendix.pdf).

This document reproduces the supplementary appendix of the paper **"BrainAgent: Multi-Agent Fusion for Glioma Report Generation from Multi-Sequence MRI."** It is a complete, verbatim record of the prompts supplied to the report-generation agents, the consolidation orchestrators, and the LLM judge, together with the model inventory and the hyperparameters used for report generation, automated evaluation, and image analysis.

## Table of Contents

- [Reproducibility](#reproducibility)
  - [Multi-Agent Architecture](#multi-agent-architecture)
  - [Models](#models)
  - [Generation and Sampling Settings](#generation-and-sampling-settings)
  - [Report-Generation Agent Prompts](#report-generation-agent-prompts)
  - [Orchestrator Prompts](#orchestrator-prompts)
  - [LLM-as-a-Judge: Finding-Level F1](#llm-as-a-judge-finding-level-f1)
  - [Computer-Vision Hyperparameters](#computer-vision-hyperparameters)
    - [Tumor detection and localization (ConvNextLocator)](#tumor-detection-and-localization-convnextlocator)
    - [Mask refinement (MedSAM2 / SAM2.1 Hiera-Tiny)](#mask-refinement-medsam2--sam21-hiera-tiny)
    - [Preprocessing](#preprocessing)
    - [Skull stripping](#skull-stripping)
    - [Registration and atlases](#registration-and-atlases)
    - [Tumor metrics](#tumor-metrics)
  - [Reproducibility Notes](#reproducibility-notes)
- [Extended Methods (Full Detail Condensed from the Main Text)](#extended-methods-full-detail-condensed-from-the-main-text)
  - [Input Handling and DICOM Conversion](#input-handling-and-dicom-conversion)
  - [Image Preprocessing](#image-preprocessing)
    - [RAS+ Reorientation](#ras-reorientation)
    - [N4 Bias-Field Correction](#n4-bias-field-correction)
    - [Isotropic Resampling](#isotropic-resampling)
  - [Skull Stripping and ANTs](#skull-stripping-and-ants)
  - [Abnormality Segmentation Pipeline](#abnormality-segmentation-pipeline)
    - [Dataset](#dataset)
    - [Automatic Box-Prompting and Segmentation Pipeline](#automatic-box-prompting-and-segmentation-pipeline)
    - [Per-Plane Inference](#per-plane-inference)
    - [Multi-Plane Majority Voting](#multi-plane-majority-voting)
    - [MedSAM2 Refinement](#medsam2-refinement)
    - [Mask Post-Processing](#mask-post-processing)
  - [Tumor Metrics Computation](#tumor-metrics-computation)
    - [Volumetry](#volumetry)
    - [Centroid and Hemisphere Determination](#centroid-and-hemisphere-determination)
  - [MNI-152 Registration](#mni-152-registration)
    - [Registration Procedure](#registration-procedure)
    - [Coordinate Conventions (RAS vs. LPS)](#coordinate-conventions-ras-vs-lps)
    - [Threading and Reproducibility](#threading-and-reproducibility)
    - [MNI Brain Mask](#mni-brain-mask)
  - [Atlas-Based Region Analysis](#atlas-based-region-analysis)
    - [Atlases](#atlases)
    - [Overlap Computation](#overlap-computation)
    - [Per-Region Hemisphere Determination](#per-region-hemisphere-determination)
  - [Slice Extraction and Batch Image Assembly](#slice-extraction-and-batch-image-assembly)
  - [Coordinate System Conventions](#coordinate-system-conventions)
  - [Evaluation Metrics: Finding-Level Precision, Recall, and F1](#evaluation-metrics-finding-level-precision-recall-and-f1)
    - [Per-Category Metrics](#per-category-metrics)

---

## Reproducibility

This appendix provides a reproducible record of (a) every prompt supplied to the report-generation agents, the consolidation orchestrators, and the large-language-model (LLM) judge, and (b) the models and hyperparameters used for report generation, automated evaluation, and image analysis. Brace tokens such as `{patient_id}` are runtime placeholders interpolated when the prompt is rendered. Where a prompt contains an internal inconsistency, both forms are reported and the discrepancy is noted in [Reproducibility Notes](#reproducibility-notes). Implementation-level source code is released separately with the accompanying repository.

### Multi-Agent Architecture

Report generation uses a three-tier hierarchical multi-agent design with a deterministic axial-reference consolidation rule.

- **Tier 1: per-modality/per-view writer agents.** One agent per (modality × view) combination, with modalities {T1, T1c, T2, FLAIR} and views {Axial, Coronal, Sagittal}, giving twelve agents. Each agent produces one initial report and then performs iterative vision-refinement passes over successive image batches.
- **Tier 2: view-level consolidation agents (×3).** One per anatomical plane. Each fuses the four modality reports acquired in its plane into a single coherent view-level report.
- **Tier 3: master orchestrator (×1).** Fuses the axial, coronal, and sagittal view reports into the final report, treating the axial report as the primary reference and resolving contradictions in its favour unless another plane provides unambiguous complementary evidence.

An analysis-mode switch selects the prompt set per patient: when the tumor volume is below 1 cm³ (or no mask is produced), the agents use the *General-Brain* prompt set; otherwise they use the *Tumor-Analysis* prompt set. The visual input to each Tier-1 agent is a sequence of batch grid images, each tiling six slices in a 3×2 layout, which balances anatomical coverage against the input-resolution limits of the vision model. Reasoning ("thinking") mode is disabled for the twelve Tier-1 image-analysis agents and enabled for the Tier-2 and Tier-3 consolidation agents, where cross-modal and cross-planar synthesis benefits from it. The table below summarizes the end-to-end per-patient processing flow.

*End-to-end per-patient processing flow*

| Stage | Description |
|---|---|
| Input handling | File grouping by patient; DICOM-to-NIfTI conversion where required |
| Preprocessing | Reorientation to RAS+, N4 bias-field correction, 1 mm isotropic resampling |
| Skull stripping | Optional removal of non-brain tissue |
| Tumor segmentation | ConvNextLocator detection/localization with MedSAM2 refinement |
| MNI-152 registration | SyN registration to the ICBM152 template |
| Region analysis | Atlas-based labelling (Harvard-Oxford and Pauli) |
| Tumor metrics | Volume, area, centroid, and hemisphere determination |
| Visual assembly | 2D slice extraction and batch-grid image construction |
| Report generation | Three-tier multi-agent generation (this appendix) |
| Output | Final report and 3D surface meshes |

### Models

All twelve writer agents, the three view-level consolidation agents, and the master orchestrator share a single vision-language backbone, Qwen3.5-35B-A3B (a mixture-of-experts model). The finding-level F1 analysis is run with a single LLM judge, gpt-5.4; the six-criterion rubric is completed by the human evaluators (described in the Evaluation section of the main paper), not by the LLM. The remaining entries in the table below are the supporting image-analysis models. The agentic architecture is backbone-agnostic, so the vision-language model can be replaced without changing the pipeline.

*Models used for generation, evaluation, and image analysis.*

| Role | Model | Notes |
|---|---|---|
| Generation backbone (all agents) | Qwen3.5-35B-A3B | MoE vision-language model |
| Finding-level F1 judge | gpt-5.4 | Atomic-finding TP/FP/FN classification |
| Tumor detection/localization | ConvNextLocator | ConvNeXt-Tiny backbone with dual head |
| Mask refinement | MedSAM2 (SAM2.1 Hiera-Tiny) | Box-prompted segmentation |
| Registration | ANTs SyN | Nonlinear MNI-152 registration |
| Skull stripping | HD-BET / ANTs / SynthStrip | Selectable; HD-BET default |
| Registration target | MNI ICBM152 2009a T1 | Fixed template |
| Cortical atlas | Harvard-Oxford | $\approx 48$ cortical regions |
| Subcortical atlas | Pauli 2017 | 16 subcortical nuclei |

### Generation and Sampling Settings

The table below reports the decoding settings for each agent tier. Nucleus sampling uses top-$p = 0.95$; no top-$k$, repetition, presence, or frequency penalties are applied, and no fixed decoding seed is set, so reproducibility relies on the low sampling temperatures together with the fixed registration seed. The LLM judge runs at temperature 0.1.

*Per-tier generation settings. "Reasoning" denotes the chain-of-thought mode.*

| Agent tier | Temperature | Max tokens | Reasoning |
|---|---|---|---|
| Tier 1 initial report | 0.3 | 2500 | off |
| Tier 1 vision refinement | 0.2 | 2500 | off |
| Tier 2 view consolidation | 0.2 | 20000 | on |
| Tier 3 master orchestrator | 0.2 | 20000 | on |

### Report-Generation Agent Prompts

Two short system messages frame the Tier-1 writer-agent calls.

**System Message A.1 — Initial-report system message**

```text
You are an expert radiologist with extensive experience in multi-modal medical imaging analysis, anatomical interpretation, and automated image analysis integration.

IMAGE ORIENTATION: All images use RADIOLOGICAL convention - the patient's RIGHT side appears on the viewer's LEFT, and the patient's LEFT side appears on the viewer's RIGHT. 'R' and 'L' markers on images confirm this convention. When correlating visual observations with the provided region analysis data, trust the automated lateralization labels ('Left'/'Right' prefixes on region names) as these were computed from MNI-152 atlas coordinates.
```

**System Message A.2 — Refinement system message**

```text
You are an expert radiologist specializing in brain tumor analysis with extensive experience in multimodal neuroimaging interpretation.
```

**Tumor-Analysis prompt set**

**Prompt A.1 — Tumor-Analysis: initial report**

```text
You are an expert radiologist generating a comprehensive medical imaging analysis report.
You have been provided with:
1. Quantitative analysis metadata from a VALIDATED AUTOMATED PIPELINE
2. Automated modality detection results
3. A batch image showing multiple slices of the scan WITH TUMOR SEGMENTATION OVERLAYS

PATIENT ID: {patient_id}

TUMOR SEGMENTATION OVERLAY LEGEND (APPLIED TO ALL IMAGES):
- Light Green overlay: Necrotic/Non-Enhancing Tumor Core (NCR/NET) - Class 1
- Light Yellow overlay: Peritumoral Edema (ED) - Class 2
- Light Blue overlay: GD-Enhancing Tumor (ET) - Class 3

These semi-transparent colored overlays have been applied to help you identify and localize
the different tumor components on the MRI slices. The underlying grayscale MRI signal is
still visible through the overlays. Use these overlays to correlate the tumor location
with the quantitative metrics provided below (Note that the quantitative metrics are derived from the validated automated pipeline).

AUTOMATED ANALYSIS RESULTS:
{organ_info}
{modality_info}

QUANTITATIVE ANALYSIS SUMMARY (from validated automated pipeline - TRUST THESE VALUES):
- Tumor Metrics: {tumor_metrics_json}
- Region Analysis: {region_analysis_json}
- Common Locations: {common_locations}

IMPORTANT: The tumor metrics and region analysis above were computed through a validated automated
image processing pipeline using MNI-152 registration and Harvard-Oxford/Pauli atlas mapping.
These quantitative measurements are accurate and should be trusted. Your role is to interpret
these findings in clinical context and correlate them with the visual observations from the image.

REPORTING GUIDELINES FOR QUANTITATIVE DATA:
- Report the tumor VOLUME (in cm3) as a key measurement
- List the affected brain regions with their involvement percentages
- Do NOT include centroid coordinates (X, Y, Z) in the report - these are internal pipeline values
- Focus on clinical interpretation: which regions are affected and what that means clinically

IMAGING CONTEXT:
{context_info_json}

PATIENT & STUDY INFORMATION (from DICOM metadata):
{metadata_summary_json}

NOTE: The metadata_summary above contains:
- patient_info: patient demographics (age, sex, study date)
- scanner_info: MRI scanner details (manufacturer, model, field strength)
- tumor_metrics: automated analysis results
- region_analysis: brain region involvement data

VISUAL ANALYSIS INSTRUCTIONS:
- Carefully analyze the provided batch image showing multiple slices WITH COLORED OVERLAYS
- Use the overlay colors to identify different tumor components (necrotic core=green, edema=yellow, enhancing=blue)
- Correlate visual observations with the quantitative metrics provided
- Note any additional findings visible in the images not captured by automated analysis
- Verify that visual observations are consistent with the reported tumor location and extent
- Describe the spatial relationship between the different tumor components visible through the colored overlays

Please generate a comprehensive initial radiology report including:

1. CLINICAL HISTORY & INDICATION
   - Include detected organ system and imaging approach

2. TECHNIQUE & METHODOLOGY
   - Note automated analysis methods used

3. FINDINGS:
   - Organ-specific observations based on detected anatomy and which areas of the brain were affected by the tumor
   - Tumor Characteristics (size, location, morphology) - correlate with provided metrics
   - Tumor Class Analysis (ET, ED, NCR/NET distribution)
   - Spatial Distribution Analysis
   - Visual observations from the batch image
   - Modality-specific findings

4. IMPRESSION
   - Integrate automated detection results with quantitative analysis and visual findings
   - Consider organ-specific differential diagnoses

5. RECOMMENDATIONS
   - Organ and modality-appropriate follow-up suggestions

IMPORTANT CONSIDERATIONS:
- Use the detected organ type ({primary_organ}) to guide anatomical interpretation
- Incorporate the identified modalities ({detected_modalities_list}) in technical descriptions
- Maintain professional medical language and standard radiology terminology
- The region analysis is comprehensive - address the major affected regions of the brain and classify the tumor based on this information
- DO NOT contradict the quantitative measurements unless you observe a clear discrepancy in the images
- Briefly note the status of the following structures (you should comment about them): bone, brain stem, cerebellum, ventricles, pituitary gland, and vessels. If these structures appear normal, a single sentence noting their normal appearance is sufficient. Do NOT fabricate detailed descriptions of structures you cannot clearly evaluate from the provided images.
```

**Prompt A.2 — Tumor-Analysis: iterative refinement**

```text
You are an expert radiologist performing iterative improvement of a medical imaging analysis report.

PATIENT CONTEXT:
- Primary organ: {detected_organ}
- Detected modalities: {detected_modalities_list}

TUMOR SEGMENTATION OVERLAY LEGEND (APPLIED TO ALL IMAGES):
- Light Green overlay: Necrotic/Non-Enhancing Tumor Core (NCR/NET) - Class 1
- Light Yellow overlay: Peritumoral Edema (ED) - Class 2
- Light Blue overlay: GD-Enhancing Tumor (ET) - Class 3

The MRI slices have semi-transparent colored overlays showing tumor segmentation. Use these
to identify and localize different tumor components while observing the underlying MRI signal.

CURRENT REPORT STATUS:
{current_report}

IMPROVEMENT STAGE: {stage_number}
CURRENT BATCH: {image_name}

CRITICAL CONTEXT ABOUT QUANTITATIVE DATA:
The tumor metrics (volume) and region analysis (affected brain regions) were already provided in the
initial report generation. These values were computed through a VALIDATED AUTOMATED IMAGE PROCESSING
PIPELINE using MNI-152 registration and Harvard-Oxford/Pauli atlas mapping.

DO NOT modify, contradict, or re-estimate these quantitative values. They are accurate measurements
from the automated pipeline. Your task is to REFINE THE VISUAL DESCRIPTIONS AND CLINICAL INTERPRETATION
based on the new batch of images, while keeping the quantitative data consistent.

REMINDER: Report tumor volume and affected regions with percentages. Do NOT add centroid coordinates to the report.

INSTRUCTIONS:
1. Carefully analyze the provided medical imaging batch WITH COLORED OVERLAYS
2. Use the overlay colors to identify tumor components (green=necrotic, yellow=edema, blue=enhancing)
3. Compare visual findings with the current report
4. Leverage the known organ type ({detected_organ}) and modalities for accurate interpretation
5. Improve the report by:
   - Adding NEW specific visual observations from this batch
   - Describing the spatial distribution of different tumor components (green, yellow, blue regions)
   - Enhancing descriptions based on organ-specific anatomy visible in this batch
   - Refining qualitative descriptions (e.g., signal characteristics, enhancement patterns)
   - Adding modality-specific clinical correlations
   - Incorporating organ-appropriate differential considerations

6. MAINTAIN the overall structure and ALL quantitative measurements from the initial report
7. Remove "INITIAL REPORT - PENDING VISUAL CONFIRMATION" if still present
8. Add specific visual observations from this batch
9. Integrate findings coherently with previous observations

FOCUS AREAS FOR {detected_organ} IMAGING:
- Organ-specific anatomical landmarks and normal variants
- Pathology patterns typical for this organ system
- Modality-specific signal characteristics and enhancement patterns
- Spatial relationships relevant to this anatomical region
- Clinical significance in the context of this organ system
- Briefly note the status of the following structures (you should comment about them): bone, brain stem, cerebellum, ventricles, pituitary gland, and vessels. If these structures appear normal, a single sentence noting their normal appearance is sufficient. Do NOT fabricate detailed descriptions of structures you cannot clearly evaluate from the provided images.

Provide the COMPLETE IMPROVED REPORT (not just changes). Preserve all quantitative measurements exactly as stated in the current report.
```

**General-Brain prompt set**

Used when the tumor volume is below 1 cm³ or no tumor is detected.

**Prompt A.3 — General-Brain: initial report**

```text
You are an expert neuroradiologist generating a comprehensive brain MRI analysis report.

IMPORTANT CLINICAL CONTEXT:
The automated tumor detection pipeline did NOT identify a significant tumor mass in this scan.
This could mean:
1. The brain appears normal with no detectable lesions
2. Any abnormalities are subtle or non-specific
3. The findings may represent normal variants, artifacts, or incidental findings

YOUR TASK: Perform a THOROUGH general brain analysis looking for ANY abnormalities, not just tumors.

PATIENT ID: {patient_id}

PATIENT & STUDY INFORMATION (from DICOM metadata):
{metadata_summary_json}

NOTE: The metadata_summary above contains:
- patient_info: patient demographics (age, sex, study date)
- scanner_info: MRI scanner details (manufacturer, model, field strength)
- tumor_metrics: automated detection results (volume was below threshold)

IMAGING PARAMETERS:
{context_info_json}

VISUAL ANALYSIS INSTRUCTIONS:
You are being shown a comprehensive set of brain MRI slices from the {modality_name} sequence.
Since no significant tumor was detected by automated analysis, carefully examine:

1. BRAIN PARENCHYMA:
   - Gray matter / white matter differentiation
   - Any focal or diffuse signal abnormalities
   - Presence of encephalomalacia, gliosis, or atrophy
   - Symmetry between hemispheres

2. VENTRICULAR SYSTEM:
   - Size and configuration of lateral ventricles
   - Third and fourth ventricles
   - Any hydrocephalus or asymmetry
   - Periventricular signal changes

3. EXTRA-AXIAL SPACES:
   - Sulci and cisterns (prominent? effaced?)
   - Subdural or epidural collections
   - Meningeal enhancement (if contrast used)

4. VASCULAR STRUCTURES:
   - Major arteries and venous sinuses
   - Any vascular malformations or aneurysms
   - Flow voids

5. SKULL AND CALVARIUM:
   - Bone marrow signal
   - Calvarial thickness and integrity
   - Skull base abnormalities

6. POSTERIOR FOSSA:
   - Cerebellum (vermis and hemispheres)
   - Brain stem (midbrain, pons, medulla)
   - Cerebellar tonsils position

7. SELLA AND PARASELLAR REGION:
   - Pituitary gland size and signal
   - Optic chiasm and nerves
   - Cavernous sinuses

8. ORBITS AND PARANASAL SINUSES:
   - Globe and optic nerve appearance
   - Sinus aeration and mucosal thickening

9. INCIDENTAL FINDINGS:
   - White matter hyperintensities
   - Choroid plexus cysts
   - Pineal cysts
   - Arachnoid cysts
   - Developmental venous anomalies

Please generate a comprehensive radiology report including:

1. CLINICAL HISTORY & INDICATION
   - Note the clinical question and any relevant patient history

2. TECHNIQUE
   - Describe the MRI sequences analyzed ({detected_modalities_list})
   - Note any technical limitations

3. FINDINGS:
   - Systematic evaluation of all brain structures
   - Document normal findings as well as any abnormalities
   - Be thorough - even if the brain appears grossly normal, document this systematically

4. IMPRESSION
   - Summary of significant findings (or confirmation of normal study)
   - Any recommendations for additional imaging or clinical correlation

5. RECOMMENDATIONS
   - Follow-up if needed
   - Clinical correlation suggestions

CRITICAL REMINDER:
- The automated analysis did NOT detect a significant tumor
- Your role is to provide a comprehensive brain evaluation
- Document both normal and abnormal findings
- If the brain truly appears normal, state this confidently
- Look for subtle findings that automated detection might miss
```

**Prompt A.4 — General-Brain: iterative refinement**

```text
You are an expert neuroradiologist performing iterative improvement of a general brain MRI analysis report.

CONTEXT:
- No significant tumor was detected by automated analysis
- You are analyzing {modality_name} images for any brain abnormalities
- This is improvement stage {stage_number}
- Current batch: {image_name}

CURRENT REPORT STATUS:
{current_report}

MODALITY BEING ANALYZED: {detected_modalities_list}

INSTRUCTIONS:
1. Carefully analyze the provided brain MRI slices
2. Compare findings with the current report
3. Look for ANY abnormalities, not just tumors:
   - Signal changes in white matter
   - Atrophy patterns
   - Vascular abnormalities
   - Extra-axial collections
   - Ventricular abnormalities
   - Skull/bone findings
   - Incidental findings

4. Improve the report by:
   - Adding NEW specific observations from this batch
   - Refining descriptions of normal or abnormal structures
   - Ensuring comprehensive coverage of all visible anatomy
   - Adding modality-specific findings (e.g., FLAIR for white matter, T1c for enhancement)

5. MAINTAIN the overall structure and completeness of the report
6. Document NORMAL findings - stating "normal" is clinically important

FOCUS AREAS FOR THIS BATCH:
- Anatomical structures visible in these slices
- Any subtle signal abnormalities
- Comparison with expected normal appearance
- Documentation of normal structures (equally important as abnormalities)

Provide the COMPLETE IMPROVED REPORT (not just changes).
```

### Orchestrator Prompts

**Tier-1 structured mini-report.** The Tier-1 agents emit a structured per-modality report validated against a schema (embedded in the prompt as `{escaped_schema_json}`).

**System Message A.3 — Tier-1 structured mini-report (system)**

```text
You are a board-certified radiologist writing a meticulous, objective, and consistent mini-report for a patient with a brain tumor. ONLY for the assigned modality. Use precise medical terminology. Analyze both the text and the image (if provided). If data is insufficient, explicitly state limitations. Prioritize safety-critical findings. Return a structured JSON that validates against the provided schema.
```

**Prompt A.5 — Tier-1 structured mini-report (user)**

```text
Modality: {modality}
Clinical history/indication: {patient_context}
Clinical question: {clinical_question}
Comparison: {comparison}
Modality-specific input (raw text):
{modality_input}

Instructions:
- Analyze the provided image in conjunction with the text.
- Technique: briefly describe acquisition relevant to this modality.
- Findings: objective bullet points based on text and image; do not include interpretation here.
- Impression: numbered, prioritized, clinically relevant interpretations for THIS modality only.
- Recommendations: follow-up imaging/clinical steps when appropriate; be concise.
- Limitations: note artifacts, motion, limited coverage, contrast timing issues, etc.
- Critical findings: list urgent findings explicitly, if any.
Ensure factual consistency with provided input and history.

Output requirements:
- Return ONLY a valid JSON object conforming to this JSON Schema:
{escaped_schema_json}
- Do not include any explanatory text or markdown code fences.
```

**Tier-2 view-level consolidation**

**System Message A.4 — View-level consolidation (system)**

```text
You are a senior neuroradiologist writing a consolidated report for a specific anatomical view ({view_name}) of a brain MRI study.

You are provided with reports from multiple MRI sequences (T1, T1c, T2, FLAIR) all acquired in the {view_name} plane.
Your task is to synthesize these modality-specific reports into a single, coherent view-level report.

IMPORTANT CONTEXT:
- Tumor volume and affected brain regions were computed through a VALIDATED AUTOMATED IMAGE PROCESSING
  PIPELINE using MNI-152 registration and Harvard-Oxford/Pauli atlas mapping.
- These measurements are ACCURATE and should be preserved exactly as reported.
- Your role is to integrate findings across modalities, not to re-estimate measurements.
- Do NOT include centroid coordinates (X, Y, Z) in the report.

SYNTHESIS GUIDELINES:
1. Integrate complementary information from each modality:
   - T1: Anatomical detail, gray/white matter differentiation
   - T1c (contrast-enhanced): Enhancement patterns, blood-brain barrier disruption, tumor vascularity
   - T2: Edema, fluid content, cystic components
   - FLAIR: Periventricular lesions, subtle edema, infiltrative margins

2. Resolve any apparent inconsistencies by considering modality-specific characteristics
3. Highlight findings that are consistently observed across multiple modalities
4. Note any modality-specific findings that provide unique diagnostic information
5. Do NOT mention segmentation overlays, colored overlays (green, yellow, blue), or any visualization
   artifacts - these are internal pipeline tools, not clinical findings. Use standard clinical terms
   (necrotic core, enhancing component, peritumoral edema) instead.

Write in professional medical language. Create a comprehensive view-specific report.
```

**Prompt A.6 — View-level consolidation (user)**

```text
Patient context: {patient_context}
Anatomical View: {view_name}
Clinical question: {clinical_question}
Comparison: {comparison}

Modality reports for {view_name} view (from T1, T1c, T2, FLAIR sequences):
{modality_reports_json}

Based on the above modality reports for the {view_name} view, write a consolidated report with:
1. REPORT FINDINGS: Single comprehensive paragraph describing all findings observed in this anatomical plane
2. REPORT IMPRESSION: Single paragraph with clinical interpretation specific to observations in this view

Output requirements:
- Return ONLY a valid JSON object conforming to this JSON Schema:
{json_schema}
- Do not include any explanatory text or markdown code fences.
```

**Tier-3 master orchestrator**

The axial report is treated as the reference.

**System Message A.5 — Master orchestrator (system)**

```text
You are the chief neuroradiologist providing the FINAL authoritative interpretation of a brain MRI study.

You are provided with three consolidated view-level reports:
1. AXIAL VIEW REPORT: This is the GOLD STANDARD and PRIMARY reference
2. CORONAL VIEW REPORT: Secondary/supplementary view
3. SAGITTAL VIEW REPORT: Secondary/supplementary view

YOUR ROLE: Synthesize the view reports into ONE UNIFIED clinical narrative. You are NOT comparing
views or listing findings per-sequence. Write as if you personally reviewed the entire study and
are dictating a single, cohesive report.

CRITICAL RULES:

1. **AXIAL IS THE GOLD STANDARD**:
   - The Axial view report is the PRIMARY and most reliable source of information
   - All quantitative measurements from the Axial report should be preserved EXACTLY as stated
   - If Coronal or Sagittal reports CONTRADICT the Axial report -> TRUST THE AXIAL REPORT
   - If Coronal or Sagittal provide COMPLEMENTARY details consistent with Axial -> Include them

2. **QUANTITATIVE DATA**:
   - Report tumor VOLUME (cm3) this is the only numerical measurement to include
   - List only the TOP 5 (or fewer) MOST affected brain regions by name - no percentages, no coordinates
   - Do NOT include centroid coordinates (X, Y, Z)
   - Do NOT include region involvement percentages

3. **DO NOT MENTION**:
   - Segmentation overlays, colored overlays, green/yellow/blue regions, or any visualization artifacts
     (these are internal pipeline tools for the AI, NOT clinical findings)
   - Individual MRI sequences by name (T1, T2, FLAIR, T1c) as separate analysis sections - instead,
     describe signal characteristics naturally (e.g., "hyperintense on T2/FLAIR", "enhancing on post-contrast")
   - Which view (axial, coronal, sagittal) a finding came from - present findings as unified observations
   - Raw data like slice indices, voxel counts, or class labels (Class 1, Class 2, etc.)

4. **REPORT STYLE**:
   - Write in professional medical language as if dictating the final official radiology report
   - Use complete sentences in paragraph form (no bullet points or numbered lists)
   - The report should read as a unified interpretation by a single radiologist, not as a collation of sub-reports
   - Describe tumor components using standard clinical terms (necrotic core, enhancing component,
     peritumoral edema) NOT overlay colors or segmentation class names
```

**Prompt A.7 — Master orchestrator (user)**

```text
Patient context: {patient_context}
Clinical question: {clinical_question}
Comparison: {comparison}

=== AXIAL VIEW REPORT (GOLD STANDARD - PRIMARY REFERENCE) ===
{axial_report_json}

=== CORONAL VIEW REPORT (SUPPLEMENTARY) ===
{coronal_report_json}

=== SAGITTAL VIEW REPORT (SUPPLEMENTARY) ===
{sagittal_report_json}

INSTRUCTIONS:
1. Use the AXIAL report as your primary source of truth
2. Check Coronal and Sagittal reports for ADDITIONAL information that complements the Axial findings
3. If Coronal/Sagittal contradict Axial -> Use Axial findings only
4. Preserve tumor volume exactly as stated in the Axial report
5. List only the top 7 (or fewer) most affected brain regions by name - NO percentages
6. Do NOT mention segmentation overlays, colored regions, or visualization tools
7. Write a UNIFIED narrative - do not attribute findings to specific views or sequences separately

Write your FINAL radiology report with:
1. REPORT FINDINGS: Single comprehensive paragraph describing all findings (Axial as base + supplementary info from other views)
2. REPORT IMPRESSION: Single paragraph with clinical interpretation and recommendations

Output requirements:
- Return ONLY a valid JSON object conforming to this JSON Schema:
{json_schema}
- Do not include any explanatory text or markdown code fences.
```

### LLM-as-a-Judge: Finding-Level F1

The finding-level judge decomposes both the reference and AI reports into atomic findings across eleven categories (primary lesion presence, lesion location, lesion size, signal characteristics, mass effect, edema, structural involvement, secondary observations, diagnosis/impression, differential diagnoses, and recommendations), classifies each finding as true positive, false positive, false negative, or true negative by semantic matching, and reports

$$\mathrm{Precision} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}, \qquad \mathrm{Recall} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}, \qquad F_1 = \frac{2\,\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}.$$

Micro metrics pool counts across patients before applying these formulas; macro metrics average per-patient values; per-category metrics are computed likewise. True negatives are recorded but excluded from the metrics.

**Judge Prompt A.1 — Finding-level F1 judge: system prompt**

```text
You are a senior neuroradiologist serving as an impartial judge for an
automated evaluation of AI-generated brain MRI radiology reports. Your task
is to convert each report pair into a structured finding-level inventory
and classify every finding as True Positive (TP), False Positive (FP), or
False Negative (FN) according to the protocol below.

=== ATOMIC FINDING EXTRACTION ===
Decompose BOTH the reference report and the AI report into discrete atomic
findings. Each finding must be a SINGLE indivisible observation. For
example, the sentence "large mass in the right frontal lobe with
surrounding edema" decomposes into:
  - (primary_lesion_presence) mass present
  - (lesion_location) right frontal lobe
  - (lesion_size) large mass
  - (edema) perilesional edema present

Assign every finding to exactly ONE of these 11 categories:
  1. primary_lesion_presence    - mass/tumor/lesion present or absent
  2. lesion_location            - anatomical location, laterality, lobe, region
  3. lesion_size                - measurements, dimensions, size descriptors
  4. signal_characteristics     - T1/T2/FLAIR signal, enhancement pattern
  5. mass_effect                - midline shift, compression, sulcal effacement
  6. edema                      - perilesional/vasogenic edema presence and extent
  7. structural_involvement     - corpus callosum, white matter tracts, specific structures
  8. secondary_observations     - hydrocephalus, herniation, vascular encasement, etc.
  9. diagnosis_impression       - primary diagnosis, tumor type, WHO grade
 10. differential_diagnoses     - listed differentials and their ranking/likelihood
 11. recommendations            - follow-up imaging, surgery, biopsy, consultations

=== CLASSIFICATION RULES ===

EMIT EXACTLY ONE CLASSIFICATION PER UNIQUE JUDGMENT. Do NOT emit mirror
entries for the same match (e.g., once from the reference side and once
from the AI side) -- each matched pair is ONE TP, not two.

For each MATCHED PAIR (a reference finding that has a semantically
equivalent AI finding):
  - Emit exactly ONE classification entry:
      classification = "TP"
      finding_text   = <the AI-side finding>
      matched_to     = <the reference-side finding>
      source         = "ai"
      category       = <one of the 11 categories>
  - Do NOT emit a separate TP entry for the reference side.

For each AI finding that was not matched to a reference finding:
  - If the AI finding CONTRADICTS the reference (e.g., opposite laterality,
    different tumor size, different diagnosis category, a pathology the
    reference explicitly rules out) -> classification="FP", source="ai".
  - If the AI finding is ADDITIONAL DETAIL that neither contradicts nor is
    supported-or-denied by the reference (the reference is simply silent
    on it) -> classification="TP", source="ai", matched_to=null. The AI
    report in this pipeline is known to be more sophisticated than the
    reference; radiologists have validated that its extra secondary
    diagnoses / secondary observations / precise measurements / deeper
    anatomical detail are clinically correct. Do NOT penalize additional
    non-contradicting detail. Err on the side of TP when uncertain whether
    the reference is silent vs. contradicting.

For each REFERENCE finding that was NOT matched (the AI omitted it):
  - IF the reference finding is a NORMAL-STRUCTURE MENTION or a ROUTINE
    NEGATIVE FINDING (see rule below), and the AI is simply SILENT on it
    (AI neither confirms nor contradicts) -> classification="TN",
    finding_text=<reference finding>, source="reference". TN entries are
    NOT counted in Precision / Recall / F1, so they do not penalize the
    AI for not enumerating every normal structure.
  - OTHERWISE (the reference finding is a POSITIVE ABNORMAL finding that
    the AI failed to report, OR a clinically load-bearing negative that
    the AI ignored) -> classification="FN", source="reference".

===== NORMAL-STRUCTURE / ROUTINE-NEGATIVE RULE (TN vs FN) =====
The AI pipeline is not instructed to exhaustively enumerate normal
anatomy. When the reference describes a structure or region as NORMAL /
UNREMARKABLE / PRESERVED / PATENT / INTACT / SYMMETRIC, and the AI
report is SILENT on that structure, treat the reference statement as a
TRUE NEGATIVE (agreement by omission), NOT a false negative.

Apply the TN rule to reference findings such as:
  - "Brainstem is normal" / "Cerebellum is normal" / "Vermis is normal"
  - "Basal ganglia are normal" / "Thalami are normal"
  - "Pituitary gland / stalk is normal"
  - "Optic chiasm is normal" / "Orbits are normal"
  - "Paranasal sinuses are normal" / "Calvarium is intact"
  - "Gray-white matter differentiation is preserved"
  - "Cerebral sulci and gyri are normal"
  - "Subarachnoid spaces are normal"
  - "Ventricular system is normal in size"
  - Routine negative checklist items the reference lists reflexively:
    "No cortical infarction", "No lacunar infarction", "No intra-axial
    hemorrhage", "No extra-axial hemorrhage", "No acute infarction",
    "No hydrocephalus" UNLESS the AI also addresses the same item
    (in which case it's TP or FP as usual) or the AI contradicts it
    (in which case the AI claim is FP).

Do NOT apply the TN rule to:
  - POSITIVE ABNORMAL findings in the reference (mass present, edema
    present, shift present, restricted diffusion present, abnormal
    enhancement, infarction present, hemorrhage present, lesion in X
    location). If the AI omits a positive abnormal finding, it's FN.
  - Clinically load-bearing negatives that the AI directly addresses 
    these follow the normal TP/FP match logic.
  - Any finding where the AI explicitly contradicts the reference 
    these remain FP on the AI side.

Expected total classification entries = (# AI findings) + (# unmatched
reference findings). Of those, TP + FP + FN are the metric-contributing
entries; TN entries are recorded for audit but excluded from metrics.
One AI statement MAY satisfy multiple reference findings; in that case
emit multiple TP entries, each pointing at a different reference
finding via matched_to.

=== SEMANTIC MATCHING ===
Match findings by meaning, not by exact string. A reference finding and an
AI finding match if they describe the same underlying observation:
  - "right frontal mass" matches "lesion in the right frontal lobe".
  - "3.2 cm tumor" matches "approximately 3 cm tumor" (small rounding ok).
  - "high-grade glioma" matches "GBM / Grade IV glioma" (more specific AI
    version of a reference finding counts as TP).
  - "midline shift" matches "mass effect with shift of midline structures".
Partial matches where the CORE observation is the same count as TP.

=== TRUE NEGATIVES ===
Do not enumerate TNs. The space of absent findings is effectively infinite;
metrics rely only on TP, FP, and FN.

=== OUTPUT ===
Return ONLY a single valid JSON object (no markdown fences, no prose before
or after). Every finding in the reference_findings array must receive a
classification (TP or FN) in the classifications array. Every finding in
the ai_findings array must receive a classification (TP or FP) in the
classifications array. The counts you report are sanity checks; the final
metrics will be recomputed from your classifications array.
```

**Judge Prompt A.2 — Finding-level F1 judge: user prompt**

```text
## REFERENCE REPORT (Ground Truth)
{reference_report}

## AI-GENERATED REPORT (To Be Evaluated)
{ai_report}

---

Decompose both reports into atomic findings, assign each to one of the 11
categories, then classify every finding as TP, FP, FN, or TN per the
rules in the system prompt. Remember:
  - Additional non-contradicting AI detail is TP, not FP.
  - Reference findings that just list structures as normal / unremarkable
    / preserved or are routine negative checklist items, and that the AI
    is silent on, are TN (not FN). TNs do not count against the AI.
  - Only direct contradictions count as FP.

Return ONLY this JSON structure:

{
  "patient_id": "{patient_id}",
  "reference_findings": [
    {"category": "<category>", "text": "<atomic finding>"}
  ],
  "ai_findings": [
    {"category": "<category>", "text": "<atomic finding>"}
  ],
  "classifications": [
    {
      "finding_text": "<text of the finding being classified>",
      "category": "<one of the 11 categories>",
      "classification": "TP" | "FP" | "FN" | "TN",
      "source": "reference" | "ai" | "both",
      "matched_to": "<counterpart text if TP, else null>",
      "rationale": "<short reason>"
    }
  ],
  "extra_ai_detail": [
    {"category": "<category>", "text": "<atomic finding>"}
  ],
  "counts": {"tp": <int>, "fp": <int>, "fn": <int>, "tn": <int>},
  "notes": "<1-3 sentences on notable patterns, e.g. which categories dominated FN or FP>"
}

Notes on the structure:
- extra_ai_detail lists the AI findings that were credited as TP because
  they were additional (non-contradicting) detail absent from the
  reference. These are a SUBSET of the AI findings already labeled TP in
  classifications; do not double-count them elsewhere.
- Emit ONE classification per unique finding-level judgment. Matched
  pairs produce a single TP entry (attached to the AI-side text with
  matched_to=<reference-side text>), NOT two. Total entries in
  classifications should equal TP + FP + FN.
```

### Computer-Vision Hyperparameters

#### Tumor detection and localization (ConvNextLocator)

The detector uses a ConvNeXt-Tiny backbone with a classification head (tumor presence) and a bounding-box regression head, operating on 2.5D input formed by stacking the previous, current, and next slices as three channels. The table below lists the operative settings. Twelve weight sets are maintained, one per (modality × plane) combination; per-plane masks are combined by majority voting (at least two of three planes).

*ConvNextLocator settings.*

| Parameter | Value |
|---|---|
| Input size | 256×256, three channels (previous/current/next slice) |
| Confidence threshold | 0.5 (slice classified tumor-positive if $\sigma(\mathrm{logit}) \ge 0.5$) |
| GPU batch size | 16 |
| Classification head dropout | 0.4 |
| Bounding-box output | $[c_x, c_y, w, h]$ via sigmoid, normalized to $[0,1]$ |
| Per-slice normalization | min-max to $[0,255]$ |
| Resize interpolation | bilinear (image), nearest-neighbor (mask) |
| Multi-plane combination | majority vote ($\ge 2$ of 3 planes) |
| Primary modality | FLAIR (median-volume consensus when all modalities used) |
| Volume conversion | $\mathrm{cm^3} = \text{voxels} \times \prod(\text{spacing})/1000$ |

#### Mask refinement (MedSAM2 / SAM2.1 Hiera-Tiny)

The detected bounding box is supplied as the sole prompt to MedSAM2 (no point prompts), with multi-mask output enabled and the highest-scoring mask retained. The table below lists the principal configuration values.

*MedSAM2 / SAM2.1 Hiera-Tiny configuration.*

| Setting | Value | Setting | Value |
|---|---|---|---|
| Image size | 1024 | Memory-attention layers | 4 |
| Backbone embed dim | 96 | Feed-forward dim | 2048 |
| Backbone heads | 1 | RoPE theta | 10000 |
| Backbone stages | [1,2,7,2] | RoPE feature sizes | [64,64] |
| Global attention blocks | [5,7,9] | Memory-encoder out dim | 64 |
| Neck model dim | 256 | Mask-memory length | 7 |
| Neck channel list | [768,384,192,96] | Sigmoid scale (mem.) | 20.0 |
| Neck top-down levels | [2,3] | Sigmoid bias (mem.) | -10.0 |

#### Preprocessing

Volumes are reoriented to RAS+, bias-corrected, and resampled to 1 mm isotropic spacing (see the table below).

*Image-preprocessing parameters.*

| Parameter | Value |
|---|---|
| Target orientation | RAS+ |
| Target spacing | 1.0 × 1.0 × 1.0 mm |
| N4 shrink factor | 4 |
| N4 convergence iterations / tolerance | [50,50,30,20] / $10^{-7}$ |
| N4 B-spline knot spacing | 200 mm |
| N4 tissue-mask threshold | 10% of mean intensity |
| Minimum volume threshold | 1000 voxels |
| Resample interpolation | linear (images) / nearest-neighbor (masks) |

#### Skull stripping

The default method is HD-BET (test-time augmentation enabled), with ANTs template-based extraction and SynthStrip as alternatives. Quality control accepts a brain-to-image volume ratio in $[0.15, 0.95]$ by default ($[0.15, 0.90]$ for T1/T1c, $[0.12, 0.85]$ for T2/FLAIR) and an absolute brain volume of 800–2000 cm³, warning on removal below 5 cm³ and failing on removal above 1500 cm³.

#### Registration and atlases

Registration to the MNI-152 ICBM152 2009a template uses ANTs symmetric diffeomorphic normalization (SyN); the random seed is fixed for reproducibility (see the table below).

*Registration and atlas parameters.*

| Parameter | Value |
|---|---|
| Transform | SyN (symmetric diffeomorphic), mutual-information metric |
| Interpolation | linear (anatomical) / nearest-neighbor (mask) |
| Registration seed | fixed |
| MNI template | ICBM152 2009a, 1 mm resolution |
| Brain-mask threshold | 1% of template maximum intensity |
| Midline threshold | 5.0 mm ($X < -5$ left, $X > 5$ right, $\lvert X\rvert \le 5$ bilateral) |
| Cortical atlas | Harvard-Oxford, max-probability at 25% |
| Subcortical atlas | Pauli 2017, probability threshold 0.05 |

#### Tumor metrics

Tumor sub-compartments follow the BraTS labelling convention: necrotic/non-enhancing tumor core (NCR/NET), peritumoral edema (ED), and GD-enhancing tumor (ET). Per-class and total volumes are obtained by multiplying the voxel count by the voxel volume and converting to cm³; the centroid and hemisphere are determined in registered space using the same 5 mm midline threshold as the region analysis.

### Reproducibility Notes

2. **Reasoning mode.** Reasoning ("thinking") mode is disabled for the twelve Tier-1 image-analysis agents and enabled for the Tier-2 and Tier-3 consolidation agents.
3. **Decoding determinism.** No fixed decoding seed is set for the language models; reproducibility relies on the low sampling temperatures (0.1–0.3) together with the fixed registration seed.

---

## Extended Methods (Full Detail Condensed from the Main Text)

To meet the journal page limit, the main paper presents the imaging pipeline in condensed form. This section reproduces the full technical detail of the modules that were shortened, for reproducibility. The content is reproduced from the pre-condensation manuscript.

### Input Handling and DICOM Conversion

The pipeline supports five different folder structures to handle varied clinical and research data sources. These include institutional structures where patient identifiers organize subdirectories with per-modality NIfTI files. It also includes flat naming conventions that encode patient and modality information in the filename. Additionally, it supports DICOM-based formats from clinical PACS systems, which consist of hierarchical study/series directory structures and flat single-folder organizations.

The metadata of the patient is usually extracted from the DICOM files if they are available, especially the age, as this might provide a better context for the agents, which might lead to a better analysis of the patient. For convenience, DICOM images are converted to NIfTI.

### Image Preprocessing

All input volumes (image preprocessing is performed in 3D) undergo a three-stage standardization pipeline designed to match a unified characterization and handle any data inconsistencies (1 mm isotropic resolution in RAS+ orientation).

#### RAS+ Reorientation

All images are reoriented to the Right-Anterior-Superior (RAS+) canonical orientation through analyzing the NIfTI affine matrix and applying the necessary axis permutations and flips. In the RAS+ convention, the X-axis encodes right (+) to left ($-$), the Y-axis encodes anterior (+) to posterior ($-$), and the Z-axis encodes superior (+) to inferior ($-$). This standardizing process is critical for three downstream operations: (i) hemisphere determination in tumor metrics, where centroid $X < 0$ indicates left hemisphere; (ii) MNI-152 registration, which assumes RAS+ input; and (iii) consistent laterality marker placement on visualization slices.

#### N4 Bias-Field Correction

Intensity non-uniformity arising from RF coil inhomogeneity is corrected using the N4 algorithm (N4ITK). The correction operates at four multi-resolution levels with iteration numbers of [50, 50, 30, 20] and convergence tolerance of $10^{-7}$, using a shrink factor of 4 for computational efficiency and a B-spline knot spacing of 200 mm. A non-zero tissue mask is generated by thresholding at 10% of the mean image intensity. Bias correction is performed before skull stripping because the additional tissue context (including the skull) provides a more complete basis for estimating the low-frequency bias field. The estimated bias field is saved as a separate volume for quality verification.

#### Isotropic Resampling

Volumes are resampled to 1 mm isotropic voxel spacing using linear interpolation for anatomical images. This target spacing is configurable via the pipeline configuration settings. Segmentation masks are resampled with nearest-neighbor interpolation to preserve discrete label values. A minimum volume threshold of 1,000 voxels is enforced to reject corrupt or truncated input files.

### Skull Stripping and ANTs

Skull stripping is an optional preprocessing step; it can be disabled if the data is already skull-stripped, even though, if left enabled, it would automatically detect whether the skull is stripped. When enabled, the module supports three algorithms: (i) **HD-BET**, a deep-learning approach using a U-Net architecture that serves as the default method due to its proven high accuracy across different MRI contrasts; (ii) **ANTs** template-based brain extraction, which propagates a template brain mask to patient space via registration; and (iii) **SynthStrip** from the FreeSurfer suite. Skull stripping operates on preprocessed (bias-corrected, resampled) images rather than raw inputs, because cleaner input produces higher-quality brain masks. The module implements graceful fallback: if skull stripping fails for a given modality, the pipeline reverts to using the preprocessed image with skull intact.

### Abnormality Segmentation Pipeline

Tumor segmentation applies a multi-plane voting strategy with the use of ConvNextLocator, a convolutional neural network built on the ConvNeXt feature-extraction backbone.

![Overview of the proposed fully automatic segmentation pipeline.](Segmentation.png)

*Overview of the proposed fully automatic segmentation pipeline, which runs in three successive stages: tumor-presence detection, bounding-box localization, and prompt-based segmentation. A ConvNeXt-Tiny detector operating on 2.5D input (the previous, current, and next slices stacked as three channels) classifies each slice for tumor presence and, where a tumor is detected, regresses a bounding box around it. The predicted box is passed as the sole prompt to MedSAM2, which produces a contour-following tumor mask. Inference is performed independently in the axial, coronal, and sagittal planes, and the three per-plane masks are fused by majority voting, labeling a voxel as tumor when at least two of the three planes agree (see Multi-Plane Majority Voting below).*

#### Dataset

We conducted experiments on the UCSF-PDGM dataset, which contains preoperative multi-parametric brain MRI scans with corresponding expert tumor annotations for patients with diffuse glioma. In this study, we used four MRI sequences available for each case, namely T1, T1c, T2, and FLAIR. These modalities provide complementary anatomical and pathological information and are widely used in brain tumor analysis.

To match the proposed 2.5D formulation, each 3D volume was processed as a sequence of 2D slices. For every central slice, its adjacent previous and next slices were stacked to form a three-channel input. We further evaluated the method in axial, coronal, and sagittal views to examine its robustness across anatomical planes. Ground-truth tumor masks were used to generate slice-level labels for tumor presence and to derive bounding boxes for localization supervision. Slices containing no tumor region were considered negative examples, whereas slices with tumor pixels were used for both classification and box regression training.

This dataset configuration supports the full pipeline studied in this work, including slice-level tumor detection, automatic bounding-box prediction, and prompt-driven segmentation with MedSAM2. Evaluating the framework across multiple MRI sequences and views provides a detailed understanding of its generalization behavior under varying image contrasts and anatomical orientations.

#### Automatic Box-Prompting and Segmentation Pipeline

Our proposed pipeline is designed to achieve fully automatic brain tumor segmentation from MRI without manual interaction at inference time. The framework consists of three successive stages: tumor-presence detection, bounding-box localization, and prompt-based segmentation. Instead of relying on a user-provided box, the system first analyzes each MRI slice to determine whether tumor tissue is present. When an abnormality is detected, the model predicts a bounding box around the suspicious region, which is then used as the prompt for MedSAM2 to generate the final tumor mask.

The detection-localization module is based on a ConvNeXt-Tiny backbone and uses 2.5D input formed by stacking the previous, current, and next slices as three channels. This design allows the model to capture short-range inter-slice context while preserving the efficiency of 2D processing. On top of the shared backbone, the network uses two task-specific branches. The classification branch determines whether a slice is tumor-positive, while the localization branch predicts the bounding-box coordinates. For localization, the regression head operates on the stage3 feature map. It begins with Group Normalization, followed by a $3 \times 3$ convolution with padding 1 that reduces the channel dimension from 384 to 128 while refining spatial information. A GELU activation is then applied, after which the feature map is flattened and passed through a fully connected layer that projects it into a 256-dimensional hidden representation. After a second GELU activation, a final linear layer produces four outputs, and a sigmoid activation constrains them to $[0,1]$, resulting in a normalized bounding box in center-coordinate format.

During training, the classification branch is optimized using focal loss to address the strong imbalance between tumor-positive and tumor-negative slices. The localization branch is trained with a hybrid loss combining L1 and Complete IoU, encouraging both coordinate accuracy and geometric agreement between predicted and target boxes. In addition, area-aware weighted sampling is employed to increase the representation of slices containing small tumors, which are generally harder to localize and more sensitive to prompt errors. Once the predicted box is obtained, it is provided to MedSAM2 as an automatic prompt, enabling a fully automated segmentation workflow that preserves the strengths of prompt-based foundation models while removing the need for manual initialization.

#### Per-Plane Inference

For each of the three canonical anatomical planes (axial, coronal, sagittal), the 3D volume is decomposed into 2D slices along the corresponding axis (Z for axial, Y for coronal, X for sagittal). Each slice undergoes preprocessing to match the training data pipeline: a 90-degree rotation for display orientation, intensity normalization to the $[0, 255]$ range, and bilinear resize to $256 \times 256$ pixels. Slices are processed in GPU batches of 16. For each slice, the model outputs a confidence score $c \in [0, 1]$ and a normalized bounding box $[\hat{c}_x, \hat{c}_y, \hat{w}, \hat{h}]$. Slices with $c > 0.5$ (configurable threshold) are classified as tumor-containing, and the bounding box is converted to pixel coordinates to generate a binary mask within the detected region. After inference, the *inverse* of the preprocessing rotation is applied to map each 2D mask back into the correct 3D volume coordinates.

A critical implementation detail is that the ConvNextLocator models were trained on PNG slices extracted with a specific 90-degree rotation transform. To maintain inference compatibility, the identical rotation is applied during slice extraction, and its inverse is applied when reassembling the 3D mask.

Twelve sets of model weights are maintained, one per combination of four modalities (T1, T1c, T2, FLAIR) and three planes (axial, coronal, sagittal), enabling modality-specific and view-specific inference.

#### Multi-Plane Majority Voting

The three per-plane 3D masks are combined using majority voting: a voxel is classified as tumor if and only if at least two out of three planes agree on its tumor status:

$$M_{\text{final}}(v) = \mathbb{1}\!\left[\sum_{p \in \{\text{ax}, \text{cor}, \text{sag}\}} \mathbb{1}[M_p(v) > 0] \geq 2\right]$$

where $M_p(v)$ denotes the binary mask value at voxel $v$ from plane $p$. This strategy significantly reduces false positives arising from partial-volume artifacts or imaging artifacts in a single viewing plane, while maintaining sensitivity by requiring agreement from only two of three independent perspectives.

#### MedSAM2 Refinement

When enabled, the Segment Anything Model for Medical images (MedSAM2), configured with the tiny hierarchical design variant, refines the bounding-box detections from ConvNextLocator. For each tumor-containing slice, the detected bounding box is passed to the MedSAM2 predictor, which generates a precise segmentation mask with anatomically aware boundaries within the bounding box. This replaces the simple rectangular fill with a contour-following mask. If MedSAM2 refinement fails for any slice, the system falls back to the ConvNextLocator box mask for that slice.

#### Mask Post-Processing

After segmentation, a critical alignment step resamples the generated mask to match the spatial frame (orientation, resolution, grid) of the preprocessed structural images using nearest-neighbor interpolation. This prevents geometric errors including left-right flips, when the MNI warp is subsequently applied to the mask.

The complete segmentation process generates full traceability records including per-slice confidence scores, bounding box coordinates, tumor pixel counts, and per-plane processing times, enabling post-hoc quality assessment of every inference decision.

### Tumor Metrics Computation

The tumor metrics module computes quantitative measurements from the segmentation mask following the BraTS multi-class labeling convention: label 1 for necrotic/non-enhancing tumor core (NCR/NET), label 2 for peritumoral edema (ED), label 3 for gadolinium-enhancing tumor (ET), and label 4 as an alternative enhancing tumor label used in some datasets. It is very important to note that the tumor is calculated for each sequence separately, and then, based on the results, we obtain a tumor size for each MRI sequence. Consequently, this module calculates the median tumor size and, based on it, selects the tumor segmentation closest to the median as the primary sequence for the remaining modules.

#### Volumetry

Total and per-class tumor volumes are computed as:

$$V_k = N_k \cdot \Delta x \cdot \Delta y \cdot \Delta z \cdot 10^{-3} \quad [\text{cm}^3],$$

where $N_k$ is the number of voxels with label $k$ and $(\Delta x, \Delta y, \Delta z)$ are the voxel dimensions in millimeters obtained from the image header. Per-slice tumor areas are computed along the axial axis as $A_s = n_s \cdot \Delta x \cdot \Delta y$ [mm²], where $n_s$ is the number of tumor voxels in slice $s$.

#### Centroid and Hemisphere Determination

The tumor centroid is computed in world coordinates using the NIfTI affine transformation matrix $\mathbf{A}$:

$$\mathbf{c}_{\text{world}} = \mathbf{A} \cdot \begin{bmatrix} \bar{i} \\ \bar{j} \\ \bar{k} \\ 1 \end{bmatrix},$$

where $(\bar{i}, \bar{j}, \bar{k})$ is the mean voxel position of all tumor voxels. When available, the MNI-registered mask is preferred for centroid computation because MNI space provides a reliable anatomical midline at $X = 0$, independent of potentially inconsistent native-space DICOM orientation headers. Hemisphere determination uses a 5 mm threshold around the midline: $c_x < -5$ mm indicates left hemisphere, $c_x > +5$ mm indicates right hemisphere, and $|c_x| \leq 5$ mm indicates midline/bilateral involvement.

### MNI-152 Registration

Patient brain volumes are registered to the MNI-152 standard space (ICBM 2009a, non-linear symmetric, 1 mm resolution, $197 \times 233 \times 189$ voxels) using the ANTs Symmetric Normalization (SyN) algorithm. SyN is a diffeomorphic registration method that produces smooth, invertible deformation fields, preserving brain topology while permitting large inter-subject anatomical differences.

#### Registration Procedure

The registration is performed on a reference structural image, preferring skull-stripped data (registration also runs on skull-retained images but yields better results on skull-stripped inputs, as used in our pipeline), and produces two transform components: a non-linear warp field stored as a volumetric deformation image and a linear affine matrix. The combined transform maps any point from native patient space to MNI space. These transforms are then applied to all remaining modality volumes and the segmentation mask. Linear interpolation is used for anatomical images; nearest-neighbor interpolation is used for segmentation masks to preserve discrete label values.

#### Coordinate Conventions (RAS vs. LPS)

A critical technical consideration concerns the handling of coordinate conventions. All NIfTI volumes are stored in the RAS (Right-Anterior-Superior) convention, whereas ANTs operates internally in the LPS (Left-Posterior-Superior) convention. The NIfTI loading mechanism within ANTs automatically handles this RAS-to-LPS conversion. An early implementation discovered that directly constructing ANTs image objects from numpy arrays with spatial metadata extracted from NIfTI headers in RAS convention caused systematic left-right flips, because the registration framework interpreted the RAS-encoded origin and direction vectors as LPS. The solution is to always load images from disk through the standard ANTs file reader, which performs the conversion automatically. Additionally, input volumes must be cast to float32 prior to loading, as the registration framework may return all-zero results for unsigned or signed 16-bit integer data types.

#### Threading and Reproducibility

The registration module dynamically distributes 80% of available CPU cores (minimum of ~4) and enforces deterministic performance by fixing the random seed. Thread counts are propagated to ITK, OpenMP, MKL, OpenBLAS, and NumExpr through environment variables at module load time, assuring consistent thread utilization across all numerical libraries involved in the registration.

#### MNI Brain Mask

After registration, a binary brain mask is created by thresholding the MNI template at 1% of its maximum intensity. This mask is applied to registered anatomical images to remove registration artifacts outside the brain boundary. Importantly, the mask is *not* applied to registered segmentation masks, as tumor regions extending beyond the template brain boundary need to be preserved.

### Atlas-Based Region Analysis

Tumor extent is mapped to named anatomical brain regions using two complementary standard brain atlases.

#### Atlases

The Harvard-Oxford cortical atlas provides 48 cortical regions as a maximum probability map thresholded at 25%, with 0-based label indexing where the first label ("Background") maps to atlas value 0. The Pauli 2017 subcortical atlas provides 16 subcortical nuclei (including caudate nucleus, putamen, globus pallidus, thalamus, hippocampus, amygdala, nucleus accumbens, and subthalamic nucleus) with 1-based label indexing where the first label in the list maps to atlas value 1 rather than 0. This indexing distinction is handled via an explicit atlas-type parameter, eliminating a class of subtle off-by-one errors that arise when both atlases are processed with the same indexing logic. In other words, based on this module, we provide the affected regions of the tumor to the agents to minimize the room for errors and hallucinations.

#### Overlap Computation

Each atlas is first resampled to the MNI-registered mask grid using nearest-neighbor interpolation to ensure voxel-by-voxel alignment. For each atlas region, the overlap is computed as the number of voxels where both the tumor mask and the region mask are non-zero:

$$O_r = \sum_v \mathbb{1}[M(v) > 0] \cdot \mathbb{1}[A(v) = r],$$

where $M(v)$ is the tumor mask value, $A(v)$ is the atlas label, and $r$ is the region index. The overlap volume is computed as $V_r = O_r \cdot \Delta v \cdot 10^{-3}$ [cm³] where $\Delta v$ is the voxel volume in mm³, and the percentage involvement is $P_r = 100 \cdot O_r / N_{\text{total}}$. Results are sorted by percentage involvement in descending order and output as both tabular data (per-region metrics with region name, voxel count, volume, and percentage) and a human-readable markdown summary (which is fed to the agents to provide better context on which brain regions are affected).

#### Per-Region Hemisphere Determination

Hemisphere assignment is performed per-region rather than globally, because a tumor spanning the midline may overlap with different atlas regions in different hemispheres. For each region with non-zero overlap, the mean MNI-space position of the overlapping voxels is calculated using the affine matrix and classified using the same 5 mm midline threshold described in Tumor Metrics Computation. Region labels are prefixed with the determined hemisphere (e.g., "Left Frontal Pole", "Right Insular Cortex", or "Bilateral Thalamus").

### Slice Extraction and Batch Image Assembly

Two stages prepare visual inputs for the LLM agents. First, individual 2D PNG slices are extracted from each modality volume across all three anatomical orientations, with a semi-transparent, color-coded bounding box surrounding the necrotic core, colored light green. This semi-transparency level provides sufficient overlay visibility without obscuring the underlying MRI signal-intensity patterns that the vision-language agents must interpret. Laterality markers ("R" and "L") are added according to radiological convention: on axial and coronal views, "R" appears on the viewer's left, corresponding to the patient's right hemisphere (to provide correct orientation to the agents).

Second, individual slices are assembled into grid images ($3 \times 2 = 6$ slices per grid), producing batch images that serve as the primary visual input to the LLM vision agents. This batching strategy harmonizes information density — ensuring sufficient anatomical coverage per image — with the input-resolution constraints of current vision-language models.

### Coordinate System Conventions

All images in the pipeline are standardized to the RAS+ convention (Right-Anterior-Superior), consistent with the NIfTI standard and MNI-152 space. The X-axis encodes right (+) to left ($-$), the Y-axis encodes anterior (+) to posterior ($-$), and the Z-axis encodes superior (+) to inferior ($-$). ANTs internally operates in the LPS (Left-Posterior-Superior) convention; the conversion between RAS and LPS is handled automatically during standard NIfTI file loading within the ANTs framework. An early implementation identified a systematic left-right flip caused by supplying RAS-encoded spatial metadata directly to an interface that expected LPS-encoded values, stressing the importance of using the standard file-loading pathway that performs convention conversion automatically.

### Evaluation Metrics: Finding-Level Precision, Recall, and F1

From the aggregate counts (TP, FP, FN), we compute standard information-retrieval metrics. Precision quantifies the proportion of reported findings that are correct,

$$\mathrm{Precision} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}},$$

such that low Precision signals hallucination. Recall quantifies the proportion of reference findings that were captured,

$$\mathrm{Recall} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}},$$

such that low Recall signals omission. The two are combined through their harmonic mean, which penalizes imbalance between Precision and Recall and yields a single balanced score,

$$F_{1} = 2 \cdot \frac{\mathrm{Precision} \cdot \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}.$$

#### Per-Category Metrics

In addition to the aggregate triple $(\mathrm{Precision}, \mathrm{Recall}, F_1)$, we compute per-category $(\mathrm{Precision}_c, \mathrm{Recall}_c, F_{1,c})$ for each of the eleven finding categories $c$. Category-level metrics expose systematic failure modes that aggregate scores may obscure: uniformly high $F_1$ with degraded performance restricted to the lesion-location category, for example, would indicate a lateralization or localization failure rather than a global reporting deficit, and likewise for diagnosis, recommendations, and other categories. For each evaluated patient, the judge emits a structured output containing the two per-report finding inventories with category labels, the per-finding classifications with short rationales, the aggregate TP, FP, and FN counts, the overall $(\mathrm{Precision}, \mathrm{Recall}, F_1)$ triple, and the per-category metrics.
