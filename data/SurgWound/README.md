---
license: cc-by-sa-4.0
task_categories:
- question-answering
- text-generation
language:
- en
tags:
- medical
- surgical
- wound
pretty_name: SurgWound
size_categories:
- 1K<n<10K
---

# Dataset Card for SurgWound

<!-- Provide a quick summary of the dataset. -->

**SurgWound** is the first open-source dataset for surgical wound analysis across multiple procedure types.
SurgWound comprises 686 surgical wound images, each annotated by surgical experts at The Ohio State University Wexner Medical Center (OSWUMC).
Each image is accompanied by high-quality labels covering six surgical wound characteristic attributes and two diagnostic outcomes attributes.

**SurgWound-Bench** is the first multimodal benchmark for surgical wound analysis, which includes two tasks: SurgWound-VQA and SurgWound-Report

**Published paper:** [npj Digital Medicine](https://doi.org/10.1038/s41746-026-02791-3)
**Preprint:** [arXiv:2508.15189](https://arxiv.org/abs/2508.15189)
**Code:** [GitHub](https://github.com/xuxuxuxuxuxjh/SurgWound)

## Dataset Details

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->


<!-- - **Curated by:** [More Information Needed]
- **Funded by [optional]:** [More Information Needed]
- **Shared by [optional]:** [More Information Needed]
- **Language(s) (NLP):** [More Information Needed]
- **License:** [More Information Needed] -->

**SurgWound**:

SurgWound comprises 686 surgical wound images, accompanied by high-quality labels covering six surgical wound characteristic attributes and two diagnostic outcomes attributes.

+ Location (Abdomen, Patella, Ankle, Facial region, Manus, Cervical region, Other, Uncertain)
+ Healing Status (Healed, Not Healed)
+ Closure Method (Invisible, Sutures, Staples, Adhesives, Uncertain)
+ Exudate Type (Non-existent, Serous, Sanguineous, Purulent, Seropurulent, Uncertain)
+ Erythema (Non-existent, Existent, Uncertain)
+ Edema (Non-existent, Existent, Uncertain)
+ Infection Risk Assessment (Low, Medium, High)
+ Urgency Level (Home Care (Green): Manage with routine care, Clinic Visit (Yellow): Requires professional evaluation within 48 hours, Emergency Care (Red): Seek immediate medical attention)

**SurgWound-Bench**

SurgWound is divided into training, validation, and test sets in a 7:1:2 ratio.
The test set comprises 137 surgical wound images.
Using these images and their corresponding annotations, we design two tasks: 

+ Visual Question Answering (VQA) Task
+ Report Generation Task


<!-- ### Dataset Sources [optional] -->

<!-- Provide the basic links for the dataset. -->

<!-- - **Repository:** [https://github.com/xuxuxuxuxuxjh/SurgWound]
- **Paper [optional]:** [More Information Needed]
- **Demo [optional]:** [More Information Needed] --> 



<!-- ## Uses -->

<!-- Address questions around how the dataset is intended to be used. -->

<!-- ### Direct Use -->

<!-- This section describes suitable use cases for the dataset. -->

<!-- [More Information Needed] -->

<!-- ### Out-of-Scope Use -->

<!-- This section addresses misuse, malicious use, and uses that the dataset will not work well for. -->

<!-- [More Information Needed] -->




## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->

### Data Instances

The following is an example from SurgWound-Bench VQA Task:

```
"id": "76.jpg_closure_method",
"image_name": "76.jpg",
"field": "Closure Method",
"task_type": "multi_choice",
"image": base64encoded_image,
"question": "What is the closure method of this surgical wound?",
"options": [
    "Invisible",
    "Sutures",
    "Staples",
    "Adhesives",
    "Uncertain"
],
"answer": "Sutures"
```

The following is an example from SurgWound-Bench Report Generation Task:

```
"id": "606.jpg_report",
"image_name": "606.jpg",
"field": "Medical Report",
"task_type": "report_generation",
"image": base64encoded_image,
"question": "\nGiven a surgical wound image, generate a detailed medical report that includes the following aspects: wound location, wound status, closure method, exudate characteristics, presence of erythema, presence of edema, urgency level, infection risk assessment.\n",
"answer": "The wound assessment of the healed abdominal wound, closed with sutures, reveals normal imaging status with no signs of exudate, erythema, or edema. The infection risk assessment is low, indicating a favorable prognosis. The urgency level is classified as home care (Green), requiring routine care management. It is recommended to continue monitoring for any signs of infection or delayed healing, and to maintain good hygiene practices to prevent complications. If there are any concerns or changes in the wound condition, prompt follow-up with a healthcare provider is advised."
```

### Data Split

SurgWound is divided into training, validation, and test sets in a 7:1:2 ratio.

+ Train
  + VQA (3435)
  + Report Generation (480)
+ Val
  + VQA (500)
  + Report Generation (69)
+ Test
  + VQA (979)
  + Report Generation (137)

## Dataset Creation

<!-- ### Curation Rationale -->

<!-- Motivation for the creation of this dataset. -->

<!-- [More Information Needed] -->

### Source Data

<!-- This section describes the source data (e.g. news text and headlines, social media posts, translated sentences, ...). -->

We utilize a collection of domain-specific hashtags (e.g., *#surgicalwoundinfection*) and keywords (e.g., *postoperative wound*) to extract relevant content from various platforms, including *RedNote*, *Twitter*, *Facebook*, *Instagram*, and *Reddit*.

#### Data Collection and Processing

<!-- This section describes the data collection and processing process such as data selection criteria, filtering and normalization methods, tools and libraries used, etc. -->

**Data Collection**

We collect surgical wound images from publicly available content on social media platforms.
Specifically, we utilize a collection of domain-specific hashtags (e.g., *#surgicalwoundinfection*) and keywords (e.g., *postoperative wound*) to extract relevant content from various platforms, including *RedNote*, *Twitter*, *Facebook*, *Instagram*, and *Reddit*.
In addition, we further expand the dataset by collecting images specifically from the social media accounts of surgeons and other medical professionals, where postoperative wound cases are often shared for educational or awareness purposes.

**Data Filtering**

We perform a two-stage filtering process involving both AI expert and human expert review to ensure that only high-quality images containing visible surgical wounds are included in the dataset.
First, we leverage GPT-4o as an AI expert to automatically assess whether an image depicts a clear surgical wound, filtering out low-resolution images or those lacking any wound-related content.
Subsequently, three surgeons serve as human experts to manually review the remaining images and exclude any that low-resolution or do not depict authentic surgical wounds.

**Expert Annotation**

To ensure high-quality annotation while optimizing expert effort, we estimate the difficulty level of each image based on the predicted Risk Level outputs from three MLLMs: GPT-4o, Claude 3.5, and Gemini 2.
Images for which all three models consistently predict a Low risk level are considered low-difficulty, while those with inconsistent predictions or predicted as Medium or High risk are categorized as high-difficulty.
For low-difficulty cases, a single surgeon is randomly assigned to perform the annotation.
For high-difficulty cases, the image is independently annotated by two randomly assigned surgeons.
If any disagreement arises between their annotations, a third surgeon is introduced to review both sets of annotations and make the final decision.


<!-- #### Who are the source data producers? -->

<!-- This section describes the people or systems who originally created the data. It should also include self-reported demographic or identity information for the source data creators if this information is available. -->

<!-- [More Information Needed] -->

<!-- ### Annotations [optional] -->

<!-- If the dataset contains annotations which are not part of the initial data collection, use this section to describe them. -->

<!-- #### Annotation process -->

<!-- This section describes the annotation process such as annotation tools used in the process, the amount of data annotated, annotation guidelines provided to the annotators, interannotator statistics, annotation validation, etc. -->

<!-- [More Information Needed] -->

<!-- #### Who are the annotators? -->

<!-- This section describes the people or systems who created the annotations. -->

<!-- [More Information Needed] -->

<!-- #### Personal and Sensitive Information -->

<!-- State whether the dataset contains data that might be considered personal, sensitive, or private (e.g., data that reveals addresses, uniquely identifiable names or aliases, racial or ethnic origins, sexual orientations, religious beliefs, political opinions, financial or health data, etc.). If efforts were made to anonymize the data, describe the anonymization process. -->

<!-- [More Information Needed] -->

<!-- ## Bias, Risks, and Limitations -->

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

<!-- [More Information Needed] -->

<!-- ### Recommendations -->

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

<!-- Users should be made aware of the risks, biases and limitations of the dataset. More information needed for further recommendations. -->

## Citation

```
@article{xu2026surgwoundbench,
      title={SurgWound-Bench: a benchmark for surgical wound diagnosis},
      author={Xu, Jiahao and Yin, Changchang and Chatzipanagiotou, Odysseas P. and Tsilimigras, Diamantis I. and Clear, Kevin and Yao, Bingsheng and Cao, Weidan and Wang, Dakuo and Pawlik, Timothy M. and Zhang, Ping},
      journal={npj Digital Medicine},
      year={2026},
      month={May},
      day={23},
      doi={10.1038/s41746-026-02791-3},
      url={https://doi.org/10.1038/s41746-026-02791-3},
}
```

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

<!-- **BibTeX:** -->

<!-- [More Information Needed] -->

<!-- **APA:** -->

<!-- [More Information Needed] -->

<!-- ## Glossary [optional] -->

<!-- If relevant, include terms and calculations in this section that can help readers understand the dataset or dataset card. -->

<!-- [More Information Needed] -->

<!-- ## More Information [optional] -->

<!-- [More Information Needed] -->

<!-- ## Dataset Card Authors [optional] -->

<!-- [More Information Needed] -->

<!-- ## Dataset Card Contact -->

<!-- [More Information Needed] -->