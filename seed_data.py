import json
import shutil
from datetime import datetime
from pathlib import Path
import cv2

from database import DB_PATH, STORAGE_PATH, get_connection, init_db
from app import _process_image

PROJECT_ROOT = Path(__file__).resolve().parent
TEST_IMG_DIR = PROJECT_ROOT / "test_images"


def seed_database():
    print("Initializing database schema...")
    init_db()
    conn = get_connection()
    cur = conn.cursor()

    # Clear existing demo data to ensure fresh seed
    cur.execute("DELETE FROM clinician_reviews")
    cur.execute("DELETE FROM clinical_assessments")
    cur.execute("DELETE FROM ai_analyses")
    cur.execute("DELETE FROM visits")
    cur.execute("DELETE FROM wounds")
    cur.execute("DELETE FROM patients")
    conn.commit()

    print("Pre-processing test images for seed data...")
    # Cache AI results for the 5 test images
    img_names = ["fusc_0096.png", "fusc_0106.png", "fusc_0061.png", "fusc_0032.png", "fusc_0048.png"]
    processed = {}
    for name in img_names:
        img_path = TEST_IMG_DIR / name
        if not img_path.exists():
            print(f"Warning: {img_path} does not exist")
            continue
        b = img_path.read_bytes()
        res = _process_image(b, name)
        processed[name] = res

    now_iso = datetime.now().isoformat()

    # 1. Patients
    patients = [
        ("P-00042", "David Miller", 1, 62, "Male", "Type 2 Diabetes Mellitus, Peripheral Neuropathy", "+1-555-0142", "2026-04-20"),
        ("P-00018", "Radhika Sharma", 1, 58, "Female", "Chronic Venous Insufficiency, Bilateral Lower Extremity Edema", "+1-555-0118", "2026-05-10"),
        ("P-00029", "Saranya Patel", 1, 45, "Female", "Post-operative Cesarean Wound Dehiscence", "+1-555-0129", "2026-07-28"),
        ("P-00055", "Marcus Vance", 1, 71, "Male", "Spinal Cord Injury, Immobile, Stage II Sacral Pressure Ulcer", "+1-555-0155", "2026-09-12"),
    ]
    cur.executemany("INSERT INTO patients VALUES (?, ?, ?, ?, ?, ?, ?, ?)", patients)

    # 2. Wounds
    wounds = [
        ("W-0007", "P-00042", "Diabetic Foot Ulcer", "Right plantar foot", "Diabetic Neuropathic", "2026-04-20", "Active"),
        ("W-0003", "P-00018", "Venous Leg Ulcer", "Left medial malleolus", "Venous Stasis", "2026-03-15", "Active"),
        ("W-0012", "P-00029", "Surgical Incision Wound", "Lower abdominal midline", "Post-surgical Dehiscence", "2026-08-01", "Active"),
        ("W-0021", "P-00055", "Pressure Injury", "Sacral prominence", "Pressure / Shear Injury", "2026-09-12", "Active"),
    ]
    cur.executemany("INSERT INTO wounds VALUES (?, ?, ?, ?, ?, ?, ?)", wounds)
    conn.commit()

    # Helper to insert visit + AI + clinical + review
    def insert_visit(visit_id, wound_id, patient_id, visit_date, week_label, img_file, source,
                     clinical_data, review_data, patient_inputs=None):
        img_source = TEST_IMG_DIR / img_file
        img_dest = STORAGE_PATH / f"{visit_id}_{img_file}"
        shutil.copy2(img_source, img_dest)

        p_pain = patient_inputs.get("pain") if patient_inputs else None
        p_symp = patient_inputs.get("symptoms", "") if patient_inputs else ""
        p_comm = patient_inputs.get("comments", "") if patient_inputs else ""

        cur.execute("""
            INSERT INTO visits (id, wound_id, patient_id, visit_date, week_label, image_path, source,
                                patient_pain_score, patient_symptoms, patient_comments, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (visit_id, wound_id, patient_id, visit_date, week_label, str(img_dest), source, p_pain, p_symp, p_comm, now_iso))

        # AI Result
        ai = processed.get(img_file)
        if ai:
            mask_dest = STORAGE_PATH / f"{visit_id}_mask.png"
            vis_dest = STORAGE_PATH / f"{visit_id}_vis.jpg"
            cv2.imwrite(str(mask_dest), ai["segmentation"].mask)
            if Path(ai["visualization_path"]).exists():
                shutil.copy2(ai["visualization_path"], vis_dest)

            area = float(cv2.countNonZero(ai["segmentation"].mask))
            peri_area = float(ai["peri_wound"].peri_wound_area)
            conf = float(ai["segmentation"].confidence or 0.0)

            cur.execute("""
                INSERT INTO ai_analyses (
                    id, visit_id, wound_area, peri_wound_area, detection_confidence,
                    healing_probability, not_healed_probability, healing_class,
                    infection_low_prob, infection_med_prob, infection_high_prob, infection_class,
                    healing_component, infection_component, tissue_component, btis,
                    mask_path, visualization_path, top_shap_json, handcrafted_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"AI-{visit_id}", visit_id, area, peri_area, conf,
                float(ai["healing_probability"]), float(ai["not_healed_probability"]), str(ai["healing_class"]),
                float(ai["infection_probability"][0]), float(ai["infection_probability"][1]), float(ai["infection_probability"][2]),
                str(ai["infection_class"]),
                float(ai["healing_component"]), float(ai["infection_component"]), float(ai["tissue_component"]), float(ai["btis"]),
                str(mask_dest), str(vis_dest),
                json.dumps(ai.get("healing_top", [])),
                json.dumps({k: float(v) for k, v in ai.get("handcrafted", {}).items() if isinstance(v, (int, float))}),
            ))

        # Clinical Assessment
        cur.execute("""
            INSERT INTO clinical_assessments (
                id, visit_id, wound_bed, periwound_condition, exudate_amount, exudate_type,
                pain_score, odor, perfusion, neuropathy, infection_signs, wound_depth,
                undermining_tunneling, dressing_management, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"CA-{visit_id}", visit_id,
            clinical_data.get("wound_bed", "Granulation tissue 60%, slough 40%"),
            clinical_data.get("periwound", "Mild erythema, intact edges"),
            clinical_data.get("exudate_amount", "Moderate"),
            clinical_data.get("exudate_type", "Serosanguinous"),
            clinical_data.get("pain", 4),
            clinical_data.get("odor", "None"),
            clinical_data.get("perfusion", "Dorsalis pedis pulse 2+, capillary refill < 2s"),
            clinical_data.get("neuropathy", "Intact sensation"),
            clinical_data.get("infection_signs", "None observed"),
            clinical_data.get("depth", "Partial thickness, 2mm"),
            clinical_data.get("undermining", "None observed"),
            clinical_data.get("dressing", "Hydrofiber with silver dressing, offloading diabetic boot"),
            now_iso
        ))

        # Clinician Review
        cur.execute("""
            INSERT INTO clinician_reviews (
                id, visit_id, review_status, overall_clinical_status, clinician_confirmation,
                clinical_notes, follow_up_date, reviewed_by, review_date, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"CR-{visit_id}", visit_id,
            review_data.get("review_status", "Reviewed"),
            review_data.get("overall_status", "Stable"),
            review_data.get("confirmation", "Agreed with AI findings"),
            review_data.get("notes", "Routine check completed."),
            review_data.get("follow_up", "2026-09-24"),
            review_data.get("reviewed_by", "Dr. Eleanor Vance, MD (Wound Care)"),
            review_data.get("review_date", visit_date),
            now_iso
        ))

    # --- Seed Visits for David Miller (P-00042, Diabetic Foot Wound W-0007) ---
    print("Seeding exactly 10 distinct visits for David Miller...")
    # 10 weeks with progressing, distinct wound photos and clinical trajectory:
    insert_visit("V-DAVID-01", "W-0007", "P-00042", "2026-05-12", "Week 1 — Initial Presentation", "fusc_0096.png", "Clinician Visit",
        {"wound_bed": "Granulation 75%, Slough 25%", "periwound": "Mild maceration", "pain": 3, "depth": "Partial thickness 1.5mm"},
        {"review_status": "Reviewed", "overall_status": "Stable", "confirmation": "Baseline wound established", "notes": "Initial wound baseline documented. Offloading footwear prescribed."})

    insert_visit("V-DAVID-02", "W-0007", "P-00042", "2026-05-26", "Week 2 — Early Monitoring", "fusc_0061.png", "Clinician Visit",
        {"wound_bed": "Granulation 70%, Slough 30%", "periwound": "Intact edges", "pain": 4, "depth": "Partial thickness 2.0mm"},
        {"review_status": "Reviewed", "overall_status": "Stable", "confirmation": "Confirmed offloading compliance", "notes": "Patient reports minor discomfort after walking. Reinforced offloading."})

    insert_visit("V-DAVID-03", "W-0007", "P-00042", "2026-06-09", "Week 3 — Active Treatment", "fusc_0032.png", "Clinician Visit",
        {"wound_bed": "Granulation 80%, Slough 20%", "periwound": "Clean, healthy margin", "pain": 3, "depth": "Partial thickness 1.8mm"},
        {"review_status": "Reviewed", "overall_status": "Improving", "confirmation": "Satisfactory wound bed", "notes": "Epithelialization advancing along lateral edge."})

    insert_visit("V-DAVID-04", "W-0007", "P-00042", "2026-06-23", "Week 4 — Treatment Check", "fusc_0106.png", "Clinician Visit",
        {"wound_bed": "Granulation 65%, Slough 35%", "periwound": "Mild peri-wound warmth", "pain": 4, "depth": "Partial thickness 2.0mm"},
        {"review_status": "Reviewed", "overall_status": "Stable", "confirmation": "Continued surveillance required", "notes": "Area slightly expanded. Swab culture negative."})

    insert_visit("V-DAVID-05", "W-0007", "P-00042", "2026-07-07", "Week 5 — Plateau Phase", "fusc_0048.png", "Clinician Visit",
        {"wound_bed": "Granulation 70%, Slough 30%", "periwound": "Intact edges", "pain": 3, "depth": "Partial thickness 1.9mm"},
        {"review_status": "Reviewed", "overall_status": "Stable", "confirmation": "Plateau observed", "notes": "Dressing refreshed."})

    insert_visit("V-DAVID-06", "W-0007", "P-00042", "2026-07-21", "Week 6 (Patient Upload)", "fusc_0096.png", "Patient Upload",
        {"wound_bed": "Granulation 75%, Slough 25%", "periwound": "Clean", "pain": 2, "depth": "Partial thickness 1.7mm"},
        {"review_status": "Reviewed", "overall_status": "Improving", "confirmation": "Good healing", "notes": "Home photo verified."})

    insert_visit("V-DAVID-07", "W-0007", "P-00042", "2026-08-04", "Week 7 (Patient Upload)", "fusc_0061.png", "Patient Upload",
        {"wound_bed": "Granulation 80%, Slough 20%", "periwound": "Healthy", "pain": 2, "depth": "Partial thickness 1.5mm"},
        {"review_status": "Reviewed", "overall_status": "Improving", "confirmation": "Continuing reduction", "notes": "Wound area contracting."})

    insert_visit("V-DAVID-08", "W-0007", "P-00042", "2026-08-18", "Week 8 (Patient Upload)", "fusc_0032.png", "Patient Upload",
        {"wound_bed": "Granulation 85%, Slough 15%", "periwound": "Intact", "pain": 1, "depth": "Superficial 1.2mm"},
        {"review_status": "Reviewed", "overall_status": "Improving", "confirmation": "Epithelial coverage >80%", "notes": "Marked improvement."})

    insert_visit("V-DAVID-09", "W-0007", "P-00042", "2026-09-01", "Week 9 (Patient Upload)", "fusc_0106.png", "Patient Upload",
        {"wound_bed": "Granulation 90%, Slough 10%", "periwound": "Normal", "pain": 1, "depth": "Superficial 0.8mm"},
        {"review_status": "Reviewed", "overall_status": "Improving", "confirmation": "Near epithelialization", "notes": "Marginal contraction."})

    insert_visit("V-DAVID-10", "W-0007", "P-00042", "2026-09-17", "Week 10 — Clinical Concern ⚠️", "fusc_0048.png", "Clinician Visit",
        {"wound_bed": "Granulation 55%, Slough 45%, friable base", "periwound": "Marked periwound erythema", "exudate_amount": "Heavy", "exudate_type": "Purulent", "pain": 6, "depth": "Partial to full 3.0mm"},
        {"review_status": "Review required", "overall_status": "Worsening", "confirmation": "Pending review", "notes": "Acute enlargement."})
    # --- Seed Visits for Radhika Sharma (P-00018, Venous Leg Ulcer W-0003) ---
    print("Seeding visits for Radhika Sharma...")
    insert_visit(
        "V-RADHIKA-01", "W-0003", "P-00018", "2026-06-01", "Week 1", "fusc_0048.png", "Clinician Visit",
        {"wound_bed": "Slough 50%, Granulation 50%", "periwound": "Hemosiderin staining, venous dermatitis", "pain": 5, "depth": "Shallow 1.0mm"},
        {"review_status": "Reviewed", "overall_status": "Stable", "confirmation": "Initiated 4-layer compression", "notes": "Initiated compression bandaging."}
    )
    insert_visit(
        "V-RADHIKA-02", "W-0003", "P-00018", "2026-07-15", "Week 7", "fusc_0032.png", "Clinician Visit",
        {"wound_bed": "Granulation 75%, Slough 25%", "periwound": "Reduced edema", "pain": 3, "depth": "Superficial 0.8mm"},
        {"review_status": "Reviewed", "overall_status": "Improving", "confirmation": "Edema decreased", "notes": "Area decreased significantly with compression."}
    )
    insert_visit(
        "V-RADHIKA-03", "W-0003", "P-00018", "2026-09-10", "Current (Week 15)", "fusc_0106.png", "Clinician Visit",
        {"wound_bed": "Granulation 90%, Slough 10%", "periwound": "Healthy epithelial rim", "pain": 1, "depth": "Superficial 0.5mm"},
        {"review_status": "Reviewed", "overall_status": "Improving", "confirmation": "Healing well", "notes": "Wound area reduced by 39.8%. Compression therapy effective.", "follow_up": "2026-10-08"}
    )

    # --- Seed Visits for Saranya Patel (P-00029, Surgical Incision W-0012) ---
    print("Seeding visits for Saranya Patel...")
    insert_visit(
        "V-SARANYA-01", "W-0012", "P-00029", "2026-08-15", "Week 1", "fusc_0061.png", "Clinician Visit",
        {"wound_bed": "Clean surgical bed, minimal slough", "periwound": "Well-approximated edges", "pain": 2, "depth": "Partial thickness"},
        {"review_status": "Reviewed", "overall_status": "Stable", "confirmation": "Clean surgical site", "notes": "Post-surgical check. No signs of deep infection."}
    )
    insert_visit(
        "V-SARANYA-02", "W-0012", "P-00029", "2026-09-05", "Current (Week 4)", "fusc_0061.png", "Clinician Visit",
        {"wound_bed": "Epithelializing nicely, no drainage", "periwound": "Dry, minimal pinkness", "pain": 1, "depth": "Superficial"},
        {"review_status": "Reviewed", "overall_status": "Stable", "confirmation": "Healing on schedule", "notes": "Incision line consolidating well.", "follow_up": "2026-09-26"}
    )

    # --- Seed Visit for Marcus Vance (P-00055, Pressure Injury W-0021) ---
    # Tests the FIRST ASSESSMENT rule! No previous visit!
    print("Seeding visits for Marcus Vance...")
    insert_visit(
        "V-MARCUS-01", "W-0021", "P-00055", "2026-09-16", "First Assessment", "fusc_0096.png", "Clinician Visit",
        {"wound_bed": "Granulation 60%, Slough 40%", "periwound": "Erythema over sacrum", "pain": 4, "depth": "Stage II shallow"},
        {"review_status": "Review required", "overall_status": "Uncertain", "confirmation": "First assessment baseline", "notes": "Initial staging and pressure redistribution mattress ordered.", "follow_up": "2026-09-23"}
    )

    conn.commit()
    conn.close()
    print("Database seeding completed successfully!")


if __name__ == "__main__":
    seed_database()
