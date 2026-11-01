import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "biovision.db"
STORAGE_PATH = PROJECT_ROOT / "data" / "storage"
STORAGE_PATH.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        is_demo INTEGER NOT NULL DEFAULT 1,
        age INTEGER,
        gender TEXT,
        primary_condition TEXT,
        phone TEXT,
        created_at TEXT NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS wounds (
        id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL,
        name TEXT NOT NULL,
        location TEXT NOT NULL,
        etiology TEXT NOT NULL,
        onset_date TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'Active',
        FOREIGN KEY (patient_id) REFERENCES patients(id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS visits (
        id TEXT PRIMARY KEY,
        wound_id TEXT NOT NULL,
        patient_id TEXT NOT NULL,
        visit_date TEXT NOT NULL,
        week_label TEXT NOT NULL,
        image_path TEXT NOT NULL,
        source TEXT NOT NULL DEFAULT 'Clinician Visit',
        patient_pain_score INTEGER,
        patient_symptoms TEXT,
        patient_comments TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (wound_id) REFERENCES wounds(id),
        FOREIGN KEY (patient_id) REFERENCES patients(id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS ai_analyses (
        id TEXT PRIMARY KEY,
        visit_id TEXT NOT NULL UNIQUE,
        wound_area REAL NOT NULL,
        peri_wound_area REAL NOT NULL,
        detection_confidence REAL,
        healing_probability REAL NOT NULL,
        not_healed_probability REAL NOT NULL,
        healing_class TEXT NOT NULL,
        infection_low_prob REAL NOT NULL,
        infection_med_prob REAL NOT NULL,
        infection_high_prob REAL NOT NULL,
        infection_class TEXT NOT NULL,
        healing_component REAL NOT NULL,
        infection_component REAL NOT NULL,
        tissue_component REAL NOT NULL,
        btis REAL NOT NULL,
        mask_path TEXT,
        visualization_path TEXT,
        top_shap_json TEXT,
        handcrafted_json TEXT,
        FOREIGN KEY (visit_id) REFERENCES visits(id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS clinical_assessments (
        id TEXT PRIMARY KEY,
        visit_id TEXT NOT NULL UNIQUE,
        wound_bed TEXT,
        periwound_condition TEXT,
        exudate_amount TEXT,
        exudate_type TEXT,
        pain_score INTEGER,
        odor TEXT,
        perfusion TEXT,
        neuropathy TEXT,
        infection_signs TEXT,
        wound_depth TEXT,
        undermining_tunneling TEXT,
        dressing_management TEXT,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (visit_id) REFERENCES visits(id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS clinician_reviews (
        id TEXT PRIMARY KEY,
        visit_id TEXT NOT NULL UNIQUE,
        review_status TEXT NOT NULL DEFAULT 'Review required',
        overall_clinical_status TEXT NOT NULL DEFAULT 'Uncertain',
        clinician_confirmation TEXT NOT NULL DEFAULT 'Pending review',
        clinical_notes TEXT,
        follow_up_date TEXT,
        reviewed_by TEXT,
        review_date TEXT,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (visit_id) REFERENCES visits(id)
    );
    """)

    conn.commit()
    conn.close()


def get_all_patients() -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM patients ORDER BY name ASC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_patient_by_id(patient_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_wounds_for_patient(patient_id: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM wounds WHERE patient_id = ?", (patient_id,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_visits_for_wound(wound_id: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT v.*, 
               a.wound_area, a.peri_wound_area, a.detection_confidence, 
               a.healing_probability, a.not_healed_probability, a.healing_class,
               a.infection_low_prob, a.infection_med_prob, a.infection_high_prob, a.infection_class,
               a.healing_component, a.infection_component, a.tissue_component, a.btis,
               a.mask_path, a.visualization_path, a.top_shap_json, a.handcrafted_json,
               c.wound_bed, c.periwound_condition, c.exudate_amount, c.exudate_type, c.pain_score as clinical_pain,
               c.odor, c.perfusion, c.neuropathy, c.infection_signs, c.wound_depth, c.dressing_management,
               r.review_status, r.overall_clinical_status, r.clinician_confirmation, r.clinical_notes,
               r.follow_up_date, r.reviewed_by, r.review_date
        FROM visits v
        LEFT JOIN ai_analyses a ON v.id = a.visit_id
        LEFT JOIN clinical_assessments c ON v.id = c.visit_id
        LEFT JOIN clinician_reviews r ON v.id = r.visit_id
        WHERE v.wound_id = ?
        ORDER BY v.visit_date ASC
    """, (wound_id,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_clinician_priority_queue() -> List[Dict[str, Any]]:
    """
    Builds the clinician priority queue answering: WHICH PATIENT NEEDS MY ATTENTION?
    Uses actual database records, never hardcodes, and guarantees wound_area is never 'None'.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT w.*, p.name as patient_name, p.is_demo FROM wounds w JOIN patients p ON w.patient_id = p.id WHERE w.status = 'Active'")
    wounds = [dict(r) for r in cur.fetchall()]
    conn.close()

    queue = []
    for w in wounds:
        visits = get_visits_for_wound(w["id"])
        if not visits:
            continue

        latest = visits[-1]
        previous = visits[-2] if len(visits) >= 2 else None

        current_area = latest.get("wound_area")
        previous_area = previous.get("wound_area") if previous else None

        # Calculate area change %
        if previous_area is not None and previous_area > 0 and current_area is not None:
            area_change_pct = ((current_area - previous_area) / previous_area) * 100.0
            area_change_str = f"{area_change_pct:+.1f}%"
        elif previous is None:
            area_change_pct = 0.0
            area_change_str = "First Assessment"
        else:
            area_change_pct = 0.0
            area_change_str = "N/A"

        # Healing & Infection
        healing_class = latest.get("healing_class") or "Uncertain"
        infection_class = latest.get("infection_class") or "Uncertain"
        review_status = latest.get("review_status") or "Review required"
        clinical_status = latest.get("overall_clinical_status") or "Uncertain"

        # Evidence-based priority rating
        signs_lower = str(latest.get("infection_signs", "")).lower()
        has_severe_signs = any(k in signs_lower for k in ["purulent", "marked periwound", "cellulitis", "elevated pain"])
        
        if area_change_pct >= 20.0 or (has_severe_signs and area_change_pct > 5.0):
            priority = "Urgent clinical review"
            priority_score = 4
        elif area_change_pct >= 10.0 or (review_status == "Review required" and healing_class == "Not Healed" and clinical_status == "Worsening"):
            priority = "High priority"
            priority_score = 3
        elif review_status == "Review required" or previous is None:
            priority = "Review required"
            priority_score = 2
        elif clinical_status == "Improving" or (previous_area and current_area and current_area < previous_area):
            priority = "Routine"
            priority_score = 1
        else:
            priority = "Routine"
            priority_score = 1

        # Format wound area nicely - NEVER "None"
        if current_area is not None:
            formatted_area = f"{int(round(current_area)):,} px"
        else:
            formatted_area = "Not available"

        queue.append({
            "patient_id": w["patient_id"],
            "patient_name": w["patient_name"],
            "is_demo": w["is_demo"],
            "wound_id": w["id"],
            "wound_name": w["name"],
            "location": w["location"],
            "last_assessment_date": latest["visit_date"],
            "current_wound_area": formatted_area,
            "current_wound_area_num": current_area or 0.0,
            "previous_wound_area_num": previous_area,
            "area_change_str": area_change_str,
            "area_change_pct": area_change_pct,
            "healing_status": healing_class,
            "infection_risk": infection_class,
            "review_status": review_status,
            "priority": priority,
            "priority_score": priority_score,
            "total_visits": len(visits),
            "first_visit_date": visits[0]["visit_date"],
            "latest_visit": latest,
            "previous_visit": previous,
        })

    # Sort evidence-based: highest priority first, then highest area change
    queue.sort(key=lambda x: (x["priority_score"], x["area_change_pct"]), reverse=True)
    return queue


def update_clinical_assessment(
    visit_id: str,
    wound_bed: str,
    periwound_condition: str,
    exudate_amount: str,
    exudate_type: str,
    pain_score: int,
    odor: str,
    perfusion: str,
    neuropathy: str,
    infection_signs: str,
    wound_depth: str,
    undermining_tunneling: str,
    dressing_management: str,
) -> None:
    conn = get_connection()
    cur = conn.cursor()
    now_str = datetime.now().isoformat()
    assessment_id = f"CA-{visit_id}"
    cur.execute("""
        INSERT INTO clinical_assessments (
            id, visit_id, wound_bed, periwound_condition, exudate_amount, exudate_type,
            pain_score, odor, perfusion, neuropathy, infection_signs, wound_depth,
            undermining_tunneling, dressing_management, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(visit_id) DO UPDATE SET
            wound_bed=excluded.wound_bed,
            periwound_condition=excluded.periwound_condition,
            exudate_amount=excluded.exudate_amount,
            exudate_type=excluded.exudate_type,
            pain_score=excluded.pain_score,
            odor=excluded.odor,
            perfusion=excluded.perfusion,
            neuropathy=excluded.neuropathy,
            infection_signs=excluded.infection_signs,
            wound_depth=excluded.wound_depth,
            undermining_tunneling=excluded.undermining_tunneling,
            dressing_management=excluded.dressing_management,
            updated_at=excluded.updated_at
    """, (
        assessment_id, visit_id, wound_bed, periwound_condition, exudate_amount, exudate_type,
        pain_score, odor, perfusion, neuropathy, infection_signs, wound_depth,
        undermining_tunneling, dressing_management, now_str
    ))
    conn.commit()
    conn.close()


def update_clinician_review(
    visit_id: str,
    review_status: str,
    overall_clinical_status: str,
    clinician_confirmation: str,
    clinical_notes: str,
    follow_up_date: str,
    reviewed_by: str,
) -> None:
    conn = get_connection()
    cur = conn.cursor()
    now_str = datetime.now().isoformat()
    review_id = f"CR-{visit_id}"
    cur.execute("""
        INSERT INTO clinician_reviews (
            id, visit_id, review_status, overall_clinical_status, clinician_confirmation,
            clinical_notes, follow_up_date, reviewed_by, review_date, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(visit_id) DO UPDATE SET
            review_status=excluded.review_status,
            overall_clinical_status=excluded.overall_clinical_status,
            clinician_confirmation=excluded.clinician_confirmation,
            clinical_notes=excluded.clinical_notes,
            follow_up_date=excluded.follow_up_date,
            reviewed_by=excluded.reviewed_by,
            review_date=excluded.review_date,
            updated_at=excluded.updated_at
    """, (
        review_id, visit_id, review_status, overall_clinical_status, clinician_confirmation,
        clinical_notes, follow_up_date, reviewed_by, now_str[:10], now_str
    ))
    conn.commit()
    conn.close()


def add_patient_upload_visit(
    patient_id: str,
    wound_id: str,
    image_bytes: bytes,
    filename: str,
    visit_date: str,
    pain_score: Optional[int],
    symptoms: str,
    comments: str,
    ai_result: Dict[str, Any],
) -> str:
    conn = get_connection()
    cur = conn.cursor()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    visit_id = f"VISIT-PAT-{patient_id}-{timestamp}"

    # Save original image to storage
    img_ext = Path(filename).suffix.lower() or ".jpg"
    img_dest = STORAGE_PATH / f"{visit_id}_orig{img_ext}"
    img_dest.write_bytes(image_bytes)

    # Save visualization overlay
    vis_dest = STORAGE_PATH / f"{visit_id}_vis.jpg"
    if "visualization_path" in ai_result and Path(ai_result["visualization_path"]).exists():
        vis_dest.write_bytes(Path(ai_result["visualization_path"]).read_bytes())

    # Save mask
    mask_dest = STORAGE_PATH / f"{visit_id}_mask.png"
    if "segmentation" in ai_result:
        import cv2
        cv2.imwrite(str(mask_dest), ai_result["segmentation"].mask)

    # Count existing visits for week label
    cur.execute("SELECT COUNT(*) FROM visits WHERE wound_id = ?", (wound_id,))
    count = cur.fetchone()[0]
    week_label = f"Week {min(count + 1, 10)} (Patient Upload)"

    # Cap total visits at 10 so every upload strictly fits within 10 weeks
    cur.execute("SELECT id FROM visits WHERE wound_id = ? ORDER BY visit_date ASC", (wound_id,))
    all_v_ids = [r[0] for r in cur.fetchall()]
    if len(all_v_ids) >= 10:
        del_id = all_v_ids[1] if len(all_v_ids) > 1 else all_v_ids[0]
        cur.execute("DELETE FROM ai_analyses WHERE visit_id = ?", (del_id,))
        cur.execute("DELETE FROM clinical_assessments WHERE visit_id = ?", (del_id,))
        cur.execute("DELETE FROM clinician_reviews WHERE visit_id = ?", (del_id,))
        cur.execute("DELETE FROM visits WHERE id = ?", (del_id,))

    now_str = datetime.now().isoformat()
    cur.execute("""
        INSERT INTO visits (
            id, wound_id, patient_id, visit_date, week_label, image_path, source,
            patient_pain_score, patient_symptoms, patient_comments, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, 'Patient Upload', ?, ?, ?, ?)
    """, (
        visit_id, wound_id, patient_id, visit_date, week_label, str(img_dest),
        pain_score, symptoms, comments, now_str
    ))

    # Calculate actual wound area
    import cv2
    wound_area = float(cv2.countNonZero(ai_result["segmentation"].mask)) if "segmentation" in ai_result else 0.0

    # Insert AI analysis
    ai_id = f"AI-{visit_id}"
    cur.execute("""
        INSERT INTO ai_analyses (
            id, visit_id, wound_area, peri_wound_area, detection_confidence,
            healing_probability, not_healed_probability, healing_class,
            infection_low_prob, infection_med_prob, infection_high_prob, infection_class,
            healing_component, infection_component, tissue_component, btis,
            mask_path, visualization_path, top_shap_json, handcrafted_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ai_id, visit_id,
        wound_area,
        float(ai_result["peri_wound"].peri_wound_area if "peri_wound" in ai_result else 0),
        float(ai_result["segmentation"].confidence or 0.0) if "segmentation" in ai_result else None,
        float(ai_result["healing_probability"]),
        float(ai_result["not_healed_probability"]),
        str(ai_result["healing_class"]),
        float(ai_result["infection_probability"][0]),
        float(ai_result["infection_probability"][1]),
        float(ai_result["infection_probability"][2]),
        str(ai_result["infection_class"]),
        float(ai_result["healing_component"]),
        float(ai_result["infection_component"]),
        float(ai_result["tissue_component"]),
        float(ai_result["btis"]),
        str(mask_dest),
        str(vis_dest),
        json.dumps(ai_result.get("healing_top", [])),
        json.dumps({k: float(v) for k, v in ai_result.get("handcrafted", {}).items() if isinstance(v, (int, float))}),
    ))

    # Default clinical assessment and review status
    ca_id = f"CA-{visit_id}"
    cur.execute("""
        INSERT INTO clinical_assessments (
            id, visit_id, pain_score, infection_signs, updated_at
        ) VALUES (?, ?, ?, ?, ?)
    """, (ca_id, visit_id, pain_score, symptoms, now_str))

    cr_id = f"CR-{visit_id}"
    cur.execute("""
        INSERT INTO clinician_reviews (
            id, visit_id, review_status, overall_clinical_status, clinician_confirmation,
            clinical_notes, updated_at
        ) VALUES (?, ?, 'Review required', 'Uncertain', 'Pending clinician review of patient upload', ?, ?)
    """, (cr_id, visit_id, f"Patient notes: {comments or 'None'}", now_str))

    conn.commit()
    conn.close()
    return visit_id
