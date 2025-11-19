
use anyhow::Result;
use image::{open, GrayImage, Luma, GenericImageView, ImageOutputFormat};
use serde::Serialize;
use serde_json::json;
use std::{fs, path::Path, process::Command};

#[derive(Debug, Serialize, Clone)]
pub struct Event {
    pub frame_index: usize,
    pub raw_score: f32,
    pub contextual_score: f32,
    pub velocity: f32,
    pub final_score: f32,
    pub ml_label: Option<String>,       
    pub ml_confidence: Option<f32>,     
    pub prev_frame: String,
    pub curr_frame: String,
    pub bbox: (u32, u32, u32, u32),
}


const MOTION_GATE_THRESHOLD: f32 = 0.25;

// stabilizer params
const BLOCK_SIZE: u32 = 16;
const SEARCH: i32 = 6;
const SMOOTH_ALPHA: f32 = 0.6;

// crop params
const MIN_CROP: u32 = 192;
const MAX_CROP: u32 = 768;
const CROP_TARGET: u32 = 512;



fn analyze_deformity(prev: &str, curr: &str, crop: &str) -> Option<(String, f32)> {
    let output = Command::new("python3")
        .arg("deform_analyzer.py")
        .arg(prev)
        .arg(curr)
        .arg(crop)
        .output()
        .ok()?;

    if !output.status.success() {
        eprintln!(
            "deform_analyzer.py error: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }

    let s = String::from_utf8_lossy(&output.stdout);
    let v: serde_json::Value = serde_json::from_str(&s).ok()?;

    let label = v["label"].as_str()?.to_string();
    let score = v["confidence"].as_f64()? as f32;

    Some((label, score))
}


fn motion_centroid(
    a: &GrayImage,
    b: &GrayImage,
    block: u32,
    search: i32,
) -> (f32, f32, f32, f32) {
    let (w, h) = a.dimensions();
    if w < block || h < block {
        return ((w / 2) as f32, (h / 2) as f32, 0.0, 0.0);
    }

    let mut total_w = 0.0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut mag_sum = 0.0;
    let mut active_blocks = 0usize;
    let mut total_blocks = 0usize;

    for by in (0..h - block + 1).step_by(block as usize) {
        for bx in (0..w - block + 1).step_by(block as usize) {
            total_blocks += 1;

            let mut best_sad = u64::MAX;
            let mut best_dx = 0;
            let mut best_dy = 0;

            for dy in -search..=search {
                for dx in -search..=search {
                    let tx = bx as i32 + dx;
                    let ty = by as i32 + dy;
                    if tx < 0 || ty < 0 { continue; }
                    let txu = tx as u32;
                    let tyu = ty as u32;
                    if txu + block > w || tyu + block > h { continue; }

                    let mut sad = 0u64;
                    for yy in 0..block {
                        for xx in 0..block {
                            let p1 = a.get_pixel(bx + xx, by + yy)[0] as i32;
                            let p2 = b.get_pixel(txu + xx, tyu + yy)[0] as i32;
                            sad += (p1 - p2).abs() as u64;
                        }
                    }

                    if sad < best_sad {
                        best_sad = sad;
                        best_dx = dx;
                        best_dy = dy;
                    }
                }
            }

            let mag = ((best_dx * best_dx + best_dy * best_dy) as f32).sqrt();
            let weight = 1.0 / (1.0 + best_sad as f32 / 50000.0);

            if weight > 0.001 {
                active_blocks += 1;
                total_w += weight;
                let cx = bx as f32 + block as f32 / 2.0 + best_dx as f32;
                let cy = by as f32 + block as f32 / 2.0 + best_dy as f32;
                sum_x += weight * cx;
                sum_y += weight * cy;
                mag_sum += mag;
            }
        }
    }

    if total_w == 0.0 {
        return ((w / 2) as f32, (h / 2) as f32, 0.0, 0.0);
    }

    let cx = sum_x / total_w;
    let cy = sum_y / total_w;
    let mean_mag = mag_sum / active_blocks.max(1) as f32;
    let area_frac = active_blocks as f32 / total_blocks.max(1) as f32;

    (cx, cy, mean_mag, area_frac)
}

// --------------------------------------------
// Simple raw diff score (no masking)
// --------------------------------------------
fn simple_diff(a: &GrayImage, b: &GrayImage) -> f32 {
    let mut sum = 0.0;
    let mut count = 0.0;

    for (p1, p2) in a.pixels().zip(b.pixels()) {
        sum += (p1[0] as f32 - p2[0] as f32).abs();
        count += 1.0;
    }
    (sum / count / 255.0).min(1.0)
}

// --------------------------------------------
// Compute crop size
// --------------------------------------------
fn compute_crop_size(area: f32, mag: f32, fw: u32, fh: u32) -> u32 {
    let mut s = MIN_CROP as f32 + area * 600.0 + mag * 40.0;
    if s < MIN_CROP as f32 { s = MIN_CROP as f32; }
    if s > MAX_CROP as f32 { s = MAX_CROP as f32; }
    let max_allowed = fw.min(fh) as f32;
    if s > max_allowed { s = max_allowed; }
    s as u32
}

// --------------------------------------------
// MAIN EVENT DETECTOR (Rust-only)
// --------------------------------------------
pub fn detect_events(processed_dir: &Path, threshold: f32) -> Result<Vec<Event>> {
    // load frames
    let mut frames: Vec<_> = fs::read_dir(processed_dir)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().unwrap_or_default() == "png")
        .collect();
    frames.sort();

    let n = frames.len();
    if n < 3 {
        println!("Need >= 3 frames");
        return Ok(vec![]);
    }
    println!("Loaded {} frames", n);

    let mut raw = vec![0.0; n];
    let mut ctx = vec![0.0; n];
    let mut vel = vec![0.0; n];
    let mut bboxes = vec![(0, 0, 0, 0); n];

    // centroid smoothing
    let mut sm_cx = 0.0;
    let mut sm_cy = 0.0;
    let mut init = false;

    // STEP 1 — raw & centroid
    for i in 0..(n - 1) {
        let a = open(&frames[i])?.to_luma8();
        let b = open(&frames[i + 1])?.to_luma8();

        raw[i] = simple_diff(&a, &b);

        let (cx, cy, mag, area) = motion_centroid(&a, &b, BLOCK_SIZE, SEARCH);

        if !init {
            sm_cx = cx;
            sm_cy = cy;
            init = true;
        } else {
            sm_cx = SMOOTH_ALPHA * cx + (1.0 - SMOOTH_ALPHA) * sm_cx;
            sm_cy = SMOOTH_ALPHA * cy + (1.0 - SMOOTH_ALPHA) * sm_cy;
        }

        let full = open(&frames[i + 1])?.to_rgb8();
        let (fw, fh) = full.dimensions();

        let cs = compute_crop_size(area, mag, fw, fh);
        let half = cs / 2;
        let mut x1 = (sm_cx as i32 - half as i32).max(0) as u32;
        let mut y1 = (sm_cy as i32 - half as i32).max(0) as u32;
        if x1 + cs > fw { x1 = fw - cs; }
        if y1 + cs > fh { y1 = fh - cs; }

        let x2 = x1 + cs;
        let y2 = y1 + cs;
        bboxes[i] = (x1, y1, x2, y2);
    }

    // STEP 2 — contextual smoothing
    const W: [f32; 5] = [0.1, 0.25, 0.3, 0.25, 0.1];
    for i in 0..(n - 1) {
        let mut s = 0.0;
        let mut wsum = 0.0;
        for (k, off) in [-2, -1, 0, 1, 2].iter().enumerate() {
            let idx = i as isize + off;
            if idx >= 0 && (idx as usize) < (n - 1) {
                s += W[k] * raw[idx as usize];
                wsum += W[k];
            }
        }
        ctx[i] = if wsum > 0.0 { s / wsum } else { raw[i] };
    }

    // STEP 3 — velocity
    let mut maxv = 0.0;
    for i in 0..(n - 1) {
        let v = if i == 0 {
            raw[i + 1] - raw[i]
        } else if i + 1 == n - 1 {
            raw[i] - raw[i - 1]
        } else {
            (raw[i + 1] - raw[i - 1]) * 0.5
        };
        vel[i] = v;
        if v > maxv { maxv = v; }
    }
    if maxv == 0.0 { maxv = 1.0; }

    // STEP 4 — final score
    let mut fscores = vec![0.0; n];
    for i in 0..(n - 1) {
        let vn = vel[i] / maxv;
        fscores[i] = (0.8 * ctx[i] + 0.6 * vn).clamp(0.0, 1.0);
    }

    // STEP 5 — find bursts
    let mut bursts = Vec::<Vec<usize>>::new();
    let mut cur = Vec::<usize>::new();

    for i in 0..(n - 1) {
        if fscores[i] >= threshold {
            if cur.is_empty() || i == cur.last().cloned().unwrap() + 1 {
                cur.push(i);
            } else {
                bursts.push(cur.clone());
                cur.clear();
                cur.push(i);
            }
        } else if !cur.is_empty() {
            bursts.push(cur.clone());
            cur.clear();
        }
    }
    if !cur.is_empty() {
        bursts.push(cur);
    }

    if bursts.is_empty() {
        println!("No events");
        return Ok(vec![]);
    }

    // STEP 6 — choose 1 best event
    let mut events = vec![];

    for (idx, burst) in bursts.into_iter().enumerate() {
        let mut peak = burst[0];
        for &i in &burst {
            if fscores[i] > fscores[peak] {
                peak = i;
            }
        }

        let edir = Path::new("events").join(format!("event_{:04}", idx + 1));
        fs::create_dir_all(&edir)?;

        // copy frames
        fs::copy(&frames[peak], edir.join("prev.png"))?;
        fs::copy(&frames[peak + 1], edir.join("curr.png"))?;

        // crop stabilized object
        let full = open(&frames[peak + 1])?.to_rgb8();
        let (x1, y1, x2, y2) = bboxes[peak];
        let crop = full.view(x1, y1, x2 - x1, y2 - y1).to_image();
        let norm = image::imageops::resize(
            &crop,
            CROP_TARGET,
            CROP_TARGET,
            image::imageops::FilterType::Lanczos3,
        );
        let crop_path = edir.join("crop.png");
        let mut f = fs::File::create(&crop_path)?;
        norm.write_to(&mut f, ImageOutputFormat::Png)?;

        let mut ml_label = None;
        let mut ml_conf = None;

        if raw[peak] >= MOTION_GATE_THRESHOLD {
            if let Some((label, conf)) =
                analyze_deformity(
                    &frames[peak].to_string_lossy(),
                    &frames[peak + 1].to_string_lossy(),
                    crop_path.to_string_lossy().as_ref(),
                )
            {
                ml_label = Some(label);
                ml_conf = Some(conf);
            } else {
                ml_label = Some("DeformityAnalysisError".to_string());
            }
        } else {
            ml_label = Some("LowMotion".to_string());
        }

        fs::write(
            edir.join("ml.json"),
            serde_json::to_string_pretty(&json!({
                "label": ml_label,
                "confidence": ml_conf
            }))?,
        )?;

        events.push(Event {
            frame_index: peak,
            raw_score: raw[peak],
            contextual_score: ctx[peak],
            velocity: vel[peak],
            final_score: fscores[peak],
            ml_label,
            ml_confidence: ml_conf,
            prev_frame: frames[peak].file_name().unwrap().to_string_lossy().to_string(),
            curr_frame: frames[peak + 1].file_name().unwrap().to_string_lossy().to_string(),
            bbox: bboxes[peak],
        });
    }

    // return ONLY best event
    events.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap());
    Ok(vec![events[0].clone()])
}
