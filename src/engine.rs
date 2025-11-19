use anyhow::Result;
use std::path::Path;

/// The main pipeline controller
pub fn run(video_path: &str) -> Result<()> {
    let frames_dir = Path::new("frames");
    let processed_dir = Path::new("processed");

    println!("\n--- VISUAL DIFF ENGINE ---");

    // 1. Extract raw frames
    println!("1. Extracting frames...");
    crate::preprocess::extract_frames_cli(video_path, frames_dir)?;

    // 2. Preprocess frames
    println!("2. Preprocessing frames...");
    crate::preprocess::preprocess_directory(frames_dir, processed_dir)?;

    // 3. Detect ALL significant events
    println!("3. Detecting significant events...");

    let score_threshold = 0.55;

    
    let events = crate::event::detect_events(processed_dir, score_threshold)?;

    if events.is_empty() {
        println!("No events detected above threshold {}", score_threshold);
    } else {
        println!("\nDetected {} events:", events.len());
        for ev in &events {
            println!("{:#?}", ev);
        }
    }

    println!("DONE: Visual Diff Engine complete.");
    Ok(())
}
