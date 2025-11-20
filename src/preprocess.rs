use anyhow::{anyhow, Result};
use std::{fs, path::Path, process::Command};

use image::{
    self, DynamicImage, GenericImageView, ImageBuffer, Luma,
};
use image::imageops::{crop_imm, FilterType, overlay};
use imageproc::geometric_transformations::{rotate_about_center, Interpolation};



pub fn extract_frames_cli(video_path: &str, out_dir: &Path) -> Result<()> {
    if out_dir.exists() {
        fs::remove_dir_all(out_dir)?;
    }
    fs::create_dir_all(out_dir)?;


    let status = Command::new("ffmpeg")
        .args([
            "-i", video_path,
            "-vf", "fps=15",             
            &format!("{}/frame_%04d.png", out_dir.display()),
        ])
        .status()?;

    if !status.success() {
        return Err(anyhow!("FFmpeg failed to extract frames"));
    }

    println!("Extracted {} frames", fs::read_dir(out_dir)?.count());
    Ok(())
}
pub fn preprocess_directory(input: &Path, output: &Path) -> Result<()> {
    if output.exists() {
        fs::remove_dir_all(output)?;
    }
    fs::create_dir_all(output)?;

    let mut frames: Vec<_> = fs::read_dir(input)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().map(|ext| ext == "png").unwrap_or(false))
        .collect();

    frames.sort();

    for path in frames {
        let out = output.join(path.file_name().unwrap());
        preprocess_single(&path, &out)?;
    }

    Ok(())
}

pub fn preprocess_single(input: &Path, output: &Path) -> Result<()> {
    let img = image::open(input)?;

    let (w, h) = img.dimensions();
    let target = 512;


    let min_side = w.min(h).max(1);
    let scale = target as f32 / min_side as f32;

    let nw = ((w as f32 * scale).round()).max(1.0) as u32;
    let nh = ((h as f32 * scale).round()).max(1.0) as u32;

    let resized = img.resize_exact(nw, nh, FilterType::Triangle);
    let cx = if nw > 512 { (nw - 512) / 2 } else { 0 };
    let cy = if nh > 512 { (nh - 512) / 2 } else { 0 };

    let cw = nw.min(512);
    let ch = nh.min(512);

    let cropped = crop_imm(&resized, cx, cy, cw, ch).to_image();
    let gray = DynamicImage::ImageRgba8(cropped).to_luma8();

    let mut padded = ImageBuffer::<Luma<u8>, Vec<u8>>::new(520, 520);
    for p in padded.pixels_mut() {
        *p = Luma([0]);
    }

    let rx = ((520 - gray.width()) / 2).max(0);
    let ry = ((520 - gray.height()) / 2).max(0);

    overlay(&mut padded, &gray, rx as i64, ry as i64);

    padded.save(output)?;
    Ok(())
}
