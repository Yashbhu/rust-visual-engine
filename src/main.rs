mod engine;
mod diff;
mod preprocess;
mod event;
mod ipc;

fn main() {
    if let Err(err) = engine::run("input.mp4") {
        eprintln!("\n[ERROR] Visual Diff Engine failed:\n{:#?}", err);
    }
}
