@echo off
cd /d "%~dp0.."
cargo run --release -- generate --labels 3 --max-nodes 7 --strategy optimal --out .\output\optimal_large --export-json
