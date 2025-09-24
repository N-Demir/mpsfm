#!/usr/bin/env python3
"""
Script to predownload all required models for MPSFM reconstruction.
This prevents the need to download models during runtime, which can be slow and interrupt the workflow.

Usage:
    python download_models.py                    # Download all models (with confirmation)
    python download_models.py --check            # Check existing models without downloading
    python download_models.py --estimate         # Show estimated download sizes
    python download_models.py --skip-large       # Skip large models (>1GB)

Models downloaded:
    - Metric3D v2 (Large): ~5.1GB - Depth estimation model
    - Metric3D v2 (Small): ~1.2GB - Smaller depth estimation model
    - MASt3R: ~1.8GB - 3D reconstruction model
    - DSINE: ~200MB - Surface normal estimation
    - Depth Anything V2 (Large): ~1.2GB - Depth estimation
    - Depth Anything V2 (Small): ~300MB - Smaller depth estimation
    - Depth Pro: ~100MB - Apple's depth estimation
    - Sky Segmentation: ~10MB - Sky mask generation
    - NetVLAD: ~500MB - Image retrieval features
    - RoMa models: ~1.5GB - Line matching
    - LightGlue: ~45MB - Feature matching

Total estimated size: ~12GB

Requirements:
    - wget (for downloading from URLs)
    - gdown (pip install gdown, for Google Drive downloads)
    - Sufficient disk space (~15GB recommended)
    - Stable internet connection
"""

import os
import subprocess
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from mpsfm.vars import gvars

def ensure_dir(path):
    """Ensure directory exists"""
    path.mkdir(parents=True, exist_ok=True)

def download_with_wget(url, output_path):
    """Download file using wget"""
    print(f"📥 Downloading via wget: {url}")
    try:
        subprocess.run(
            ["wget", url, "-O", str(output_path)], 
            stdout=sys.stdout, 
            stderr=sys.stderr, 
            check=True
        )
        print(f"✅ Successfully downloaded: {output_path.name}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download {url}: {e}")
        return False
    return True

def download_with_gdown(gdown_id, output_path):
    """Download file using gdown"""
    print(f"📥 Downloading via gdown: {gdown_id}")
    try:
        subprocess.run(
            ["gdown", gdown_id, "-O", str(output_path)], 
            check=True
        )
        print(f"✅ Successfully downloaded: {output_path.name}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download {gdown_id}: {e}")
        return False
    return True

def download_netvlad_models():
    """Download NetVLAD models"""
    print("\n🔽 Downloading NetVLAD models...")
    
    # NetVLAD models are downloaded via torch.hub
    # They go to ~/.cache/torch/hub/netvlad/
    # We'll trigger the download by importing and using the model
    try:
        from mpsfm.extraction.imagewise.features.models.netvlad import NetVLAD
        
        # Create a temporary config to trigger download
        conf = {"model_name": "VGG16-NetVLAD-Pitts30K", "whiten": True, "require_download": False}
        model = NetVLAD(conf)
        print("✅ NetVLAD model downloaded successfully")
    except Exception as e:
        print(f"❌ Failed to download NetVLAD model: {e}")

def download_roma_models():
    """Download RoMa models"""
    print("\n🔽 Downloading RoMa models...")
    
    roma_path = gvars.ROOT / "third_party/RoMa"
    pretrained_path = roma_path / "pretrained"
    ensure_dir(pretrained_path)
    
    # RoMa outdoor model
    roma_outdoor_url = "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth"
    roma_outdoor_path = pretrained_path / "roma_outdoor.pth"
    
    if not roma_outdoor_path.exists():
        download_with_wget(roma_outdoor_url, roma_outdoor_path)
    else:
        print(f"✅ RoMa outdoor model already exists: {roma_outdoor_path.name}")
    
    # DINOv2 model
    dinov2_url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"
    dinov2_path = pretrained_path / "dinov2_vitl14_pretrain.pth"
    
    if not dinov2_path.exists():
        download_with_wget(dinov2_url, dinov2_path)
    else:
        print(f"✅ DINOv2 model already exists: {dinov2_path.name}")

def download_lightglue_model():
    """Download LightGlue model"""
    print("\n🔽 Downloading LightGlue model...")
    
    # LightGlue model is downloaded via torch.hub
    # It goes to ~/.cache/torch/hub/checkpoints/
    try:
        import torch
        # This will trigger the download
        torch.hub.load("cvg/LightGlue", "superpoint_lightglue", pretrained=True)
        print("✅ LightGlue model downloaded successfully")
    except Exception as e:
        print(f"❌ Failed to download LightGlue model: {e}")

def check_existing_models():
    """Check which models are already downloaded"""
    print("\n🔍 Checking existing models...")
    
    models_dir = gvars.ROOT / "local" / "weights"
    roma_path = gvars.ROOT / "third_party/RoMa/pretrained"
    
    # Check main models directory
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pth")) + list(models_dir.glob("*.pt")) + list(models_dir.glob("*.onnx"))
        if model_files:
            print(f"Found {len(model_files)} model files in {models_dir}:")
            for file in model_files:
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  ✅ {file.name} ({size_mb:.1f} MB)")
        else:
            print(f"No model files found in {models_dir}")
    else:
        print(f"Models directory does not exist: {models_dir}")
    
    # Check RoMa models
    if roma_path.exists():
        roma_files = list(roma_path.glob("*.pth"))
        if roma_files:
            print(f"Found {len(roma_files)} RoMa model files:")
            for file in roma_files:
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  ✅ {file.name} ({size_mb:.1f} MB)")
    
    # Check torch hub cache
    try:
        import torch
        hub_dir = Path(torch.hub.get_dir())
        if hub_dir.exists():
            netvlad_dir = hub_dir / "netvlad"
            lightglue_dir = hub_dir / "checkpoints"
            
            if netvlad_dir.exists():
                netvlad_files = list(netvlad_dir.glob("*.mat"))
                if netvlad_files:
                    print(f"Found {len(netvlad_files)} NetVLAD files:")
                    for file in netvlad_files:
                        size_mb = file.stat().st_size / (1024 * 1024)
                        print(f"  ✅ {file.name} ({size_mb:.1f} MB)")
            
            if lightglue_dir.exists():
                lightglue_files = list(lightglue_dir.glob("*lightglue*"))
                if lightglue_files:
                    print(f"Found {len(lightglue_files)} LightGlue files:")
                    for file in lightglue_files:
                        size_mb = file.stat().st_size / (1024 * 1024)
                        print(f"  ✅ {file.name} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"Could not check torch hub cache: {e}")

def estimate_total_download_size():
    """Estimate total download size"""
    print("\n📏 Estimated download sizes:")
    
    models_to_download = [
        {"name": "Metric3D v2 (Large)", "size": "~5.1GB"},
        {"name": "Metric3D v2 (Small)", "size": "~1.2GB"},
        {"name": "MASt3R", "size": "~1.8GB"},
        {"name": "DSINE", "size": "~200MB"},
        {"name": "Depth Anything V2 (Large)", "size": "~1.2GB"},
        {"name": "Depth Anything V2 (Small)", "size": "~300MB"},
        {"name": "Depth Pro", "size": "~100MB"},
        {"name": "Sky Segmentation", "size": "~10MB"},
        {"name": "NetVLAD", "size": "~500MB"},
        {"name": "RoMa models", "size": "~1.5GB"},
        {"name": "LightGlue", "size": "~45MB"},
    ]
    
    total_estimated = 0
    for model in models_to_download:
        size_str = model["size"]
        if "GB" in size_str:
            size_gb = float(size_str.replace("~", "").replace("GB", ""))
            total_estimated += size_gb
        elif "MB" in size_str:
            size_mb = float(size_str.replace("~", "").replace("MB", ""))
            total_estimated += size_mb / 1024
    
    print(f"Total estimated size: ~{total_estimated:.1f} GB")
    print("Note: Actual sizes may vary slightly.")

def download_required_models(skip_large=False):
    """Download all models that require downloads"""
    print("🚀 Starting model downloads...")
    
    # Create models directory
    models_dir = gvars.ROOT / "local" / "weights"
    ensure_dir(models_dir)
    
    # List of models to download with their configurations
    models_to_download = [
        {
            "name": "Metric3D v2 (Large)",
            "filename": "metric_depth_vit_giant2_800k.pth",
            "url": "https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_giant2_800k.pth",
            "method": "wget",
            "size": "~5.1GB",
            "is_large": True
        },
        {
            "name": "Metric3D v2 (Small)",
            "filename": "metric_depth_vit_small_800k.pth",
            "url": "https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_small_800k.pth",
            "method": "wget",
            "size": "~1.2GB",
            "is_large": True
        },
        {
            "name": "MASt3R",
            "filename": "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
            "url": "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
            "method": "wget",
            "size": "~1.8GB",
            "is_large": True
        },
        {
            "name": "DSINE",
            "filename": "dsine.pth",
            "gdown_id": "1u8TdKXkR7-0zzRRcx-3x3rPN7gvAAM9N",
            "method": "gdown",
            "size": "~200MB",
            "is_large": False
        },
        {
            "name": "Depth Anything V2 (Large)",
            "filename": "depth_anything_v2_metric_vkitti_vitl.pth",
            "url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth",
            "method": "wget",
            "size": "~1.2GB",
            "is_large": True
        },
        {
            "name": "Depth Anything V2 (Small)",
            "filename": "depth_anything_v2_metric_vkitti_vits.pth",
            "url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/resolve/main/depth_anything_v2_metric_vkitti_vits.pth",
            "method": "wget",
            "size": "~300MB",
            "is_large": False
        },
        {
            "name": "Depth Pro",
            "filename": "depth_pro.pt",
            "url": "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt",
            "method": "wget",
            "size": "~100MB",
            "is_large": False
        },
        {
            "name": "Sky Segmentation",
            "filename": "skyseg.onnx",
            "gdown_id": "1jJpcRXAHaTR1zk4xD1kVYXtnO1-C982K",
            "method": "gdown",
            "size": "~10MB",
            "is_large": False
        }
    ]
    
    # Filter out large models if requested
    if skip_large:
        models_to_download = [m for m in models_to_download if not m.get("is_large", False)]
        print(f"⚠️  Skipping large models. Downloading {len(models_to_download)} smaller models only.")
    
    # Download each model
    for model_info in models_to_download:
        print(f"\n🔽 Downloading {model_info['name']}...")
        output_path = models_dir / model_info['filename']
        
        if output_path.exists():
            print(f"✅ {model_info['name']} already exists: {model_info['filename']}")
            continue
        
        success = False
        if model_info['method'] == 'wget':
            success = download_with_wget(model_info['url'], output_path)
        elif model_info['method'] == 'gdown':
            success = download_with_gdown(model_info['gdown_id'], output_path)
        
        if not success:
            print(f"❌ Failed to download {model_info['name']}")
    
    # Download models that use different mechanisms
    download_netvlad_models()
    download_roma_models()
    download_lightglue_model()
    
    print("\n🎉 Model download process completed!")
    print(f"Models are stored in: {models_dir}")
    
    # Print summary
    print("\n📊 Download Summary:")
    total_size = 0
    downloaded_count = 0
    
    for model_info in models_to_download:
        output_path = models_dir / model_info['filename']
        status = "✅ Downloaded" if output_path.exists() else "❌ Failed"
        size_info = ""
        
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            size_info = f" ({size_mb:.1f} MB)"
            total_size += size_mb
            downloaded_count += 1
        else:
            size_info = f" (Expected: {model_info.get('size', 'Unknown')})"
        
        print(f"  {model_info['name']}: {status}{size_info}")
    
    print(f"\n📈 Total downloaded: {downloaded_count}/{len(models_to_download)} models")
    print(f"💾 Total size: {total_size:.1f} MB ({total_size/1024:.1f} GB)")
    
    if downloaded_count == len(models_to_download):
        print("\n🎉 All models downloaded successfully!")
        print("You can now run reconstruction without waiting for model downloads.")
    else:
        print(f"\n⚠️  {len(models_to_download) - downloaded_count} models failed to download.")
        print("You may need to run the script again or download them manually.")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download all required models for MPSFM reconstruction")
    parser.add_argument("--check", action="store_true", help="Check existing models without downloading")
    parser.add_argument("--estimate", action="store_true", help="Show estimated download sizes")
    parser.add_argument("--skip-large", action="store_true", help="Skip downloading large models (>1GB)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("        MPSFM MODEL DOWNLOADER")
    print("=" * 60)
    print("This script will download all required models for MPSFM reconstruction.")
    print("This prevents downloads during runtime and speeds up the reconstruction process.")
    print("=" * 60)
    
    if args.check:
        check_existing_models()
        return
    
    if args.estimate:
        estimate_total_download_size()
        return
    
    # Check if required tools are available
    try:
        subprocess.run(["wget", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ wget is not available. Please install wget to download models.")
        return
    
    try:
        subprocess.run(["gdown", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ gdown is not available. Please install gdown to download models.")
        print("   Install with: pip install gdown")
        return
    
    # Show estimated sizes first
    estimate_total_download_size()
    
    print("\n⚠️  This will download approximately 12+ GB of model files.")
    print("Make sure you have sufficient disk space and a stable internet connection.")
    print("Starting download automatically...")
    
    download_required_models(skip_large=args.skip_large)

if __name__ == "__main__":
    main()
