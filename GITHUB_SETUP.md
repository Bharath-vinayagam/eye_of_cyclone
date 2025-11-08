# GitHub Setup Commands

## Step 1: Initialize Git (if not already done in this directory)
```bash
cd c:\Users\beast\Documents\Projects\cyclone_eye_detection
git init
```

## Step 2: Add all files (respecting .gitignore)
```bash
git add .
```

## Step 3: Make your first commit
```bash
git commit -m "Initial commit: Cyclone Eye Detection with ResNet50"
```

## Step 4: Create a repository on GitHub
1. Go to https://github.com/new
2. Create a new repository (e.g., "cyclone-eye-detection")
3. **DO NOT** initialize with README, .gitignore, or license
4. Copy the repository URL (e.g., `https://github.com/yourusername/cyclone-eye-detection.git`)

## Step 5: Add remote and push
```bash
# Replace YOUR_USERNAME and REPO_NAME with your actual GitHub username and repository name
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

## Alternative: If you already have a GitHub repository
```bash
# Just add the remote and push
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

## Quick One-Liner Commands (PowerShell)
```powershell
cd c:\Users\beast\Documents\Projects\cyclone_eye_detection
git init
git add .
git commit -m "Initial commit: Cyclone Eye Detection with ResNet50"
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

## Note:
- The `.gitignore` file will automatically exclude large files like:
  - Model files (*.h5, *.pt)
  - Dataset files
  - Cache files
  - Training outputs in `runs/` folder

