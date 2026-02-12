# Pre-Defense Checklist

Complete this checklist before your final year defense presentation.

---

## 📤 Asset Upload Tasks

### 1. Prepare Asset Archives

- [ ] Create `dataset.zip`:
  ```bash
  cd data/processed
  zip -r ../../dataset.zip train/ val/ test/ dataset_info.json
  cd ../..
  ```

- [ ] Create `model_v1.zip`:
  ```bash
  cd models/v1
  zip -r ../../model_v1.zip best_model.pth training_metadata.json
  cd ../..
  ```

- [ ] (Optional) Create `mlflow_backup.zip`:
  ```bash
  cd mlflow_data
  zip -r ../mlflow_backup.zip mlflow.db artifacts/ mlruns/
  cd ..
  ```

### 2. Upload to Google Drive

- [ ] Upload `dataset.zip` to Google Drive
- [ ] Upload `model_v1.zip` to Google Drive
- [ ] Set sharing to "Anyone with the link"
- [ ] Copy shareable links

### 3. Generate Checksums

- [ ] Generate checksums:
  ```bash
  sha256sum dataset.zip > checksums_temp.txt
  sha256sum model_v1.zip >> checksums_temp.txt
  cat checksums_temp.txt
  ```

- [ ] Update `checksums.txt` with actual hash values
- [ ] Delete temporary file: `rm checksums_temp.txt`

### 4. Update Documentation

- [ ] Replace `REPLACE_WITH_YOUR_LINK` in `README.md` with actual Google Drive links
- [ ] Replace `REPLACE_WITH_YOUR_LINK` in `ASSETS.md` with actual links
- [ ] Replace `REPLACE_WITH_ACTUAL_HASH` in `checksums.txt` with actual checksums
- [ ] Update `REPLACE_WITH_ACTUAL_LINK` in `DEVELOPMENT.md` if referencing dataset source

---

## 🧪 Testing on Fresh Environment

### 1. Test Complete Setup Flow

- [ ] Clone repository to a new directory (simulate fresh clone)
- [ ] Download assets from your Google Drive links
- [ ] Run setup script:
  ```bash
  chmod +x scripts/setup_assets.sh
  ./scripts/setup_assets.sh
  ```
- [ ] Follow Quick Start in README
- [ ] Verify all services start correctly
- [ ] Run a test batch prediction
- [ ] Check Grafana dashboard populates

### 2. Verify Documentation Accuracy

- [ ] Every command in README works as written
- [ ] All file paths are correct (absolute vs relative)
- [ ] Screenshots in documentation are up-to-date
- [ ] Links to external resources work
- [ ] Code examples run without errors

### 3. Test Edge Cases

- [ ] What happens if MLflow isn't running? (Model Service should show clear error)
- [ ] What if Model Service is down? (Orchestrator should fail gracefully)
- [ ] What if wrong file paths are used? (Clear error messages)
- [ ] What if checkpoint is corrupted? (System should handle or warn)

---

## 📊 Prepare Demo Materials

### 1. Screenshot Collection

Take screenshots of:

- [ ] Grafana dashboard with populated metrics
- [ ] Streamlit UI upload page
- [ ] Streamlit results table
- [ ] MLflow experiments page
- [ ] Model Service Swagger docs
- [ ] Prometheus targets page (showing UP status)
- [ ] Terminal showing batch job running

Save to: `docs/screenshots/` (not tracked in Git)

### 2. Demo Data

Prepare demo images:

- [ ] Copy 20-30 diverse test images to a separate folder
- [ ] Test that they classify correctly
- [ ] Know which categories they belong to (for explaining results)

### 3. Presentation Slides

Create slides covering:

- [ ] Problem statement & business context
- [ ] System architecture diagram
- [ ] Technology stack choices (why each tool?)
- [ ] Key design decisions (batch vs real-time, etc.)
- [ ] Model performance metrics
- [ ] Live demo flow
- [ ] Monitoring & production readiness
- [ ] Challenges & solutions
- [ ] Future improvements

---

## 🎯 Defense Preparation

### 1. Practice Demo Flow

Rehearse this exact sequence:

1. **Show Architecture** (5 min)
   - Explain components
   - Show how they communicate
   - Highlight production patterns

2. **Live Demo** (10 min)
   - Start with: "All services are already running..."
   - Open Streamlit UI
   - Upload sample images (have them ready)
   - Click "Run Classification"
   - Show results table
   - Switch to Grafana, show metrics update
   - (Optional) Show MLflow registry

3. **Code Walkthrough** (5 min if asked)
   - Show orchestrator checkpoint logic
   - Show Model Service startup/loading
   - Show metrics instrumentation

### 2. Anticipate Questions

**Technical Questions:**

- [ ] "Why batch processing instead of real-time?"
  - **Answer:** Business requirement (overnight processing), cost efficiency, GPU batching

- [ ] "How do you handle model updates?"
  - **Answer:** MLflow aliases, register new version, promote to production, restart service

- [ ] "What happens if the system crashes mid-batch?"
  - **Answer:** Checkpoint recovery, can resume from last completed batch

- [ ] "Why separate Model Service from Orchestrator?"
  - **Answer:** Testability, independent deployment, technology flexibility

- [ ] "How do you monitor the system?"
  - **Answer:** Prometheus metrics, Grafana dashboards, tracking latency/throughput/errors

**Business Questions:**

- [ ] "Can this scale to 10x more images?"
  - **Answer:** Yes, current architecture handles it. Could parallelize if needed beyond that.

- [ ] "What's the cost to run this?"
  - **Answer:** Currently free (local), cloud would be ~$X/month (estimate based on usage)

- [ ] "How accurate is the model?"
  - **Answer:** 96.5% test accuracy, confidence scores help flag uncertain predictions

**Process Questions:**

- [ ] "How long did this take to build?"
  - **Answer:** ~5 weeks (show development timeline from DEVELOPMENT.md)

- [ ] "What was the hardest part?"
  - **Answer:** [Be honest - maybe WSL networking, or checkpoint logic, or monitoring setup]

- [ ] "What would you do differently?"
  - **Answer:** [Could mention starting with Docker earlier, or more testing, etc.]

### 3. Know Your Numbers

Memorize these metrics:

- [ ] Test accuracy: **96.53%**
- [ ] Dataset size: **2,500 images** (5 classes × 500 each)
- [ ] Training time: **~12 minutes** (Colab T4 GPU)
- [ ] Inference speed: **~5 seconds per batch** (10 images on CPU)
- [ ] Model size: **50 MB**
- [ ] Code lines: ~**1,500 lines** (estimate with `cloc .`)

---

## 📝 Documentation Final Review

### README.md

- [ ] Quick Start instructions are accurate
- [ ] All download links work
- [ ] Commands have correct syntax
- [ ] Troubleshooting section is helpful

### ARCHITECTURE.md

- [ ] Diagrams are clear and accurate
- [ ] Design decisions are well-explained
- [ ] Scalability discussion is realistic

### DEVELOPMENT.md

- [ ] Timeline reflects actual work done
- [ ] Technology choices are justified
- [ ] Lessons learned are honest and insightful

### ASSETS.md

- [ ] Download instructions are clear
- [ ] Verification steps work
- [ ] Troubleshooting covers common issues

---

## 🚀 Day-Of Defense Checklist

**Morning of defense:**

- [ ] Fully charge laptop
- [ ] Test WiFi/projector connection if presenting in person
- [ ] Have backup plan (recorded video of demo)
- [ ] Start all services before presentation:
  ```bash
  # Terminal 1: MLflow
  mlflow-start
  
  # Terminal 2: Model Service  
  cd model-service && uvicorn app:app --host 0.0.0.0 --port 8000
  
  # Terminal 3: Monitoring
  cd monitoring && docker-compose up -d
  
  # Terminal 4: Streamlit (when ready to demo)
  streamlit run streamlit-ui/app.py
  ```

- [ ] Pre-load Grafana dashboard in browser tab
- [ ] Pre-load MLflow UI in browser tab
- [ ] Have sample images ready in a folder
- [ ] Close unnecessary browser tabs/applications

**During defense:**

- [ ] Speak clearly and confidently
- [ ] Don't rush through the demo
- [ ] If something breaks, explain what should have happened
- [ ] Refer to documentation when needed
- [ ] Admit when you don't know something
- [ ] Emphasize production patterns (monitoring, versioning, recovery)

---

## ✅ Final Verification

**The night before:**

- [ ] Run complete system end-to-end one final time
- [ ] All services start without errors
- [ ] Demo images classify correctly
- [ ] Grafana shows metrics
- [ ] Documentation links are correct
- [ ] Code is clean (no commented-out debug statements)
- [ ] Git repository is up to date
- [ ] Backup of entire project folder created

**You're ready! 🎓**

---

## 📞 Emergency Contacts

If something goes wrong during defense:

**Technical Issues:**
- Model won't load → Check MLflow is running, production alias is set
- No metrics → Check Prometheus targets, Model Service /metrics endpoint
- Streamlit crashes → Restart: `streamlit run streamlit-ui/app.py`

**Fallback Plan:**
- If live demo fails → Show screenshots
- If WiFi fails → Run everything locally (already setup for this)
- If computer fails → Have slides exported as PDF on USB drive

---

## 🎊 Post-Defense

After successful defense:

- [ ] Celebrate! 🎉
- [ ] Upload final version to GitHub (with asset links)
- [ ] Consider writing a blog post about the project
- [ ] Add to portfolio/resume
- [ ] Thank anyone who helped (classmates, advisors)
- [ ] Reflect on what you learned

**Good luck with your defense!** You've built something impressive. Trust in the work you've done. 🚀