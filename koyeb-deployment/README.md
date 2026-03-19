# Koyeb Deployment

Two services: **model-service** (FastAPI) and **streamlit-ui** (Streamlit).
Deploy model-service first, then streamlit-ui pointing at it.

---

## Prerequisites

- Docker installed and running
- Docker Hub account (free) — hub.docker.com
- Koyeb account — koyeb.com

---

## Step 1 — Copy the model weights

From your project root, run:

```bash
cp models/v1/best_model.pth koyeb-deployment/model-service/best_model.pth
```

This bakes the model into the Docker image at build time.

---

## Step 2 — Build and push the model service

```bash
cd koyeb-deployment/model-service

docker build -t YOUR_DOCKERHUB_USERNAME/refund-model-service:latest .

docker push YOUR_DOCKERHUB_USERNAME/refund-model-service:latest
```

Replace `YOUR_DOCKERHUB_USERNAME` with your actual Docker Hub username.

---

## Step 3 — Deploy model service on Koyeb

1. Go to koyeb.com → **Create Service**
2. Choose **Docker** as the deployment method
3. Image: `YOUR_DOCKERHUB_USERNAME/refund-model-service:latest`
4. Port: `8000`
5. Click **Deploy**
6. Wait for it to go green, then copy the public URL — it looks like:
   `https://refund-model-service-yourorg.koyeb.app`

---

## Step 4 — Build and push the Streamlit UI

```bash
cd koyeb-deployment/streamlit-ui

docker build -t YOUR_DOCKERHUB_USERNAME/refund-streamlit-ui:latest .

docker push YOUR_DOCKERHUB_USERNAME/refund-streamlit-ui:latest
```

---

## Step 5 — Deploy Streamlit UI on Koyeb

1. Go to koyeb.com → **Create Service**
2. Choose **Docker**
3. Image: `YOUR_DOCKERHUB_USERNAME/refund-streamlit-ui:latest`
4. Port: `8501`
5. Under **Environment Variables**, add:
   - Key: `MODEL_SERVICE_URL`
   - Value: the URL from Step 3 (no trailing slash)
     e.g. `https://refund-model-service-yourorg.koyeb.app`
6. Click **Deploy**

---

## Step 6 — Verify

Open the Streamlit URL. The sidebar should show **✅ Model Service: Online**.

Upload any product image and click **Run Classification**.

---

## Troubleshooting

**Model service sidebar shows offline**
- Check the `MODEL_SERVICE_URL` env var has no trailing slash
- Check the model service is green on Koyeb dashboard

**Build fails on `best_model.pth`**
- Make sure you ran the `cp` command in Step 1 before building

**Koyeb free tier goes to sleep**
- Free instances sleep after 1 hour of inactivity — first request after sleep
  takes ~30 seconds. This is normal on the free tier.